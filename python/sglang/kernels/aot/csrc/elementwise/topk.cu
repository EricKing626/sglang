/**
 * @NOTE: This file is adapted from
 * https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py
 * We:
 * 1. adapt from tilelang to pure cuda
 * 2. optimize the performance a little
 * 3. fix the potential illegal memory access
 */
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace {

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;

#ifdef USE_ROCM
// On ROCm, the per-workgroup LDS budget depends on the target arch, so we inject a
// per-arch value from `setup_rocm.py` via `-DSGL_TOPK_DYNAMIC_SMEM_BYTES=...`.
#ifdef SGL_TOPK_DYNAMIC_SMEM_BYTES
constexpr size_t kSmem = static_cast<size_t>(SGL_TOPK_DYNAMIC_SMEM_BYTES);
#else
constexpr size_t kSmem = 48 * 1024;  // bytes
#endif
#else
// Reduced from 128KB to 32KB to improve occupancy.
// Each radix pass needs at most ~TopK candidates in the threshold bin,
// so 4K entries per round (2 rounds = 8K entries = 32KB) is sufficient.
constexpr size_t kSmem = 8 * 1024 * sizeof(uint32_t);  // 32KB (bytes)
#endif

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, TopK]
  int32_t* __restrict__ lengths;           // [B]
  int64_t input_stride;
};

// when length <= TopK, we can directly write the indices
__device__ void naive_topk_cuda(const float* __restrict__ score, int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform(
    const float* __restrict__ score,
    int32_t length,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform_ragged(
    const float* __restrict__ score, int32_t length, int32_t* __restrict__ topk_indices_ragged, int32_t offset) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    topk_indices_ragged[i] = (i < length) ? static_cast<int32_t>(i) + offset : -1;
  }
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Lanes per wave/warp, used by the deterministic tie-break scan below. On ROCm
// it can be injected per-arch from setup_rocm.py the same way the dynamic LDS
// budget is; it is not read from a target macro because this is a host+device
// translation unit and ROCm 7.0 deprecates both __AMDGCN_WAVEFRONT_SIZE__ and
// the constexpr `warpSize` variable.
#ifdef USE_ROCM
#ifdef SGL_TOPK_WAVE_SIZE
constexpr int kWaveSize = SGL_TOPK_WAVE_SIZE;
#else
constexpr int kWaveSize = 64;  // gfx9/CDNA, the default build target
#endif
#else
constexpr int kWaveSize = 32;
#endif

// Deterministic tie-breaking costs an extra ballot scan over the row, so it is
// opt-in. When it is off, ties among equal keys are resolved by whichever thread
// claims the slot first, which is what this kernel has always done.
#ifndef SGL_TOPK_DETERMINISTIC_TIES
#define SGL_TOPK_DETERMINISTIC_TIES 0
#endif
constexpr bool kDeterministicTies = SGL_TOPK_DETERMINISTIC_TIES != 0;

// Block-wide exclusive rank of a predicate, plus the block-wide total.
// Two-level ballot scan: no cub/rocPRIM dependency, and the only extra LDS is
// one counter per wave. Every thread of the block must reach this.
__device__ __forceinline__ void block_scan_flag(bool pred, int* __restrict__ s_wave, int& rank, int& total) {
  constexpr int kNumWaves = kThreadsPerBlock / kWaveSize;
  const int lane = static_cast<int>(threadIdx.x) & (kWaveSize - 1);
  const int wave = static_cast<int>(threadIdx.x) / kWaveSize;
#ifdef USE_ROCM
  const uint64_t ballot = __ballot(pred);
  const int lane_rank = __popcll(ballot & ((1ull << lane) - 1ull));
  const int wave_count = __popcll(ballot);
#else
  const uint32_t ballot = __ballot_sync(0xffffffffu, pred);
  const int lane_rank = __popc(ballot & ((1u << lane) - 1u));
  const int wave_count = __popc(ballot);
#endif
  if (lane == 0) s_wave[wave] = wave_count;
  __syncthreads();
  int prefix = 0;
  int sum = 0;
#pragma unroll
  for (int w = 0; w < kNumWaves; ++w) {
    const int c = s_wave[w];
    if (w < wave) prefix += c;
    sum += c;
  }
  rank = prefix + lane_rank;
  total = sum;
  __syncthreads();
}

__device__ void fast_topk_cuda_tl(const float* __restrict__ input, int* __restrict__ index, int row_start, int length) {
  // An optimized topk kernel copied from tilelang kernel, restructured to follow
  // the guess-verify-refine shape used by flashinfer's FilteredTopK:
  //   guess  - one 8-bit coarse histogram over the whole row picks the bin that
  //            straddles the top-k threshold;
  //   verify - a suffix scan tells us exactly how many winners are above that bin
  //            and how many still have to come out of it;
  //   refine - the straddling bin is descended one byte at a time.
  //
  // The survivor set of a refine round normally lives in an LDS buffer of fixed
  // capacity. A clustered row can put far more than that into the straddling
  // bin, and the surplus used to be dropped without any indication, so the
  // returned top-k was silently incomplete. The descent now keeps running
  // straight off the row when the survivors do not fit, and re-enters the buffer
  // as soon as they do: such a row costs extra passes rather than a wrong answer.
  //
  // We assume length > TopK here, or it will crash
  int topk = TopK;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;

  // A single shared histogram serialises hard: a wave puts 64 lanes on ~64 of
  // the 256 bins, and 256 ints cover the 32 LDS banks eight times over, so the
  // hardware replays each atomic once per colliding bank. Keeping several
  // copies of the histogram and picking one per lane fixes that, but only if
  // the copies sit in different banks - at a stride of RADIX every copy aliases
  // the same banks and the contention is exactly unchanged. Hence RADIX + 1.
  constexpr int kHistStride = RADIX + 1;
#ifndef SGL_TOPK_HIST_COPIES
#define SGL_TOPK_HIST_COPIES (kSmem >= 96 * 1024 ? 16 : (kSmem >= 24 * 1024 ? 8 : 4))
#endif
#ifndef SGL_TOPK_LOOK_COPIES
#define SGL_TOPK_LOOK_COPIES (kSmem >= 96 * 1024 ? 8 : (kSmem >= 24 * 1024 ? 4 : 2))
#endif
  constexpr int kHistCopies = SGL_TOPK_HIST_COPIES;
  constexpr int kLookCopies = SGL_TOPK_LOOK_COPIES;
  constexpr int kHistInts = kHistCopies * kHistStride;
  constexpr int kLookInts = kLookCopies * kHistStride;
  // The copies are carved out of the same dynamic allocation as the candidate
  // buffer, so the block's LDS footprint is unchanged and every architecture
  // keeps the occupancy it had before.
  constexpr int kBufTotal = static_cast<int>((kSmem - (kHistInts + kLookInts) * sizeof(int)) / sizeof(int));
  // The two buffers ping-pong, but they do not carry comparable loads: the
  // first holds the entire straddling bin while every later round holds what
  // survived a byte split of it. Splitting the space evenly therefore wastes
  // half of it on a set that has already collapsed, and pushes bins that would
  // otherwise have fit into the much more expensive off-row descent. A bin that
  // genuinely fails to shrink still lands there, and is still answered
  // correctly.
  constexpr int kBufCap0 = kBufTotal * 3 / 4;
  constexpr int kBufCap1 = kBufTotal - kBufCap0;
  // fp32 ordered keys are 4 bytes and the coarse key consumes none of them (it
  // comes from a lossy fp16 conversion), so the descent takes 4 rounds.
  constexpr int NUM_ROUNDS = 4;
  constexpr int FIRST_SHIFT = 24;

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  // Histogram of the second refine byte, gathered in the same pass as the first.
  // The coarse fp16 key is sign+exp+2 mantissa bits while the top fp32 byte is
  // sign+exp minus its low bit, so one coarse bin is eight times narrower than
  // one fp32 byte: round 0 nearly always has nothing left to split. When that
  // holds, this histogram already describes round 1 and its pass is skipped.
  alignas(128) __shared__ int s_hist_lookahead[RADIX + 1];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];
  alignas(128) __shared__ int s_last_remain;
  alignas(128) __shared__ int s_spill;
  alignas(128) __shared__ int s_eq_counter;
  alignas(128) __shared__ int s_wave_scan[kThreadsPerBlock / kWaveSize];

  auto& s_histogram = s_histogram_buf[0];
  extern __shared__ int s_dynamic[];
  int* const s_hist_copies = s_dynamic;
  int* const s_look_copies = s_dynamic + kHistInts;
  // allocate for two rounds
  int* const s_buf0 = s_dynamic + kHistInts + kLookInts;
  int* const s_buf1 = s_buf0 + kBufCap0;
  const auto buf_of = [&](int i) { return i == 0 ? s_buf0 : s_buf1; };
  const auto cap_of = [](int i) { return i == 0 ? kBufCap0 : kBufCap1; };

  const int tx = threadIdx.x;
  const float* const row = input + row_start;

  // Zeroing and folding the copies costs the same whether the pass counts four
  // thousand elements or a hundred thousand, so a short pass is charged for
  // contention it never had. Each pass therefore takes only as many copies as
  // its own element count justifies, which keeps short rows on the cheap path.
#ifndef SGL_TOPK_COPY_WORK
#define SGL_TOPK_COPY_WORK 2048
#endif
  constexpr int kCopyWork = SGL_TOPK_COPY_WORK;
  const auto copies_for = [](int work, int cap) {
    int c = 1;
    while (c < cap && work >= c * kCopyWork) c <<= 1;
    return c;
  };

  int hist_live = 1;
  int look_live = 1;
  int* hist_lane = s_hist_copies;
  int* look_lane = s_look_copies;
  const auto hist_begin = [&](int work) {
    hist_live = copies_for(work, kHistCopies);
    hist_lane = s_hist_copies + (tx & (hist_live - 1)) * kHistStride;
    for (int i = tx; i < hist_live * kHistStride; i += BLOCK_SIZE) s_hist_copies[i] = 0;
  };
  const auto look_begin = [&](int work) {
    look_live = copies_for(work, kLookCopies);
    look_lane = s_look_copies + (tx & (look_live - 1)) * kHistStride;
    for (int i = tx; i < look_live * kHistStride; i += BLOCK_SIZE) s_look_copies[i] = 0;
  };
  const auto hist_bump = [&](int bin) { ::atomicAdd(&hist_lane[bin], 1); };
  const auto look_bump = [&](int bin) { ::atomicAdd(&look_lane[bin], 1); };
  // Fold the copies back into the histogram the cumsum works on. Slot RADIX is
  // the suffix-scan sentinel and is always read one past the last bin.
  const auto hist_fold = [&] {
    if (tx < RADIX) {
      int sum = 0;
      for (int r = 0; r < hist_live; ++r) sum += s_hist_copies[r * kHistStride + tx];
      s_histogram[tx] = sum;
    } else if (tx == RADIX) {
      s_histogram[RADIX] = 0;
    }
  };
  const auto look_fold = [&] {
    if (tx < RADIX) {
      int sum = 0;
      for (int r = 0; r < look_live; ++r) sum += s_look_copies[r * kHistStride + tx];
      s_hist_lookahead[tx] = sum;
    } else if (tx == RADIX) {
      s_hist_lookahead[RADIX] = 0;
    }
  };

  // Visit every element of the row, four at a time. A single float per thread
  // per step leaves the memory pipeline mostly idle: one block owns a whole row,
  // so the only way to keep enough loads in flight is to widen them. The row
  // base is not guaranteed to be 16B aligned, so the unaligned head and the
  // ragged tail are walked one element at a time.
  const auto for_each_element = [&](auto&& fn) {
    const int head = static_cast<int>((16u - (reinterpret_cast<uintptr_t>(row) & 15u)) & 15u) / 4;
    const int lead = head < length ? head : length;
    for (int i = tx; i < lead; i += BLOCK_SIZE) fn(i, row[i]);
    const int n4 = (length - lead) / 4;
    const float4* const row4 = reinterpret_cast<const float4*>(row + lead);
    for (int i = tx; i < n4; i += BLOCK_SIZE) {
      const float4 v = row4[i];
      const int base = lead + i * 4;
      fn(base + 0, v.x);
      fn(base + 1, v.y);
      fn(base + 2, v.z);
      fn(base + 3, v.w);
    }
    for (int i = lead + n4 * 4 + tx; i < length; i += BLOCK_SIZE) fn(i, row[i]);
  };

  // stage 1: 8bit coarse histogram
  hist_begin(length);
  __syncthreads();

  for_each_element([&](int, float value) { hist_bump(convert_to_uint8(value)); });
  __syncthreads();
  hist_fold();
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  // Append a winner. Callers only ever emit at most `topk` of these, so the slot
  // is in range by construction; the check is here because a silent
  // out-of-bounds write is exactly the failure this rewrite is meant to remove.
  const auto emit_winner = [&](int idx) {
    const auto pos = ::atomicAdd(&s_counter, 1);
    if (C10_LIKELY(pos < TopK)) index[pos] = idx;
  };

  // Claim one of the remaining equal-key slots, filling backwards from the end
  // of the output so that it can never collide with the winners growing forward
  // from s_counter. Exactly `s_last_remain` of them are handed out.
  const auto claim_tie_slot = [&](int idx) {
    const auto pos = ::atomicAdd(&s_last_remain, -1);
    if (pos > 0) index[TopK - pos] = idx;
  };

  // Deterministic tie-break: fill the last `eq_needed` slots with the lowest
  // indices whose ordered key equals the pivot. The row is walked in tiles of
  // BLOCK_SIZE with thread tx owning idx = base + tx, so the emission order is
  // index order regardless of how the hardware schedules the waves. This one
  // stays scalar: the ballot scan ranks one flag per thread, and widening the
  // load would put four indices behind each flag.
  const auto collect_det_eq_pivot = [&](uint32_t pivot, int eq_needed) {
    if (eq_needed <= 0) return;
    if (tx == 0) s_eq_counter = 0;
    __syncthreads();
    for (int base = 0; base < length; base += BLOCK_SIZE) {
      const int idx = base + tx;
      const bool pred = (idx < length) && (convert_to_uint32(input[idx + row_start]) == pivot);
      int rank = 0;
      int total = 0;
      block_scan_flag(pred, s_wave_scan, rank, total);
      const int start = s_eq_counter;
      if (pred) {
        const int pos = start + rank;
        if (pos < eq_needed) index[TopK - eq_needed + pos] = idx;
      }
      __syncthreads();
      if (tx == 0) s_eq_counter = start + total;
      __syncthreads();
      if (s_eq_counter >= eq_needed) break;
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
    s_spill = 0;
  }
  __syncthreads();

  const int threshold_bin = s_threshold_bin_id;
  const int bin_count = s_histogram[threshold_bin] - s_histogram[threshold_bin + 1];
  topk -= s_histogram[threshold_bin + 1];

  if (topk == 0) {
    for_each_element([&](int idx, float value) {
      if (static_cast<int>(convert_to_uint8(value)) > threshold_bin) emit_winner(idx);
    });
    __syncthreads();
    return;
  }

  __syncthreads();
  hist_begin(bin_count);
  look_begin(bin_count);
  __syncthreads();

  // stage 2: collect the threshold bin. Both fused histograms count every element
  // of the bin, including the ones the buffer had no room for, so an overflowing
  // bin still leaves an exact picture of it behind.
  for_each_element([&](int idx, float value) {
    const auto bin = static_cast<int>(convert_to_uint8(value));
    if (bin > threshold_bin) {
      emit_winner(idx);
    } else if (bin == threshold_bin) {
      const auto key = convert_to_uint32(value);
      const auto pos = ::atomicAdd(&s_num_input[0], 1);
      if (C10_LIKELY(pos < kBufCap0)) {
        s_buf0[pos] = idx;
      } else {
        ::atomicOr(&s_spill, 1);
      }
      /// NOTE: (dark) fuse the histogram computation here
      hist_bump((key >> FIRST_SHIFT) & 0xFF);
      look_bump((key >> (FIRST_SHIFT - 8)) & 0xFF);
    }
  });
  __syncthreads();
  hist_fold();
  look_fold();
  __syncthreads();

  // stage 3: refine with 8bit radix passes
  int cur = 0;
  int num_input = s_num_input[0] < kBufCap0 ? s_num_input[0] : kBufCap0;
  bool from_row = s_spill != 0;
  uint32_t prefix = 0;
  uint32_t prefix_mask = 0;
  int survivors = bin_count;

  // Walk whatever holds the current survivor set: the LDS buffer, or the row
  // itself for a bin that never fit in it.
  const auto for_each_survivor = [&](auto&& fn) {
    if (from_row) {
      for_each_element([&](int idx, float value) {
        if (static_cast<int>(convert_to_uint8(value)) != threshold_bin) return;
        const auto key = convert_to_uint32(value);
        if ((key & prefix_mask) != prefix) return;
        fn(idx, key);
      });
    } else {
      const int* const buf = buf_of(cur);
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = buf[i];
        fn(idx, convert_to_uint32(row[idx]));
      }
    }
  };

#pragma unroll
  for (int round = 0; round < NUM_ROUNDS; ++round) {
    const int offset = FIRST_SHIFT - round * 8;

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_last_remain = topk - s_histogram[tx + 1];
      s_num_input[cur ^ 1] = 0;
      s_spill = 0;
    }
    __syncthreads();

    const int threshold = s_threshold_bin_id;
    const int next_survivors = s_histogram[threshold] - s_histogram[threshold + 1];
    topk -= s_histogram[threshold + 1];
    // Every thread has to finish reading the histogram before anyone below
    // overwrites it, either with zeros or with the lookahead.
    __syncthreads();

    if (topk == 0) {
      for_each_survivor([&](int idx, uint32_t key) {
        if (static_cast<int>((key >> offset) & 0xFF) > threshold) emit_winner(idx);
      });
      __syncthreads();
      return;
    }

    if (round == NUM_ROUNDS - 1) {
      // Last round: the key is fully resolved, so anything above the threshold
      // byte is an outright winner and everything still on it is a genuine tie.
      const uint32_t pivot = prefix | (static_cast<uint32_t>(threshold) << offset);
      for_each_survivor([&](int idx, uint32_t key) {
        const auto bin = static_cast<int>((key >> offset) & 0xFF);
        if (bin > threshold) {
          emit_winner(idx);
        } else if (bin == threshold && !kDeterministicTies) {
          claim_tie_slot(idx);
        }
      });
      __syncthreads();
      if (kDeterministicTies) collect_det_eq_pivot(pivot, topk);
      return;
    }

    const int next_offset = offset - 8;

    // Nothing was split off, so the survivor set is unchanged and the lookahead
    // histogram already describes the next byte over exactly this set. Skip the
    // partition pass entirely; this is the common case for round 0.
    if (round == 0 && next_survivors == survivors) {
      prefix |= static_cast<uint32_t>(threshold) << offset;
      prefix_mask |= 0xFFu << offset;
      if (tx < RADIX + 1) s_histogram[tx] = s_hist_lookahead[tx];
      __syncthreads();
      continue;
    }

    hist_begin(survivors);
    __syncthreads();

    for_each_survivor([&](int idx, uint32_t key) {
      const auto bin = static_cast<int>((key >> offset) & 0xFF);
      if (bin > threshold) {
        emit_winner(idx);
      } else if (bin == threshold) {
        const auto pos = ::atomicAdd(&s_num_input[cur ^ 1], 1);
        if (C10_LIKELY(pos < cap_of(cur ^ 1))) {
          buf_of(cur ^ 1)[pos] = idx;
        } else {
          ::atomicOr(&s_spill, 1);
        }
        /// NOTE: (dark) fuse the histogram computation here
        hist_bump((key >> next_offset) & 0xFF);
      }
    });
    __syncthreads();
    hist_fold();
    __syncthreads();

    // Only now that the round's winners are out may the filter tighten: until
    // this point `prefix` has to describe the set entering the round, because
    // the row scan uses it to rediscover exactly that set.
    prefix |= static_cast<uint32_t>(threshold) << offset;
    prefix_mask |= 0xFFu << offset;
    cur ^= 1;
    from_row = s_spill != 0;
    num_input = s_num_input[cur] < cap_of(cur) ? s_num_input[cur] : cap_of(cur);
    survivors = next_survivors;
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // topk
    void topk_kernel(const FastTopKParams params) {
  const auto& [input, row_starts, indices, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_cuda(score, indice, length);
  } else {
    return fast_topk_cuda_tl(score, indice, row_start, length);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // decode
    void topk_transform_decode_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride) {
  const auto& [input, _1, _2, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = 0;
  const auto length = lengths[bid];
  const auto src_page_entry = src_page_table + bid * src_stride;
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill
    void topk_transform_prefill_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride,
        const int32_t* __restrict__ cu_seqlens_q,
        const int64_t prefill_bs) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;

  /// NOTE: prefill bs is usually small, we can just use a simple loop here
  /// We ensure that last cu_seqlens is equal to number of blocks launched
  __shared__ const int32_t* s_src_page_entry;
  if (C10_LIKELY(prefill_bs <= kThreadsPerBlock)) {
    if (tid < prefill_bs) {
      if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid + 1]) {
        s_src_page_entry = src_page_table + tid * src_stride;
      }
    }
  } else {
    for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
      if (bid >= cu_seqlens_q[i] && bid < cu_seqlens_q[i + 1]) {
        s_src_page_entry = src_page_table + i * src_stride;
      }
    }
  }
  __syncthreads();
  const auto src_page_entry = s_src_page_entry;

  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill, ragged kv
    void topk_transform_prefill_ragged_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ topk_indices_ragged,
        const int32_t* __restrict__ topk_indices_offset) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto dst_indices_entry = topk_indices_ragged + bid * TopK;
  const auto score = input + bid * input_stride;
  const auto offset = topk_indices_offset[bid];

  if (length <= TopK) {
    return naive_topk_transform_ragged(score, length, dst_indices_entry, offset);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_indices_entry[idx_0] = pos_0 + offset;
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_indices_entry[idx_1] = pos_1 + offset;
  }
}

auto get_params(
    const at::Tensor& score,
    const at::Tensor& lengths,
    std::optional<at::Tensor> row_starts_opt = std::nullopt,
    std::optional<at::Tensor> indices_opt = std::nullopt) -> FastTopKParams {
  const auto B = score.size(0);
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1);
  if (row_starts_opt.has_value()) {
    const auto& row_starts = row_starts_opt.value();
    TORCH_CHECK(row_starts.dim() == 1);
    TORCH_CHECK(row_starts.size(0) == B);
  }
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous());
  TORCH_CHECK(lengths.size(0) == B);
  int32_t* indices_data_ptr = nullptr;
  if (indices_opt.has_value()) {
    const auto& indices = indices_opt.value();
    TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
    TORCH_CHECK(indices.size(0) == B);
    TORCH_CHECK(indices.size(1) == TopK);
    indices_data_ptr = indices.data_ptr<int32_t>();
  }

  return FastTopKParams{
      .input = score.data_ptr<float>(),
      .row_starts = row_starts_opt.has_value() ? row_starts_opt->data_ptr<int32_t>() : nullptr,
      .indices = indices_data_ptr,
      .lengths = lengths.data_ptr<int32_t>(),
      .input_stride = score.stride(0),
  };
}

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result = [] {
#ifdef USE_ROCM
    // hipify will turn cudaFuncSetAttribute -> hipFuncSetAttribute. On ROCm,
    // hipFuncSetAttribute expects `const void*` and hipcc does not accept passing
    // a function pointer directly, so cast explicitly.
    return ::cudaFuncSetAttribute(
        reinterpret_cast<const void*>(f), ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#else
    // CUDA: keep original behavior (no cast needed).
    return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
  }();
  TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}

}  // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void fast_topk_interface(
    const at::Tensor& score, at::Tensor& indices, const at::Tensor& lengths, std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(indices);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }
  CHECK_CUDA(lengths);
  const auto params = get_params(score, lengths, row_starts_opt, indices);
  const auto B = score.size(0);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  setup_kernel_smem_once<topk_kernel, kSmem>();
  topk_kernel<<<grid, block, kSmem, stream>>>(params);
  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(dst_page_table);
  CHECK_CUDA(src_page_table);
  CHECK_CUDA(cu_seqlens_q);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }
  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  TORCH_CHECK(dst_page_table.dim() == 2 && dst_page_table.is_contiguous());
  TORCH_CHECK(src_page_table.dim() == 2 && src_page_table.stride(1) == 1);
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous());
  const auto prefill_bs = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(dst_page_table.size(0) == B);
  TORCH_CHECK(dst_page_table.size(1) == TopK);
  TORCH_CHECK(src_page_table.size(0) == prefill_bs);
  TORCH_CHECK(prefill_bs <= B);  // prefill_bs should be smaller than expanded bs

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  const auto src_stride = src_page_table.stride(0);

  // dispatch to decode or prefill
  // extend and draft extend: row_starts_opt is not null, invokes the prefill kernel
  // decode: row_starts_opt is null, invokes the decode kernel
  // target verify: row_starts_opt is null, invokes the prefill kernel
  const auto is_decode = !row_starts_opt.has_value() && prefill_bs == B;
  if (is_decode) {
    setup_kernel_smem_once<topk_transform_decode_kernel, kSmem>();
    topk_transform_decode_kernel<<<grid, block, kSmem, stream>>>(
        params, dst_page_table.data_ptr<int32_t>(), src_page_table.data_ptr<int32_t>(), src_stride);
  } else {
    setup_kernel_smem_once<topk_transform_prefill_kernel, kSmem>();
    topk_transform_prefill_kernel<<<grid, block, kSmem, stream>>>(
        params,
        dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>(),
        src_stride,
        cu_seqlens_q.data_ptr<int32_t>(),
        prefill_bs);
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_ragged_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& topk_indices_ragged,
    const at::Tensor& topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(topk_indices_ragged);
  CHECK_CUDA(topk_indices_offset);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }

  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  TORCH_CHECK(topk_indices_ragged.dim() == 2 && topk_indices_ragged.is_contiguous());
  TORCH_CHECK(topk_indices_offset.dim() == 1);

  TORCH_CHECK(topk_indices_ragged.size(0) == B);
  TORCH_CHECK(topk_indices_ragged.size(1) == TopK);
  TORCH_CHECK(topk_indices_offset.size(0) == B);

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};

  setup_kernel_smem_once<topk_transform_prefill_ragged_kernel, kSmem>();
  topk_transform_prefill_ragged_kernel<<<grid, block, kSmem, stream>>>(
      params, topk_indices_ragged.data_ptr<int32_t>(), topk_indices_offset.data_ptr<int32_t>());

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}
