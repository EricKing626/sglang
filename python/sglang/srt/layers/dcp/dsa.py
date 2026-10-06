# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Decode context parallel (DCP) helpers for DSA sparse attention (GLM-5 / V3.2).

Owner rule matches the widened allocator: slot % W == rank, local row = slot // W.
"""

from typing import Callable, List, NamedTuple, Tuple

import torch

from sglang.kernels.ops.attention.dcp_kernels import (
    dcp_topk_merge,
    dcp_topk_owned_all,
    dcp_topk_pack,
)
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.runtime_context import get_parallel


class DcpOwnedTopk(NamedTuple):
    """This rank's share of the global top-k, as the DCP sparse decode reads it."""

    rows: torch.Tensor  # [bs, topk] int32 local KV rows, packed first, -1 tail
    lens: torch.Tensor  # [bs] int32 number of owned rows


def dcp_localize_write_loc(loc: torch.Tensor) -> torch.Tensor:
    """Widened write loc -> this rank's row; non-owned ids go to reserved row 0."""
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return loc
    w, r = parallel.attn_dcp_size, parallel.attn_dcp_rank
    return torch.where(loc % w == r, loc // w, torch.zeros_like(loc))


def dcp_local_index_block_table(page_table_1: torch.Tensor, page_size: int):
    """Per-request local index-K page table and its column capacity in tokens.

    One widened page (page_size * W slots) holds page_size local tokens per rank.
    """
    span = page_size * get_parallel().attn_dcp_size
    block_tables = (page_table_1[:, ::span] // span).to(torch.int32).contiguous()
    return block_tables, block_tables.shape[1] * page_size


def dcp_exchange_topk(
    local_logits: torch.Tensor,
    local_lens: torch.Tensor,
    topk: int,
    topk_func: Callable,
    page_table_1: torch.Tensor,
) -> DcpOwnedTopk:
    """Local top-k -> all-gather scores -> this rank's share of the global top-k.

    A token in the global top-k is in its owner's local top-k, so the merge is
    exact. ``page_table_1`` maps positions to widened slots; local position j of
    rank r is global position j * W + r.
    """
    parallel = get_parallel()
    w, r = parallel.attn_dcp_size, parallel.attn_dcp_rank
    rows = local_logits.shape[0]
    if local_logits.shape[1] < topk:
        local_logits = torch.nn.functional.pad(
            local_logits, (0, topk - local_logits.shape[1]), value=-float("inf")
        )
    local_idx = topk_func(local_logits, local_lens, topk)
    if local_logits.is_cuda:
        send = dcp_topk_pack(local_logits, local_idx, local_lens)
        recv = parallel.dcp_group.all_gather(send, dim=0)
        return DcpOwnedTopk(
            *dcp_topk_merge(recv, local_idx, local_lens, page_table_1, w, r, topk_func)
        )

    valid = (local_idx >= 0) & (local_idx < local_lens.view(rows, 1))
    safe_idx = torch.where(valid, local_idx, torch.zeros_like(local_idx)).long()
    send = torch.where(
        valid, local_logits.gather(1, safe_idx), torch.tensor(-float("inf"))
    ).float()
    recv = parallel.dcp_group.all_gather(send, dim=0)
    scores = recv.view(w, rows, topk).permute(1, 0, 2).reshape(rows, w * topk)
    _, pick = torch.topk(scores, topk, dim=1)
    col = pick - r * topk
    mine = (col >= 0) & (col < topk)
    col = col.clamp(0, topk - 1)
    owned = mine & valid.gather(1, col)
    pos = (safe_idx.gather(1, col) * w + r).clamp(max=page_table_1.shape[1] - 1)
    local_rows = torch.where(owned, page_table_1.long().gather(1, pos) // w, -1)
    order = torch.sort((~owned).int(), dim=1, stable=True).indices
    return DcpOwnedTopk(
        local_rows.gather(1, order).int(), owned.sum(dim=1, dtype=torch.int32)
    )


def dcp_owned_topk_all(
    seq_lens: torch.Tensor, topk: int, page_table_1: torch.Tensor
) -> DcpOwnedTopk:
    """The ``kv_len <= topk`` shortcut: every token is selected, so no exchange."""
    parallel = get_parallel()
    w, r = parallel.attn_dcp_size, parallel.attn_dcp_rank
    local_lens = get_dcp_lens(seq_lens, w, r).to(torch.int32)
    if page_table_1.is_cuda:
        return DcpOwnedTopk(*dcp_topk_owned_all(local_lens, page_table_1, topk, w, r))
    local = torch.arange(topk, device=page_table_1.device)
    owned = local < local_lens.view(-1, 1)
    pos = (local * w + r).clamp(max=page_table_1.shape[1] - 1)
    local_rows = torch.where(owned, page_table_1.long()[:, pos] // w, -1)
    return DcpOwnedTopk(local_rows.int(), local_lens)


def dcp_gather_index_k_prefill(
    pool,
    layer_id: int,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    page_table_1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rebuild the full-sequence index K (flat, per-request order) from DCP shards.

    Every rank reads ceil(len / W) local tokens per request so the all-gather is even.
    """
    parallel = get_parallel()
    w = parallel.attn_dcp_size
    block_tables, _ = dcp_local_index_block_table(page_table_1, pool.page_size)
    pad_lens_cpu = (seq_lens_cpu + w - 1) // w
    pad_sum = int(pad_lens_cpu.sum())
    k_fp8, k_scale = pool.get_index_k_scale_buffer(
        layer_id,
        ((seq_lens + w - 1) // w).to(torch.int32),
        block_tables,
        pad_sum,
        int(pad_lens_cpu.max()),
    )
    k_all = parallel.dcp_group.all_gather(k_fp8.contiguous().view(torch.uint8), dim=0)
    s_all = parallel.dcp_group.all_gather(k_scale.contiguous(), dim=0)
    src = _dcp_flat_gather_index(
        seq_lens_cpu.tolist(), pad_lens_cpu.tolist(), pad_sum, w
    )
    src = src.to(k_fp8.device, non_blocking=True)
    return k_all.index_select(0, src).view(k_fp8.dtype), s_all.index_select(0, src)


def _dcp_flat_gather_index(
    seq_lens: List[int], pad_lens: List[int], pad_sum: int, w: int
) -> torch.Tensor:
    """Row in the rank-major gathered buffer for each (request, position)."""
    parts, base = [], 0
    for seq_len, pad_len in zip(seq_lens, pad_lens):
        pos = torch.arange(seq_len, dtype=torch.int64)
        parts.append((pos % w) * pad_sum + base + pos // w)
        base += pad_len
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.int64)


def dcp_prefill_page_table(
    dcp_kv_indptr: torch.Tensor,
    dcp_kv_indices: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """[bs, width] position -> row of the all-gathered ``dcp_kv_buffer``."""
    bs = seq_lens_cpu.shape[0]
    device = dcp_kv_indices.device
    table = torch.zeros((bs, width), dtype=torch.int32, device=device)
    seq_lens = seq_lens_cpu.to(device=device, dtype=torch.int64)
    rows = torch.repeat_interleave(torch.arange(bs, device=device), seq_lens)
    cols = torch.arange(rows.shape[0], device=device) - dcp_kv_indptr[:-1].long()[rows]
    table[rows, cols] = dcp_kv_indices[: rows.shape[0]].to(torch.int32)
    return table
