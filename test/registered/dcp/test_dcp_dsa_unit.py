"""CPU unit test for the DSA decode-context-parallel (DCP) helpers.

Simulates a W-rank DCP group on one process: each rank's ``all_gather`` sees
the tensors every rank sent, so the helpers in ``layers/dcp/dsa.py`` can be
checked against a single-rank reference built from the full sequence.

Usage:
    python -m pytest test_dcp_dsa_unit.py -v
    python test_dcp_dsa_unit.py
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.dcp import dsa as dcp_dsa
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeGroup:
    """``all_gather`` along dim 0, fed by the per-rank sends of a previous pass."""

    def __init__(self, gathered=None):
        self.gathered = gathered
        self.sent = []

    def all_gather(self, t, dim=0):
        assert dim == 0
        self.sent.append(t.clone())
        if self.gathered is None:
            return t.repeat((len(self._world),) + (1,) * (t.dim() - 1))
        return torch.cat([g[len(self.sent) - 1] for g in self.gathered], dim=0)


def _parallel(w, r, group=None):
    return SimpleNamespace(
        dcp_enabled=w > 1, attn_dcp_size=w, attn_dcp_rank=r, dcp_group=group
    )


def _run_collective(w, fn):
    """Run ``fn(rank)`` on every rank twice: once to record sends, once to gather."""
    sends = []
    for r in range(w):
        group = _FakeGroup()
        group._world = range(w)
        with patch.object(dcp_dsa, "get_parallel", return_value=_parallel(w, r, group)):
            fn(r)
        sends.append(group.sent)
    outs = []
    for r in range(w):
        group = _FakeGroup(gathered=sends)
        with patch.object(dcp_dsa, "get_parallel", return_value=_parallel(w, r, group)):
            outs.append(fn(r))
    return outs


def _widened_slots(seq_len, w, page_size, first_page):
    """Widened slots of one request: pages of page_size * W, page-aligned start."""
    span = page_size * w
    pos = torch.arange(seq_len)
    return (first_page + pos // span) * span + pos % span


class TestDcpLocalize(CustomTestCase):
    def test_write_loc_owner_rule(self):
        loc = torch.arange(0, 40)
        for w in (2, 4, 8):
            for r in range(w):
                with patch.object(
                    dcp_dsa, "get_parallel", return_value=_parallel(w, r)
                ):
                    out = dcp_dsa.dcp_localize_write_loc(loc)
                owned = loc % w == r
                self.assertTrue(torch.equal(out[owned], loc[owned] // w))
                self.assertTrue(torch.all(out[~owned] == 0))

    def test_write_loc_identity_without_dcp(self):
        loc = torch.arange(7)
        with patch.object(dcp_dsa, "get_parallel", return_value=_parallel(1, 0)):
            self.assertIs(dcp_dsa.dcp_localize_write_loc(loc), loc)

    def test_local_index_block_table(self):
        for page_size in (1, 4, 64):
            w = 4
            slots = _widened_slots(5 * page_size * w, w, page_size, first_page=3)
            table = slots.view(1, -1)
            with patch.object(dcp_dsa, "get_parallel", return_value=_parallel(w, 1)):
                bt, cap = dcp_dsa.dcp_local_index_block_table(table, page_size)
            self.assertEqual(bt.tolist(), [[3, 4, 5, 6, 7]])
            self.assertEqual(cap, 5 * page_size)
            # Local token j of rank r is global position j*W + r, at row slot // W.
            for r in range(w):
                pos = torch.arange(r, slots.numel(), w)
                rows = slots[pos] // w
                j = pos // w
                self.assertTrue(
                    torch.equal(rows, bt[0, j // page_size] * page_size + j % page_size)
                )


def _shuffled_page_table(rows, local_width, w, g):
    """Widened pages of W slots in shuffled order: position p of row b sits at
    slot page * W + p % W, so its owner is p % W."""
    pages = torch.randperm(rows * local_width, generator=g) + 1
    pos = torch.arange(local_width * w)
    return torch.stack(
        [pages[b * local_width + pos // w] * w + pos % w for b in range(rows)]
    ).int()


class TestDcpExchangeTopk(CustomTestCase):
    def _check(self, w, seq_lens, topk, seed=0, device="cpu"):
        g = torch.Generator().manual_seed(seed)
        rows = len(seq_lens)
        max_len = max(seq_lens)
        global_logits = torch.randn((rows, max_len), generator=g)
        lens = torch.tensor(seq_lens, dtype=torch.int32)
        backend = DSATopKBackend.TORCH if device == "cpu" else DSATopKBackend.SGL_KERNEL

        def topk_func(score, lengths, k):
            # Each rank runs twice here (record sends, then gather), and the merge
            # reads this rank's local top-k by its position in the send. Pin
            # fast_topk_v2's unspecified output order so both runs agree.
            return backend.topk_func(score, lengths, k).sort(dim=1).values

        local_width = (max_len + w - 1) // w
        page_table = _shuffled_page_table(rows, local_width, w, g)

        def rank_fn(r):
            local = torch.full((rows, local_width), float("nan"))
            pos = torch.arange(r, max_len, w)
            local[:, : pos.numel()] = global_logits[:, pos]
            local_lens = torch.clamp((lens - r + w - 1) // w, min=0).int()
            out = dcp_dsa.dcp_exchange_topk(
                local.to(device),
                local_lens.to(device),
                topk,
                topk_func,
                page_table.to(device),
            )
            return out.rows.cpu(), out.lens.cpu()

        outs = _run_collective(w, rank_fn)
        for b, n in enumerate(seq_lens):
            k = min(topk, n)
            ref_pos = torch.topk(global_logits[b, :n], k).indices
            n_owned = 0
            for r, (local_rows, local_lens) in enumerate(outs):
                self.assertEqual(local_rows.shape, (rows, topk))
                owned = ref_pos[ref_pos % w == r]
                ref = (page_table[b, owned] // w).sort().values
                m = int(local_lens[b])
                # Selection order is unspecified (fast_topk_v2 is unordered).
                got = local_rows[b, :m].sort().values
                self.assertTrue(torch.equal(got, ref), f"row {b} rank {r}")
                self.assertTrue(torch.all(local_rows[b, m:] == -1))
                n_owned += m
            self.assertEqual(n_owned, k)

    def test_long_rows(self):
        self._check(w=4, seq_lens=[300, 257, 1000], topk=64)

    def test_short_rows_padded(self):
        # Rows shorter than topk and a local width below topk exercise the pad.
        self._check(w=4, seq_lens=[1, 3, 10, 70], topk=32)

    def test_dcp8(self):
        self._check(w=8, seq_lens=[2048, 5, 4097], topk=128, seed=1)

    @unittest.skipUnless(torch.cuda.is_available(), "Triton kernels need a GPU")
    def test_gpu_kernels(self):
        # fast_topk_v2 only supports topk == 2048 (the DSA index_topk).
        self._check(w=4, seq_lens=[300, 2047, 1000, 3], topk=2048, device="cuda")
        self._check(w=4, seq_lens=[9000, 20, 2049], topk=2048, device="cuda")
        self._check(w=8, seq_lens=[2048, 5, 40000], topk=2048, seed=1, device="cuda")

    def test_short_context_selects_all(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for w, seq_lens in ((4, [1, 3, 2048, 700]), (8, [2045, 9])):
                topk = 2048
                g = torch.Generator().manual_seed(w)
                page_table = _shuffled_page_table(len(seq_lens), topk // w, w, g)
                lens = torch.tensor(seq_lens, dtype=torch.int32)
                for r in range(w):
                    with patch.object(
                        dcp_dsa, "get_parallel", return_value=_parallel(w, r)
                    ):
                        out = dcp_dsa.dcp_owned_topk_all(
                            lens.to(device), topk, page_table.to(device)
                        )
                    for b, n in enumerate(seq_lens):
                        ref = page_table[b, r:n:w] // w
                        m = int(out.lens[b])
                        self.assertEqual(m, ref.numel(), f"{device} row {b} rank {r}")
                        self.assertTrue(torch.equal(out.rows[b, :m].cpu(), ref))
                        self.assertTrue(torch.all(out.rows[b, m:] == -1))


class _FakeIndexPool:
    """Index-K store: ``get_index_k_scale_buffer`` gathers rows by page table."""

    def __init__(self, k, scale, page_size):
        self.k, self.scale, self.page_size = k, scale, page_size

    def get_index_k_scale_buffer(
        self, layer_id, seq_lens, block_tables, total, max_len
    ):
        ks, ss = [], []
        for b, n in enumerate(seq_lens.tolist()):
            j = torch.arange(n)
            rows = block_tables[b, j // self.page_size].long() * self.page_size
            rows += j % self.page_size
            ks.append(self.k[rows])
            ss.append(self.scale[rows])
        k, s = torch.cat(ks), torch.cat(ss)
        assert k.shape[0] == total
        return k, s


class TestDcpGatherIndexKPrefill(CustomTestCase):
    def _check(self, w, page_size, seq_lens):
        g = torch.Generator().manual_seed(0)
        span = page_size * w
        pages = [(n + span - 1) // span for n in seq_lens]
        width = max(pages) * span
        first_pages = [1 + sum(pages[:b]) for b in range(len(seq_lens))]
        table = torch.full((len(seq_lens), width), -1, dtype=torch.int64)
        for b, n in enumerate(seq_lens):
            table[b, : pages[b] * span] = _widened_slots(
                pages[b] * span, w, page_size, first_pages[b]
            )
        total_slots = (sum(pages) + 1) * span
        global_k = torch.randint(0, 255, (total_slots, 128), generator=g).to(
            torch.uint8
        )
        global_s = torch.randn((total_slots, 1), generator=g)

        def rank_fn(r):
            # Rank r's local store holds the slots it owns, at row slot // W.
            owned = torch.arange(r, total_slots, w)
            pool = _FakeIndexPool(global_k[owned], global_s[owned], page_size)
            lens = torch.tensor(seq_lens, dtype=torch.int32)
            return dcp_dsa.dcp_gather_index_k_prefill(pool, 0, lens, lens, table)

        outs = _run_collective(w, rank_fn)
        ref_slots = torch.cat([table[b, :n] for b, n in enumerate(seq_lens)])
        for k, s in outs:
            self.assertTrue(torch.equal(k, global_k[ref_slots]))
            self.assertTrue(torch.equal(s, global_s[ref_slots]))

    def test_page1(self):
        self._check(w=4, page_size=1, seq_lens=[13, 4, 1, 9])

    def test_page64(self):
        self._check(w=4, page_size=64, seq_lens=[300, 17, 513])

    def test_dcp8_uneven(self):
        self._check(w=8, page_size=4, seq_lens=[3, 33, 64, 65])


def _aiter_gfx95():
    if not (torch.cuda.is_available() and torch.version.hip):
        return False
    from sglang.srt.utils import is_gfx95_supported

    return is_gfx95_supported()


class TestDcpQrepAbsorb(CustomTestCase):
    @unittest.skipUnless(_aiter_gfx95(), "fp8 absorb kernel needs aiter on gfx950")
    def test_fp8_w_kc_qrep_matches_per_rank(self):
        from sglang.srt.layers.quantization.fp8_utils import input_to_float8
        from sglang.srt.models.deepseek_common.attention_forward_methods import (
            forward_mla_rocm as rocm,
        )

        torch.manual_seed(0)
        w, heads, nope, lora = 4, 16, 192, 512
        ranks = []
        for r in range(w):
            wk = torch.randn(heads, nope, lora, device="cuda") * (0.02 * (1 + r % 2))
            wq, s = input_to_float8(wk.bfloat16(), dtype=torch.float8_e4m3fn)
            ranks.append(
                SimpleNamespace(
                    w_kc=wq.transpose(1, 2).contiguous().transpose(1, 2), w_scale=s
                )
            )

        def rank_fn(r):
            attn = SimpleNamespace(w_kc=ranks[r].w_kc, w_scale=ranks[r].w_scale)
            rocm.prepare_w_kc_qrep_fp8(attn, self._group)
            return attn

        sends = []
        for r in range(w):
            self._group = _FakeGroup()
            self._group._world = range(w)
            rank_fn(r)
            sends.append(self._group.sent)
        attns = []
        for r in range(w):
            self._group = _FakeGroup(gathered=sends)
            attns.append(rank_fn(r))

        q_nope = torch.randn(8, heads * w, nope, device="cuda", dtype=torch.bfloat16)
        per_rank = torch.cat(
            [
                rocm.rocm_absorb_q_bmm(
                    ranks[r],
                    q_nope[:, r * heads : (r + 1) * heads].contiguous(),
                    is_capture_mode=True,
                ).transpose(0, 1)
                for r in range(w)
            ],
            dim=1,
        ).float()
        for attn in attns:
            self.assertEqual(attn.w_kc_qrep.shape, (heads * w, nope, lora))
            self.assertTrue(attn.w_kc_qrep.transpose(-1, -2).is_contiguous())
            out = rocm.rocm_absorb_q_bmm_qrep(attn, q_nope).float()
            self.assertEqual(out.shape, (8, heads * w, lora))
            rel = (out - per_rank).norm() / per_rank.norm()
            self.assertLess(rel.item(), 3e-2)

    @unittest.skipUnless(_aiter_gfx95(), "fp8 absorb kernel needs aiter on gfx950")
    def test_project_before_merge_matches_merge_then_project(self):
        from sglang.srt.layers.quantization.fp8_utils import input_to_float8
        from sglang.srt.models.deepseek_common.attention_forward_methods import (
            forward_mla_rocm as rocm,
        )

        torch.manual_seed(0)
        w, heads, lora, v_dim, bs = 4, 16, 512, 256, 8
        w_vc = []
        for r in range(w):
            wv = torch.randn(heads, v_dim, lora, device="cuda") * (0.02 * (1 + r % 2))
            wq, s = input_to_float8(wv.bfloat16(), dtype=torch.float8_e4m3fn)
            w_vc.append((wq.transpose(1, 2), s))

        def rank_fn(r):
            attn = SimpleNamespace(
                w_vc=w_vc[r][0], w_scale=w_vc[r][1], use_deep_gemm_bmm=False
            )
            self.assertTrue(rocm.prepare_w_vc_dcp(attn, self._group))
            return attn

        sends = []
        for r in range(w):
            self._group = _FakeGroup()
            self._group._world = range(w)
            rank_fn(r)
            sends.append(self._group.sent)
        attns = []
        for r in range(w):
            self._group = _FakeGroup(gathered=sends)
            attns.append(rank_fn(r))

        # Rank r's partial latent output over all W * heads heads, and its LSE.
        o = torch.randn(w, bs, heads * w, lora, device="cuda").bfloat16()
        lse = torch.randn(w, bs, heads * w, device="cuda")
        weight = torch.softmax(lse, dim=0).unsqueeze(-1)
        merged = (weight * o.float()).sum(0)
        ref = torch.cat(
            [
                torch.bmm(
                    merged[:, q * heads : (q + 1) * heads].transpose(0, 1),
                    w_vc[q][0].float() * w_vc[q][1],
                ).transpose(0, 1)
                for q in range(w)
            ],
            dim=1,
        )
        for attn in attns:
            self.assertEqual(attn.w_vc_dcp.shape, (heads * w, lora, v_dim))
            self.assertTrue(attn.w_vc_dcp.transpose(-1, -2).is_contiguous())
        projected = torch.stack(
            [rocm.rocm_absorb_v_bmm_dcp(attns[r], o[r]).float() for r in range(w)]
        )
        self.assertEqual(projected.shape, (w, bs, heads * w, v_dim))
        out = (weight * projected).sum(0)
        # Merge-then-project through the same fp8 kernel with each rank's own
        # w_vc: the error budget PBM has to stay within.
        merge_first = torch.cat(
            [
                rocm.rocm_absorb_v_bmm_dcp(
                    SimpleNamespace(w_vc_dcp=w_vc[q][0], w_scale_vc_dcp=w_vc[q][1]),
                    merged[:, q * heads : (q + 1) * heads].bfloat16(),
                ).float()
                for q in range(w)
            ],
            dim=1,
        )
        rel = ((out - ref).norm() / ref.norm()).item()
        rel_base = ((merge_first - ref).norm() / ref.norm()).item()
        # The kernel takes one scalar w_scale, so the gathered w_vc is
        # requantized to one scale and the ranks with a 2x smaller scale here
        # lose precision; a layout bug would be off by O(1) instead.
        self.assertLess(rel, 1.5 * rel_base, (rel, rel_base))
        self.assertLess(rel, 5e-2)


class TestDcpPrefillPageTable(CustomTestCase):
    def test_rows_follow_indptr(self):
        seq_lens = torch.tensor([3, 1, 4])
        indptr = torch.tensor([0, 3, 4, 8], dtype=torch.int32)
        indices = torch.tensor([10, 11, 12, 20, 30, 31, 32, 33], dtype=torch.int32)
        table = dcp_dsa.dcp_prefill_page_table(indptr, indices, seq_lens, width=6)
        self.assertEqual(
            table.tolist(),
            [[10, 11, 12, 0, 0, 0], [20, 0, 0, 0, 0, 0], [30, 31, 32, 33, 0, 0]],
        )


if __name__ == "__main__":
    unittest.main()
