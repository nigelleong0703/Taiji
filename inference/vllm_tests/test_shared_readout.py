"""Readout correctness across cached prefixes, chunks and mixed batches."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vllm_patch"))
from taiji_readout import TaijiHead, TaijiReadout, validate_spec


class SharedReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.head = TaijiHead(12, dim=8).eval()
        self.head.gate.data.fill_(0.7)
        self.hidden = torch.randn(18, 12)
        self.spec = {"positions": [8, 12, 15], "query_index": 17, "num_tokens": 18,
                     "keys": ["a", "b", "c"]}
        self.params = SimpleNamespace(extra_args={"taiji_readout": self.spec}, n=1)
        self.requests = {"decision": SimpleNamespace(sampling_params=self.params,
                                                     prompt_token_ids=list(range(18)))}

    def collect(self, readout, start, end):
        batch = SimpleNamespace(req_ids=["decision"], num_computed_tokens_cpu=[start])
        return readout.collect(self.hidden[start:end], batch, self.requests, [0, end-start], [end-start])

    def expected(self):
        with torch.inference_mode():
            return self.head(self.hidden[[8, 12, 15]], self.hidden[17].expand(3, -1))

    def test_cached_prefix_and_chunked_prefill(self):
        readout = TaijiReadout(self.head)
        scores, selected = self.collect(readout, 8, 11)
        self.assertEqual(scores, {})
        self.assertEqual(selected, {"decision"})
        self.collect(readout, 11, 14)
        scores, _ = self.collect(readout, 14, 18)
        torch.testing.assert_close(torch.tensor(scores["decision"]), self.expected())
        self.assertEqual(readout.rows, {})

    def test_mixed_batch_offsets_and_no_cross_request_leak(self):
        readout = TaijiReadout(self.head)
        self.requests["text"] = SimpleNamespace(sampling_params=SimpleNamespace(extra_args=None))
        batch = SimpleNamespace(req_ids=["text", "decision"], num_computed_tokens_cpu=[0, 8])
        packed = torch.cat([torch.randn(5, 12), self.hidden[8:]])
        scores, selected = readout.collect(packed, batch, self.requests, [0, 5, 15], [5, 10])
        torch.testing.assert_close(torch.tensor(scores["decision"]), self.expected())
        self.assertEqual(selected, {"decision"})

    def test_missing_cached_marker_raises(self):
        with self.assertRaisesRegex(RuntimeError, "missing marker states"):
            self.collect(TaijiReadout(self.head), 10, 18)

    def test_preemption_discards_old_rows(self):
        readout = TaijiReadout(self.head)
        self.collect(readout, 8, 14)
        self.hidden[8:14] += 2
        self.collect(readout, 8, 14)
        scores, _ = self.collect(readout, 14, 18)
        torch.testing.assert_close(torch.tensor(scores["decision"]), self.expected())

    def test_invalid_indices_and_prompt_are_rejected(self):
        for replacement in ({"positions": [-1]}, {"positions": []}, {"query_index": 5},
                            {"num_tokens": 19}, {"keys": ["a"]}, {"positions": [True]}):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                validate_spec({**self.spec, **replacement}, list(range(18)))

    def test_validate_spec_accepts_a_prompt_length(self):
        """The V2 runner only exposes the prompt length, not the token ids."""
        validate_spec(self.spec, None, 18)
        for prompt_len in (19, None):
            with self.subTest(prompt_len=prompt_len), self.assertRaises(ValueError):
                validate_spec(self.spec, None, prompt_len)

    def v2_batch(self, computed, scheduled, start=0):
        return SimpleNamespace(req_ids=["decision"], num_computed_tokens_np=[computed],
                               num_scheduled_tokens=[scheduled], query_start_loc_np=[start])

    def test_v2_runner_collects_across_chunks(self):
        """The V2 runner reads the same rows off InputBatch absolute offsets."""
        readout = TaijiReadout(self.head)
        readout.register("decision", self.spec)
        scores, selected = readout.collect_v2(self.hidden[8:11], self.v2_batch(8, 3))
        self.assertEqual(scores, {})
        self.assertEqual(selected, {"decision"})
        readout.collect_v2(self.hidden[11:14], self.v2_batch(11, 3))
        scores, _ = readout.collect_v2(self.hidden[14:18], self.v2_batch(14, 4))
        torch.testing.assert_close(torch.tensor(scores["decision"]), self.expected())
        self.assertEqual(readout.rows, {})
        self.assertEqual(readout.specs, {})

    def test_v2_mixed_batch_offsets(self):
        readout = TaijiReadout(self.head)
        readout.register("decision", self.spec)
        batch = SimpleNamespace(req_ids=["text", "decision"],
                                num_computed_tokens_np=[0, 8],
                                num_scheduled_tokens=[5, 10],
                                query_start_loc_np=[0, 5])
        packed = torch.cat([torch.randn(5, 12), self.hidden[8:]])
        scores, selected = readout.collect_v2(packed, batch)
        torch.testing.assert_close(torch.tensor(scores["decision"]), self.expected())
        self.assertEqual(selected, {"decision"})

    def test_v2_ignores_batches_without_readouts(self):
        readout = TaijiReadout(self.head)
        batch = SimpleNamespace(req_ids=["text"], num_computed_tokens_np=[0],
                                num_scheduled_tokens=[5], query_start_loc_np=[0])
        scores, selected = readout.collect_v2(torch.randn(5, 12), batch)
        self.assertEqual(scores, {})
        self.assertEqual(selected, set())


if __name__ == "__main__":
    unittest.main()
