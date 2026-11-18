"""Offline checks for order filtering, score fitting, and resumable VLM calls."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import evaluate_vlm as vlm


class VlmProtocolChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name, color in (("reference", "white"), ("a_source", "red"), ("a_near", "blue"),
                            ("b_source", "green"), ("b_near", "yellow")):
            Image.new("RGB", (32, 24), color).save(self.root / f"{name}.png")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def args(self, candidate: list[list[str]]) -> argparse.Namespace:
        return argparse.Namespace(
            reference=self.root / "reference.png", candidate=candidate, output=self.root / "result.json",
            model=["mock-judge"], base_url="https://example.invalid/api/v1", api_key_env="VLMTOKEN_TEST",
            tile_width=128, timeout=1.0, http_retries=0, parse_retries=0,
            preview_only=False, overwrite=False,
        )

    def test_order_consistency_tie_and_resume(self) -> None:
        candidates = [["method_a", str(self.root / "a_source.png"), str(self.root / "a_near.png")],
                      ["method_b", str(self.root / "b_source.png"), str(self.root / "b_near.png")]]
        answers = iter(["A", "tie", "B", "B", "tie", "B"])
        calls = []

        def answer(*unused):
            calls.append(True)
            return json.dumps({"winner": next(answers), "reason": "image evidence"}), {"response_model": "mock-judge"}

        with patch.dict("os.environ", {"VLMTOKEN_TEST": "test-value"}), patch.object(vlm, "call_vlm", side_effect=answer):
            report = vlm.run(self.args(candidates))
        self.assertEqual(len(calls), 6)
        self.assertEqual(report["status"], "complete")
        self.assertAlmostEqual(report["scores"]["completeness"]["mean_centered_bt_scores"]["method_a"], math.log(3) / 2)
        self.assertEqual(report["scores"]["layout"]["mean_centered_bt_scores"], {"method_a": 0.0, "method_b": 0.0})
        self.assertIsNone(report["scores"]["appearance"]["mean_centered_bt_scores"])
        self.assertEqual(report["scores"]["appearance"]["discarded_order_inconsistent_pair_judgments"], 1)
        with patch.dict("os.environ", {"VLMTOKEN_TEST": "test-value"}), patch.object(vlm, "call_vlm", side_effect=AssertionError("Should resume")):
            resumed = vlm.run(self.args(candidates))
        self.assertEqual(resumed["scores"], report["scores"])

    def test_single_candidate_is_not_bt(self) -> None:
        candidates = [["method_a", str(self.root / "a_source.png"), str(self.root / "a_near.png")]]
        answers = iter([4, 3, 2])

        def answer(*unused):
            return json.dumps({"score": next(answers), "reason": "image evidence"}), {}

        with patch.dict("os.environ", {"VLMTOKEN_TEST": "test-value"}), patch.object(vlm, "call_vlm", side_effect=answer):
            report = vlm.run(self.args(candidates))
        self.assertEqual(report["manifest"]["protocol"], "single_candidate_supplemental_rubric")
        self.assertEqual(report["scores"]["completeness"]["mean_0_to_5"], 4)
        self.assertEqual(report["scores"]["appearance"]["mean_0_to_5"], 2)


if __name__ == "__main__":
    unittest.main()
