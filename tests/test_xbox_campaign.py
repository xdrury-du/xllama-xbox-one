# Copyright (c) 2026 xllama contributors
# SPDX-License-Identifier: MIT
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("campaign", Path(__file__).parents[1] / "scripts/xbox-campaign.py")
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


def run(rate, **changes):
    metrics = {"decode_tokens": 1200, "decode_ms": 1200000 / rate,
               "sampled_peak_working_set_mib": 3000, "configured_gpu_layers": 0,
               "gpu_dispatches": 0, "active_kv_q8": False}
    metrics.update(changes)
    return {"xbox_metrics": metrics, "error": None}


class VerdictTests(unittest.TestCase):
    def test_speed_threshold_and_quality_are_required(self):
        self.assertTrue(CAMPAIGN.verdict("threads-4", run(2), run(2.2), True)[0])
        self.assertFalse(CAMPAIGN.verdict("threads-4", run(2), run(2.05), True)[0])
        self.assertFalse(CAMPAIGN.verdict("threads-4", run(2), run(3), False)[0])

    def test_gpu_fallback_and_missing_metrics_never_count_as_improvements(self):
        self.assertFalse(CAMPAIGN.verdict("gpu-8", run(2), run(4), True)[0])
        actual = run(4, configured_gpu_layers=8, gpu_dispatches=500)
        self.assertTrue(CAMPAIGN.verdict("gpu-8", run(2), actual, True)[0])
        self.assertFalse(CAMPAIGN.verdict("threads-4", {}, {}, True)[0])
        self.assertFalse(CAMPAIGN.verdict("threads-4", {"error": "failed"}, run(4), True)[0])

    def test_memory_improvement_requires_real_q8_and_bounded_speed_regression(self):
        self.assertFalse(CAMPAIGN.verdict("kv-q8", run(2), run(2, sampled_peak_working_set_mib=2500), True)[0])
        self.assertTrue(CAMPAIGN.verdict("kv-q8", run(2), run(1.95, sampled_peak_working_set_mib=2500, active_kv_q8=True), True)[0])
        self.assertFalse(CAMPAIGN.verdict("kv-q8", run(2), run(1.5, sampled_peak_working_set_mib=2500, active_kv_q8=True), True)[0])

    def test_microbatch_prefill_improvement_requires_matched_work(self):
        baseline = run(2, prefill_tokens=100, prefill_ms=1000)
        candidate = run(1.95, prefill_tokens=100, prefill_ms=800)
        self.assertTrue(CAMPAIGN.verdict("ubatch-256", baseline, candidate, True)[0])
        candidate["xbox_metrics"]["prefill_tokens"] = 1
        self.assertFalse(CAMPAIGN.verdict("ubatch-256", baseline, candidate, True)[0])

    def test_short_or_repetitive_answers_fail_the_long_response_gate(self):
        checks = CAMPAIGN.quality_checks("word " * 1100, 1500)
        self.assertFalse(CAMPAIGN.quality_passed(checks))
        self.assertFalse(CAMPAIGN.quality_checks("short", 2)["at_least_1024_output_tokens"])

    def test_repetitions_use_median_rates_and_preserve_failed_runs(self):
        summary = CAMPAIGN.aggregate([run(1), run(3), run(10)])
        metrics = summary["xbox_metrics"]
        self.assertAlmostEqual(metrics["decode_tokens"] / (metrics["decode_ms"] / 1000), 3)
        failed = CAMPAIGN.aggregate([run(1), {"error":"disconnected"}])
        self.assertIsNotNone(failed["error"])
        self.assertEqual(failed["xbox_metrics"], {})

    def test_stress_requires_length_all_sections_and_natural_completion(self):
        record = {"text":"\n".join(f"### {i}. Section" for i in range(1,25)) + "\nEND OF MANUAL",
                  "usage":{"completion_tokens":2500}, "finish_reason":"stop", "error":None}
        self.assertTrue(CAMPAIGN.quality_passed(CAMPAIGN.stress_checks(record)))
        record["finish_reason"] = "length"
        self.assertFalse(CAMPAIGN.quality_passed(CAMPAIGN.stress_checks(record)))


if __name__ == "__main__":
    unittest.main()
