"""Targeted tests using fabricated validation records, never study data."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from opal2.faithful_stopping import ARMS, evaluate_records, inspect_run


def record(epoch, physical=2., standardized=1.):
    return {
        "epoch": epoch,
        "arms": {
            arm: {"n": 96, "measurement": {
                "physical": {"overall": {"mse": physical}},
                "standardized": {"overall": {"mse": standardized}},
            }} for arm in ARMS
        },
    }


class FaithfulStoppingTests(unittest.TestCase):
    def test_not_enough_records(self):
        self.assertEqual(evaluate_records([])["decision"], "NEEDS_REVIEW")
        result = evaluate_records([record(epoch) for epoch in range(5, 30, 5)])
        self.assertEqual(result["decision"], "CONTINUE")
        self.assertFalse(result["eligible"])
        self.assertEqual(result["latest_epoch"], 25)

    def test_common_plateau_can_stop(self):
        result = evaluate_records([record(epoch) for epoch in range(5, 35, 5)])
        self.assertEqual(result["decision"], "STOP_ELIGIBLE")
        self.assertTrue(result["eligible"])
        for metrics in result["per_metric"].values():
            for state in metrics.values():
                self.assertEqual(state["last_significant_epoch"], 5)
                self.assertEqual(state["stale_checks"], 5)

    def test_one_arm_one_metric_improves_so_all_continue(self):
        records = [record(epoch) for epoch in range(5, 35, 5)]
        for index, row in enumerate(records):
            row["arms"]["J_JOINT"]["measurement"]["standardized"]["overall"]["mse"] = .99**index
        result = evaluate_records(records)
        self.assertFalse(result["eligible"])
        state = result["per_metric"]["J_JOINT"]["standardized"]
        self.assertEqual(state["last_significant_epoch"], 30)
        self.assertEqual(state["stale_checks"], 0)

    def test_small_improvements_accumulate_against_anchor(self):
        records = [record(epoch, physical=value) for epoch, value in
                   zip((5, 10, 15, 20), (100., 99.8, 99.6, 99.4))]
        before = evaluate_records(records[:3])["per_metric"]["J_JOINT"]["physical"]
        self.assertEqual(before["anchor"], 100.)
        self.assertEqual(before["stale_checks"], 2)
        after = evaluate_records(records)["per_metric"]["J_JOINT"]["physical"]
        self.assertEqual(after["anchor"], 99.4)
        self.assertEqual(after["last_significant_epoch"], 20)
        self.assertEqual(after["stale_checks"], 0)

    def test_exact_half_percent_counts_and_zero_does_not_reset(self):
        result = evaluate_records([record(5, 100.), record(10, 99.5)])
        self.assertEqual(result["per_metric"]["J_JOINT"]["physical"]["stale_checks"], 0)
        zeros = evaluate_records([record(epoch, 0., 0.) for epoch in range(5, 35, 5)])
        self.assertTrue(zeros["eligible"])

    def test_invalid_records_need_review(self):
        invalid = []
        invalid.append([record(10)])
        invalid.append([record(5), record(15)])
        invalid.append([record(5), record(5)])
        for value in (float("nan"), float("inf"), -1., None, True):
            invalid.append([record(5, physical=value)])
        for mutation in ("n", "arm", "m_f", "metric"):
            row = deepcopy(record(5))
            if mutation == "n":
                row["arms"]["J_JOINT"]["n"] = 95
            elif mutation == "arm":
                del row["arms"]["J_JOINT"]
            elif mutation == "m_f":
                row["arms"]["M_MEAN_ONLY"]["measurement"]["physical"]["overall"]["mse"] = 2.1
            else:
                del row["arms"]["J_JOINT"]["measurement"]["physical"]
            invalid.append([row])
        for rows in invalid:
            with self.subTest(rows=rows):
                result = evaluate_records(rows)
                self.assertEqual(result["decision"], "NEEDS_REVIEW")
                self.assertFalse(result["eligible"])

    def test_reader_sorts_existing_json_and_rejects_bad_history(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "validation"
            folder.mkdir()
            for epoch in (30, 5, 25, 10, 20, 15):
                (folder / f"epoch_{epoch:04d}.json").write_text(json.dumps(record(epoch)))
            before = {path.name: path.read_bytes() for path in folder.iterdir()}
            self.assertTrue(inspect_run(directory)["eligible"])
            self.assertEqual(before, {path.name: path.read_bytes() for path in folder.iterdir()})
            (folder / "epoch_0010.json").write_text("{")
            self.assertEqual(inspect_run(directory)["decision"], "NEEDS_REVIEW")

    def test_absolute_script_cli_emits_only_json(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "validation"
            folder.mkdir()
            (folder / "epoch_0005.json").write_text(json.dumps(record(5)))
            script = Path(__file__).resolve().parents[1] / "opal2" / "faithful_stopping.py"
            result = subprocess.run([sys.executable, str(script), "--output", directory],
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stderr, "")
            payload = json.loads(result.stdout)
            self.assertEqual(payload["decision"], "CONTINUE")
            self.assertEqual(payload["latest_epoch"], 5)


if __name__ == "__main__":
    unittest.main()
