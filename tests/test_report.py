import json
import tempfile
import unittest
from pathlib import Path

import report


class LoadRunsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_runs_dir = report.RUNS_DIR
        report.RUNS_DIR = Path(self.temp_dir.name)

    def tearDown(self):
        report.RUNS_DIR = self.original_runs_dir
        self.temp_dir.cleanup()

    def test_loads_only_run_summaries(self):
        (report.RUNS_DIR / "valid.json").write_text(
            json.dumps({"name": "model", "dataset": "ledgar", "accuracy": 0.8})
        )
        (report.RUNS_DIR / "other.json").write_text(json.dumps({"with_label_list": 1.0}))
        (report.RUNS_DIR / "broken.json").write_text("not json")

        rows = report.load_runs()

        self.assertEqual(rows, [{"name": "model", "dataset": "ledgar", "accuracy": 0.8}])


if __name__ == "__main__":
    unittest.main()
