from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurogolf.tasks import iter_task_paths, summarize_dataset, summarize_task  # noqa: E402


class TaskLoadingTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "data" / "task001.json").exists(), "local Kaggle data is not available")
    def test_dataset_has_400_task_files(self) -> None:
        paths = iter_task_paths(ROOT / "data")
        self.assertEqual(len(paths), 400)

    @unittest.skipUnless((ROOT / "data" / "task001.json").exists(), "local Kaggle data is not available")
    def test_task_001_summary_matches_schema(self) -> None:
        summary = summarize_task("001", ROOT / "data")
        self.assertEqual(summary["task_id"], 1)
        self.assertIn("train", summary["pair_counts"])
        self.assertIn("test", summary["pair_counts"])
        self.assertIn("arc-gen", summary["pair_counts"])

    @unittest.skipUnless((ROOT / "data" / "task001.json").exists(), "local Kaggle data is not available")
    def test_dataset_summary_validates_all_tasks(self) -> None:
        summary = summarize_dataset(ROOT / "data")
        self.assertEqual(summary["valid_tasks"], 400)
        self.assertEqual(summary["missing_task_ids"], [])
        self.assertEqual(summary["invalid_tasks"], [])


if __name__ == "__main__":
    unittest.main()
