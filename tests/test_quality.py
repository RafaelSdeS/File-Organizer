"""End-to-end clustering quality with the real embedding model.

Slow (loads intfloat/multilingual-e5-small, a ~470 MB download the first
time), so skipped unless RUN_QUALITY=1. Runs the analyzer with its default
settings on examples/demo and compares the folders it proposes with the
known topics in examples/demo_topics.tsv. The unit tests mock the model, so
this is what catches regressions like "pt+en folders split by language".
"""
import csv
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEMO = os.path.join(ROOT, "examples", "demo")
TOPICS = os.path.join(ROOT, "examples", "demo_topics.tsv")
# Set just below the measured value, so it fails on a real regression but
# not on numeric noise across platforms.
ARI_FLOOR = 0.8


@unittest.skipUnless(os.environ.get("RUN_QUALITY") == "1", "set RUN_QUALITY=1 to run (loads the real model)")
class TestDemoQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.document_analyzer.analyzer import DocumentAnalyzer

        with open(TOPICS, encoding="utf-8") as f:
            cls.truth = {row["file"]: row for row in csv.DictReader(f, delimiter="\t")}
        cls.analyzer = DocumentAnalyzer()  # defaults: what a user gets with no flags
        cls.folders = cls.analyzer.analyze_directory(DEMO)
        cls.folder_of = {path: folder for folder, paths in cls.folders.items() for path in paths}

    def test_folders_match_topics(self):
        from sklearn.metrics import adjusted_rand_score

        files = sorted(self.truth)
        # Each one-off is its own topic: grouping two of them is a mistake.
        topics = [self.truth[f]["topic"] if self.truth[f]["topic"] != "one-off" else f for f in files]
        ari = adjusted_rand_score(topics, [self.folder_of[f] for f in files])
        self.assertGreaterEqual(ari, ARI_FLOOR, f"folders: {self.folders}")

    def test_folders_are_not_split_by_language(self):
        """Every topic mixes pt and en, so any sizeable folder should too."""
        for folder, paths in self.folders.items():
            if len(paths) >= 4:
                languages = {self.truth[p]["language"] for p in paths}
                self.assertEqual(languages, {"pt", "en"}, f"{folder} is monolingual: {paths}")

    def test_report_renders(self):
        from src.document_analyzer.report import save_report

        with tempfile.TemporaryDirectory() as out_dir:
            saved = save_report(self.analyzer.last_run, self.folders, out_dir)
            self.assertEqual(sorted(os.path.basename(p) for p in saved),
                             ["k_selection.png", "map.png", "similarity.png"])


if __name__ == "__main__":
    unittest.main()
