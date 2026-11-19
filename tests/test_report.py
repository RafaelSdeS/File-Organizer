import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_analyzer.report import _group_outline, save_report


class TestSaveReport(unittest.TestCase):
    def test_writes_all_three_charts(self):
        rng = np.random.default_rng(0)
        paths = [f"doc{i}.txt" for i in range(9)]
        last_run = {
            "paths": paths,
            "embeddings": rng.random((9, 5)),
            "k_scores": {
                "k": [2, 3, 4], "inertia": [9.0, 5.0, 4.0],
                "silhouette": [0.2, 0.4, 0.3], "davies_bouldin": [1.5, 0.9, 1.1],
                "votes": {"silhouette": 3, "davies_bouldin": 3, "elbow": 3}, "chosen": 3,
            },
        }
        structure = {"Alpha": paths[:4], "Beta": paths[4:8], "Misc": paths[8:]}

        with tempfile.TemporaryDirectory() as out_dir:
            saved = save_report(last_run, structure, out_dir)
            self.assertEqual([os.path.basename(p) for p in saved],
                             ["map.png", "k_selection.png", "similarity.png"])
            for path in saved:
                self.assertGreater(os.path.getsize(path), 0)

    def test_skips_k_chart_when_k_was_not_searched(self):
        """<= 2 documents short-circuit to k=1 with no scores to plot."""
        last_run = {"paths": ["a.txt", "b.txt"], "embeddings": np.eye(2), "k_scores": None}
        with tempfile.TemporaryDirectory() as out_dir:
            saved = save_report(last_run, {"Only": ["a.txt", "b.txt"]}, out_dir)
            self.assertEqual([os.path.basename(p) for p in saved], ["similarity.png"])

    def test_requires_a_run(self):
        with self.assertRaises(ValueError):
            save_report(None, {}, "unused")

    def test_group_outline_handles_degenerate_groups(self):
        """1 point, 2 points and collinear points have no convex hull; they
        must fall back to a stroke instead of raising QhullError."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            for points in ([[0, 0]], [[0, 0], [1, 1]], [[0, 0], [1, 1], [2, 2]],
                           [[0, 0], [1, 0], [0, 1], [1, 1]]):
                _group_outline(ax, np.array(points, dtype=float))
            # Every group drew something (a polygon or a stroke), none raised.
            self.assertEqual(len(ax.patches) + len(ax.lines), 4)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
