import logging
import os
import sys
import tempfile
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import unittest
from unittest.mock import patch

from src.document_analyzer import main as main_module


class TestMainCLI(unittest.TestCase):
    def setUp(self):
        # Point CONFIG_PATH at a nonexistent file inside a fresh temp dir by
        # default, so tests never depend on (or write to) the real
        # ~/.config/file-organizer.toml on the host running the tests.
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.config_path = Path(self.tmp_dir.name) / "file-organizer.toml"
        config_patcher = patch('src.document_analyzer.main.CONFIG_PATH', self.config_path)
        config_patcher.start()
        self.addCleanup(config_patcher.stop)

        patcher = patch('src.document_analyzer.main.DocumentAnalyzer')
        self.mock_analyzer_cls = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_analyzer = self.mock_analyzer_cls.return_value
        self.mock_analyzer.analyze_directory.return_value = {"Cluster_one": ["a.txt"]}

    def _run(self, argv):
        with patch('sys.argv', ["main.py"] + argv):
            main_module.main()

    def test_dry_run_skips_move(self):
        self._run(["some/dir", "--dry-run"])
        self.mock_analyzer.organize_files.assert_not_called()

    def test_declining_prompt_skips_move(self):
        with patch('builtins.input', return_value="n"):
            self._run(["some/dir"])
        self.mock_analyzer.organize_files.assert_not_called()

    def test_confirming_prompt_moves(self):
        with patch('builtins.input', return_value="y"):
            self._run(["some/dir"])
        self.mock_analyzer.organize_files.assert_called_once_with(
            {"Cluster_one": ["a.txt"]}, "some/dir", None
        )

    def test_yes_flag_skips_prompt(self):
        with patch('builtins.input', side_effect=AssertionError("should not prompt")):
            self._run(["some/dir", "--yes"])
        self.mock_analyzer.organize_files.assert_called_once_with(
            {"Cluster_one": ["a.txt"]}, "some/dir", None
        )

    def test_target_dir_passed_through(self):
        self._run(["some/dir", "--target-dir", "other/dir", "-y"])
        self.mock_analyzer.organize_files.assert_called_once_with(
            {"Cluster_one": ["a.txt"]}, "some/dir", "other/dir"
        )

    def test_tuning_flags_passed_to_analyzer(self):
        self._run([
            "some/dir", "-y",
            "--path-weight", "3",
            "--max-clusters", "6",
            "--keyword-ngram", "1",
            "--keyword-count", "3",
        ])
        self.mock_analyzer_cls.assert_called_once_with(
            path_weight=3, max_clusters=6, yake_ngram=1, yake_top=3,
            skip_noise=True, exclude_patterns=[]
        )

    def test_include_all_disables_skip_noise(self):
        self._run(["some/dir", "-y", "--include-all"])
        self.mock_analyzer_cls.assert_called_once_with(
            path_weight=2, max_clusters=10, yake_ngram=2, yake_top=5,
            skip_noise=False, exclude_patterns=[]
        )

    def test_exclude_patterns_passed_to_analyzer(self):
        self._run(["some/dir", "-y", "--exclude", "*.log", "--exclude", "tmp_*"])
        self.mock_analyzer_cls.assert_called_once_with(
            path_weight=2, max_clusters=10, yake_ngram=2, yake_top=5,
            skip_noise=True, exclude_patterns=["*.log", "tmp_*"]
        )

    def test_verbose_sets_debug_level(self):
        self._run(["some/dir", "-y", "--verbose"])
        self.assertEqual(logging.getLogger().getEffectiveLevel(), logging.DEBUG)

    def test_quiet_sets_warning_level(self):
        self._run(["some/dir", "-y", "--quiet"])
        self.assertEqual(logging.getLogger().getEffectiveLevel(), logging.WARNING)

    def test_default_verbosity_is_info(self):
        self._run(["some/dir", "-y"])
        self.assertEqual(logging.getLogger().getEffectiveLevel(), logging.INFO)

    def test_analysis_failure_does_not_move_files(self):
        self.mock_analyzer.analyze_directory.side_effect = ValueError("bad path")
        self._run(["some/dir", "-y"])
        self.mock_analyzer.organize_files.assert_not_called()

    def test_config_file_provides_defaults(self):
        self.config_path.write_text("path_weight = 7\nmax_clusters = 4\n")
        self._run(["some/dir", "-y"])
        self.mock_analyzer_cls.assert_called_once_with(
            path_weight=7, max_clusters=4, yake_ngram=2, yake_top=5,
            skip_noise=True, exclude_patterns=[]
        )

    def test_cli_flag_overrides_config_file(self):
        self.config_path.write_text("path_weight = 7\n")
        self._run(["some/dir", "-y", "--path-weight", "9"])
        self.mock_analyzer_cls.assert_called_once_with(
            path_weight=9, max_clusters=10, yake_ngram=2, yake_top=5,
            skip_noise=True, exclude_patterns=[]
        )

    def test_missing_config_file_uses_hardcoded_defaults(self):
        # self.config_path is never written to in setUp - exercises the
        # not-found branch of _load_config.
        self._run(["some/dir", "-y"])
        self.mock_analyzer_cls.assert_called_once_with(
            path_weight=2, max_clusters=10, yake_ngram=2, yake_top=5,
            skip_noise=True, exclude_patterns=[]
        )

    def test_malformed_config_file_warns_and_falls_back(self):
        self.config_path.write_text("this is not valid toml [[[")
        self._run(["some/dir", "-y"])
        self.mock_analyzer_cls.assert_called_once_with(
            path_weight=2, max_clusters=10, yake_ngram=2, yake_top=5,
            skip_noise=True, exclude_patterns=[]
        )

    def test_undo_flag_calls_undo_organize_and_skips_pipeline(self):
        with patch('src.document_analyzer.main.undo_organize') as mock_undo:
            self._run(["--undo", "manifest.json"])
        mock_undo.assert_called_once_with("manifest.json")
        self.mock_analyzer_cls.assert_not_called()

    def test_invalid_max_clusters_does_not_crash(self):
        """DocumentAnalyzer(max_clusters=2) raises ValueError - main() should log
        and return cleanly instead of letting the traceback escape."""
        self.mock_analyzer_cls.side_effect = ValueError("max_clusters must be at least 3")
        self._run(["some/dir", "--max-clusters", "2", "-y"])
        self.mock_analyzer.analyze_directory.assert_not_called()
        self.mock_analyzer.organize_files.assert_not_called()


if __name__ == '__main__':
    unittest.main()
