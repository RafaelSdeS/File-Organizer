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

# Analyzer knobs the tests below don't set: --no-cache, --depth, --llm-names.
NEW_KNOB_DEFAULTS = {"use_cache": True, "depth": 1, "llm_names": None}


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

    def test_output_survives_a_legacy_console_encoding(self):
        """Windows consoles/pipes can be cp1252, which has no folder emoji:
        printing the structure used to raise UnicodeEncodeError."""
        import io
        raw = io.BytesIO()
        console = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
        with patch('sys.stdout', console):
            self._run(["some/dir", "--dry-run"])
            console.flush()
        self.assertIn(b"? Cluster_one", raw.getvalue())

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
            {"Cluster_one": ["a.txt"]}, "some/dir", None, copy=False
        )

    def test_yes_flag_skips_prompt(self):
        with patch('builtins.input', side_effect=AssertionError("should not prompt")):
            self._run(["some/dir", "--yes"])
        self.mock_analyzer.organize_files.assert_called_once_with(
            {"Cluster_one": ["a.txt"]}, "some/dir", None, copy=False
        )

    def test_target_dir_passed_through(self):
        self._run(["some/dir", "--target-dir", "other/dir", "-y"])
        self.mock_analyzer.organize_files.assert_called_once_with(
            {"Cluster_one": ["a.txt"]}, "some/dir", "other/dir", copy=False
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
            skip_noise=True, exclude_patterns=[], lang="auto", misc_threshold=2.0, **NEW_KNOB_DEFAULTS
        )

    def test_include_all_disables_skip_noise(self):
        self._run(["some/dir", "-y", "--include-all"])
        self.mock_analyzer_cls.assert_called_once_with(
            path_weight=2, max_clusters=10, yake_ngram=2, yake_top=5,
            skip_noise=False, exclude_patterns=[], lang="auto", misc_threshold=2.0, **NEW_KNOB_DEFAULTS
        )

    def test_exclude_patterns_passed_to_analyzer(self):
        self._run(["some/dir", "-y", "--exclude", "*.log", "--exclude", "tmp_*"])
        self.mock_analyzer_cls.assert_called_once_with(
            path_weight=2, max_clusters=10, yake_ngram=2, yake_top=5,
            skip_noise=True, exclude_patterns=["*.log", "tmp_*"], lang="auto", misc_threshold=2.0, **NEW_KNOB_DEFAULTS
        )

    def test_lang_passed_to_analyzer(self):
        self._run(["some/dir", "-y", "--lang", "pt"])
        self.mock_analyzer_cls.assert_called_once_with(
            path_weight=2, max_clusters=10, yake_ngram=2, yake_top=5,
            skip_noise=True, exclude_patterns=[], lang="pt", misc_threshold=2.0, **NEW_KNOB_DEFAULTS
        )

    def test_misc_threshold_passed_to_analyzer(self):
        self._run(["some/dir", "-y", "--misc-threshold", "0"])
        self.assertEqual(self.mock_analyzer_cls.call_args.kwargs["misc_threshold"], 0.0)

    def test_copy_flag_passed_to_organize_files(self):
        self._run(["some/dir", "-y", "--copy"])
        self.mock_analyzer.organize_files.assert_called_once_with(
            {"Cluster_one": ["a.txt"]}, "some/dir", None, copy=True
        )

    def test_report_flag_saves_charts_even_on_dry_run(self):
        with patch('src.document_analyzer.report.save_report', return_value=["out/map.png"]) as mock_save:
            self._run(["some/dir", "--dry-run", "--report", "out"])
        mock_save.assert_called_once_with(
            self.mock_analyzer.last_run, {"Cluster_one": ["a.txt"]}, "out"
        )
        self.mock_analyzer.organize_files.assert_not_called()

    def test_report_failure_does_not_move_files(self):
        with patch('src.document_analyzer.report.save_report', side_effect=ImportError("no matplotlib")):
            self._run(["some/dir", "-y", "--report", "out"])
        self.mock_analyzer.organize_files.assert_not_called()

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
            skip_noise=True, exclude_patterns=[], lang="auto", misc_threshold=2.0, **NEW_KNOB_DEFAULTS
        )

    def test_cli_flag_overrides_config_file(self):
        self.config_path.write_text("path_weight = 7\n")
        self._run(["some/dir", "-y", "--path-weight", "9"])
        self.mock_analyzer_cls.assert_called_once_with(
            path_weight=9, max_clusters=10, yake_ngram=2, yake_top=5,
            skip_noise=True, exclude_patterns=[], lang="auto", misc_threshold=2.0, **NEW_KNOB_DEFAULTS
        )

    def test_missing_config_file_uses_hardcoded_defaults(self):
        # self.config_path is never written to in setUp - exercises the
        # not-found branch of _load_config.
        self._run(["some/dir", "-y"])
        self.mock_analyzer_cls.assert_called_once_with(
            path_weight=2, max_clusters=10, yake_ngram=2, yake_top=5,
            skip_noise=True, exclude_patterns=[], lang="auto", misc_threshold=2.0, **NEW_KNOB_DEFAULTS
        )

    def test_malformed_config_file_warns_and_falls_back(self):
        self.config_path.write_text("this is not valid toml [[[")
        self._run(["some/dir", "-y"])
        self.mock_analyzer_cls.assert_called_once_with(
            path_weight=2, max_clusters=10, yake_ngram=2, yake_top=5,
            skip_noise=True, exclude_patterns=[], lang="auto", misc_threshold=2.0, **NEW_KNOB_DEFAULTS
        )

    def test_history_lists_runs_without_loading_the_model(self):
        entries = [
            {"time": "2026-11-19T10:00:00", "source_dir": "/d", "target_dir": "/d",
             "manifest": "/d/.file_organizer_manifest_x.json", "files": 3, "copy": False},
        ]
        with patch('src.document_analyzer.main.read_history', return_value=entries), \
             patch('builtins.print') as mock_print:
            self._run(["--history"])
        printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list)
        self.assertIn("3 files moved", printed)
        self.assertIn("undone", printed)  # that manifest doesn't exist on disk
        self.mock_analyzer_cls.assert_not_called()

    def test_version_flag_prints_version_and_exits(self):
        with patch('builtins.print'), self.assertRaises(SystemExit) as exit_info, \
             patch('sys.stdout'):
            self._run(["--version"])
        self.assertEqual(exit_info.exception.code, 0)
        self.mock_analyzer_cls.assert_not_called()

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

    def test_plan_writes_a_plan_and_moves_nothing(self):
        self.mock_analyzer.last_run = {"scores": {"a.txt": 0.5}, "sensitive": {}}
        plan = os.path.join(self.tmp_dir.name, "plan.txt")
        with patch('builtins.print'):
            self._run(["some/dir", "--plan", plan])
        with open(plan, encoding="utf-8") as f:
            self.assertIn("[Cluster_one]\n0.50\ta.txt", f.read())
        self.mock_analyzer.organize_files.assert_not_called()

    def test_apply_moves_files_as_the_plan_says_without_loading_the_model(self):
        source = Path(self.tmp_dir.name) / "src"
        source.mkdir()
        (source / "a.txt").write_text("x")
        plan = Path(self.tmp_dir.name) / "plan.txt"
        plan.write_text(f"# organize-files plan\tsource={source}\ttarget={source}\n[Renamed by hand]\na.txt\n", encoding="utf-8")
        with patch('src.document_analyzer.analyzer.HISTORY_PATH', str(Path(self.tmp_dir.name) / "history.jsonl")), \
             patch('builtins.print'):
            self._run(["--apply", str(plan), "-y"])
        self.assertTrue((source / "Renamed by hand" / "a.txt").exists())
        self.mock_analyzer_cls.assert_not_called()

    def test_apply_refuses_an_unsafe_plan(self):
        plan = Path(self.tmp_dir.name) / "plan.txt"
        plan.write_text(f"# organize-files plan\tsource={self.tmp_dir.name}\n[../outside]\nplan.txt\n", encoding="utf-8")
        with patch('builtins.print'), patch('src.document_analyzer.main.organize_files') as organize:
            with patch('sys.argv', ["main.py", "--apply", str(plan), "-y"]):
                self.assertEqual(main_module.main(), 1)
        organize.assert_not_called()

    def test_into_analyzes_against_and_moves_into_existing_folders(self):
        self._run(["some/dir", "-y", "--into", "archive"])
        self.mock_analyzer.analyze_directory.assert_called_once_with("some/dir", into="archive")
        self.mock_analyzer.organize_files.assert_called_once_with({"Cluster_one": ["a.txt"]}, "some/dir", "archive", copy=False)

    def test_new_knobs_reach_the_analyzer(self):
        self._run(["some/dir", "-y", "--no-cache", "--depth", "2", "--llm-names", "qwen2.5:1.5b"])
        kwargs = self.mock_analyzer_cls.call_args.kwargs
        self.assertEqual((kwargs["use_cache"], kwargs["depth"], kwargs["llm_names"]), (False, 2, "qwen2.5:1.5b"))


if __name__ == '__main__':
    unittest.main()
