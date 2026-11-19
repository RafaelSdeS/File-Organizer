import shutil
import sys
import os

# Add the parent directory of tests to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from src.document_analyzer.analyzer import DocumentAnalyzer, read_history, undo_organize, _center_by_language, _detect_languages, _elbow_k, _folder_name, _misc_labels, _stopwords, ORGANIZED_MARKER, DEFAULT_SKIP_NAMES, MISC_FOLDER, AUTO_LANGS
from src.document_analyzer.utils import read_file, create_weighted_text

class TestDocumentAnalyzer(unittest.TestCase):
    def setUp(self):
        # Patch where the names are looked up (analyzer.py imports them at
        # module scope), and start the patches before DocumentAnalyzer() runs
        # so every test gets the mocks instead of the real, slow-to-load model.
        model_patcher = patch('src.document_analyzer.analyzer.SentenceTransformer')
        kw_patcher = patch('src.document_analyzer.analyzer.yake.KeywordExtractor')
        mock_model_cls = model_patcher.start()
        mock_kw_cls = kw_patcher.start()
        self.addCleanup(model_patcher.stop)
        self.addCleanup(kw_patcher.stop)

        # Never append to the real ~/.local/state history from tests.
        history_dir = tempfile.TemporaryDirectory()
        self.addCleanup(history_dir.cleanup)
        self.history_path = os.path.join(history_dir.name, "history.jsonl")
        history_patcher = patch('src.document_analyzer.analyzer.HISTORY_PATH', self.history_path)
        history_patcher.start()
        self.addCleanup(history_patcher.stop)

        self.analyzer = DocumentAnalyzer()
        self.mock_model = mock_model_cls.return_value
        self.mock_kw_extractor = mock_kw_cls.return_value

    def test_init(self):
        """Test that DocumentAnalyzer initializes correctly"""
        self.assertIsNotNone(self.analyzer.model)
        self.assertIsNotNone(self.analyzer.kw_extractor)

    def test_lang_auto_detects_among_common_languages(self):
        self.assertTrue(self.analyzer.auto_lang)
        self.assertEqual(list(self.analyzer.stopword_sets), AUTO_LANGS)

    def test_explicit_lang_list_restricts_detection(self):
        analyzer = DocumentAnalyzer(lang="pt, en")
        self.assertFalse(analyzer.auto_lang)
        self.assertEqual(list(analyzer.stopword_sets), ["pt", "en"])

    def test_init_rejects_low_max_clusters(self):
        """max_clusters < 3 leaves no candidate k for _find_optimal_clusters to test."""
        with self.assertRaises(ValueError):
            DocumentAnalyzer(max_clusters=2)

    def test_analyze_folder_reads_nested_files(self):
        """Regression: os.path.join(folder_path, DirEntry) used to double the path
        (DirEntry.__fspath__ already returns the full path), so any subfolder
        containing a readable file raised FileNotFoundError."""
        root = Path("test_analyze_folder_regression")
        try:
            sub = root / "sub"
            sub.mkdir(parents=True, exist_ok=True)
            (sub / "note.txt").write_text("hello from a subfolder")

            result = self.analyzer._analyze_folder(str(sub))
            self.assertIn("note.txt", result["Content"])
            self.assertIn("hello from a subfolder", result["Content"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_analyze_folder_degrades_on_unreadable_file(self):
        """A single unreadable file inside a subfolder should not blow up the
        whole subfolder - it should degrade to filename-only, like the
        top-level scan in analyze_directory already does."""
        root = Path("test_analyze_folder_degrades")
        try:
            sub = root / "sub"
            sub.mkdir(parents=True, exist_ok=True)
            (sub / "good.txt").write_text("fine")
            (sub / "bad.docx").write_text("not a real docx, will fail to parse")

            result = self.analyzer._analyze_folder(str(sub))
            self.assertIn("good.txt", result["Content"])
            self.assertIn("fine", result["Content"])
            self.assertIn("bad.docx", result["Content"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_analyze_directory(self):
        """Test directory analysis with mocked components"""
        # Two clusters of two: a one-file cluster would now go to Misc.
        self.mock_model.encode.return_value = np.array(
            [[0, 0], [0, 1], [10, 10], [10, 11]]
        )
        self.mock_kw_extractor.extract_keywords.return_value = [
            ('mock_keyword', 0.5)
        ]

        test_dir = Path("test_directory")

        try:
            test_dir.mkdir(exist_ok=True)
            for name in ("test1.pdf", "test2.pdf", "test3.pdf", "test4.pdf"):
                (test_dir / name).touch(exist_ok=True)

            result = self.analyzer.analyze_directory(str(test_dir))
            self.assertIsInstance(result, dict)
            # Both clusters extract the same mocked keyword, so they should
            # merge into one folder rather than one overwriting the other.
            self.assertEqual(len(result), 1)
            self.assertEqual(sum(len(files) for files in result.values()), 4)

            self.mock_model.encode.assert_called_once()
            self.assertEqual(self.mock_kw_extractor.extract_keywords.call_count, 2)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_analyze_folder_stops_reading_once_it_has_enough_text(self):
        root = Path("test_analyze_folder_read_cap")
        try:
            root.mkdir(exist_ok=True)
            for marker in "ABC":
                (root / f"{marker}.txt").write_text(marker * 50)
            with patch('src.document_analyzer.analyzer.READ_CHARS', 60):
                content = self.analyzer._analyze_folder(str(root))["Content"]
            # Two files fill the budget; the third only adds its name.
            self.assertEqual(sum(marker * 50 in content for marker in "ABC"), 2)
            for marker in "ABC":
                self.assertIn(f"{marker}.txt", content)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_analyze_folder_skips_symlinked_directory_cycle(self):
        """Regression: a symlinked directory pointing back at an ancestor used
        to recurse forever (entry.is_dir() follows symlinks by default)."""
        root = Path("test_analyze_folder_symlink_cycle")
        try:
            root.mkdir(exist_ok=True)
            (root / "note.txt").write_text("hello")
            loop = root / "loop"
            # Absolute target: a relative one resolves from inside root and
            # dangles, which never formed a cycle at all.
            try:
                loop.symlink_to(root.resolve(), target_is_directory=True)
            except OSError:  # Windows without symlink privileges
                self.skipTest("creating symlinks isn't permitted here")
            self.assertTrue(loop.resolve() == root.resolve())

            result = self.analyzer._analyze_folder(str(root))
            self.assertIn("note.txt", result["Content"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_analyze_directory_skips_previously_organized_folder(self):
        """A folder bearing organize_files' marker file is this tool's own
        prior output - it should be skipped, not re-clustered as a document."""
        self.mock_model.encode.return_value = np.array([[0, 0], [1, 1], [2, 2]])
        self.mock_kw_extractor.extract_keywords.return_value = [('mock_keyword', 0.5)]

        test_dir = Path("test_directory_marker")
        try:
            test_dir.mkdir(exist_ok=True)
            for name in ("test1.pdf", "test2.pdf", "test3.pdf"):
                (test_dir / name).touch(exist_ok=True)

            organized = test_dir / "Cluster_old"
            organized.mkdir()
            (organized / ORGANIZED_MARKER).touch()
            (organized / "leftover.txt").write_text("should not be read")

            self.analyzer.analyze_directory(str(test_dir))

            encoded_texts = self.mock_model.encode.call_args[0][0]
            self.assertEqual(len(encoded_texts), 3)
            self.assertFalse(any("leftover" in t for t in encoded_texts))
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_analyze_directory_skips_hidden_and_noise_entries_by_default(self):
        """Dotfiles/dotdirs and known noise dirs (__pycache__, node_modules)
        shouldn't be embedded and clustered as if they were real documents."""
        self.mock_model.encode.return_value = np.array([[0, 0], [1, 1], [2, 2]])
        self.mock_kw_extractor.extract_keywords.return_value = [('mock_keyword', 0.5)]

        test_dir = Path("test_directory_noise")
        try:
            test_dir.mkdir(exist_ok=True)
            for name in ("test1.pdf", "test2.pdf", "test3.pdf"):
                (test_dir / name).touch(exist_ok=True)

            (test_dir / ".hidden.txt").write_text("dotfile, should be skipped")
            (test_dir / ".git").mkdir()
            (test_dir / ".git" / "config").write_text("git internals")
            for noise_name in DEFAULT_SKIP_NAMES:
                noise_dir = test_dir / noise_name
                noise_dir.mkdir()
                (noise_dir / "junk.txt").write_text("noise")

            self.analyzer.analyze_directory(str(test_dir))

            encoded_texts = self.mock_model.encode.call_args[0][0]
            self.assertEqual(len(encoded_texts), 3)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_analyze_directory_include_all_disables_noise_filtering(self):
        """skip_noise=False (--include-all) should process hidden entries too."""
        self.mock_model.encode.return_value = np.array([[0, 0], [1, 1]])
        self.mock_kw_extractor.extract_keywords.return_value = [('mock_keyword', 0.5)]

        analyzer = DocumentAnalyzer(skip_noise=False)

        test_dir = Path("test_directory_noise_included")
        try:
            test_dir.mkdir(exist_ok=True)
            (test_dir / "test1.pdf").touch()
            (test_dir / ".hidden.txt").write_text("dotfile")

            analyzer.analyze_directory(str(test_dir))

            encoded_texts = self.mock_model.encode.call_args[0][0]
            self.assertEqual(len(encoded_texts), 2)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_analyze_directory_exclude_pattern_applies_regardless_of_skip_noise(self):
        """--exclude patterns are an explicit user ask, so they should still
        apply even with skip_noise=False (--include-all)."""
        self.mock_model.encode.return_value = np.array([[0, 0], [1, 1]])
        self.mock_kw_extractor.extract_keywords.return_value = [('mock_keyword', 0.5)]

        analyzer = DocumentAnalyzer(skip_noise=False, exclude_patterns=["*.log"])

        test_dir = Path("test_directory_exclude")
        try:
            test_dir.mkdir(exist_ok=True)
            (test_dir / "test1.pdf").touch()
            (test_dir / "test2.pdf").touch()
            (test_dir / "debug.log").write_text("noisy log output")

            analyzer.analyze_directory(str(test_dir))

            encoded_texts = self.mock_model.encode.call_args[0][0]
            self.assertEqual(len(encoded_texts), 2)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_analyze_directory_with_exceptions(self):
        """Test directory analysis raises for a missing directory"""
        with self.assertRaises(FileNotFoundError):
            self.analyzer.analyze_directory("nonexistent_directory")

    def test_find_optimal_clusters(self):
        """Test optimal cluster detection"""
        # Create sample embeddings
        embeddings = np.random.rand(100, 10).astype('float32')

        optimal_k = self.analyzer._find_optimal_clusters(embeddings)
        self.assertGreater(optimal_k, 1)
        # max_k default is 10, inclusive since the off-by-one fix - k=10 is
        # now a legitimately reachable candidate, not just k<10.
        self.assertLessEqual(optimal_k, 10)

    def test_find_optimal_clusters_small_input(self):
        """Regression: 2-4 embeddings used to crash (elbow method needs >=3
        k-candidates, i.e. >=5 documents, for a non-empty 2nd derivative)."""
        for n in (1, 2, 3, 4, 5):
            embeddings = np.random.rand(n, 10).astype('float32')
            optimal_k = self.analyzer._find_optimal_clusters(embeddings)
            self.assertGreaterEqual(optimal_k, 1)
            self.assertLess(optimal_k, n if n > 1 else 2)

    def test_find_optimal_clusters_reaches_configured_max(self):
        """Regression: K = range(2, min(n, max_k)) excluded max_k itself, so
        --max-clusters never tested its own value. With 4 well-separated,
        tight true clusters and max_k=4, k=4 must be reachable and win."""
        rng = np.random.default_rng(0)
        centers = [(0, 0), (1000, 1000), (2000, 2000), (3000, 3000)]
        embeddings = np.vstack([
            rng.normal(loc=center, scale=0.1, size=(5, 2)) for center in centers
        ])

        optimal_k = self.analyzer._find_optimal_clusters(embeddings, max_k=4)
        self.assertEqual(optimal_k, 4)

    def test_elbow_votes_for_sharpest_bend(self):
        """Regression: K[argmin(2nd diff)] picked the flattest stretch and was
        off by one; the elbow is the max 2nd difference, centred on K[i+1].
        Here inertia barely drops 2->3, collapses 3->4, then flattens: k=4.
        The old formula returned 2 for this curve."""
        self.assertEqual(_elbow_k(range(2, 7), [100, 90, 20, 18, 17]), 4)

    def test_elbow_needs_three_points(self):
        self.assertIsNone(_elbow_k(range(2, 4), [100, 50]))

    def test_find_optimal_clusters_records_scores_for_report(self):
        rng = np.random.default_rng(0)
        embeddings = rng.random((12, 3))
        k = self.analyzer._find_optimal_clusters(embeddings, max_k=5)

        scores = self.analyzer.k_scores
        self.assertEqual(scores["k"], [2, 3, 4, 5])
        self.assertEqual(scores["chosen"], k)
        for metric in ("inertia", "silhouette", "davies_bouldin"):
            self.assertEqual(len(scores[metric]), 4)
        self.assertEqual(set(scores["votes"]), {"silhouette", "davies_bouldin", "elbow"})

    def test_far_document_goes_to_misc(self):
        """KMeans glues a one-off to its nearest cluster; one far from its
        centroid (> mean + 2 std) should land in Misc instead."""
        rng = np.random.default_rng(0)
        groups = [rng.normal(loc=c, scale=0.01, size=(5, 2)) for c in ((0, 0), (100, 0), (0, 100))]
        outlier = np.array([[6.0, 0.0]])  # nearest to group (0, 0), but well outside it
        self.mock_model.encode.return_value = np.vstack(groups + [outlier])
        self.mock_kw_extractor.extract_keywords.return_value = [("topic", 0.5)]

        test_dir = Path("test_misc_directory")
        try:
            test_dir.mkdir(exist_ok=True)
            for i in range(16):
                (test_dir / f"doc{i:02d}.txt").write_text("x")
            with patch.object(self.analyzer, "_find_optimal_clusters", return_value=3):
                result = self.analyzer.analyze_directory(str(test_dir))

            # scandir order is arbitrary; last_run["paths"] says which file
            # got the outlier's embedding (the 16th encode() row).
            self.assertEqual(result[MISC_FOLDER], [self.analyzer.last_run["paths"][15]])
            # Misc gets no keyword extraction: only the 3 real clusters do.
            self.assertEqual(self.mock_kw_extractor.extract_keywords.call_count, 3)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_misc_takes_documents_closer_to_another_cluster(self):
        """Negative silhouette: on average nearer the other cluster's
        members than its own, so it belongs to neither."""
        emb = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10], [6, 6]], dtype=float)
        labels = np.array([0, 0, 0, 1, 1, 1, 0])  # [6, 6] glued to cluster 0
        centers = np.array([emb[labels == c].mean(axis=0) for c in (0, 1)])
        result = _misc_labels(emb, labels, centers, z=100)  # distance rule out of play
        self.assertEqual(result.tolist(), [0, 0, 0, 1, 1, 1, -1])

    def test_misc_takes_a_document_left_alone_in_its_cluster(self):
        emb = np.array([[0, 0], [0, 1], [10, 10]], dtype=float)
        labels = np.array([0, 0, 1])
        centers = np.array([[0, 0.5], [10, 10]])
        self.assertEqual(_misc_labels(emb, labels, centers, z=100).tolist(), [0, 0, -1])

    def test_misc_threshold_zero_disables_misc(self):
        self.analyzer.misc_threshold = 0
        rng = np.random.default_rng(0)
        groups = [rng.normal(loc=c, scale=0.01, size=(5, 2)) for c in ((0, 0), (100, 0), (0, 100))]
        self.mock_model.encode.return_value = np.vstack(groups + [np.array([[6.0, 0.0]])])
        self.mock_kw_extractor.extract_keywords.return_value = [("topic", 0.5)]

        test_dir = Path("test_misc_disabled_directory")
        try:
            test_dir.mkdir(exist_ok=True)
            for i in range(16):
                (test_dir / f"doc{i:02d}.txt").write_text("x")
            with patch.object(self.analyzer, "_find_optimal_clusters", return_value=3):
                result = self.analyzer.analyze_directory(str(test_dir))
            self.assertNotIn(MISC_FOLDER, result)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_organize_clusters(self):
        """Test cluster organization"""
        # Create sample DataFrame
        df = pd.DataFrame({
            'Path': ['doc', 'doc', 'doc'],
            'Content': ['1', '2', '3'],
            'Text': ['doc1', 'doc2', 'doc3'],
            'Cluster': [0, 0, 1]
        })
        
        result = self.analyzer._organize_clusters(df)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)  # Should have 2 clusters
        
    def test_extract_keywords(self):
        """Test keyword extraction"""
        self.mock_kw_extractor.extract_keywords.return_value = [
            ('keyword1', 0.5),
            ('keyword2', 0.3)
        ]

        text_list = ['This is a test document', 'Another test']
        keywords = self.analyzer._extract_keywords(text_list)

        self.assertEqual(keywords, ['keyword1', 'keyword2'])
        
    def test_create_weighted_text(self):
        """Test weighted text creation"""
        path = "test.pdf"
        content = "Test content"

        weighted_text = create_weighted_text(path, content)
        self.assertIn("test", weighted_text)
        self.assertIn(content, weighted_text)

    def test_create_weighted_text_separates_repetitions(self):
        """Repeated filenames must be space-separated, not glued together."""
        self.assertEqual(
            create_weighted_text("a.pdf", None, path_weight=3),
            "a a a",
        )

    def test_create_weighted_text_uses_name_words_without_extension(self):
        """Separators become spaces and the extension is dropped, so YAKE
        can't put 'feijoada.txt' into a folder name."""
        self.assertEqual(
            create_weighted_text("receita_pão-de.queijo.txt", None, path_weight=1),
            "receita pão de queijo",
        )

    def test_folder_name_drops_words_shared_between_keywords(self):
        """Overlapping YAKE keywords shouldn't repeat words in the folder name."""
        self.assertEqual(
            _folder_name(["engenharia eletrica", "energia eletrica", "ignored"]),
            "Engenharia_eletrica_energia",
        )
        self.assertEqual(_folder_name(["contrato", "imobiliario"]), "Contrato_imobiliario")
        self.assertEqual(_folder_name([]), "")

    def test_organize_files_moves_files_in_place(self):
        """organize_files should move each listed file into its cluster folder."""
        source_dir = Path("test_organize_source")
        try:
            source_dir.mkdir(exist_ok=True)
            (source_dir / "a.txt").write_text("a")
            (source_dir / "b.txt").write_text("b")

            self.analyzer.organize_files(
                {"Cluster_one": ["a.txt", "b.txt"]}, str(source_dir)
            )

            dest_dir = source_dir / "Cluster_one"
            self.assertTrue((dest_dir / "a.txt").exists())
            self.assertTrue((dest_dir / "b.txt").exists())
            self.assertFalse((source_dir / "a.txt").exists())
        finally:
            shutil.rmtree(source_dir, ignore_errors=True)

    def test_organize_files_target_dir(self):
        """organize_files should support moving into a separate target directory."""
        source_dir = Path("test_organize_source2")
        target_dir = Path("test_organize_target2")
        try:
            source_dir.mkdir(exist_ok=True)
            (source_dir / "a.txt").write_text("a")

            self.analyzer.organize_files(
                {"Cluster_one": ["a.txt"]}, str(source_dir), str(target_dir)
            )

            self.assertTrue((target_dir / "Cluster_one" / "a.txt").exists())
            self.assertFalse((source_dir / "a.txt").exists())
        finally:
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(target_dir, ignore_errors=True)

    def test_organize_files_missing_source_skipped(self):
        """A file listed in folder_structure but absent on disk should be skipped, not raise."""
        source_dir = Path("test_organize_source3")
        try:
            source_dir.mkdir(exist_ok=True)
            # No files created - "missing.txt" doesn't exist on disk.
            result = self.analyzer.organize_files(
                {"Cluster_one": ["missing.txt"]}, str(source_dir)
            )
            self.assertFalse((source_dir / "Cluster_one" / "missing.txt").exists())
            self.assertIsNone(result)  # nothing moved -> no manifest written
        finally:
            shutil.rmtree(source_dir, ignore_errors=True)

    def test_organize_files_skips_existing_destination(self):
        """shutil.move -> os.rename on POSIX silently clobbers an existing
        destination; organize_files must check first and skip instead."""
        source_dir = Path("test_organize_overwrite_source")
        try:
            source_dir.mkdir(exist_ok=True)
            (source_dir / "a.txt").write_text("new content")

            dest_dir = source_dir / "Cluster_one"
            dest_dir.mkdir()
            (dest_dir / "a.txt").write_text("original content")

            self.analyzer.organize_files({"Cluster_one": ["a.txt"]}, str(source_dir))

            self.assertEqual((dest_dir / "a.txt").read_text(), "original content")
            self.assertTrue((source_dir / "a.txt").exists())
        finally:
            shutil.rmtree(source_dir, ignore_errors=True)

    def test_organize_files_writes_manifest_and_undo_restores(self):
        """organize_files should record moves in a manifest that undo_organize
        can replay in reverse."""
        source_dir = Path("test_organize_manifest_source")
        try:
            source_dir.mkdir(exist_ok=True)
            (source_dir / "a.txt").write_text("a")

            manifest_path = self.analyzer.organize_files(
                {"Cluster_one": ["a.txt"]}, str(source_dir)
            )

            self.assertIsNotNone(manifest_path)
            self.assertTrue(os.path.exists(manifest_path))

            dest_file = source_dir / "Cluster_one" / "a.txt"
            self.assertTrue(dest_file.exists())
            self.assertFalse((source_dir / "a.txt").exists())

            undo_organize(manifest_path)

            self.assertTrue((source_dir / "a.txt").exists())
            self.assertFalse(dest_file.exists())
            # Nothing left behind: the emptied cluster folder (marker and
            # all) and the fully-replayed manifest are both removed.
            self.assertFalse((source_dir / "Cluster_one").exists())
            self.assertFalse(os.path.exists(manifest_path))
        finally:
            shutil.rmtree(source_dir, ignore_errors=True)

    def test_organize_files_records_history_newest_first(self):
        source_dir = Path("test_history_source")
        try:
            source_dir.mkdir(exist_ok=True)
            for name in ("a.txt", "b.txt"):
                (source_dir / name).write_text(name)
            first = self.analyzer.organize_files({"One": ["a.txt"]}, str(source_dir))
            with open(self.history_path, "a") as f:
                f.write("not json\n")  # a corrupt line must not break --history
            second = self.analyzer.organize_files({"Two": ["b.txt"]}, str(source_dir), copy=True)

            entries = read_history()
            self.assertEqual([e["manifest"] for e in entries],
                             [os.path.abspath(second), os.path.abspath(first)])
            self.assertEqual((entries[0]["files"], entries[0]["copy"]), (1, True))
            self.assertEqual(entries[1]["source_dir"], os.path.abspath(source_dir))
        finally:
            shutil.rmtree(source_dir, ignore_errors=True)

    def test_undo_keeps_manifest_and_folder_when_an_entry_is_skipped(self):
        source_dir = Path("test_undo_partial_source")
        try:
            source_dir.mkdir(exist_ok=True)
            (source_dir / "a.txt").write_text("a")
            (source_dir / "b.txt").write_text("b")
            manifest_path = self.analyzer.organize_files(
                {"Cluster_one": ["a.txt", "b.txt"]}, str(source_dir)
            )
            # Someone recreated a.txt at its original spot: that restore must
            # be skipped, so the folder still holds a file and the manifest stays.
            (source_dir / "a.txt").write_text("new a")

            undo_organize(manifest_path)

            self.assertEqual((source_dir / "a.txt").read_text(), "new a")
            self.assertTrue((source_dir / "b.txt").exists())
            self.assertTrue((source_dir / "Cluster_one" / "a.txt").exists())
            self.assertTrue(os.path.exists(manifest_path))
        finally:
            shutil.rmtree(source_dir, ignore_errors=True)

    def test_copy_keeps_originals_and_undo_removes_only_copies(self):
        source_dir = Path("test_copy_source")
        try:
            source_dir.mkdir(exist_ok=True)
            (source_dir / "a.txt").write_text("a")
            (source_dir / "sub").mkdir()
            (source_dir / "sub" / "inner.txt").write_text("inner")

            manifest_path = self.analyzer.organize_files(
                {"Cluster_one": ["a.txt", "sub"]}, str(source_dir), copy=True
            )

            # Originals untouched, copies (file and whole subfolder) in place.
            self.assertTrue((source_dir / "a.txt").exists())
            self.assertTrue((source_dir / "sub" / "inner.txt").exists())
            self.assertEqual((source_dir / "Cluster_one" / "a.txt").read_text(), "a")
            self.assertEqual((source_dir / "Cluster_one" / "sub" / "inner.txt").read_text(), "inner")

            undo_organize(manifest_path)

            self.assertTrue((source_dir / "a.txt").exists())
            self.assertTrue((source_dir / "sub" / "inner.txt").exists())
            self.assertFalse((source_dir / "Cluster_one").exists())
        finally:
            shutil.rmtree(source_dir, ignore_errors=True)

    def test_undo_never_deletes_a_copy_whose_original_is_gone(self):
        source_dir = Path("test_copy_orphan_source")
        try:
            source_dir.mkdir(exist_ok=True)
            (source_dir / "a.txt").write_text("a")
            manifest_path = self.analyzer.organize_files(
                {"Cluster_one": ["a.txt"]}, str(source_dir), copy=True
            )
            (source_dir / "a.txt").unlink()  # the copy is now the only one

            undo_organize(manifest_path)

            self.assertTrue((source_dir / "Cluster_one" / "a.txt").exists())
        finally:
            shutil.rmtree(source_dir, ignore_errors=True)

class TestLanguageCentering(unittest.TestCase):
    STOPWORDS = {"pt": {"de", "o", "a", "que"}, "en": {"the", "of", "and", "is"}}

    def test_detect_languages_by_stopword_count(self):
        texts = ["o gato de que a casa", "the cat is out of the house", "receita.txt"]
        detected = _detect_languages(texts, self.STOPWORDS)
        # A text with no stopwords at all has no detectable language.
        self.assertEqual(detected.tolist(), ["pt", "en", ""])

    def test_centering_removes_language_offset(self):
        """Two topics x two languages, with a language offset bigger than the
        topic difference: raw, same-language pairs are closest; centred, the
        same topic across languages is."""
        topic = {"A": np.array([1.0, 0, 0]), "B": np.array([0, 1.0, 0])}
        offset = {"pt": np.array([0, 0, 3.0]), "en": np.array([0, 0, -3.0])}
        rows = [(t, l) for l in ("pt", "en") for t in ("A", "B")]
        emb = np.array([topic[t] + offset[l] for t, l in rows])
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        languages = np.array([l for _, l in rows])

        def cos(m, i, j):
            return m[i] @ m[j]
        a_pt, b_pt, a_en = 0, 1, 2
        self.assertGreater(cos(emb, a_pt, b_pt), cos(emb, a_pt, a_en))

        centred = _center_by_language(emb, languages)
        self.assertGreater(cos(centred, a_pt, a_en), cos(centred, a_pt, b_pt))
        np.testing.assert_allclose(np.linalg.norm(centred, axis=1), 1.0)

    def test_single_language_left_unchanged(self):
        emb = np.random.default_rng(0).random((4, 3))
        same = _center_by_language(emb, np.array(["pt", "pt", "pt", "en"]))
        self.assertIs(same, emb)


class TestStopwords(unittest.TestCase):
    """Uses the real yake.KeywordExtractor, unlike TestDocumentAnalyzer."""

    def test_single_language_defers_to_yake(self):
        self.assertIsNone(_stopwords(["pt"]))

    def test_unions_both_languages(self):
        merged = _stopwords(["pt", "en"])
        self.assertIn("para", merged)   # Portuguese
        self.assertIn("the", merged)    # English

    def test_unknown_language_falls_back_instead_of_emptying(self):
        merged = _stopwords(["en", "zzz"])
        self.assertIn("the", merged)


if __name__ == '__main__':
    unittest.main()