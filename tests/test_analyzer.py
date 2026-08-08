import shutil
import sys
import os

# Add the parent directory of tests to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from src.document_analyzer.analyzer import DocumentAnalyzer, undo_organize, ORGANIZED_MARKER, DEFAULT_SKIP_NAMES
from src.document_analyzer.utils import read_file, analyze_document_content, create_weighted_text

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

        self.analyzer = DocumentAnalyzer()
        self.mock_model = mock_model_cls.return_value
        self.mock_kw_extractor = mock_kw_cls.return_value

    def test_init(self):
        """Test that DocumentAnalyzer initializes correctly"""
        self.assertIsNotNone(self.analyzer.model)
        self.assertIsNotNone(self.analyzer.kw_extractor)

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
        self.mock_model.encode.return_value = np.array(
            [[0, 0], [0, 1], [10, 10]]
        )
        self.mock_kw_extractor.extract_keywords.return_value = [
            ('mock_keyword', 0.5)
        ]

        test_dir = Path("test_directory")

        try:
            test_dir.mkdir(exist_ok=True)
            for name in ("test1.pdf", "test2.pdf", "test3.pdf"):
                (test_dir / name).touch(exist_ok=True)

            result = self.analyzer.analyze_directory(str(test_dir))
            self.assertIsInstance(result, dict)
            # Both clusters extract the same mocked keyword, so they should
            # merge into one folder rather than one overwriting the other.
            self.assertEqual(len(result), 1)
            self.assertEqual(sum(len(files) for files in result.values()), 3)

            self.mock_model.encode.assert_called_once()
            self.assertEqual(self.mock_kw_extractor.extract_keywords.call_count, 2)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_analyze_folder_skips_symlinked_directory_cycle(self):
        """Regression: a symlinked directory pointing back at an ancestor used
        to recurse forever (entry.is_dir() follows symlinks by default)."""
        root = Path("test_analyze_folder_symlink_cycle")
        try:
            root.mkdir(exist_ok=True)
            (root / "note.txt").write_text("hello")
            loop = root / "loop"
            loop.symlink_to(root, target_is_directory=True)

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
        self.assertLess(optimal_k, 10)  # max_k default is 10

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
        
    def test_analyze_document_content(self):
        """Test document content analysis"""
        text = "This is a test document.\n\nWith multiple sections."
        length = analyze_document_content(text)
        
        self.assertGreater(length, 1000)  # Base length
        self.assertLess(length, 4000)  # Should be less than max length
        
    def test_create_weighted_text(self):
        """Test weighted text creation"""
        path = "test.pdf"
        content = "Test content"

        weighted_text = create_weighted_text(path, content)
        self.assertIn(path, weighted_text)
        self.assertIn(content, weighted_text)

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
        finally:
            shutil.rmtree(source_dir, ignore_errors=True)

if __name__ == '__main__':
    unittest.main()