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
from src.document_analyzer.analyzer import DocumentAnalyzer
from src.document_analyzer.utils import read_file, analyze_document_content, create_weighted_text

class TestDocumentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = DocumentAnalyzer()
        
    def test_init(self):
        """Test that DocumentAnalyzer initializes correctly"""
        self.assertIsNotNone(self.analyzer.model)
        self.assertIsNotNone(self.analyzer.kw_extractor)
        
    @patch('sentence_transformers.SentenceTransformer')
    @patch('yake.KeywordExtractor')
    def test_analyze_directory(self, mock_kw_extractor, mock_model):
        """Test directory analysis with mocked components"""
        # Configure mocks
        mock_model.return_value.encode.side_effect = Exception("Mocked error")
        mock_kw_extractor.return_value.extract_keywords.return_value = [
            ('mock_keyword', 0.5)
        ]
        
        test_dir = Path("test_directory")
        test_file = test_dir / "test.pdf"
        
        try:
            test_dir.mkdir(exist_ok=True)
            test_file.touch(exist_ok=True)
            
            # Verify initial state
            self.assertTrue(test_dir.exists())
            self.assertTrue(test_file.exists())
            
            result = self.analyzer.analyze_directory(str(test_dir))
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 1)  # More specific assertion
            
            mock_model.return_value.encode.assert_called_once()
            mock_kw_extractor.return_value.extract_keywords.assert_called_once()
            
        except Exception as e:
            self.fail(f"Unexpected exception: {str(e)}")
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
                
    @patch('sentence_transformers.SentenceTransformer')
    @patch('yake.KeywordExtractor')
    def test_analyze_directory_with_exceptions(self, mock_kw_extractor, mock_model):
        """Test directory analysis handles exceptions gracefully"""
        mock_model.return_value.encode.side_effect = Exception("Mocked error")
        
        result = self.analyzer.analyze_directory("nonexistent_directory")
        self.assertIsInstance(result, dict)
        
    def test_find_optimal_clusters(self):
        """Test optimal cluster detection"""
        # Create sample embeddings
        embeddings = np.random.rand(100, 10).astype('float32')
        
        optimal_k = self.analyzer._find_optimal_clusters(embeddings)
        self.assertGreater(optimal_k, 1)
        self.assertLess(optimal_k, 10)  # max_k default is 10
        
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
        
    @patch('yake.KeywordExtractor')
    def test_extract_keywords(self, mock_kw_extractor):
        """Test keyword extraction"""
        mock_kw_extractor.return_value.extract_keywords.return_value = [
            ('keyword1', 0.5),
            ('keyword2', 0.3)
        ]
        
        text_list = ['This is a test document', 'Another test']
        keywords = self.analyzer._extract_keywords(text_list)
        
        self.assertEqual(keywords, ['test document', 'test', 'document'])
        
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

if __name__ == '__main__':
    unittest.main()