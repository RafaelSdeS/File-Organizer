import os
import sys
import tempfile
import zipfile

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import unittest
from unittest.mock import patch, MagicMock

from src.document_analyzer.utils import (
    read_file,
    read_audio_metadata,
    transcribe_audio,
    set_transcription_language,
)
import src.document_analyzer.utils as utils_module


class TestReadFile(unittest.TestCase):
    def test_txt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "note.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("hello world")
            self.assertEqual(read_file(path), "hello world")

    def test_empty_file_returns_empty_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.txt")
            open(path, "w").close()
            self.assertEqual(read_file(path), "")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            read_file("/nonexistent/path/file.txt")

    def test_unsupported_extension_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.csv")
            with open(path, "w") as f:
                f.write("a,b,c")
            with self.assertRaises(ValueError):
                read_file(path)

    def test_docx_file(self):
        document_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body><w:p><w:r><w:t>Hello world</w:t></w:r></w:p></w:body>'
            '</w:document>'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "doc.docx")
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("word/document.xml", document_xml)
            self.assertEqual(read_file(path), "Hello world")

    @patch('src.document_analyzer.utils.PyPDF2.PdfReader')
    def test_pdf_file(self, mock_reader_cls):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page text"
        mock_reader_cls.return_value.pages = [mock_page]

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "doc.pdf")
            with open(path, "wb") as f:
                f.write(b"%PDF-1.4 fake content")
            self.assertEqual(read_file(path), "Page text")


class TestAudioMetadata(unittest.TestCase):
    def test_unsupported_extension_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "note.txt")
            open(path, "w").close()
            self.assertIsNone(read_audio_metadata(path))

    def test_missing_file_returns_none(self):
        self.assertIsNone(read_audio_metadata("/nonexistent/file.mp3"))

    @patch('src.document_analyzer.utils.transcribe_audio')
    @patch('src.document_analyzer.utils.TINYTAG_AVAILABLE', True)
    def test_combines_metadata_and_transcription(self, mock_transcribe):
        mock_transcribe.return_value = "spoken words"
        fake_tag = MagicMock(title="Song", artist="Artist", album=None, genre=None, composer=None)
        with patch('src.document_analyzer.utils.TinyTag', create=True) as mock_tinytag_cls:
            mock_tinytag_cls.get.return_value = fake_tag
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "track.mp3")
                open(path, "wb").close()
                result = read_audio_metadata(path)

        self.assertIn("Song Artist", result)
        self.assertIn("spoken words", result)

    @patch('src.document_analyzer.utils.transcribe_audio')
    @patch('src.document_analyzer.utils.TINYTAG_AVAILABLE', False)
    def test_no_content_extracted_returns_none(self, mock_transcribe):
        mock_transcribe.return_value = None
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "track.mp3")
            open(path, "wb").close()
            self.assertIsNone(read_audio_metadata(path))

    def test_transcribe_audio_without_vosk_returns_none(self):
        with patch('src.document_analyzer.utils.VOSK_AVAILABLE', False):
            self.assertIsNone(transcribe_audio("whatever.mp3"))


class TestTranscriptionLanguage(unittest.TestCase):
    def test_valid_language_code(self):
        set_transcription_language("es")
        self.assertEqual(utils_module._vosk_language, "es")

    def test_invalid_language_code_falls_back_to_english(self):
        set_transcription_language("xx-not-real")
        self.assertEqual(utils_module._vosk_language, "en")


if __name__ == '__main__':
    unittest.main()
