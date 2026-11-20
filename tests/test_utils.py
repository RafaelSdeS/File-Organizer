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
            path = os.path.join(tmp, "data.xyz")
            with open(path, "w") as f:
                f.write("a,b,c")
            with self.assertRaises(ValueError):
                read_file(path)

    def test_markdown_and_csv_read_as_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, body in (("notes.md", "# Title\nbody"), ("data.csv", "a,b\n1,2")):
                path = os.path.join(tmp, name)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(body)
                self.assertEqual(read_file(path), body)

    def test_pptx_file_reads_slides_in_numeric_order(self):
        slide = (
            '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{}</a:t></a:r></a:p>'
            '</p:txBody></p:sp></p:spTree></p:cSld></p:sld>'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "deck.pptx")
            with zipfile.ZipFile(path, "w") as zf:
                # Written out of order: slide10 must still come after slide2.
                zf.writestr("ppt/slides/slide10.xml", slide.format("Ten"))
                zf.writestr("ppt/slides/slide2.xml", slide.format("Two"))
                zf.writestr("ppt/slides/_rels/slide2.xml.rels", "<Relationships/>")
            self.assertEqual(read_file(path), "Two\nTen")

    def test_odt_file_keeps_paragraphs_separate(self):
        content_xml = (
            '<office:document-content '
            'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
            'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
            '<office:body><office:text>'
            '<text:h>Heading</text:h>'
            '<text:p>Hello <text:span>world</text:span></text:p>'
            '</office:text></office:body></office:document-content>'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "doc.odt")
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("content.xml", content_xml)
            self.assertEqual(read_file(path), "Heading\nHello world")

    def test_windows_line_endings_become_newlines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dos.txt")
            with open(path, "wb") as f:
                f.write(b"line one\r\nline two\rline three\n")
            self.assertEqual(read_file(path), "line one\nline two\nline three")
            self.assertEqual(read_file(path, max_chars=100), "line one\nline two\nline three")

    def test_max_chars_caps_text_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "long.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("é" * 10000)  # 2 bytes each: the capped read can split one
            self.assertEqual(read_file(path, max_chars=100), "é" * 100)

    @patch('src.document_analyzer.utils.PdfReader')
    def test_max_chars_stops_extracting_pdf_pages(self, mock_reader_cls):
        pages = [MagicMock() for _ in range(5)]
        for page in pages:
            page.extract_text.return_value = "x" * 60
        mock_reader_cls.return_value.pages = pages
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "big.pdf")
            with open(path, "wb") as f:
                f.write(b"%PDF-1.4 fake")
            self.assertEqual(read_file(path, max_chars=100), "x" * 100)
        # Two pages already give 120 characters: the rest are never parsed.
        self.assertEqual([p.extract_text.called for p in pages], [True, True, False, False, False])

    def test_xlsx_file_reads_shared_and_inline_strings(self):
        ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "budget.xlsx")
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("xl/workbook.xml", f'<workbook {ns}><sheets><sheet name="S1"/></sheets></workbook>')
                zf.writestr("xl/sharedStrings.xml",
                            f'<sst {ns}><si><t>Aluguel</t></si><si><r><t>Mercado</t></r></si></sst>')
                zf.writestr("xl/worksheets/sheet1.xml",
                            f'<worksheet {ns}><sheetData><row><c t="inlineStr"><is><t>Total</t></is></c>'
                            f'<c><v>42</v></c></row></sheetData></worksheet>')
            self.assertEqual(read_file(path).split("\n"), ["Aluguel", "Mercado", "Total"])

    def test_xlsx_with_numbers_only_reads_as_empty(self):
        ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "numbers.xlsx")
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("xl/workbook.xml", f"<workbook {ns}/>")
                zf.writestr("xl/worksheets/sheet1.xml", f"<worksheet {ns}><sheetData/></worksheet>")
            self.assertEqual(read_file(path), "")

    def test_rtf_file_drops_metadata_and_decodes_escapes(self):
        rtf = (r"{\rtf1\ansi\deff0{\fonttbl{\f0 Times New Roman;}}{\colortbl;\red0\green0\blue0;}"
               r"{\*\generator Riched20;}{\info{\author Someone}}\f0\fs24 Ol\'e1 mundo, caf\u233?."
               r"\par Segunda {\b linha}\par}")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "carta.rtf")
            with open(path, "w", encoding="ascii") as f:
                f.write(rtf)
            text = read_file(path)
        self.assertEqual(text.split("\n")[0].strip(), "Olá mundo, café.")
        self.assertIn("Segunda linha", text)
        for noise in ("Times New Roman", "Riched20", "Someone", "red0"):
            self.assertNotIn(noise, text)

    def test_office_file_without_expected_part_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.pptx")
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("docProps/app.xml", "<Properties/>")
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

    @patch('src.document_analyzer.utils.PdfReader')
    def test_pdf_file(self, mock_reader_cls):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page text"
        mock_reader_cls.return_value.pages = [mock_page]

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "doc.pdf")
            with open(path, "wb") as f:
                f.write(b"%PDF-1.4 fake content")
            self.assertEqual(read_file(path), "Page text")

    @patch('src.document_analyzer.utils.PdfReader')
    def test_malformed_pdf_raises_value_error(self, mock_reader_cls):
        from pypdf.errors import PdfReadError
        mock_reader_cls.side_effect = PdfReadError("EOF marker not found")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "doc.pdf")
            with open(path, "wb") as f:
                f.write(b"not a real pdf")
            with self.assertRaises(ValueError):
                read_file(path)


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


class TestOcr(unittest.TestCase):
    def test_no_tesseract_means_no_text(self):
        with patch.object(utils_module, "TESSERACT", None):
            self.assertIsNone(utils_module.ocr_image("scan.png"))
            self.assertEqual(utils_module._ocr_pdf(MagicMock()), "")

    def test_ocr_image_runs_tesseract_and_drops_noise(self):
        utils_module._ocr_langs.cache_clear()
        self.addCleanup(utils_module._ocr_langs.cache_clear)
        langs = MagicMock(stdout="List of available languages (3):\neng\npor\nosd\n")
        with patch.object(utils_module, "TESSERACT", "/usr/bin/tesseract"), \
             patch.object(utils_module.subprocess, "run", side_effect=[langs, MagicMock(stdout=" Nota fiscal eletrônica nº 123 \n")]) as run:
            self.assertEqual(utils_module.ocr_image("scan.png"), "Nota fiscal eletrônica nº 123")
        self.assertEqual(run.call_args.args[0], ["/usr/bin/tesseract", "scan.png", "stdout", "-l", "eng+por"])
        with patch.object(utils_module, "TESSERACT", "/usr/bin/tesseract"), \
             patch.object(utils_module.subprocess, "run", return_value=MagicMock(stdout="~ |")):
            self.assertIsNone(utils_module.ocr_image("photo.jpg"))  # a photo, not a document

    def test_scanned_pdf_falls_back_to_ocr_of_page_images(self):
        page = MagicMock()
        page.extract_text.return_value = ""
        page.images = [MagicMock(data=b"jpeg bytes")]
        page.images[0].name = "Im0.jpg"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scan.pdf")
            with open(path, "wb") as f:
                f.write(b"%PDF-1.4")
            with patch.object(utils_module, "PdfReader") as reader, \
                 patch.object(utils_module, "TESSERACT", "/usr/bin/tesseract"), \
                 patch.object(utils_module, "ocr_image", return_value="Contracheque março") as ocr:
                reader.return_value.pages = [page]
                self.assertEqual(read_file(path), "Contracheque março")
            self.assertTrue(ocr.call_args.args[0].endswith(".jpg"))
            self.assertEqual(os.listdir(tmp), ["scan.pdf"])  # temp image removed


class TestSensitiveKinds(unittest.TestCase):
    def test_cpf_needs_valid_check_digits(self):
        self.assertEqual(utils_module.sensitive_kinds("CPF: 529.982.247-25"), {"cpf"})
        self.assertEqual(utils_module.sensitive_kinds("CPF: 52998224725"), {"cpf"})
        self.assertEqual(utils_module.sensitive_kinds("CPF: 529.982.247-26"), set())
        self.assertEqual(utils_module.sensitive_kinds("000.000.000-00"), set())

    def test_card_needs_luhn_and_a_card_network_prefix(self):
        self.assertEqual(utils_module.sensitive_kinds("cartão 4111 1111 1111 1111"), {"card"})
        self.assertEqual(utils_module.sensitive_kinds("cartão 4111 1111 1111 1112"), set())
        self.assertEqual(utils_module.sensitive_kinds("ISBN 978-0-306-40615-7"), set())

    def test_longer_digit_runs_are_not_cards(self):
        # e.g. a boleto's digitable line: 47 digits in a row
        self.assertEqual(utils_module.sensitive_kinds("4" * 47), set())


if __name__ == '__main__':
    unittest.main()
