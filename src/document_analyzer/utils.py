from pypdf import PdfReader
from pypdf.errors import PdfReadError
import pandas as pd
import fnmatch
import logging
import os
import re
import chardet
import zipfile
import xml.etree.ElementTree as ET

try:
    from tinytag import TinyTag
    TINYTAG_AVAILABLE = True
except ImportError:
    TINYTAG_AVAILABLE = False

try:
    import vosk
    import wave
    import json
    import subprocess
    VOSK_AVAILABLE = True
    _vosk_model = None
except ImportError:
    VOSK_AVAILABLE = False
    _vosk_model = None

logger = logging.getLogger(__name__)
# pypdf warns about every recoverable quirk in a PDF ("Ignoring wrong pointing
# object ..."), dozens per real-world file; only real failures matter here.
logging.getLogger("pypdf").setLevel(logging.ERROR)

audio_extensions = [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".aiff", ".opus"]

# Office formats are zips of XML parts: (member glob, tags holding the text).
# pptx has one part per slide (slide1.xml, slide2.xml, ...).
_ODF_TEXT = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
ZIPPED_XML_FORMATS = {
    ".docx": ("word/document.xml", {"{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"}),
    ".pptx": ("ppt/slides/slide*.xml", {"{http://schemas.openxmlformats.org/drawingml/2006/main}t"}),
    ".odt": ("content.xml", {_ODF_TEXT + "p", _ODF_TEXT + "h"}),
    # Shared strings plus inline strings in the sheets; workbook.xml always
    # matches, so a numbers-only workbook reads as "" instead of an error.
    ".xlsx": ("xl/*.xml", {"{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"}),
}

# RTF groups that hold metadata, not document text.
_RTF_SKIP = {"fonttbl", "colortbl", "stylesheet", "info", "pict", "header", "footer",
             "listtable", "listoverridetable", "rsidtbl", "themedata", "latentstyles",
             "datastore", "xmlnstbl", "colorschememapping", "generator"}
_RTF_TOKEN = re.compile(r"\\([a-zA-Z]+)(-?\d+)? ?|\\'([0-9a-fA-F]{2})|\\(.)|([{}])|([^\\{}]+)", re.S)


def _rtf_to_text(rtf):
    """Plain text of an RTF document: a minimal reader (no dependency) that
    drops metadata groups, decodes \\'hh (cp1252) and \\uN escapes, and turns
    paragraph marks into newlines. Enough for clustering, not for rendering."""
    out, stack, skipping, swallow = [], [], False, 0
    for word, arg, hexcode, symbol, brace, text in _RTF_TOKEN.findall(rtf):
        if brace == "{":
            stack.append(skipping)
        elif brace == "}":
            skipping = stack.pop() if stack else False
        elif symbol == "*" or word in _RTF_SKIP:
            skipping = True
        elif skipping:
            continue
        elif swallow and (hexcode or text):
            # The ANSI fallback right after a \\uN escape.
            swallow -= 1
            if text and len(text) > 1:
                out.append(text[1:].replace("\r", "").replace("\n", ""))
        elif word in ("par", "line", "sect", "page", "row"):
            out.append("\n")
        elif word in ("tab", "cell"):
            out.append("\t")
        elif word == "u" and arg:
            out.append(chr(int(arg) % 65536))
            swallow = 1
        elif hexcode:
            out.append(bytes([int(hexcode, 16)]).decode("cp1252", errors="replace"))
        elif symbol:
            out.append({"~": " ", "_": "-", "-": ""}.get(symbol, symbol))
        elif text:
            out.append(text.replace("\r", "").replace("\n", ""))
    return re.sub(r"[ \t]+", " ", "".join(out)).strip()


def _read_zipped_xml(file_path, pattern, tags):
    """Text of every `tags` element in the zip members matching `pattern`.

    Raises KeyError if no member matches (same as ZipFile.open on a missing
    name), so callers treat it like any other malformed file.
    """
    with zipfile.ZipFile(file_path) as archive:
        members = sorted(
            (name for name in archive.namelist() if fnmatch.fnmatchcase(name, pattern)),
            key=lambda name: int(re.sub(r"\D", "", name) or 0),  # slide2 before slide10
        )
        if not members:
            raise KeyError(pattern)
        texts = []
        for member in members:
            with archive.open(member) as f:
                root = ET.parse(f).getroot()
            texts.extend("".join(node.itertext()) for node in root.iter() if node.tag in tags)
    return "\n".join(text for text in texts if text)

def read_file(file_path, max_chars=None):
    """Text content of a document; with max_chars, only (about) that much
    of its start - PDFs stop extracting pages and text files stop reading
    once they have it, which is most of the cost on large files."""
    try:
        # Check if file exists and is accessible
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Permission denied for file: {file_path}")
        
        # Get file size to check for empty files
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.info(f"Empty file detected: {file_path}")
            return ""
        
        # Determine file type
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Handle PDF files
        if ext == '.pdf':
            try:
                with open(file_path, 'rb') as file:
                    reader = PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                        if max_chars and len(text) >= max_chars:
                            break
                    if not text:
                        logger.info(f"No text found in PDF file: {file_path}")
                        return ""
                    return text[:max_chars]
            except PdfReadError as e:
                logger.error(f"Invalid PDF format: {file_path} - {str(e)}")
                raise ValueError(f"Invalid PDF format: {file_path}")
        
        # Handle Office documents (docx/pptx/odt are zips of XML parts)
        elif ext in ZIPPED_XML_FORMATS:
            try:
                text = _read_zipped_xml(file_path, *ZIPPED_XML_FORMATS[ext])
                if not text:
                    logger.info(f"No text found in {ext} file: {file_path}")
                return text[:max_chars]
            except (zipfile.BadZipFile, KeyError, ET.ParseError) as e:
                logger.error(f"Invalid {ext} format: {file_path} - {str(e)}")
                raise ValueError(f"Invalid {ext} format: {file_path}")

        elif ext == '.rtf':
            with open(file_path, 'rb') as file:
                # RTF is 7-bit ASCII; anything else is escaped inside it.
                text = _rtf_to_text(file.read().decode('latin-1'))
            if not text:
                logger.info(f"No text found in RTF file: {file_path}")
            return text[:max_chars]

        # Handle text files
        elif ext in ['.txt', '.md', '.csv', '.xml']:
            try:
                with open(file_path, 'rb') as file:
                    # Up to 4 bytes per character in UTF-8.
                    raw_content = file.read(max_chars * 4 if max_chars else -1)
                # chardet is slow on big inputs; the first 64 KB are plenty.
                encoding = chardet.detect(raw_content[:65536])['encoding'] or 'utf-8'
                # "ignore": a capped read can end mid-character.
                text = raw_content.decode(encoding, errors='ignore' if max_chars else 'strict').strip()
                if not text:
                    logger.info(f"No text found in file: {file_path}")
                    return ""
                return text[:max_chars]
            except Exception as e:
                logger.error(f"Error reading text file: {file_path} - {str(e)}")
                raise ValueError(f"Error reading text file: {file_path}")
        
        # Handle unsupported file types
        else:
            logger.warning(f"Unsupported file type: {ext}")
            raise ValueError(f"Unsupported file type: {ext}")
            
    except Exception as e:
        logger.error(f"Unexpected error reading file {file_path}: {str(e)}")
        raise

def create_weighted_text(path, content, path_weight=2):
        # The name as plain words: "receita_pao-de.queijo.txt" -> "receita pao
        # de queijo". The extension is file type, not topic, and glued
        # separators turn a name into one unknown token - for the embedding,
        # and for YAKE, which used to put "feijoada.txt" into folder names.
        name = re.sub(r"[\W_]+", " ", os.path.splitext(str(path))[0]).strip() or str(path)
        # Space-separated: "a.pdfa.pdf" tokenizes as one unknown blob, so the
        # repetition added noise instead of weight.
        weighted_path = " ".join([name] * path_weight)
        if pd.isna(content):
            return weighted_path
        return weighted_path + " " + content

# Supported language models (small/lightweight versions for efficiency)
VOSK_LANGUAGE_MODELS = {
    "en": "vosk-model-small-en-us-0.15",      # English (US)
    "zh": "vosk-model-small-cn-0.22",          # Chinese
    "ru": "vosk-model-small-ru-0.22",          # Russian
    "fr": "vosk-model-small-fr-0.22",          # French
    "de": "vosk-model-small-de-0.15",          # German
    "es": "vosk-model-small-es-0.42",          # Spanish
    "pt": "vosk-model-small-pt-0.3",           # Portuguese
    "tr": "vosk-model-small-tr-0.3",           # Turkish
    "vi": "vosk-model-small-vn-0.4",           # Vietnamese
    "it": "vosk-model-small-it-0.22",          # Italian
    "nl": "vosk-model-small-nl-0.22",          # Dutch
    "uk": "vosk-model-small-uk-v3-nano",       # Ukrainian
    "ja": "vosk-model-small-ja-0.22",          # Japanese
    "hi": "vosk-model-small-hi-0.22",          # Hindi
    "ko": "vosk-model-small-ko-0.22",          # Korean
    "ar": "vosk-model-ar-mgb2-0.4",            # Arabic
    "fa": "vosk-model-small-fa-0.5",           # Persian/Farsi
    "pl": "vosk-model-small-pl-0.22",          # Polish
    "ca": "vosk-model-small-ca-0.4",           # Catalan
    "cs": "vosk-model-small-cs-0.4-rhasspy",   # Czech
}

# Persistent, unlike the system temp dir (often wiped on reboot, which meant
# re-downloading the model every boot).
VOSK_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "vosk")

# Default language for transcription
_vosk_language = "en"
_vosk_models = {}

def set_transcription_language(lang_code):
    """
    Set the language for audio transcription.
    
    Args:
        lang_code (str): ISO 639-1 language code (e.g., 'en', 'es', 'fr', 'de', 'zh')
    
    Supported languages: en (English), zh (Chinese), ru (Russian), fr (French),
    de (German), es (Spanish), pt (Portuguese), tr (Turkish), vi (Vietnamese),
    it (Italian), nl (Dutch), uk (Ukrainian), ja (Japanese), hi (Hindi),
    ko (Korean), ar (Arabic), fa (Persian), pl (Polish), ca (Catalan), cs (Czech)
    """
    global _vosk_language
    if lang_code in VOSK_LANGUAGE_MODELS:
        _vosk_language = lang_code
        logger.info(f"Transcription language set to: {lang_code}")
    else:
        logger.warning(f"Unsupported language code: {lang_code}. Using default (en).")
        _vosk_language = "en"

def _get_vosk_model(lang_code=None):
    """Lazily load the Vosk model for the specified language."""
    global _vosk_models
    
    if not VOSK_AVAILABLE:
        return None
    
    lang = lang_code or _vosk_language
    
    if lang in _vosk_models:
        return _vosk_models[lang]
    
    import urllib.request

    model_name = VOSK_LANGUAGE_MODELS.get(lang, VOSK_LANGUAGE_MODELS["en"])
    model_path = os.path.join(VOSK_MODEL_DIR, model_name)

    if not os.path.exists(model_path):
        logger.info(f"Downloading Vosk model for language '{lang}' to {VOSK_MODEL_DIR}...")
        model_url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
        zip_path = os.path.join(VOSK_MODEL_DIR, f"{model_name}.zip")

        try:
            os.makedirs(VOSK_MODEL_DIR, exist_ok=True)
            urllib.request.urlretrieve(model_url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(VOSK_MODEL_DIR)
            os.remove(zip_path)
        except Exception as e:
            logger.error(f"Failed to download Vosk model for '{lang}': {str(e)}")
            return None
    
    logger.info(f"Loading Vosk model for language '{lang}'...")
    vosk.SetLogLevel(-1)  # Suppress Vosk logging
    _vosk_models[lang] = vosk.Model(model_path)
    return _vosk_models[lang]

def _convert_to_wav(input_path):
    """Convert audio file to WAV format suitable for Vosk (16kHz, mono, 16-bit)."""
    import tempfile

    # mkstemp creates the file atomically (mktemp only picks a name - racy).
    fd, output_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        # Use ffmpeg to convert audio to the required format
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",      # Mono
            "-sample_fmt", "s16",  # 16-bit
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return output_path
        logger.warning(f"ffmpeg conversion failed: {result.stderr}")
    except FileNotFoundError:
        logger.warning("ffmpeg not found, cannot convert audio file")
    except Exception as e:
        logger.warning(f"Error converting audio file: {str(e)}")

    os.remove(output_path)
    return None

def transcribe_audio(file_path, max_duration=60):
    """
    Transcribes audio content using Vosk (free, offline speech recognition).
    
    Args:
        file_path (str): Path to the audio file
        max_duration (int): Maximum duration in seconds to transcribe (default: 60)
        
    Returns:
        str: Transcribed text or None if transcription fails
    """
    if not VOSK_AVAILABLE:
        logger.info("Vosk not available, skipping transcription")
        return None
    
    try:
        model = _get_vosk_model()
        if model is None:
            return None
        
        logger.info(f"Transcribing audio file: {file_path}")
        
        # Convert to WAV if needed
        _, ext = os.path.splitext(file_path)
        if ext.lower() != ".wav":
            wav_path = _convert_to_wav(file_path)
            if wav_path is None:
                return None
            temp_file = True
        else:
            wav_path = file_path
            temp_file = False
        
        try:
            wf = wave.open(wav_path, "rb")
            
            # Verify audio format
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                logger.warning("Audio file must be mono 16-bit, attempting conversion...")
                wf.close()
                wav_path = _convert_to_wav(file_path)
                if wav_path is None:
                    return None
                temp_file = True
                wf = wave.open(wav_path, "rb")
            
            rec = vosk.KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)
            
            results = []
            frames_read = 0
            max_frames = max_duration * wf.getframerate()
            
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                
                frames_read += 4000
                if frames_read > max_frames:
                    break
                
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if result.get("text"):
                        results.append(result["text"])
            
            # Get final result
            final_result = json.loads(rec.FinalResult())
            if final_result.get("text"):
                results.append(final_result["text"])
            
            wf.close()
            
            text = " ".join(results).strip()
            
            if text:
                # Limit the transcription length
                max_chars = 2000
                if len(text) > max_chars:
                    text = text[:max_chars] + "..."
                return text
            
            logger.info(f"No speech detected in audio file: {file_path}")
            return None
            
        finally:
            if temp_file and wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        
    except Exception as e:
        logger.warning(f"Error transcribing audio file {file_path}: {str(e)}")
        return None

def read_audio_metadata(file_path):
    """
    Extracts metadata and transcribes content from audio files for clustering purposes.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        str: Combined metadata and transcription, or None if extraction fails
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Permission denied for file: {file_path}")
        
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext not in audio_extensions:
            logger.warning(f"Unsupported audio file type: {ext}")
            return None
        
        content_parts = []
        
        # Extract metadata using TinyTag
        if TINYTAG_AVAILABLE:
            try:
                tag = TinyTag.get(file_path)
                metadata_parts = []
                if tag.title:
                    metadata_parts.append(tag.title)
                if tag.artist:
                    metadata_parts.append(tag.artist)
                if tag.album:
                    metadata_parts.append(tag.album)
                if tag.genre:
                    metadata_parts.append(tag.genre)
                if tag.composer:
                    metadata_parts.append(tag.composer)
                
                if metadata_parts:
                    content_parts.append(" ".join(metadata_parts))
            except Exception as e:
                logger.warning(f"Error extracting metadata from {file_path}: {str(e)}")
        
        # Transcribe audio content using Vosk
        transcription = transcribe_audio(file_path)
        if transcription:
            content_parts.append(transcription)
        
        if content_parts:
            return " ".join(content_parts)
        
        logger.info(f"No content extracted from audio file: {file_path}")
        return None
        
    except Exception as e:
        logger.warning(f"Error reading audio file {file_path}: {str(e)}")
        return None