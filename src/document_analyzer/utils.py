import PyPDF2
from PyPDF2.errors import PdfReadError
import pandas as pd
import logging
import os
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

audio_extensions = [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".aiff", ".opus"]

_DOCX_TEXT_TAG = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"


def _read_docx(file_path):
    with zipfile.ZipFile(file_path) as archive:
        with archive.open("word/document.xml") as f:
            tree = ET.parse(f)
    return "\n".join(node.text for node in tree.iter(_DOCX_TEXT_TAG) if node.text)

def read_file(file_path):
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
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                    if not text:
                        logger.info(f"No text found in PDF file: {file_path}")
                        return ""
                    return text
            except PdfReadError as e:
                logger.error(f"Invalid PDF format: {file_path} - {str(e)}")
                raise ValueError(f"Invalid PDF format: {file_path}")
        
        # Handle Word documents (docx is a zip of XML parts)
        elif ext == '.docx':
            try:
                text = _read_docx(file_path)
                if not text:
                    logger.info(f"No text found in DOCX file: {file_path}")
                return text
            except (zipfile.BadZipFile, KeyError, ET.ParseError) as e:
                logger.error(f"Invalid DOCX format: {file_path} - {str(e)}")
                raise ValueError(f"Invalid DOCX format: {file_path}")

        # Handle text files
        elif ext in ['.txt', '.xml']:
            try:
                # Detect encoding automatically
                with open(file_path, 'rb') as file:
                    raw_content = file.read()
                    encoding = chardet.detect(raw_content)['encoding']
                    if not encoding:
                        encoding = 'utf-8'  # Fallback to utf-8
                
                # Read with detected encoding
                with open(file_path, 'r', encoding=encoding) as file:
                    text = file.read().strip()
                    if not text:
                        logger.info(f"No text found in file: {file_path}")
                        return ""
                    return text
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
        # Space-separated: "a.pdfa.pdf" tokenizes as one unknown blob, so the
        # repetition added noise instead of weight.
        weighted_path = " ".join([str(path)] * path_weight)
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
    
    import tempfile
    import urllib.request
    import zipfile
    
    model_name = VOSK_LANGUAGE_MODELS.get(lang, VOSK_LANGUAGE_MODELS["en"])
    model_path = os.path.join(tempfile.gettempdir(), model_name)
    
    if not os.path.exists(model_path):
        logger.info(f"Downloading Vosk model for language '{lang}'...")
        model_url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
        zip_path = os.path.join(tempfile.gettempdir(), f"vosk-model-{lang}.zip")
        
        try:
            urllib.request.urlretrieve(model_url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tempfile.gettempdir())
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
    
    output_path = tempfile.mktemp(suffix=".wav")
    
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
        
        if result.returncode != 0:
            logger.warning(f"ffmpeg conversion failed: {result.stderr}")
            return None
        
        return output_path
    except FileNotFoundError:
        logger.warning("ffmpeg not found, cannot convert audio file")
        return None
    except Exception as e:
        logger.warning(f"Error converting audio file: {str(e)}")
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