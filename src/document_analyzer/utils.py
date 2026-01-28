import PyPDF2
import pandas as pd
import logging
import os
import chardet

try:
    from tinytag import TinyTag
    TINYTAG_AVAILABLE = True
except ImportError:
    TINYTAG_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
    _whisper_model = None
except ImportError:
    WHISPER_AVAILABLE = False
    _whisper_model = None

logger = logging.getLogger(__name__)

audio_extensions = [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".aiff", ".opus"]

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
            except PyPDF2.PdfReadError as e:
                logger.error(f"Invalid PDF format: {file_path} - {str(e)}")
                raise ValueError(f"Invalid PDF format: {file_path}")
        
        # Handle text files
        elif ext in ['.txt', '.docx', '.xml']:
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

def analyze_document_content(text):
        """
        Analyzes document content to determine optimal text extraction length.
        The length is calculated based on:
        - Number of sections (paragraph breaks)
        - Number of keywords (words > 5 characters)
        """
        if not text:
            return 1000  # Default minimum length
            
        # Count content indicators
        section_markers = text.count('\n\n') + text.count('. ') + text.count(' ')
        keywords = len([word for word in text.split() if len(word) > 5])
        
        # Calculate optimal length
        base_length = 1000
        additional_length = min(
            section_markers * 200,  # Add length for each section
            keywords * 50,          # Add length for each keyword
            3000                    # Maximum additional length
        )
        return base_length + additional_length

def create_weighted_text(path, content, path_weight=2):
        if pd.isna(content):
            return str(path) * path_weight
        return (str(path) * path_weight) + " " + content

def _get_whisper_model():
    """Lazily load the Whisper model to avoid loading it multiple times."""
    global _whisper_model
    if _whisper_model is None and WHISPER_AVAILABLE:
        logger.info("Loading Whisper model (base)...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def transcribe_audio(file_path, max_duration=60):
    """
    Transcribes audio content using OpenAI Whisper.
    
    Args:
        file_path (str): Path to the audio file
        max_duration (int): Maximum duration in seconds to transcribe (default: 60)
        
    Returns:
        str: Transcribed text or None if transcription fails
    """
    if not WHISPER_AVAILABLE:
        logger.info("Whisper not available, skipping transcription")
        return None
    
    try:
        model = _get_whisper_model()
        if model is None:
            return None
        
        logger.info(f"Transcribing audio file: {file_path}")
        result = model.transcribe(file_path, fp16=False)
        
        text = result.get("text", "").strip()
        if text:
            # Limit the transcription length for very long audio files
            max_chars = 2000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            return text
        
        logger.info(f"No speech detected in audio file: {file_path}")
        return None
        
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
        
        # Transcribe audio content using Whisper
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