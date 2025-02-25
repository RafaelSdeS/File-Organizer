import PyPDF2
import pandas as pd
import logging
import os
import chardet

logger = logging.getLogger(__name__)

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