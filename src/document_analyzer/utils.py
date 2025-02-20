import PyPDF2
import pandas as pd

def read_file(file_path):
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                # Optimize text length based on content analysis
                return text[:analyze_document_content(text)]
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

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