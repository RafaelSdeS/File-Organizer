# File Organizer
AI-powered document organization system that uses natural language processing and machine learning to group similar documents together.

## Overview
A sophisticated tool that analyzes document contents and automatically organizes them into meaningful clusters based on their semantic similarity. It leverages advanced techniques including sentence embeddings, keyword extraction, and unsupervised clustering to understand document relationships and structure.

## Key Features
- **Intelligent Clustering**: Uses AI-powered sentence embeddings to understand document meaning
- **Automatic Organization**: Creates structured folders based on document content
- **Keyword Extraction**: Identifies key topics within document groups
- **Robust Error Handling**: Comprehensive logging and error management
- **Recursive Processing**: Handles nested directory structures

## Technical Architecture
The system employs several advanced technologies:



Let's visualize the system's workflow:



## Installation Requirements
```bash
pip install pandas scikit-learn yake sentence-transformers numpy
```

## Usage Example
```python
from document_analyzer import DocumentAnalyzer

# Initialize analyzer
analyzer = DocumentAnalyzer()

try:
    # Analyze directory and get cluster organization
    folder_structure = analyzer.analyze_directory("/path/to/documents")
    
    # Organize files (optional target_dir parameter organizes in a different location)
    analyzer.organize_files(folder_structure, 
                          source_dir="/path/to/documents",
                          target_dir="/path/to/organized_documents")
except Exception as e:
    logger.error(f"Analysis failed: {str(e)}")
```


## Error Handling
The system implements comprehensive error handling for various scenarios:


All errors are logged using Python's logging module for easy debugging and monitoring.
