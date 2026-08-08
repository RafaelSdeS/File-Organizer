# File Organizer
AI-powered document organization system that uses natural language processing and machine learning to group similar documents together.

## Overview
A sophisticated tool that analyzes document contents and automatically organizes them into meaningful clusters based on their semantic similarity. It leverages advanced techniques including sentence embeddings, keyword extraction, and unsupervised clustering to understand document relationships and structure.

## Key Features
- **Intelligent Clustering**: sentence embeddings + KMeans group files by semantic similarity, not just file type
- **Automatic Naming**: cluster folders are named from top YAKE keywords extracted from each cluster's content
- **Multi-format Support**: reads `.pdf`, `.txt`, `.docx`, `.xml`, and transcribes audio (`.mp3`, `.wav`, `.flac`, `.aac`, `.ogg`, `.wma`, `.m4a`, `.aiff`, `.opus`) via Vosk
- **Recursive Processing**: subdirectories are treated as a single document (their contents concatenated) for clustering
- **Dry-run / confirmation**: preview the proposed structure before any files move

## Installation
```bash
pip install -r requirements.txt
```
Or install as an editable package, which also enables `from document_analyzer import DocumentAnalyzer` and an `organize-files` console command:
```bash
pip install -e .
```
Audio transcription also needs `ffmpeg` on `PATH`. Without it (or without `tinytag`/`vosk` installed — pull them in with `pip install -e .[audio]`), audio files are still clustered by filename, just without transcribed content.

## Usage

### CLI
```bash
python -m src.document_analyzer.main [path] [--target-dir DIR] [--dry-run] [-y]
    [--path-weight N] [--max-clusters N] [--keyword-ngram N] [--keyword-count N] [--lang CODE]
```
- `path` — directory to organize (prompted interactively if omitted)
- `--target-dir` — move organized files here instead of in place
- `--dry-run` — print the proposed folder structure without moving anything
- `-y, --yes` — skip the move confirmation prompt
- `--path-weight` — how many times the filename is repeated when weighting embeddings (default: 2)
- `--max-clusters` — upper bound on the number of clusters to consider (default: 10)
- `--keyword-ngram` — max n-gram size for YAKE keyword extraction (default: 2)
- `--keyword-count` — number of keywords YAKE extracts per cluster; the top 2 become the folder name (default: 5)
- `--lang` — language of your documents, used to pick the keyword stopword list (default: `en`; e.g. `pt`, `es`). Comma-separate for mixed directories: `--lang pt,en`

### Python API
```python
from document_analyzer import DocumentAnalyzer

# All constructor args are optional; shown here at their defaults.
analyzer = DocumentAnalyzer(path_weight=2, max_clusters=10, yake_ngram=2, yake_top=5, lang="en")

try:
    folder_structure = analyzer.analyze_directory("/path/to/documents")

    # target_dir is optional; defaults to source_dir (files moved in place)
    analyzer.organize_files(folder_structure,
                          source_dir="/path/to/documents",
                          target_dir="/path/to/organized_documents")
except Exception as e:
    logger.error(f"Analysis failed: {str(e)}")
```

## Error Handling
`analyze_directory` raises `ValueError` / `FileNotFoundError` / `PermissionError` for an invalid path, and `RuntimeError` if embedding or clustering fails. Per-file read errors (unreadable PDF, bad encoding, etc.) are logged and the file is still included in clustering, just without content. All errors are logged via Python's `logging` module.

## License
[MIT](LICENSE)
