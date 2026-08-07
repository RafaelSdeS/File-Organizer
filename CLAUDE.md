# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AI-powered CLI that scans a directory of documents, clusters them by semantic content similarity (sentence embeddings + KMeans), and moves/organizes files into folders named after cluster keywords (YAKE keyword extraction).

## Commands

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the CLI (prompts interactively for a directory path):
```bash
python -m src.document_analyzer.main
```

Run all tests:
```bash
python -m unittest discover tests
```

Run a single test file / test case / test method:
```bash
python -m unittest tests.test_analyzer
python -m unittest tests.test_analyzer.TestDocumentAnalyzer
python -m unittest tests.test_analyzer.TestDocumentAnalyzer.test_extract_keywords
```

## Architecture

Single pipeline, driven from `src/document_analyzer/analyzer.py::DocumentAnalyzer`:

1. **`analyze_directory(path)`** — top-level entry point.
   - Scans the directory (`os.scandir`, non-recursive at this level for files, but recurses into subfolders via `_analyze_folder`).
   - Only files with extensions in `readable_files` (`.pdf`, `.txt`, `.docx`, `.xml`) get their content read via `utils.read_file`; others get `Content: None`.
   - Builds a `pandas.DataFrame` with one row per top-level entry (`Path`, `Content`).
   - `utils.create_weighted_text` concatenates the filename (repeated `path_weight` times) with file content into a `Text` column — this weighting biases embeddings toward filename similarity, not just content.
   - Embeds `Text` via `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")`.
   - `_find_optimal_clusters` picks a k (2..min(n,10)) by combining elbow-method (2nd derivative of inertia), silhouette score, and Calinski-Harabasz score — majority vote among the three. The elbow candidate is only included when there are ≥3 k-values to test (≥5 documents), since a 2nd derivative needs 3+ points; below that it falls back to silhouette/Calinski only, and directories with ≤2 usable documents skip clustering entirely (`k=1`, everything in one folder) rather than crashing.
   - Runs `KMeans` with that k, then `_organize_clusters` names each cluster folder from its top-2 YAKE keywords (falls back to `Cluster_<n>` if no keywords extracted). If two clusters produce the same folder name, their files are merged into that one folder (`setdefault(...).extend(...)`) rather than one cluster's files overwriting the other's.
   - Returns `{folder_name: [original_filenames]}`.

2. **`_analyze_folder(folder_path)`** — recursively walks a subdirectory, concatenating all descendant file names/contents into a single `Content` blob so the whole subfolder is treated as one document for clustering purposes.

3. **`organize_files(folder_structure, source_dir, target_dir=None)`** — creates the cluster folders (under `target_dir`, defaulting to `source_dir`) and moves (`shutil.move`) each original file into its assigned folder — works across filesystems/drives, unlike `os.rename`. This is a real filesystem mutation — moves files out of `source_dir` in place unless `target_dir` is given.

`src/document_analyzer/utils.py`:
- `read_file` — dispatches by extension: PyPDF2 for `.pdf`, a minimal stdlib `zipfile`/`xml.etree.ElementTree` extraction (`_read_docx`) for `.docx` (reads `word/document.xml`, pulls `<w:t>` text runs — no external docx library needed), `chardet`-detected-encoding text read for `.txt`/`.xml`. Raises on unsupported/unreadable files; empty files return `""`.
- `create_weighted_text` — the filename/content weighting used for embeddings (see above).
- `analyze_document_content` — computes a suggested extraction length from section/keyword counts; defined but not currently called anywhere in the pipeline.

`src/document_analyzer/main.py` is a thin interactive wrapper: prompts for a path, prints the resulting cluster structure, then calls `organize_files` on the same path (source == target, so files move in place).

## Notes for changes here

- `readable_files` in `analyzer.py` is the single source of truth for which extensions get content-read vs. treated as name-only; keep it in sync with `read_file`'s extension handling in `utils.py`.
- Tests in `tests/test_analyzer.py` patch `SentenceTransformer`/`yake.KeywordExtractor` in `setUp` (via `src.document_analyzer.analyzer.SentenceTransformer` / `...analyzer.yake.KeywordExtractor` — the names as looked up from inside `analyzer.py`, not their origin modules) before `DocumentAnalyzer()` is constructed, so `self.mock_model`/`self.mock_kw_extractor` are available in every test. Configure `.encode.return_value` / `.extract_keywords.return_value` per test as needed.
