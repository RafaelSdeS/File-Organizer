# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AI-powered CLI that scans a directory of documents, clusters them by semantic content similarity (sentence embeddings + KMeans), and moves/organizes files into folders named after cluster keywords (YAKE keyword extraction).

## Commands

Install dependencies:
```bash
pip install -r requirements.txt
```
Or `pip install -e .` (uses `pyproject.toml`) to also get a real `document_analyzer` top-level package and the `organize-files` console script.

Run the CLI (prompts interactively for a directory path if omitted; see `main.py::parse_args` for the full flag set — `--target-dir`, `--dry-run`, `-y`, `--path-weight`, `--max-clusters`, `--keyword-ngram`, `--keyword-count`):
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

0. **`DocumentAnalyzer(path_weight=2, max_clusters=10, yake_ngram=2, yake_top=5)`** — all four tuning knobs are constructor params (also exposed as CLI flags in `main.py`); defaults match the original hardcoded values.

1. **`analyze_directory(path)`** — top-level entry point.
   - Scans the directory (`os.scandir`, non-recursive at this level for files, but recurses into subfolders via `_analyze_folder`).
   - Files with extensions in `readable_files` (`.pdf`, `.txt`, `.docx`, `.xml`) get content via `utils.read_file`; extensions in `audio_files` (`.mp3`, `.wav`, `.flac`, `.aac`, `.ogg`, `.wma`, `.m4a`, `.aiff`, `.opus`) get content via `utils.read_audio_metadata`; everything else gets `Content: None`. Extension matching is case-insensitive (`ext.lower()`).
   - Builds a `pandas.DataFrame` with one row per top-level entry (`Path`, `Content`).
   - `utils.create_weighted_text` concatenates the filename (repeated `self.path_weight` times) with file content into a `Text` column — this weighting biases embeddings toward filename similarity, not just content.
   - Embeds `Text` via `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")`. Logs an info line before this step and before cluster-count search — both can be slow on larger directories.
   - `_find_optimal_clusters` picks a k (2..min(n, `self.max_clusters`)) by combining elbow-method (2nd derivative of inertia), silhouette score, and Calinski-Harabasz score — majority vote among the three. The elbow candidate is only included when there are ≥3 k-values to test (≥5 documents), since a 2nd derivative needs 3+ points; below that it falls back to silhouette/Calinski only, and directories with ≤2 usable documents skip clustering entirely (`k=1`, everything in one folder) rather than crashing.
   - Runs `KMeans` with that k, then `_organize_clusters` names each cluster folder from its top-2 YAKE keywords (falls back to `Cluster_<n>` if no keywords extracted). If two clusters produce the same folder name, their files are merged into that one folder (`setdefault(...).extend(...)`) rather than one cluster's files overwriting the other's.
   - Returns `{folder_name: [original_filenames]}`.

2. **`_analyze_folder(folder_path)`** — recursively walks a subdirectory, concatenating all descendant file names/contents into a single `Content` blob so the whole subfolder is treated as one document for clustering purposes.

3. **`organize_files(folder_structure, source_dir, target_dir=None)`** — creates the cluster folders (under `target_dir`, defaulting to `source_dir`) and moves (`shutil.move`) each original file into its assigned folder — works across filesystems/drives, unlike `os.rename`. This is a real filesystem mutation — moves files out of `source_dir` in place unless `target_dir` is given.

`src/document_analyzer/utils.py`:
- `read_file` — dispatches by extension: PyPDF2 for `.pdf`, a minimal stdlib `zipfile`/`xml.etree.ElementTree` extraction (`_read_docx`) for `.docx` (reads `word/document.xml`, pulls `<w:t>` text runs — no external docx library needed), `chardet`-detected-encoding text read for `.txt`/`.xml`. Raises on unsupported/unreadable files; empty files return `""`.
- `create_weighted_text` — the filename/content weighting used for embeddings (see above).
- `analyze_document_content` — computes a suggested extraction length from section/keyword counts; defined but not currently called anywhere in the pipeline.
- `read_audio_metadata` — for `audio_files` extensions, combines TinyTag metadata (title/artist/album/genre/composer) with a Vosk offline transcription (via `transcribe_audio`, which shells out to `ffmpeg` to normalize to 16kHz mono WAV first). `TINYTAG_AVAILABLE`/`VOSK_AVAILABLE` are set at import time from `try/except ImportError`, so the whole audio path degrades to `None` (no crash) if `tinytag`/`vosk`/`ffmpeg` aren't installed — those are optional deps in `requirements.txt`. `set_transcription_language(lang_code)` picks from `VOSK_LANGUAGE_MODELS`; models are lazily downloaded to the system temp dir on first use per language.

`src/document_analyzer/main.py` — CLI wrapper (`argparse`, see `parse_args`): takes `path` as a positional arg (prompted interactively if omitted), builds `DocumentAnalyzer` from the tuning flags, prints the proposed cluster structure, then either stops (`--dry-run`), moves without asking (`-y/--yes`), or asks for `y`/`N` confirmation before calling `organize_files` — nothing moves without an explicit opt-in. `--target-dir` is threaded through to `organize_files`; defaults to `path` (in-place) when omitted.

`src/document_analyzer/__init__.py` re-exports `DocumentAnalyzer` so `from document_analyzer import DocumentAnalyzer` works once the package is installed (`pip install -e .`, via `pyproject.toml`).

## Notes for changes here

- `readable_files` in `analyzer.py` is the single source of truth for which extensions get content-read vs. treated as name-only; keep it in sync with `read_file`'s extension handling in `utils.py`.
- Tests in `tests/test_analyzer.py` patch `SentenceTransformer`/`yake.KeywordExtractor` in `setUp` (via `src.document_analyzer.analyzer.SentenceTransformer` / `...analyzer.yake.KeywordExtractor` — the names as looked up from inside `analyzer.py`, not their origin modules) before `DocumentAnalyzer()` is constructed, so `self.mock_model`/`self.mock_kw_extractor` are available in every test. Configure `.encode.return_value` / `.extract_keywords.return_value` per test as needed.
- `tests/test_main.py` patches `src.document_analyzer.main.DocumentAnalyzer` wholesale (never loads the real model) and drives `main()` via `sys.argv`/`input` patches to test the confirm/dry-run/`--yes` control flow in isolation from the real pipeline.
- `tests/test_utils.py` covers `read_file` per extension (a real minimal `.docx` zip is built in-test; PDF is exercised via a mocked `PyPDF2.PdfReader`) and the audio path with `TINYTAG_AVAILABLE`/`VOSK_AVAILABLE`/`TinyTag` patched (`create=True` for `TinyTag` since `tinytag` isn't an installed dependency in this environment).
