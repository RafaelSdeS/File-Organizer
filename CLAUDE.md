# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AI-powered CLI that scans a directory of documents, clusters them by semantic content similarity (sentence embeddings + KMeans), and moves/organizes files into folders named after cluster keywords (YAKE keyword extraction).

Requires Python 3.11+ (`requires-python` in `pyproject.toml`) — needed for stdlib `tomllib`, used to read the optional `~/.config/file-organizer.toml` defaults file.

## Commands

Install dependencies:
```bash
pip install -r requirements.txt
```
Or `pip install -e .` (uses `pyproject.toml`) to also get a real `document_analyzer` top-level package and the `organize-files` console script.

Run the CLI (prompts interactively for a directory path if omitted; see `main.py::parse_args` for the full flag set — `--target-dir`, `--dry-run`, `-y`, `--path-weight`, `--max-clusters`, `--keyword-ngram`, `--keyword-count`, `--undo`, `--include-all`, `--exclude`, `--verbose`/`--quiet`):
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

0. **`DocumentAnalyzer(path_weight=2, max_clusters=10, yake_ngram=2, yake_top=5, skip_noise=True, exclude_patterns=None)`** — all six tuning knobs are constructor params (also exposed as CLI flags in `main.py`); defaults match the original hardcoded values (`skip_noise`/`exclude_patterns` are new and have no old behavior to match — filtering defaults on, exclude defaults to none). `max_clusters < 3` raises `ValueError` immediately (`_find_optimal_clusters`'s candidate range `range(2, max_k)` is empty otherwise, which used to surface as a bare `numpy`/`sklearn` exception).

1. **`analyze_directory(path)`** — top-level entry point.
   - Scans the directory (`os.scandir`, non-recursive at this level for files, but recurses into subfolders via `_analyze_folder`). Every entry is checked against `_is_noise_entry(name, self.skip_noise, self.exclude_patterns)`: when `skip_noise` (default on), a name starting with `.` or matching `DEFAULT_SKIP_NAMES` (`__pycache__`, `node_modules`) is skipped — this also incidentally hides `organize_files`' own `.file_organizer_created`/`.file_organizer_manifest_*.json` dotfiles from a later scan; `--include-all` (CLI) / `skip_noise=False` (constructor) disables just that default heuristic. Independently, any name matching one of `exclude_patterns` (`fnmatch` glob patterns, `--exclude PATTERN` on the CLI, repeatable) is always skipped, even under `--include-all` — it's an explicit user ask, not the default heuristic. A subfolder is skipped entirely (not recursed, not added as a document) if it's a symlink (`entry.is_symlink()` — prevents unbounded recursion on a symlink cycle) or if it contains `ORGANIZED_MARKER` (`.file_organizer_created`, dropped by `organize_files` into every cluster folder it creates — this stops a second run on an already-organized directory from re-clustering the first run's output as fresh documents); these two checks are unconditional, independent of both `skip_noise` and `exclude_patterns`.
   - Files with extensions in `readable_files` (`.pdf`, `.txt`, `.docx`, `.xml`) get content via `utils.read_file`; extensions in `audio_files` (`.mp3`, `.wav`, `.flac`, `.aac`, `.ogg`, `.wma`, `.m4a`, `.aiff`, `.opus`) get content via `utils.read_audio_metadata`; everything else gets `Content: None`. Extension matching is case-insensitive (`ext.lower()`).
   - Builds a `pandas.DataFrame` with one row per top-level entry (`Path`, `Content`).
   - `utils.create_weighted_text` concatenates the filename (repeated `self.path_weight` times) with file content into a `Text` column — this weighting biases embeddings toward filename similarity, not just content.
   - Embeds `Text` via `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")`. Logs an info line before this step and before cluster-count search — both can be slow on larger directories.
   - `_find_optimal_clusters` picks a k (2..min(`self.max_clusters`, n-1), inclusive of both ends) by combining elbow-method (2nd derivative of inertia), silhouette score, and Calinski-Harabasz score — majority vote among the three. The upper bound is capped at `n - 1` rather than `n` because `silhouette_score` requires strictly fewer clusters than samples (k == n would crash mid-loop). The elbow candidate is only included when there are ≥3 k-values to test (≥5 documents), since a 2nd derivative needs 3+ points; below that it falls back to silhouette/Calinski only, and directories with ≤2 usable documents skip clustering entirely (`k=1`, everything in one folder) rather than crashing.
   - Runs `KMeans` with that k, then `_organize_clusters` names each cluster folder from its top-2 YAKE keywords (falls back to `Cluster_<n>` if no keywords extracted). If two clusters produce the same folder name, their files are merged into that one folder (`setdefault(...).extend(...)`) rather than one cluster's files overwriting the other's.
   - Returns `{folder_name: [original_filenames]}`.

2. **`_analyze_folder(folder_path)`** — recursively walks a subdirectory, concatenating all descendant file names/contents into a single `Content` blob so the whole subfolder is treated as one document for clustering purposes. Uses `entry.path` directly (never `os.path.join(folder_path, entry)` — `DirEntry.__fspath__` already returns the full path, so joining it again used to double the path and made every subfolder containing a readable/audio file raise `FileNotFoundError`, silently dropping that whole subfolder). Per-file read errors are caught here too, same as the top-level scan in `analyze_directory`, degrading to filename-only instead of losing the rest of the subfolder. Nested symlinked directories are skipped the same way as at the top level, for the same cycle-prevention reason. Applies the same `_is_noise_entry` filtering as the top-level scan, so a nested `.git`/`__pycache__`/`node_modules`/excluded-pattern entry doesn't get folded into the subfolder's content blob either.

3. **`organize_files(folder_structure, source_dir, target_dir=None)`** — creates the cluster folders (under `target_dir`, defaulting to `source_dir`), drops `ORGANIZED_MARKER` into each one, and moves (`shutil.move`) each original file into its assigned folder — works across filesystems/drives, unlike `os.rename`. This is a real filesystem mutation — moves files out of `source_dir` in place unless `target_dir` is given. Before each move it checks whether the destination path already exists and skips (with a warning) instead of moving — `shutil.move` → `os.rename` on POSIX silently clobbers an existing destination with no error, so the `except FileExistsError` this used to rely on was dead code there (Windows-only). Every successful move is recorded as `{"source": ..., "destination": ...}` and, if anything moved, written as a JSON manifest to `target_dir` (`.file_organizer_manifest_<timestamp>.json`); the method returns that manifest's path (or `None` if nothing moved). **`undo_organize(manifest_path)`** (module-level function, not a method) reads a manifest and moves each destination back to its source, skipping (with a warning) if the destination is missing or the source path is already occupied — mirrors the same never-clobber rule in reverse.

`src/document_analyzer/utils.py`:
- `read_file` — dispatches by extension: PyPDF2 for `.pdf`, a minimal stdlib `zipfile`/`xml.etree.ElementTree` extraction (`_read_docx`) for `.docx` (reads `word/document.xml`, pulls `<w:t>` text runs — no external docx library needed), `chardet`-detected-encoding text read for `.txt`/`.xml`. Raises on unsupported/unreadable files; empty files return `""`.
- `create_weighted_text` — the filename/content weighting used for embeddings (see above).
- `read_audio_metadata` — for `audio_files` extensions, combines TinyTag metadata (title/artist/album/genre/composer) with a Vosk offline transcription (via `transcribe_audio`, which shells out to `ffmpeg` to normalize to 16kHz mono WAV first). `TINYTAG_AVAILABLE`/`VOSK_AVAILABLE` are set at import time from `try/except ImportError`, so the whole audio path degrades to `None` (no crash) if `tinytag`/`vosk`/`ffmpeg` aren't installed — those are optional deps in `requirements.txt`. `set_transcription_language(lang_code)` picks from `VOSK_LANGUAGE_MODELS`; models are lazily downloaded to the system temp dir on first use per language.

`src/document_analyzer/main.py` — CLI wrapper (`argparse`, see `parse_args`): takes `path` as a positional arg (prompted interactively if omitted), builds `DocumentAnalyzer` from the tuning flags, prints the proposed cluster structure, then either stops (`--dry-run`), moves without asking (`-y/--yes`), or asks for `y`/`N` confirmation before calling `organize_files` — nothing moves without an explicit opt-in. `--target-dir` is threaded through to `organize_files`; defaults to `path` (in-place) when omitted. `--undo MANIFEST` short-circuits all of this at the top of `main()` — if passed, it calls `undo_organize` and returns immediately, without touching `path`, `DocumentAnalyzer`, or any tuning flags. `--verbose`/`--quiet` (mutually exclusive) pick the root logging level (`DEBUG`/`WARNING`, default `INFO`); `logging.basicConfig(..., force=True)` is called at the top of `main()` (not at import time) so the level reflects that run's flags — `force=True` matters in-process too, since `basicConfig` is normally a no-op after the first call. `_load_config()` reads `CONFIG_PATH` (`~/.config/file-organizer.toml`, module-level constant so tests can patch it) via stdlib `tomllib` at the top of `parse_args()`, before the parser is built; each `add_argument`'s `default=` pulls from that dict (e.g. `config.get("path_weight", 2)`) so an explicit CLI flag always overrides the config value, and a missing or malformed config file (caught as `tomllib.TOMLDecodeError`/`OSError`, logged as a warning) just falls back to the hardcoded defaults.

`src/document_analyzer/__init__.py` re-exports `DocumentAnalyzer` so `from document_analyzer import DocumentAnalyzer` works once the package is installed (`pip install -e .`, via `pyproject.toml`).

## Notes for changes here

- `readable_files` in `analyzer.py` is the single source of truth for which extensions get content-read vs. treated as name-only; keep it in sync with `read_file`'s extension handling in `utils.py`.
- Tests in `tests/test_analyzer.py` patch `SentenceTransformer`/`yake.KeywordExtractor` in `setUp` (via `src.document_analyzer.analyzer.SentenceTransformer` / `...analyzer.yake.KeywordExtractor` — the names as looked up from inside `analyzer.py`, not their origin modules) before `DocumentAnalyzer()` is constructed, so `self.mock_model`/`self.mock_kw_extractor` are available in every test. Configure `.encode.return_value` / `.extract_keywords.return_value` per test as needed.
- `tests/test_main.py` patches `src.document_analyzer.main.DocumentAnalyzer` wholesale (never loads the real model) and drives `main()` via `sys.argv`/`input` patches to test the confirm/dry-run/`--yes` control flow in isolation from the real pipeline.
- `tests/test_utils.py` covers `read_file` per extension (a real minimal `.docx` zip is built in-test; PDF is exercised via a mocked `PyPDF2.PdfReader`) and the audio path with `TINYTAG_AVAILABLE`/`VOSK_AVAILABLE`/`TinyTag` patched (`create=True` for `TinyTag` since `tinytag` isn't an installed dependency in this environment).
