# TODO

Project identity, and the constraint every item below must respect: **this is a
local file organizer.** All processing (reading files, embedding, clustering,
moving files) happens on the user's machine. Nothing about a user's files or
folder contents gets sent to a server. The only network calls that exist today
are one-time model downloads (sentence-transformers weights from Hugging Face,
Vosk language models from alphacephei) that are cached locally after first
use — that's fine, it stays "local" after that. Any future feature that would
mean uploading file contents/paths anywhere is out of scope; don't propose it.

Priority order: **Critical** (can silently destroy or lose user data / crash)
→ **Correctness** (produces wrong results, no data loss) → **Features** →
**Housekeeping**.

## Critical — data loss / crash risks

- [x] **File moves silently overwrite existing files on Linux/Mac.**
      Verified: `shutil.move` on POSIX uses `os.rename`, which clobbers an
      existing destination with zero error and zero warning. The
      `except FileExistsError` handler in `organize_files`
      (`src/document_analyzer/analyzer.py`) is dead code on POSIX — it only
      fires on Windows. Concrete failure: two files named the same thing land in
      the same cluster folder (or you run the tool twice), and the second move
      **silently destroys** the first file's content. Fix: check
      `os.path.exists(destination_path)` before calling `shutil.move` and skip
      (or rename with a suffix) instead of relying on an exception that mostly
      doesn't fire.

- [x] **No undo for a destructive operation.** `organize_files` moves files
      with no record of where they came from. One typo'd `--target-dir` or an
      unwanted clustering result and there's no way back except manually
      reconstructing the original layout. Write a manifest (JSON: original path
      → new path) next to the run, and add an `organize-files --undo <manifest>`
      path that moves everything back.

- [x] **Symlink cycles crash `_analyze_folder`.** Verified: a symlinked
      directory pointing back at an ancestor causes unbounded recursion (tested
      — it recurses forever until Python hits `RecursionError`). `entry.is_dir()`
      follows symlinks by default. Fix: either skip symlinked directories
      entirely (`entry.is_dir(follow_symlinks=False)` check) or track visited
      real paths (`os.path.realpath`) and stop recursing on repeats.

- [x] **Re-running on an already-organized directory reprocesses its own
      output.** Nothing distinguishes a `Cluster_X` folder created by a previous
      run from a real subfolder — running the tool twice on the same directory
      clusters the previous run's output folders as if they were fresh
      documents. At minimum this should be documented; ideally the tool detects
      and skips folders it created itself (e.g. a marker file dropped in each
      cluster folder on creation).

## Correctness

- [x] **Off-by-one in cluster-count search.** `_find_optimal_clusters`
      (`analyzer.py:197`) builds `K = range(2, min(len(embeddings), max_k))`
      (`analyzer.py:205`) — `range` excludes its upper bound, so with the
      default `max_clusters=10` the search only ever tests k=2..9, never k=10.
      `--max-clusters` doesn't actually mean what its `--help` text says. Fix:
      `range(2, min(len(embeddings), max_k) + 1)`, then add a test asserting the
      configured max is reachable.

- [x] **No noise filtering when scanning.** `analyze_directory` and
      `_analyze_folder` process every `os.scandir` entry, including dotfiles,
      `.git/`, `__pycache__/`, `node_modules/`, `.DS_Store`, etc. Pointing this
      at a real project directory embeds and clusters junk. Add a default skip
      list (hidden entries + a short list of known noise directory names),
      overridable via a CLI flag.

## Features (local-only, keep them lightweight — stdlib/already-installed deps first)

- [x] **Undo/manifest support** — `--undo` done (see Critical above). Not
      done: `--history` to list past runs (would mean scanning target dirs
      for `.file_organizer_manifest_*.json` files, or keeping a central log).
- [ ] **`--exclude` pattern(s)** for the noise-filtering item above, for
      cases the default skip list doesn't cover.
- [ ] **`--verbose`/`--quiet`** — logging level is hardcoded to `INFO` in
      `main.py`; no way to get `DEBUG` detail or quiet it down for scripting.
- [ ] **A saved-defaults config file** (e.g. `~/.config/file-organizer.toml`,
      stdlib `tomllib` — Python 3.11+, already a floor we could require) so
      people don't retype `--path-weight`/`--max-clusters`/etc. every run.
- [ ] **More readable formats** (`.md`, `.csv`, `.rtf`, `.odt`, `.pptx`,
      `.xlsx`) — only if someone actually hits a directory full of these;
      don't build ahead of a real need.

## Housekeeping

- [ ] **Push the 2 local commits** (`b1369e7`, `21bf3c3` and whatever lands
      after this TODO) — `main` is ahead of `origin/main` and the new
      `.github/workflows/tests.yml` has never actually run, so CI going green is
      unverified, not confirmed.
- [ ] **Delete or wire in `analyze_document_content`** in `utils.py:116` —
      computes a suggested extraction length but is never called anywhere in the
      pipeline. Dead code; either use it or remove it.

## Explicitly not doing (considered, rejected — don't re-propose without new info)

- GUI / desktop app — large scope beyond a CLI organizer; revisit only if
  asked for.
- Any cloud/API-based keyword extraction or embeddings — violates the
  local-only constraint this project is built on.
- Duplicate/near-duplicate file detection — no evidence anyone's hit this;
  speculative.
- Sanitizing YAKE-derived folder names against path traversal — tested
  against real YAKE output; its tokenizer already strips `/`, `\`, `:`, so
  this isn't reachable in practice.
