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

## Clustering quality (unrelated files landing in the same folder)

- [x] **English-only embedding model.** `all-MiniLM-L6-v2` gave non-English
      documents near-random vectors, so KMeans was clustering noise. Swapped
      for a multilingual model.
- [x] **128-token context window.** `paraphrase-multilingual-MiniLM-L12-v2`
      truncates at 128 tokens — short enough that every letterheaded PDF
      (institution header, CNPJ, address) embedded to the same boilerplate and
      collapsed into one 40-file cluster. Now `intfloat/multilingual-e5-small`
      (512 tokens, same download size), with the `"query: "` prefix its card
      specifies for clustering.
- [x] **Embeddings weren't normalized.** KMeans minimizes Euclidean distance;
      these embeddings encode similarity as cosine, so vector magnitude
      (roughly, document length) was acting as a clustering feature.
      `normalize_embeddings=True`.
- [x] **Cluster-count vote was a coin flip.** `Counter.most_common(1)` over
      three *distinct* candidates returns whichever was inserted first
      (silhouette), never a real majority — and silhouette on one dense blob
      plus a few outliers always votes for a tiny k. Now `np.median`, which
      still returns the majority whenever two of the three agree.
- [x] **YAKE was hardcoded to `lan="en"`,** so in a Portuguese directory the
      stopwords (`de`, `do`, `para`) became folder names. `--lang`, accepting
      a comma-separated list (`pt,en`) for mixed directories.
- [x] **Folder names repeated themselves** (`Engenharia_eletrica_energia_eletrica`)
      — YAKE's top keywords overlap constantly. `_folder_name` drops words the
      top-2 share.
- [x] **`path_weight` repetitions weren't separated** — `str(path) * 2` gives
      `"a.pdfa.pdf"`, one unknown token, so filename weighting added noise
      rather than weight.
- [ ] **Chunk and mean-pool instead of truncating at 512.** The next lever if
      clusters are still too coarse: embed each document in chunks and average,
      so page 3 counts as much as the letterhead. Real work, not tuning — only
      do it if 512 tokens measurably isn't enough.
- [ ] **Strip shared boilerplate before embedding.** The letterheads are
      near-identical across editais; IDF would kill them, dense embeddings
      won't. Pairs with the item above; same "only if needed" caveat.
- [ ] **An outlier/`Misc` bucket.** KMeans forces every file into a cluster, so
      a one-off (a single `Currículo`) always gets glued to its nearest
      neighbours. ~5 lines (distance to centroid over a threshold → `Misc`),
      deferred until the fixes above have been measured.

## Features (local-only, keep them lightweight — stdlib/already-installed deps first)

- [x] **Undo/manifest support** — `--undo` done (see Critical above). Not
      done: `--history` to list past runs (would mean scanning target dirs
      for `.file_organizer_manifest_*.json` files, or keeping a central log).
- [x] **`--exclude` pattern(s)** for the noise-filtering item above, for
      cases the default skip list doesn't cover.
- [x] **`--verbose`/`--quiet`** — logging level is hardcoded to `INFO` in
      `main.py`; no way to get `DEBUG` detail or quiet it down for scripting.
- [x] **A saved-defaults config file** (e.g. `~/.config/file-organizer.toml`,
      stdlib `tomllib` — Python 3.11+, already a floor we could require) so
      people don't retype `--path-weight`/`--max-clusters`/etc. every run.
      `requires-python` bumped to `>=3.11` (CI already ran on 3.11).
- [ ] **More readable formats** (`.md`, `.csv`, `.rtf`, `.odt`, `.pptx`,
      `.xlsx`) — only if someone actually hits a directory full of these;
      don't build ahead of a real need.

## Housekeeping

- [x] **Push the 2 local commits** (`b1369e7`, `21bf3c3` and whatever lands
      after this TODO) — `main` is ahead of `origin/main` and the new
      `.github/workflows/tests.yml` has never actually run, so CI going green is
      unverified, not confirmed.
- [x] **Delete or wire in `analyze_document_content`** in `utils.py:116` —
      computes a suggested extraction length but is never called anywhere in the
      pipeline. Dead code; either use it or remove it. Deleted (no caller, no
      stated need to wire it in).

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
