# TODO

Project identity, and the constraint every item below must respect: **this is a
local file organizer.** All processing (reading files, embedding, clustering,
moving files) happens on the user's machine. Nothing about a user's files or
folder contents gets sent to a server. The only network calls that exist today
are one-time model downloads (the embedding model from Hugging Face,
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
- [x] **An outlier/`Misc` bucket.** KMeans forces every file into a cluster, so
      a one-off (a single `Currículo`) always gets glued to its nearest
      neighbours. Distance to centroid > mean + `--misc-threshold` std → `Misc`
      (global z-score; per-cluster thresholds if spreads differ a lot). On the
      demo it catches both one-offs plus one false positive (`bolsa de valores`).
- [x] **Elbow vote was wrong.** `K[np.argmin(second_derivative)]` picked the
      *flattest* stretch of the inertia curve and was off by one; the elbow is
      `K[argmax + 1]` (`_elbow_k`). One of the three k votes was near-random.
- [x] **pt+en folders clustered by language, not topic.** Found with
      `--report` on `examples/demo` (6 topics, mixed pt/en): k=2, one folder per
      language. e5 still encodes which language a text is in more strongly
      than its topic (same-language ~0.85 cosine vs ~0.78 across). Tested
      (adjusted Rand vs true topics at k=6): e5 raw 0.31, **e5 + per-language
      centering 0.91**, paraphrase-multilingual-MiniLM 1.00 (but 128 tokens, see
      above), LaBSE 0.85 (1.8 GB). Kept e5; with several `--lang` codes each
      document's language is detected from stopwords and its language mean
      subtracted (`_center_by_language`). On a real 32-file pt/en folder this
      went from 3 folders (one holding 21 unrelated files) to 9 coherent ones.
- [x] **Filename extension and separators leaked into embeddings and folder
      names** (`Eduardo_gomes_feijoada.txt`). `create_weighted_text` now uses
      the name's words: `receita_pao-de.queijo.txt` → `receita pao de queijo`.
- [ ] **Better folder names.** YAKE on a cluster's concatenated text favours
      one document's frequent phrase (`Bach_bio_piano`, `Buenos_aires_city`).
      Tried for v0.2.0 on the same clusters (demo + a real 32-file folder),
      none consistently better, so YAKE stays - don't re-try these without a
      new idea:
      - class-based TF-IDF: `Piano_musical`, `Tokyo_aires`; on real PDFs it
        surfaces extraction garbage (`Tensªo_vocœ`, `Oes_equa`);
      - share-of-files × rarity across clusters: generic words
        (`Envolve_jogadores`, `Eficaz_virus`);
      - YAKE candidates re-ranked by share of files: better on some demo
        folders (`Music_musical`, `Inflação_central_bank`), worse on real ones
        (`Brazil_tools`);
      - filename repeated once instead of twice: changes 3 of 18 names, a wash.
      The real limits are noisy PDF text and that a good name is a summary.
- [x] **Calinski-Harabasz voted k=2 on nearly every real folder**, so the
      median collapsed to min(elbow, silhouette). It bit on the 68-file demo:
      k=4, adjusted Rand 0.35. Swapped for Davies-Bouldin (lower = better):
      k=10, ARI 0.85; the 32-file personal test folder chose k=9 either way.
- [x] **One-offs got glued to a cluster** (chess and origami joined the two
      football-rules texts). Misc now also takes documents with a negative
      silhouette (closer to another cluster than their own) and any document
      left alone in its cluster (`_misc_labels`). Measured against the
      distance-only rule, per-cluster z-scores and singletons alone: demo
      one-offs caught 2/4 → 3/4 with the same one false positive; on the real
      folder it pulled out exactly the misplaced files. Chess still joins the
      football rules (it *is* about game rules).
- [x] **`--lang auto` is the default.** Without `--lang pt,en` a mixed folder
      still split by language (demo ARI 0.37). Each file's language is now
      detected among en/pt/es/fr/de/it/nl by stopword counts - 88/88 right on
      the demo plus 20 es/fr/de/it/nl excerpts, so plain counting was kept
      over exclusive-stopword or size-normalised scoring (also 88/88). Files
      with no stopwords (bare names) form their own group for centring.
- [x] **Quality regression test** with the real model on the demo
      (`tests/test_quality.py`, `RUN_QUALITY=1`, own CI job with a cached
      model). The unit tests mock the model, so before this the language and
      k-selection bugs only showed up in the charts.

## Features (local-only, keep them lightweight — stdlib/already-installed deps first)

- [x] **Undo/manifest support** — `--undo` done (see Critical above), and
      `--history` lists past runs from a central log
      (`~/.local/state/file-organizer/history.jsonl`) with each undo command.
- [x] **`--version`**, package at 0.2.0, GitHub release `v0.2.0`.
- [x] **Scale.** Reading capped at `READ_CHARS` (5000; the model only sees
      ~512 tokens anyway), silhouette sampled above 2000 files, per-stage
      timings in the INFO log. Numbers in the README.
- [x] **`--exclude` pattern(s)** for the noise-filtering item above, for
      cases the default skip list doesn't cover.
- [x] **`--verbose`/`--quiet`** — logging level is hardcoded to `INFO` in
      `main.py`; no way to get `DEBUG` detail or quiet it down for scripting.
- [x] **A saved-defaults config file** (e.g. `~/.config/file-organizer.toml`,
      stdlib `tomllib` — Python 3.11+, already a floor we could require) so
      people don't retype `--path-weight`/`--max-clusters`/etc. every run.
      `requires-python` now `>=3.12` (numpy 2.5 has no 3.11 wheels).
- [x] **More readable formats** — `.md`/`.csv` (plain text), `.pptx`/`.odt`
      (zip + XML, same reader as `.docx`: `_read_zipped_xml`), `.xlsx` (same
      reader) and `.rtf` (a ~30-line stdlib reader, `_rtf_to_text`).
- [x] **`--copy`** — copy instead of move; undo deletes the copies, never the
      last one.
- [x] **`--report DIR`** — document map, k-selection and similarity charts
      (`report.py`, matplotlib as the `[viz]` extra). Also what found the two
      clustering bugs above.
- [x] **Progress bar** on embedding (hidden under `--quiet`).
- [x] **Undo cleans up** the emptied cluster folders and the manifest.
- [x] **Vosk**: models in `~/.cache/vosk` (were re-downloaded after every
      reboot from the temp dir); transcription language follows `--lang`.

## Roadmap: why use this over an LLM-per-file sorter (plan of 2026-11-19)

The local competitors (Local-File-Organizer, AI File Sorter, LlamaFS,
SlothSort...) ask an LLM about each file: hours per 1,000 files, a 2-7 GB
model, generic categories, several with cloud fallbacks. This one sees the
whole collection, runs on a CPU in minutes and measures its own quality.
The items below make "local, provable, learns from your folders" visible.
No new pip dependency in any of them (the ONNX switch replaces two).

- [x] **Shared helpers** `_read_content`, `_embed`, `_cluster` (the read
      dispatch was duplicated in `analyze_directory` and `_analyze_folder`).
- [x] **Provably offline.** Every run used to send a request to
      huggingface.co (hub checks for updates unless `local_files_only`;
      measured: the old call retried huggingface.co a dozen times under the
      guard). Model pinned to a revision, loaded from cache, download only
      when missing; `test_quality.py` blocks and records network access.
- [x] **Embedding cache** (sqlite3, key = sha256(revision + embedded text),
      0600, rows unseen for 90 days pruned, `--no-cache`). Demo re-run:
      embedding 7.6 s → 0, and the model isn't even loaded (lazy `model`).
- [x] **Editable plan** `--plan FILE` / `--apply FILE`, per-file score
      (cosine to its folder's centroid, weakest first), shared validator for
      user-typed folder/file names. Verified end to end: rename in the
      file → apply → undo.
- [x] **Local web UI** `--ui` (stdlib `http.server` on 127.0.0.1, token in
      the URL fragment + Host check, strict CSP, single `ui.html` with no
      external URL): analyze, move via a per-file `<select>`, rename, map,
      apply, history, undo. Checked in headless Chromium at desktop and phone
      width (console clean); API driven end to end with the real model.
- [x] **`--into DIR`**: file new documents into the folders you already
      have (prototype = mean embedding per folder, same global z-rule as
      Misc); leftovers clustered as usual, < 3 leftovers → Misc. The marker
      is no longer dropped into existing folders. 10 files taken out of an
      organized demo: 9 went back; the miss came from a 1-file folder.
- [x] **OCR** for scanned PDFs and images via the `tesseract` binary (like
      ffmpeg for audio); the log says how many PDFs have no text layer.
      Checked with a real tesseract 5.5 (conda-forge): an image-only PDF
      with a JPEG page read perfectly, a 1-bit CCITT ("fax mode") one with
      two small errors, a text PNG perfectly, a noise photo -> None; the CPF
      in the OCR text was flagged; 0.3-0.5 s per file with all 7 languages.
      CCITT decoding needs Pillow (comes with `[viz]`); without it OCR of
      that PDF is skipped. JBIG2 untested.
- [x] **Flag sensitive documents** (CPF with check digits, cards with Luhn
      and a 2-6 first digit; not CNPJ - every letterhead has one). Flag only.
- [x] **Folder names from a local LLM**, one call per cluster, Ollama on
      127.0.0.1 only, proxies bypassed, YAKE fallback (`--llm-names MODEL`,
      off by default). Measured with Ollama 0.40 + qwen2.5:1.5b on the demo
      (11 requests, ~13 s): the first prompt gave "StockExch", "WorldCup2022",
      "Bachmusique" and different names every run. Now: a prompt with
      examples and explicit don'ts, temperature 0 + seed (same names every
      run, checked), hyphens count as word breaks (3-word cap), non-Latin
      script -> YAKE's name ("Classical音乐"). Result vs YAKE: 7 of 10 better
      (Classical_Music, Neural_Networks, Astronomy, Brazilian_food...), 2
      worse ("Chess" for 2 football rules + chess; "South_America" for a
      folder with Paris and Tokyo). A bigger model would likely do better
      (unmeasured).
- [x] **Hierarchy** `--depth N`: re-cluster clusters of 20+ files into
      `Parent/Child`; undo cleans up parents too. Tried on 1,970 real
      license texts (`/usr/share/doc/*/copyright`, one folder per package):
      4 min, 1.8 GB peak, 43 s re-run from the cache; 2 top folders with ~10
      subfolders each (`Gnu_lesser`, `Apache_mit`, `Psf_python`...). It
      also showed subfolders repeating their parent
      (`Public_license_general/Public_license_general`): `_folder_name` now
      skips keywords made only of the parent's words, never falls back to
      the unfiltered name.
- [x] **ONNX instead of torch**: `embed.Embedder` (tokenizers +
      onnxruntime on the repo's `onnx/model.onnx`). Cosine 1.00000 vs
      sentence-transformers on every demo doc, same ARI (0.899), same speed
      (39 s vs 39-44 s for 340 texts), peak memory 1.4 GB vs 1.6 GB, deps
      ~130 MB vs ~900 MB. Batch 32 was 15% faster but peaked at 2.9 GB, so
      batch 8. The int8 model was 38% faster but 0.987 cosine: rejected.
      Lost: automatic CUDA (onnxruntime-gpu if ever needed).
- [ ] **Benchmark against an LLM-per-file sorter** on the demo (ARI, time,
      RAM, run-to-run stability) — for the README's "why this one" claim.

## Housekeeping

- [x] **Push the 2 local commits** (`b1369e7`, `21bf3c3` and whatever lands
      after this TODO) — `main` is ahead of `origin/main` and the new
      `.github/workflows/tests.yml` has never actually run, so CI going green is
      unverified, not confirmed.
- [x] **CI never passed** — every run failed at `pip install`: `numpy==2.5.1`
      has no Python 3.11 wheels. CI now on 3.12, CPU-only torch (the PyPI build
      pulled 14 CUDA packages), pip cache. Now also Windows, macOS and
      Python 3.13.
- [x] **PyPDF2 → pypdf** (PyPDF2 is deprecated; same API), and its per-PDF
      "Ignoring wrong pointing object" warnings silenced.
- [x] **Delete or wire in `analyze_document_content`** in `utils.py:116` —
      computes a suggested extraction length but is never called anywhere in the
      pipeline. Dead code; either use it or remove it. Deleted (no caller, no
      stated need to wire it in).

## Explicitly not doing (considered, rejected — don't re-propose without new info)

- Desktop GUI toolkit (Qt/tkinter) — the UI asked for on 2026-11-19 is the
  local web page in the roadmap above (stdlib only, works on every OS).
- Any cloud/API-based keyword extraction or embeddings — violates the
  local-only constraint this project is built on.
- Duplicate/near-duplicate file detection — no evidence anyone's hit this;
  speculative.
- Sanitizing YAKE-derived folder names against path traversal — tested
  against real YAKE output; its tokenizer already strips `/`, `\`, `:`, so
  this isn't reachable in practice.
