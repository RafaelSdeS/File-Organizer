import os
# Before .embed pulls in huggingface_hub, which reads this at import.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
import yake
from sklearn.cluster import KMeans
import shutil
import json
import fnmatch
import hashlib
import re
import sqlite3
import time
import urllib.request
from collections import Counter
from datetime import datetime
import numpy as np
import logging

from .embed import Embedder
from .utils import TESSERACT, create_weighted_text, ocr_image, read_file, read_audio_metadata, sensitive_kinds, set_transcription_language

logger = logging.getLogger(__name__)

# Multilingual by default: all-MiniLM-L6-v2 is English-only, so non-English
# documents got near-random embeddings and therefore near-random clusters.
# e5-small over paraphrase-multilingual-MiniLM-L12-v2 for the context window:
# 512 tokens vs 128. At 128 every letterheaded PDF (university header, CNPJ,
# address) embedded to the same boilerplate and collapsed into one cluster.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
# Pinned: an upstream update can't silently change the clusters, and a pinned
# commit loads from the local cache with no network round trip.
EMBEDDING_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
# e5 wants a task prefix; its model card specifies "query: " for symmetric
# tasks like clustering.
EMBEDDING_PREFIX = "query: "

readable_files = [".pdf", ".txt", ".md", ".csv", ".xml", ".rtf", ".docx", ".pptx", ".odt", ".xlsx"]
audio_files = [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".aiff", ".opus"]
# Read by OCR when the tesseract binary is installed; by name otherwise.
image_files = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]

# Dropped into every cluster folder organize_files creates, so a later run
# recognizes and skips its own output instead of re-clustering it.
ORGANIZED_MARKER = ".file_organizer_created"
MANIFEST_PREFIX = ".file_organizer_manifest_"
# One JSON line per organize run, for --history (manifests live next to
# the files they moved, so there is otherwise no single place to list them).
HISTORY_PATH = os.path.join(
    os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state"),
    "file-organizer", "history.jsonl",
)
# Embeddings of documents already seen, so a re-run (or --into over a big
# archive) only encodes what changed. Vectors are derived from content, so
# the file is private to the user and rows unseen for a while are dropped.
CACHE_PATH = os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
    "file-organizer", "embeddings.sqlite",
)
CACHE_MAX_AGE_DAYS = 90
# Folder for documents too far from every cluster centroid to belong anywhere.
MISC_FOLDER = "Misc"
# With --depth 2+, clusters at least this big are split into subfolders.
# ponytail: fixed; make it a flag if 20 is wrong on real folders.
SPLIT_MIN = 20
# --llm-names: a local Ollama server. Loopback only, on purpose: a host
# option would be a way for the documents' keywords to leave the machine.
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
# How much of each document to read. The embedding model only sees ~512
# tokens (roughly 2000-2500 characters), so the rest of a long PDF was pure
# cost: on ~2000 files, reading everything took 181 s of a 411 s run.
READ_CHARS = 5000


def _elbow_k(K, distortions):
    """k at the inertia curve's sharpest bend (max 2nd difference), or None.

    np.diff(d, 2)[i] is centred on d[i + 1], hence the +1. Needs at least 3
    points (i.e. at least 5 documents) to have a 2nd difference at all.
    """
    second_derivative = np.diff(distortions, 2)
    if len(second_derivative) == 0:
        return None
    return K[int(np.argmax(second_derivative)) + 1]

# Languages --lang auto chooses among (YAKE ships stopwords for 34; these
# are the common ones, which keeps close pairs like pt/gl or no/da out).
AUTO_LANGS = ["en", "pt", "es", "fr", "de", "it", "nl"]

# Known-noise directory/file names that aren't already caught by the
# hidden-entry (dot-prefix) check - e.g. .git, .DS_Store are dotfiles already.
DEFAULT_SKIP_NAMES = {"__pycache__", "node_modules"}


def _stopword_sets(langs):
    """{lang: YAKE's bundled stopword set} for each language code.

    Reads the lists back off throwaway extractors rather than opening the
    bundled .txt files directly - their location moves between YAKE versions.
    An unknown language code falls back to YAKE's language-agnostic list.
    """
    return {lang: set(yake.KeywordExtractor(lan=lang).stopword_set) for lang in langs}


def _stopwords(langs):
    """Union of YAKE's bundled stopword lists, for mixed-language directories.

    YAKE only takes one `lan`, so a pt+en directory always had the wrong
    stopwords for half its files. Returns None (= let YAKE load its own) for
    a single language.
    """
    if len(langs) < 2:
        return None
    return set().union(*_stopword_sets(langs).values())


def _detect_languages(texts, stopword_sets):
    """Per text, the language whose stopwords it uses most (ties: first
    listed), or "" for a text with no stopwords at all - a bare filename, an
    image. Plain counting got all 88 test texts right (the 68 pt/en demo
    files plus 20 es/fr/de/it/nl Wikipedia excerpts) across AUTO_LANGS.
    """
    detected = []
    for text in texts:
        words = re.findall(r"\w+", text.lower())
        scores = {lang: sum(w in stopwords for w in words) for lang, stopwords in stopword_sets.items()}
        best = max(scores, key=scores.get)
        detected.append(best if scores[best] else "")
    return np.array(detected)


def _center_by_language(embeddings, languages):
    """Subtract each language's mean embedding, then re-normalize.

    Multilingual models still encode *which language* a text is in, strongly
    enough that same-language documents on different topics look closer than
    same-topic documents across languages - so a pt+en folder got split by
    language, not topic. Moving every language's centre to the origin removes
    that offset. On examples/demo's 64 topic texts (8 topics, pt+en):
    adjusted Rand vs. the true topics at k=8 went 0.34 -> 1.00.

    Needs two or more languages with 2+ documents each; otherwise returns the
    embeddings unchanged. ponytail: a language with a single document stays
    uncentred (its own mean would zero it), so it tends to land in Misc.
    """
    groups = [lang for lang in set(languages) if (languages == lang).sum() >= 2]
    if len(groups) < 2:
        return embeddings
    centered = np.array(embeddings, dtype=float)
    for lang in groups:
        mask = languages == lang
        centered[mask] -= centered[mask].mean(axis=0)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.maximum(norms, 1e-12)


def _misc_labels(embeddings, labels, centers, z):
    """Relabel as -1 (the Misc folder) documents that don't belong anywhere.

    KMeans forces every document into some cluster, so a one-off gets glued
    to its nearest neighbours. A document goes to Misc if it is
    - unusually far from its own centroid (distance > mean + z std), or
    - closer, on average, to another cluster than to its own (negative
      silhouette), or
    - left alone in its cluster - a one-file folder helps nobody.
    Measured on the demo: 3 of the 4 one-offs caught (the old distance-only
    rule caught 2), same single false positive; on a real 32-file folder it
    also pulled out exactly the files sitting in the wrong group.
    ponytail: silhouette_samples is O(n^2); fine to a few thousand files.
    """
    labels = np.array(labels)
    dist = np.linalg.norm(embeddings - centers[labels], axis=1)
    misc = dist > dist.mean() + z * dist.std()
    if 1 < len(set(labels)) < len(labels):
        misc |= silhouette_samples(embeddings, labels) < 0
    labels[misc] = -1
    kept, counts = np.unique(labels[labels >= 0], return_counts=True)
    labels[np.isin(labels, kept[counts < 2])] = -1
    return labels


def _folder_name(keywords, taken=()):
    """Folder name from the top-2 keywords, minus words they share.

    YAKE's top keywords overlap constantly ("engenharia eletrica" +
    "energia eletrica"), which used to produce names like
    "Engenharia_eletrica_energia_eletrica". `taken` words (a parent
    folder's) are skipped too, and so are keywords made only of them, so a
    subfolder names what sets it apart from its parent; "" if nothing does.
    """
    seen, words = set(taken), []
    keywords = [k for k in keywords if any(w.lower() not in seen for w in k.split())]
    for keyword in keywords[:2]:
        for word in keyword.split():
            if word.lower() not in seen:
                seen.add(word.lower())
                words.append(word)
    return "_".join(words).capitalize()


def _is_noise_entry(name, skip_noise, exclude_patterns):
    if skip_noise and (name.startswith(".") or name in DEFAULT_SKIP_NAMES):
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in exclude_patterns)


def _read_content(full_path):
    """A file's text for embedding, or None (not a readable type, or failed
    to read - such a file is still clustered by its name)."""
    ext = os.path.splitext(full_path)[1].lower()
    try:
        if ext in readable_files:
            return read_file(full_path, max_chars=READ_CHARS)
        if ext in audio_files:
            return read_audio_metadata(full_path)
        if ext in image_files:
            return ocr_image(full_path)
    except Exception as e:
        logger.warning(f"Failed to read {full_path}: {e}")
    return None


def _cache_key(text):
    # The model revision is part of the key, so a new model never reuses
    # vectors from the old one.
    return hashlib.sha256(f"{EMBEDDING_MODEL}@{EMBEDDING_REVISION}\n{text}".encode()).hexdigest()


def _open_cache():
    """The embedding cache, or None if it can't be opened - a cache problem
    must never fail a run."""
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        db = sqlite3.connect(CACHE_PATH)
        os.chmod(CACHE_PATH, 0o600)  # vectors are derived from document content
        db.execute("CREATE TABLE IF NOT EXISTS docs (key TEXT PRIMARY KEY, vec BLOB, last_seen REAL)")
        return db
    except (sqlite3.Error, OSError) as e:
        logger.warning(f"Embedding cache unavailable ({CACHE_PATH}): {e}")
        return None


def _assign_to_folders(embeddings, archive_embeddings, archive_folders, z):
    """For --into: per document, the existing folder it fits (or None), and
    each folder's prototype (the mean of its files' embeddings).

    A document fits its nearest prototype if it is no farther from it than
    the archive's own files usually are from theirs: mean + z std over all
    of them, the same rule (and knob) as Misc. z = 0 always takes the
    nearest folder, as it disables Misc.
    """
    names = sorted(set(archive_folders))
    folder_of = np.array(archive_folders)
    protos = np.array([archive_embeddings[folder_of == name].mean(axis=0) for name in names])
    own = np.linalg.norm(archive_embeddings - protos[[names.index(f) for f in archive_folders]], axis=1)
    # Squared distances via dot products: an n x folders x dim array would
    # be hundreds of MB on a big folder.
    sq = (embeddings ** 2).sum(axis=1)[:, None] - 2 * embeddings @ protos.T + (protos ** 2).sum(axis=1)[None, :]
    nearest = sq.argmin(axis=1)
    dist = np.sqrt(np.maximum(sq[np.arange(len(embeddings)), nearest], 0))
    fits = dist <= own.mean() + z * own.std() if z else np.ones(len(embeddings), dtype=bool)
    return [names[j] if ok else None for j, ok in zip(nearest, fits)], dict(zip(names, protos))


def _file_scores(paths, embeddings, folders, centroids=None):
    """{file: cosine similarity to its folder's centroid}, rounded - a low
    score is a placement worth checking. `centroids` overrides the mean of
    a folder's files from this run (for --into, the archive's prototype)."""
    index = {path: i for i, path in enumerate(paths)}
    scores = {}
    for folder, files in folders.items():
        rows = np.asarray(embeddings[[index[f] for f in files]], dtype=float)
        centre = (centroids or {}).get(folder)
        centre = rows.mean(axis=0) if centre is None else np.asarray(centre, dtype=float)
        centre = centre / max(np.linalg.norm(centre), 1e-12)
        sims = rows @ centre / np.maximum(np.linalg.norm(rows, axis=1), 1e-12)
        scores.update(zip(files, np.round(sims, 2).tolist()))
    return scores


def _llm_folder_name(model, keywords, files):
    """A short folder name from a local Ollama model, or None on any failure
    (the caller keeps YAKE's name). One call per cluster, not per file."""
    # Without the examples and the explicit don'ts, a 1.5B model answered
    # "StockExch", "WorldCup2022" (a made-up year), "Bachmusique".
    prompt = (
        "You name folders for a document organizer. Below are the keywords and some file names "
        "of one folder. Reply with a plain folder name of 1 to 3 common words for their shared "
        'topic, like "Astronomy", "Brazilian food" or "Football rules": no abbreviations, '
        "no dates or numbers, no file names, in the language of the keywords. "
        "Reply with the name only.\n"
        f"Keywords: {', '.join(keywords)}\n"
        f"File names: {', '.join(files)}"
    )
    request = urllib.request.Request(
        # temperature 0 + seed: the same folder gets the same name every run
        # (Ollama's default 0.8 renamed half the demo's folders between runs).
        OLLAMA_URL, data=json.dumps({"model": model, "prompt": prompt, "stream": False,
                                     "options": {"temperature": 0, "seed": 0}}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        # No proxies: urllib would otherwise honour http_proxy even for
        # 127.0.0.1 and hand the keywords to whatever proxy is configured.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=120) as response:
            reply = json.load(response).get("response", "")
    except (OSError, ValueError) as e:
        logger.warning(f"Local LLM naming failed ({e}); keeping the keyword name")
        return None
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
    # Hyphens and underscores separate words too, so the 3-word cap holds.
    words = re.sub(r"[\W_]+", " ", reply.splitlines()[0] if reply else "").split()
    name = "_".join(words[:3])[:40].strip("_-")
    # The supported languages are all Latin-script; Qwen drifted into
    # Chinese mid-name ("Classical音乐") - keep YAKE's name instead.
    if any(ord(c) > 0x24F for c in name):
        return None
    return name[:1].upper() + name[1:] or None


def _load_model():
    """The embedding model, from the local cache without touching the network.

    Without local_files_only, huggingface_hub sends a request to
    huggingface.co on every load just to check for updates, even with the
    model cached. Only the very first run (nothing cached yet) downloads.
    """
    try:
        return Embedder(EMBEDDING_MODEL, EMBEDDING_REVISION, local_files_only=True)
    except OSError:
        logger.info(f"Downloading the embedding model once (~470 MB) from huggingface.co: {EMBEDDING_MODEL}")
        return Embedder(EMBEDDING_MODEL, EMBEDDING_REVISION)


class DocumentAnalyzer:

    #Main class for analyzing and organizing documents using AI-powered clustering.
    #Uses sentence embeddings for content similarity analysis and YAKE for keyword extraction.

    def __init__(self, path_weight=2, max_clusters=10, yake_ngram=2, yake_top=5, skip_noise=True, exclude_patterns=None, lang="auto", misc_threshold=2.0, use_cache=True, depth=1, llm_names=None):
        if max_clusters < 3:
            raise ValueError("max_clusters must be at least 3 (k=2 needs to be a testable candidate)")
        self._model = None  # loaded on first use: a fully cached run never needs it
        # lang picks the stopword lists - with the wrong one, the target
        # language's stopwords ("de", "do", "para", ...) become folder names -
        # and the languages each document is detected among, so embeddings
        # can be centred per language (see _center_by_language).
        # "auto" detects among AUTO_LANGS; a comma-separated list ("pt,en")
        # restricts detection to those.
        self.auto_lang = lang.strip().lower() == "auto"
        langs = AUTO_LANGS if self.auto_lang else [p.strip() for p in lang.split(",") if p.strip()] or ["en"]
        self.stopword_sets = _stopword_sets(langs)
        self.yake_ngram, self.yake_top = yake_ngram, yake_top
        # Under auto, rebuilt in analyze_directory once the languages are known.
        self.kw_extractor = self._keyword_extractor(["en"] if self.auto_lang else langs)
        # Audio is transcribed while scanning, before any language is
        # detected, so under auto it stays English.
        set_transcription_language("en" if self.auto_lang else langs[0])
        self.path_weight = path_weight
        self.max_clusters = max_clusters
        self.skip_noise = skip_noise
        self.exclude_patterns = exclude_patterns or []
        self.misc_threshold = misc_threshold
        self.use_cache = use_cache
        self.depth = depth
        self.llm_names = llm_names
        # Filled by analyze_directory / _find_optimal_clusters, for --report.
        self.last_run = None
        self.k_scores = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model()
        return self._model

    def analyze_directory(self, path, into=None):
        """
        Analyzes all files in the specified directory and organizes them into clusters.

        Args:
            path (str): Path to the directory to analyze
            into (str): Optional folder of already-organized subfolders. A
                document that fits one of them is filed there; the rest are
                clustered into new folders as usual.

        Returns:
            dict: {folder name: [file names in path]}; with depth > 1 a
            folder name can be "Parent/Child".

        Raises:
            ValueError: If the path is invalid or directory is empty
            FileNotFoundError: If the specified path does not exist
            PermissionError: If access to the directory is denied
            RuntimeError: If clustering fails due to insufficient data
        """
        if not isinstance(path, str) or not path.strip():
            raise ValueError("Invalid directory path provided")

        try:
            # Verify directory exists and is accessible
            for folder in (path, into) if into else (path,):
                if not os.path.exists(folder):
                    raise FileNotFoundError(f"Directory not found: {folder}")
                if not os.path.isdir(folder):
                    raise ValueError(f"Path is not a directory: {folder}")
                if not os.access(folder, os.R_OK):
                    raise PermissionError(f"Permission denied for directory: {folder}")

            files_data = []
            # Per-stage timings go to the INFO log, to see where time goes on
            # big folders.
            started = time.perf_counter()

            archive = self._scan_archive(into) if into else []
            # The --into folders are destinations, not documents: never
            # cluster them (or `into` itself, if it sits inside `path`).
            destinations = {os.path.realpath(os.path.join(into, folder)) for folder, _ in archive} | {os.path.realpath(into)} if into else set()

            with os.scandir(path) as dir_iter:
                for entry in dir_iter:
                    if _is_noise_entry(entry.name, self.skip_noise, self.exclude_patterns):
                        continue
                    try:
                        if entry.is_file():
                            file_data = {"Path": entry.name, "Content": _read_content(entry.path)}
                        else:
                            if entry.is_symlink():
                                logger.info(f"Skipping symlinked directory: {entry.name}")
                                continue
                            if os.path.realpath(entry.path) in destinations:
                                continue
                            if os.path.exists(os.path.join(entry.path, ORGANIZED_MARKER)):
                                logger.info(f"Skipping previously organized folder: {entry.name}")
                                continue
                            file_data = self._analyze_folder(entry)

                        files_data.append(file_data)
                    
                    except Exception as e:
                        logger.error(f"Error processing entry {entry.name}: {str(e)}")
                        continue
            
            if not files_data:
                raise ValueError("No valid files found in directory")
            
            df = pd.DataFrame(files_data)
            if df.empty:
                raise ValueError("No data to process")
            
            df["Text"] = df.apply(
                lambda row: create_weighted_text(
                    row["Path"],
                    row["Content"],
                    path_weight=self.path_weight
                ),
                axis=1
            )

            no_text = sum(1 for d in files_data if d["Path"].lower().endswith(".pdf") and d["Content"] == "")
            if no_text:
                logger.info(f"{no_text} PDF(s) have no text layer (scans?), so only their names count"
                            + ("" if TESSERACT else " - install tesseract to OCR them"))

            # Generate embeddings for clustering. Archive files (--into) are
            # embedded and language-centred together with the new ones, so
            # both live in the same space.
            texts = df["Text"].tolist() + [text for _, text in archive]
            logger.info(f"Read {len(texts)} entries in {time.perf_counter() - started:.1f}s; generating embeddings...")
            started = time.perf_counter()
            embeddings = self._embed(texts)

            languages = _detect_languages(texts, self.stopword_sets)
            counts = Counter(lang or "none" for lang in languages.tolist())
            logger.info(f"Detected languages: {dict(counts)}")
            embeddings = _center_by_language(embeddings, languages)
            if self.auto_lang:
                found = [lang for lang in self.stopword_sets if lang in set(languages)]
                self.kw_extractor = self._keyword_extractor(found or ["en"])
            embeddings, archive_embeddings = embeddings[:len(df)], embeddings[len(df):]
            logger.info(f"Embedded in {time.perf_counter() - started:.1f}s; determining the number of clusters...")

            folders, centroids = {}, {}
            fits = [None] * len(df)
            if archive:
                fits, centroids = _assign_to_folders(embeddings, archive_embeddings, [f for f, _ in archive], self.misc_threshold)
                for file, folder in zip(df["Path"], fits):
                    if folder:
                        folders.setdefault(folder, []).append(file)
                logger.info(f"{sum(map(bool, fits))} of {len(df)} entries fit an existing folder in {into}")
            elif into:
                logger.warning(f"{into} has no subfolders with files; clustering everything into new folders")

            rest = np.array([folder is None for folder in fits])
            self.k_scores = None
            if rest.any():
                # A couple of strays don't deserve a new folder next to the user's own.
                labels = np.full(rest.sum(), -1) if archive and rest.sum() < 3 else self._cluster(embeddings[rest])
                k_scores = self.k_scores  # sub-clustering (depth > 1) overwrites it
                started = time.perf_counter()
                new = self._organize_clusters(df[rest].assign(Cluster=labels), embeddings[rest], self.depth)
                for folder, files in new.items():
                    folders.setdefault(folder, []).extend(files)
                self.k_scores = k_scores
                logger.info(f"Named folders in {time.perf_counter() - started:.1f}s")

            sensitive = {}
            for file, content in zip(df["Path"], df["Content"]):
                kinds = sensitive_kinds(content) if content else set()
                if kinds:
                    sensitive[file] = sorted(kinds)
            if sensitive:
                logger.warning(f"{len(sensitive)} file(s) contain a CPF or card number")

            self.last_run = {
                "paths": df["Path"].tolist(),
                "embeddings": embeddings,
                "k_scores": self.k_scores,
                "scores": _file_scores(df["Path"].tolist(), embeddings, folders, centroids),
                "sensitive": sensitive,
            }
            return folders

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

    def _scan_archive(self, into):
        """[(folder, text)] for every file under each top-level subfolder of
        `into` - the folders --into can file new documents into.
        ponytail: reads every archive file on each run (the cache saves the
        encoding, not the reading); nested folders aren't targets of their own.
        """
        with os.scandir(into) as entries:
            folders = [e for e in entries if e.is_dir(follow_symlinks=False)
                       and not _is_noise_entry(e.name, self.skip_noise, self.exclude_patterns)]
        docs = []
        for folder in folders:
            for root, dirs, files in os.walk(folder.path):
                dirs[:] = [d for d in dirs if not _is_noise_entry(d, self.skip_noise, self.exclude_patterns)]
                for name in files:
                    if name == ORGANIZED_MARKER or _is_noise_entry(name, self.skip_noise, self.exclude_patterns):
                        continue
                    content = _read_content(os.path.join(root, name))
                    docs.append((folder.name, create_weighted_text(name, content, path_weight=self.path_weight)))
        return docs

    def _embed(self, texts):
        """Normalized embeddings of texts, encoding only those not in the cache.

        Unit length (Embedder.encode always normalizes) makes KMeans'
        Euclidean distance monotonic in cosine distance, which is the metric
        these embeddings are trained for.
        """
        texts = [EMBEDDING_PREFIX + text for text in texts]
        keys = [_cache_key(text) for text in texts]
        db = _open_cache() if self.use_cache else None
        vectors = {}
        if db:
            try:
                for key in keys:  # ponytail: one query per key; trivial next to encoding
                    row = db.execute("SELECT vec FROM docs WHERE key = ?", (key,)).fetchone()
                    if row:
                        vectors[key] = np.frombuffer(row[0], dtype=np.float32)
            except sqlite3.Error as e:
                logger.warning(f"Embedding cache unreadable, ignoring it: {e}")
                vectors = {}

        missing = [i for i, key in enumerate(keys) if key not in vectors]
        if vectors:
            logger.info(f"{len(texts) - len(missing)} of {len(texts)} embeddings from the cache")
        if missing:
            try:
                encoded = self.model.encode(
                    [texts[i] for i in missing],
                    show_progress_bar=logger.isEnabledFor(logging.INFO),
                )
            except Exception as e:
                raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
            for i, vector in zip(missing, encoded):
                vectors[keys[i]] = np.asarray(vector, dtype=np.float32)

        if db:
            try:
                now = time.time()
                with db:
                    db.executemany("INSERT OR REPLACE INTO docs VALUES (?, ?, ?)",
                                   [(key, vectors[key].tobytes(), now) for key in set(keys)])
                    db.execute("DELETE FROM docs WHERE last_seen < ?", (now - CACHE_MAX_AGE_DAYS * 86400,))
            except sqlite3.Error as e:
                logger.warning(f"Could not update the embedding cache: {e}")
            finally:
                db.close()
        return np.array([vectors[key] for key in keys])

    def _cluster(self, embeddings):
        """Cluster labels for the embeddings: pick k, run KMeans, then send
        the documents that don't belong anywhere to Misc (label -1)."""
        started = time.perf_counter()
        try:
            k = self._find_optimal_clusters(embeddings, max_k=self.max_clusters)
        except Exception as e:
            raise RuntimeError(f"Failed to find optimal clusters: {str(e)}")
        logger.info(f"Picked k={k} in {time.perf_counter() - started:.1f}s")

        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
        except Exception as e:
            raise RuntimeError(f"Clustering failed: {str(e)}")

        if self.misc_threshold:
            labels = _misc_labels(embeddings, labels, kmeans.cluster_centers_, self.misc_threshold)
        return labels

    def _analyze_folder(self, folder_path):

        file_data = {
            "Path": os.path.basename(folder_path),
            "Content": "" 
        }

        for entry in os.scandir(folder_path):
            if _is_noise_entry(entry.name, self.skip_noise, self.exclude_patterns):
                continue

            # entry.path is already the full path - os.path.join(folder_path, entry)
            # would double it, since DirEntry.__fspath__ returns entry.path too.
            full_path = entry.path

            if entry.is_file():
                # Once the folder has enough text for the embedding, the rest
                # of its files only contribute their names.
                full = len(file_data["Content"]) >= READ_CHARS
                file_data["Content"] += entry.name + ((None if full else _read_content(full_path)) or "")

            else:
                if entry.is_symlink():
                    logger.warning(f"Skipping symlinked directory: {full_path}")
                    continue
                # Recursively process subfolders
                subfolder_data = self._analyze_folder(full_path)
                file_data["Content"] += subfolder_data["Content"]

        return file_data
    
    def _find_optimal_clusters(self, embeddings, max_k=10):
        self.k_scores = None
        # Fewer than 3 documents can't be meaningfully split into 2+ clusters.
        if len(embeddings) <= 2:
            return 1

        distortions = []
        silhouette_scores = []
        davies_bouldin_scores = []
        # Upper bound is inclusive of max_k (previously excluded it, so
        # --max-clusters never actually tested its own value). Still capped
        # at len(embeddings) - 1: silhouette_score requires strictly fewer
        # clusters than samples, so k == n would crash mid-loop.
        K = range(2, min(max_k, len(embeddings) - 1) + 1)

        # Calculate multiple metrics
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)

            distortions.append(kmeans.inertia_)
            # silhouette is O(n^2): estimate it from a sample on big folders.
            silhouette_scores.append(silhouette_score(
                embeddings, kmeans.labels_, random_state=0,
                sample_size=2000 if len(embeddings) > 2000 else None,
            ))
            davies_bouldin_scores.append(davies_bouldin_score(embeddings, kmeans.labels_))

        # Each metric votes for a k. The elbow vote is None below 5 documents
        # (no 2nd difference yet), leaving silhouette/Davies-Bouldin only.
        # Davies-Bouldin (lower = better) replaced Calinski-Harabasz, which
        # voted k=2 on nearly every real folder - so the median collapsed to
        # min(elbow, silhouette). On the 68-file demo that meant k=4 (adjusted
        # Rand 0.35) instead of k=10 (0.85); the 32-file test folder chose
        # k=9 either way.
        votes = {
            "silhouette": K[int(np.argmax(silhouette_scores))],
            "davies_bouldin": K[int(np.argmin(davies_bouldin_scores))],
            "elbow": _elbow_k(K, distortions),
        }

        # Median, not most-common: when all three metrics disagree, Counter
        # just returned the first-inserted candidate (silhouette), which on a
        # dense blob plus a few outliers always votes for a tiny k. The median
        # still returns the majority value whenever two metrics agree.
        k_optimal = int(np.median([k for k in votes.values() if k is not None]))

        # Validate result
        if k_optimal < 2:
            k_optimal = 2
        if k_optimal >= len(embeddings):
            k_optimal = len(embeddings) - 1

        self.k_scores = {
            "k": list(K),
            "inertia": distortions,
            "silhouette": silhouette_scores,
            "davies_bouldin": davies_bouldin_scores,
            "votes": votes,
            "chosen": k_optimal,
        }
        return k_optimal
    
    def _organize_clusters(self, df, embeddings=None, depth=1, parent=""):
        """{folder: [files]} from df's Cluster labels (-1 = Misc).

        With depth > 1 (and the embeddings), a cluster of SPLIT_MIN+ files is
        clustered again into "Parent/Child" folders; a file the lower level
        sends to Misc stays in the parent folder itself.
        """
        folder_structure = {}
        taken = set(re.split(r"[_/]", parent.lower())) if parent else set()
        labels = df["Cluster"].to_numpy()

        for cluster in pd.unique(labels):
            mask = labels == cluster
            cluster_files = df[mask]
            cluster_embeddings = None if embeddings is None else embeddings[mask]
            if cluster == -1:
                # Outliers have nothing in common to extract keywords from.
                name = parent or MISC_FOLDER
            else:
                name = self._cluster_name(cluster_files, cluster_embeddings, taken) or f"Cluster_{cluster}"
                name = f"{parent}/{name}" if parent else name
                if depth > 1 and cluster_embeddings is not None and mask.sum() >= SPLIT_MIN:
                    sub = cluster_files.assign(Cluster=self._cluster(cluster_embeddings))
                    for folder, files in self._organize_clusters(sub, cluster_embeddings, depth - 1, name).items():
                        folder_structure.setdefault(folder, []).extend(files)
                    continue
            # Different clusters can extract the same top keywords; merge into
            # the existing folder instead of overwriting it and losing files.
            folder_structure.setdefault(name, []).extend(cluster_files["Path"].tolist())

        return folder_structure

    def _cluster_name(self, cluster_files, embeddings, taken):
        """YAKE's name for a cluster, or a local LLM's (--llm-names) given
        YAKE's keywords and the files nearest the cluster's centre."""
        keywords = self._extract_keywords(cluster_files["Text"].tolist())
        # No fallback to the unfiltered name: on 2,000 license texts that
        # produced "Public_license_general/Public_license_general".
        name = _folder_name(keywords, taken)
        if self.llm_names and keywords:
            files = cluster_files["Path"].tolist()
            if embeddings is not None:
                order = np.argsort(-(embeddings @ embeddings.mean(axis=0)))
                files = [files[i] for i in order]
            name = _llm_folder_name(self.llm_names, keywords, files[:5]) or name
        return name
    
    def _keyword_extractor(self, langs):
        return yake.KeywordExtractor(
            lan=langs[0], n=self.yake_ngram, top=self.yake_top, stopwords=_stopwords(langs)
        )

    def _extract_keywords(self, text_list):

        full_text = " ".join(text_list)
        keywords = self.kw_extractor.extract_keywords(full_text)
        return [kw[0] for kw in keywords]
    
    # Static: applying a plan (--apply, the UI) needs no model loaded.
    @staticmethod
    def organize_files(folder_structure, source_dir, target_dir=None, copy=False):
            target_dir = target_dir or source_dir

            logger.info(f"Creating directories in: {target_dir}")
            os.makedirs(target_dir, exist_ok=True)
            for folder_name in folder_structure.keys():
                # Level by level, for "Parent/Child" folders. Only a folder
                # created here gets the marker (so a later run on the same
                # directory skips it): with --into the destinations are the
                # user's own folders, which a later run must still see.
                folder_path = target_dir
                for part in folder_name.split("/"):
                    folder_path = os.path.join(folder_path, part)
                    if not os.path.isdir(folder_path):
                        os.mkdir(folder_path)
                        with open(os.path.join(folder_path, ORGANIZED_MARKER), "w"):
                            pass

            # Move files to their respective folders
            manifest = []
            for folder_name, files in folder_structure.items():
                target_folder = os.path.join(target_dir, folder_name)
                for file in files:
                    source_path = os.path.join(source_dir, file)

                    if not os.path.exists(source_path):
                        continue

                    destination_path = os.path.join(target_folder, file)
                    # shutil.move -> os.rename on POSIX silently clobbers an
                    # existing destination with no error, so check first
                    # instead of relying on FileExistsError (Windows-only).
                    if os.path.exists(destination_path):
                        logger.warning(f"'{file}' already exists in {target_folder}, skipping to avoid overwriting it.")
                        continue

                    try:
                        if not copy:
                            shutil.move(source_path, destination_path)
                            manifest.append({"source": source_path, "destination": destination_path})
                        else:
                            # Documents can be whole subfolders, not just files.
                            if os.path.isdir(source_path):
                                shutil.copytree(source_path, destination_path, symlinks=True)
                            else:
                                shutil.copy2(source_path, destination_path)
                            manifest.append({"source": source_path, "destination": destination_path, "copy": True})
                    except Exception as e:
                        logger.error(f"Error while {'copying' if copy else 'moving'} {file}: {str(e)}")

            if not manifest:
                return None

            manifest_path = os.path.join(target_dir, f"{MANIFEST_PREFIX}{datetime.now():%Y%m%d-%H%M%S}.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"\nManifest written to {manifest_path}")
            print(f"Undo with: organize-files --undo {manifest_path}")
            _record_history({
                "time": datetime.now().isoformat(timespec="seconds"),
                "source_dir": os.path.abspath(source_dir),
                "target_dir": os.path.abspath(target_dir),
                "manifest": os.path.abspath(manifest_path),
                "files": len(manifest),
                "copy": copy,
            })

            return manifest_path


organize_files = DocumentAnalyzer.organize_files


def _record_history(entry):
    # A convenience log: failing to write it must never fail the run.
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        logger.warning(f"Could not record this run in {HISTORY_PATH}: {e}")


def read_history():
    """Past organize runs, newest first (malformed lines skipped)."""
    if not os.path.exists(HISTORY_PATH):
        return []
    entries = []
    with open(HISTORY_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries[::-1]


def undo_organize(manifest_path):
    """Reverses a previous organize_files run using its manifest JSON.

    Moved entries go back to their source. Copied entries are deleted, but
    only while the original still exists - never the last copy. Afterwards,
    cluster folders holding nothing but the marker are removed, and so is the
    manifest itself if every entry was undone (kept otherwise, for a retry).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    skipped = 0
    for entry in manifest:
        source, destination = entry["source"], entry["destination"]

        if not os.path.exists(destination):
            logger.warning(f"'{destination}' not found, skipping.")
            skipped += 1
            continue

        if entry.get("copy"):
            if not os.path.exists(source):
                logger.warning(f"Original '{source}' is gone, keeping its copy '{destination}'.")
                skipped += 1
                continue
            if os.path.isdir(destination) and not os.path.islink(destination):
                shutil.rmtree(destination)
            else:
                os.remove(destination)
            logger.info(f"Removed copy: {destination}")
            continue

        if os.path.exists(source):
            logger.warning(f"'{source}' already exists, skipping restore of '{destination}'.")
            skipped += 1
            continue

        os.makedirs(os.path.dirname(source) or ".", exist_ok=True)
        shutil.move(destination, source)
        logger.info(f"Restored: {destination} -> {source}")

    # Walk up from each destination folder, so a "Parent/Child" tree goes
    # once its last child does. Stops at the first folder holding anything
    # else - a folder the tool didn't create has no marker, so it never goes.
    for folder in {os.path.dirname(entry["destination"]) for entry in manifest}:
        while os.path.isdir(folder) and os.listdir(folder) == [ORGANIZED_MARKER]:
            os.remove(os.path.join(folder, ORGANIZED_MARKER))
            os.rmdir(folder)
            folder = os.path.dirname(folder)

    if skipped:
        print(f"Undo finished with {skipped} entr{'y' if skipped == 1 else 'ies'} skipped; manifest kept at {manifest_path}")
    else:
        os.remove(manifest_path)
        print(f"Undo complete: {len(manifest)} entr{'y' if len(manifest) == 1 else 'ies'} reverted.")
