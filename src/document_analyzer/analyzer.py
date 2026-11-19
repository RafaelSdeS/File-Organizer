import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
import yake
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import os
import shutil
import json
import fnmatch
import re
from collections import Counter
from datetime import datetime
import numpy as np
import logging

from .utils import create_weighted_text, read_file, read_audio_metadata, set_transcription_language

logger = logging.getLogger(__name__)

# Multilingual by default: all-MiniLM-L6-v2 is English-only, so non-English
# documents got near-random embeddings and therefore near-random clusters.
# e5-small over paraphrase-multilingual-MiniLM-L12-v2 for the context window:
# 512 tokens vs 128. At 128 every letterheaded PDF (university header, CNPJ,
# address) embedded to the same boilerplate and collapsed into one cluster.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
# e5 wants a task prefix; its model card specifies "query: " for symmetric
# tasks like clustering.
EMBEDDING_PREFIX = "query: "

readable_files = [".pdf", ".txt", ".md", ".csv", ".xml", ".docx", ".pptx", ".odt"]
audio_files = [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".aiff", ".opus"]

# Dropped into every cluster folder organize_files creates, so a later run
# recognizes and skips its own output instead of re-clustering it.
ORGANIZED_MARKER = ".file_organizer_created"
MANIFEST_PREFIX = ".file_organizer_manifest_"
# Folder for documents too far from every cluster centroid to belong anywhere.
MISC_FOLDER = "Misc"


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


def _folder_name(keywords):
    """Folder name from the top-2 keywords, minus words they share.

    YAKE's top keywords overlap constantly ("engenharia eletrica" +
    "energia eletrica"), which used to produce names like
    "Engenharia_eletrica_energia_eletrica".
    """
    seen, words = set(), []
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


class DocumentAnalyzer:

    #Main class for analyzing and organizing documents using AI-powered clustering.
    #Uses sentence embeddings for content similarity analysis and YAKE for keyword extraction.

    def __init__(self, path_weight=2, max_clusters=10, yake_ngram=2, yake_top=5, skip_noise=True, exclude_patterns=None, lang="auto", misc_threshold=2.0):
        if max_clusters < 3:
            raise ValueError("max_clusters must be at least 3 (k=2 needs to be a testable candidate)")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
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
        # Filled by analyze_directory / _find_optimal_clusters, for --report.
        self.last_run = None
        self.k_scores = None

    def analyze_directory(self, path):
        """
        Analyzes all files in the specified directory and organizes them into clusters.
        
        Args:
            path (str): Path to the directory to analyze
            
        Returns:
            dict: Organized cluster information'
            
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
            if not os.path.exists(path):
                raise FileNotFoundError(f"Directory not found: {path}")
            if not os.path.isdir(path):
                raise ValueError(f"Path is not a directory: {path}")
            if not os.access(path, os.R_OK):
                raise PermissionError(f"Permission denied for directory: {path}")
            
            files_data = []
            
            with os.scandir(path) as dir_iter:
                for entry in dir_iter:
                    if _is_noise_entry(entry.name, self.skip_noise, self.exclude_patterns):
                        continue
                    try:
                        if entry.is_file():
                            _, ext = os.path.splitext(entry.name)
                            ext = ext.lower()
                            if ext in readable_files:
                                try:
                                    content = read_file(os.path.join(path, entry.name))
                                    file_data = {
                                        "Path": entry.name,
                                        "Content": content
                                    }
                                except Exception as e:
                                    logger.warning(f"Failed to read file {entry.name}: {str(e)}")
                                    file_data = {
                                        "Path": entry.name,
                                        "Content": None
                                    }
                            elif ext in audio_files:
                                try:
                                    content = read_audio_metadata(os.path.join(path, entry.name))
                                    file_data = {
                                        "Path": entry.name,
                                        "Content": content
                                    }
                                except Exception as e:
                                    logger.warning(f"Failed to read audio file {entry.name}: {str(e)}")
                                    file_data = {
                                        "Path": entry.name,
                                        "Content": None
                                    }
                            else:
                                file_data = {
                                    "Path": entry.name,
                                    "Content": None
                                }
                        else:
                            if entry.is_symlink():
                                logger.info(f"Skipping symlinked directory: {entry.name}")
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

            # Generate embeddings for clustering
            logger.info(f"Read {len(files_data)} entries from {path}; generating embeddings...")
            try:
                # normalize_embeddings makes KMeans' Euclidean distance
                # monotonic in cosine distance, which is the metric these
                # embeddings are actually trained for.
                embeddings = self.model.encode(
                    [EMBEDDING_PREFIX + text for text in df["Text"]],
                    normalize_embeddings=True,
                    show_progress_bar=logger.isEnabledFor(logging.INFO),
                )
            except Exception as e:
                raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

            languages = _detect_languages(df["Text"], self.stopword_sets)
            counts = Counter(lang or "none" for lang in languages.tolist())
            logger.info(f"Detected languages: {dict(counts)}")
            embeddings = _center_by_language(embeddings, languages)
            if self.auto_lang:
                found = [lang for lang in self.stopword_sets if lang in set(languages)]
                self.kw_extractor = self._keyword_extractor(found or ["en"])

            # Find optimal number of clusters using the Elbow method
            logger.info("Determining optimal number of clusters...")
            try:
                optimal_clusters = self._find_optimal_clusters(embeddings, max_k=self.max_clusters)
            except Exception as e:
                raise RuntimeError(f"Failed to find optimal clusters: {str(e)}")
            
            # Perform K-means clustering
            try:
                kmeans = KMeans(
                    n_clusters=optimal_clusters,
                    random_state=42,
                    n_init=10
                )
                labels = kmeans.fit_predict(embeddings)
            except Exception as e:
                raise RuntimeError(f"Clustering failed: {str(e)}")

            # KMeans forces every document into some cluster, so a one-off
            # gets glued to its nearest neighbours. Anything unusually far
            # from its own centroid goes to Misc instead (label -1).
            # ponytail: global z-score over all distances; per-cluster
            # thresholds if clusters of very different spread show up.
            if self.misc_threshold:
                dist = np.linalg.norm(embeddings - kmeans.cluster_centers_[labels], axis=1)
                labels = np.where(dist > dist.mean() + self.misc_threshold * dist.std(), -1, labels)
            df["Cluster"] = labels

            self.last_run = {
                "paths": df["Path"].tolist(),
                "embeddings": embeddings,
                "k_scores": self.k_scores,
            }

            # Organize files into clusters
            return self._organize_clusters(df)
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise
    
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
                _, ext = os.path.splitext(entry.name)
                ext = ext.lower()

                if ext in readable_files:
                    try:
                        content = read_file(full_path)
                        file_data["Content"] += (entry.name + content)
                    except Exception as e:
                        logger.warning(f"Failed to read file {full_path}: {str(e)}")
                        file_data["Content"] += entry.name

                elif ext in audio_files:
                    try:
                        content = read_audio_metadata(full_path)
                        file_data["Content"] += (entry.name + (content or ""))
                    except Exception as e:
                        logger.warning(f"Failed to read audio file {full_path}: {str(e)}")
                        file_data["Content"] += entry.name

                else:
                    file_data["Content"] += entry.name

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
            silhouette_scores.append(silhouette_score(embeddings, kmeans.labels_))
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
    
    def _organize_clusters(self, df):
        
        # Organizes files into clusters based on their content similarity
        folder_structure = {}
        cluster_keywords = {}
        
        for cluster in df["Cluster"].unique():
            cluster_files = df[df["Cluster"] == cluster]
            if cluster == -1:
                # Outliers have nothing in common to extract keywords from.
                cluster_keywords[cluster] = MISC_FOLDER
            else:
                keywords = self._extract_keywords(cluster_files["Text"].tolist())
                folder_name = _folder_name(keywords)
                cluster_keywords[cluster] = folder_name if folder_name else f"Cluster_{cluster}"
            # Different clusters can extract the same top keywords; merge into
            # the existing folder instead of overwriting it and losing files.
            folder_structure.setdefault(cluster_keywords[cluster], []).extend(
                cluster_files["Path"].tolist()
            )
            
        return folder_structure
    
    def _keyword_extractor(self, langs):
        return yake.KeywordExtractor(
            lan=langs[0], n=self.yake_ngram, top=self.yake_top, stopwords=_stopwords(langs)
        )

    def _extract_keywords(self, text_list):

        full_text = " ".join(text_list)
        keywords = self.kw_extractor.extract_keywords(full_text)
        return [kw[0] for kw in keywords]
    
    def organize_files(self, folder_structure, source_dir, target_dir=None, copy=False):
            target_dir = target_dir or source_dir

            logger.info(f"Creating directories in: {target_dir}")
            # Create all necessary directories
            for folder_name in folder_structure.keys():
                folder_path = os.path.join(target_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                # Marks this as a folder the tool created, so a later run on
                # the same directory skips it instead of re-clustering it.
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

            return manifest_path


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

    for folder in {os.path.dirname(entry["destination"]) for entry in manifest}:
        if os.path.isdir(folder) and os.listdir(folder) == [ORGANIZED_MARKER]:
            os.remove(os.path.join(folder, ORGANIZED_MARKER))
            os.rmdir(folder)

    if skipped:
        print(f"Undo finished with {skipped} entr{'y' if skipped == 1 else 'ies'} skipped; manifest kept at {manifest_path}")
    else:
        os.remove(manifest_path)
        print(f"Undo complete: {len(manifest)} entr{'y' if len(manifest) == 1 else 'ies'} reverted.")
