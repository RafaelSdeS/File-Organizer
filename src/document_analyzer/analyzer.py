import pandas as pd
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import yake
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import os
import shutil
import json
import fnmatch
from datetime import datetime
import numpy as np
import logging

from .utils import create_weighted_text, read_file, read_audio_metadata

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

readable_files = [".pdf", ".txt", ".docx", ".xml"]
audio_files = [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".aiff", ".opus"]

# Dropped into every cluster folder organize_files creates, so a later run
# recognizes and skips its own output instead of re-clustering it.
ORGANIZED_MARKER = ".file_organizer_created"
MANIFEST_PREFIX = ".file_organizer_manifest_"

# Known-noise directory/file names that aren't already caught by the
# hidden-entry (dot-prefix) check - e.g. .git, .DS_Store are dotfiles already.
DEFAULT_SKIP_NAMES = {"__pycache__", "node_modules"}


def _stopwords(langs):
    """Union of YAKE's bundled stopword lists, for mixed-language directories.

    YAKE only takes one `lan`, so a pt+en directory always had the wrong
    stopwords for half its files. Returns None (= let YAKE load its own) for
    a single language.

    Reads the lists back off throwaway extractors rather than opening the
    bundled .txt files directly - their location moves between YAKE versions.
    An unknown language code falls back to YAKE's language-agnostic list.
    """
    if len(langs) < 2:
        return None

    words = set()
    for lang in langs:
        words.update(yake.KeywordExtractor(lan=lang).stopword_set)
    return words


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

    def __init__(self, path_weight=2, max_clusters=10, yake_ngram=2, yake_top=5, skip_noise=True, exclude_patterns=None, lang="en"):
        if max_clusters < 3:
            raise ValueError("max_clusters must be at least 3 (k=2 needs to be a testable candidate)")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        # lang picks YAKE's stopword list - with the wrong one, the target
        # language's stopwords ("de", "do", "para", ...) become folder names.
        # Comma-separated ("pt,en") unions the lists for mixed directories.
        langs = [part.strip() for part in lang.split(",") if part.strip()] or ["en"]
        self.kw_extractor = yake.KeywordExtractor(
            lan=langs[0], n=yake_ngram, top=yake_top, stopwords=_stopwords(langs)
        )
        self.path_weight = path_weight
        self.max_clusters = max_clusters
        self.skip_noise = skip_noise
        self.exclude_patterns = exclude_patterns or []

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
                )
            except Exception as e:
                raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

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
                df["Cluster"] = kmeans.fit_predict(embeddings)
            except Exception as e:
                raise RuntimeError(f"Clustering failed: {str(e)}")
            
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
        # Fewer than 3 documents can't be meaningfully split into 2+ clusters.
        if len(embeddings) <= 2:
            return 1

        distortions = []
        silhouette_scores = []
        calinski_scores = []
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
            calinski_scores.append(calinski_harabasz_score(embeddings, kmeans.labels_))

        # Find optimal k using multiple criteria
        silhouette_max = np.argmax(silhouette_scores)
        calinski_max = np.argmax(calinski_scores)
        k_candidates = [K[silhouette_max], K[calinski_max]]

        # The elbow method's 2nd derivative needs at least 3 distortion points
        # (i.e. at least 5 documents) to produce anything non-empty.
        second_derivative = np.diff(distortions, 2)
        if len(second_derivative) > 0:
            k_candidates.append(K[np.argmin(second_derivative)])

        # Median, not most-common: when all three metrics disagree, Counter
        # just returned the first-inserted candidate (silhouette), which on a
        # dense blob plus a few outliers always votes for a tiny k. The median
        # still returns the majority value whenever two metrics agree.
        k_optimal = int(np.median(k_candidates))

        # Validate result
        if k_optimal < 2:
            k_optimal = 2
        if k_optimal >= len(embeddings):
            k_optimal = len(embeddings) - 1

        return k_optimal
    
    def _organize_clusters(self, df):
        
        # Organizes files into clusters based on their content similarity
        folder_structure = {}
        cluster_keywords = {}
        
        for cluster in df["Cluster"].unique():
            cluster_files = df[df["Cluster"] == cluster]
            # Extract keywords for this cluster
            keywords = self._extract_keywords(cluster_files["Text"].tolist())
            # Create folder name from top keywords
            folder_name = _folder_name(keywords)
            cluster_keywords[cluster] = folder_name if folder_name else f"Cluster_{cluster}"
            # Different clusters can extract the same top keywords; merge into
            # the existing folder instead of overwriting it and losing files.
            folder_structure.setdefault(cluster_keywords[cluster], []).extend(
                cluster_files["Path"].tolist()
            )
            
        return folder_structure
    
    def _extract_keywords(self, text_list):

        full_text = " ".join(text_list)
        keywords = self.kw_extractor.extract_keywords(full_text)
        return [kw[0] for kw in keywords]
    
    def organize_files(self, folder_structure, source_dir, target_dir=None):
            target_dir = target_dir or source_dir

            print(f"\nCreating directories in: {target_dir}")
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
                        print(f"Warning: '{file}' already exists in {target_folder}, skipping to avoid overwriting it.")
                        continue

                    try:
                        shutil.move(source_path, destination_path)
                        manifest.append({"source": source_path, "destination": destination_path})
                    except Exception as e:
                        print(f"Error while moving {file}: {str(e)}")

            if not manifest:
                return None

            manifest_path = os.path.join(target_dir, f"{MANIFEST_PREFIX}{datetime.now():%Y%m%d-%H%M%S}.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"\nManifest written to {manifest_path}")
            print(f"Undo with: organize-files --undo {manifest_path}")

            return manifest_path


def undo_organize(manifest_path):
    """Reverses a previous organize_files run using its manifest JSON."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    for entry in manifest:
        source, destination = entry["source"], entry["destination"]

        if not os.path.exists(destination):
            print(f"Warning: '{destination}' not found, skipping.")
            continue
        if os.path.exists(source):
            print(f"Warning: '{source}' already exists, skipping restore of '{destination}'.")
            continue

        os.makedirs(os.path.dirname(source) or ".", exist_ok=True)
        shutil.move(destination, source)
        print(f"Restored: {destination} -> {source}")
