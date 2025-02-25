import pandas as pd
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import yake
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter
import os
import numpy as np
import logging

from .utils import create_weighted_text, read_file

logger = logging.getLogger(__name__)

readable_files = [".pdf", ".txt", ".docx", ".xml"]
class DocumentAnalyzer:

    #Main class for analyzing and organizing documents using AI-powered clustering.
    #Uses sentence embeddings for content similarity analysis and YAKE for keyword extraction.

    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=5)
        
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
                    try:
                        if entry.is_file():
                            _, ext = os.path.splitext(entry.name)
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
                            else:
                                file_data = {
                                    "Path": entry.name,
                                    "Content": None
                                }
                        else:
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
                    path_weight=2
                ),
                axis=1
            )
            
            # Generate embeddings for clustering
            try:
                embeddings = self.model.encode(df["Text"].tolist())
            except Exception as e:
                raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
            
            # Find optimal number of clusters using the Elbow method
            try:
                optimal_clusters = self._find_optimal_clusters(embeddings)
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

        for file_path in os.scandir(folder_path):
            full_path = os.path.join(folder_path, file_path)

            if os.path.isfile(full_path):
                _, ext = os.path.splitext(file_path)
                
                if ext in readable_files:
                    content = read_file(full_path)
                    file_data["Content"] += (file_path.name + content)

                else:
                    file_data["Content"] += file_path.name

            else:
#                Recursively process subfolders
                subfolder_data = self._analyze_folder(full_path)
                file_data["Content"] += subfolder_data["Content"]

        return file_data
    
    def _find_optimal_clusters(self, embeddings, max_k=10):
        distortions = []
        silhouette_scores = []
        calinski_scores = []
        K = range(2, min(len(embeddings), max_k))
        
        # Calculate multiple metrics
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            
            distortions.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(embeddings, kmeans.labels_))
            calinski_scores.append(calinski_harabasz_score(embeddings, kmeans.labels_))
        
        # Find optimal k using multiple criteria
        second_derivative = np.diff(distortions, 2)
        silhouette_max = np.argmax(silhouette_scores)
        calinski_max = np.argmax(calinski_scores)
        
        # Combine metrics for more robust result
        k_candidates = [
            K[np.argmin(second_derivative)],
            K[silhouette_max],
            K[calinski_max]
        ]
        
        # Select most frequent k or use silhouette if tie
        k_counts = Counter(k_candidates)
        k_optimal = k_counts.most_common(1)[0][0]
        
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
            folder_name = "_".join(keywords[:2]).replace(" ", "_").capitalize()
            cluster_keywords[cluster] = folder_name if folder_name else f"Cluster_{cluster}"
            folder_structure[cluster_keywords[cluster]] = cluster_files["Path"].tolist()
            
        return folder_structure
    
    def _extract_keywords(self, text_list):

        full_text = " ".join(text_list)
        keywords = self.kw_extractor.extract_keywords(full_text)
        return [kw[0] for kw in keywords]
    
    def organize_files(self, folder_structure, source_dir, target_dir=None):
        target_dir = target_dir or source_dir
        
        # Create all necessary directories
        for folder_name in folder_structure.keys():
            folder_path = os.path.join(target_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
        
        # Move files to their respective folders
        for folder_name, files in folder_structure.items():
            target_folder = os.path.join(target_dir, folder_name)
            for file in files:
                source_path = os.path.join(source_dir, file)
                if os.path.exists(source_path):
                    os.rename(source_path, target_folder)
