import pandas as pd
import yake
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter
import os
import numpy as np

from .utils import create_weighted_text, read_file

readable_files = [".pdf"] #, ".txt", ".docx", ".xml"

class DocumentAnalyzer:

    #Main class for analyzing and organizing documents using AI-powered clustering.
    #Uses sentence embeddings for content similarity analysis and YAKE for keyword extraction.

    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=5)
        
    def analyze_directory(self, path):

        # Analyzes all files in the specified directory and organizes them into clusters.
        # Initialize data structure to store file information

        files_data = []

        for file_path in os.scandir(path):
            if not file_path.is_file():
                file_data = self._analyze_folder(file_path)
            else:
                _, ext = os.path.splitext(file_path)
                if ext in readable_files:
                    content = read_file(os.path.join(path, file_path))
                    file_data = {
                        "Path": file_path.name,
                        "Content": content
                    }
                else:
                    file_data = {
                        "Path": file_path.name,
                        "Content": None
                    }
            files_data.append(file_data)
            
        df = pd.DataFrame(files_data)
        
        # Create weighted text combining path and content
        df["Text"] = df.apply(
            lambda row: create_weighted_text(
                row["Path"],
                row["Content"],
                path_weight=2
            ),
            axis=1
        )
        
        # Generate embeddings for clustering
        embeddings = self.model.encode(df["Text"].tolist())
        
        # Find optimal number of clusters using the Elbow method
        optimal_clusters = self._find_optimal_clusters(embeddings)
        
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=optimal_clusters,
            random_state=42,
            n_init=10
        )
        df["Cluster"] = kmeans.fit_predict(embeddings)
        
        # Organize files into clusters
        return self._organize_clusters(df)
    
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
                    file_data["Content"] += content
                else:
                    file_data["Content"] += file_path.name
            else:
#                Recursively process subfolders
                subfolder_data = self._analyze_folder(full_path)
                file_data["Content"] += subfolder_data["Content"]
        return file_data
    
    def _find_optimal_clusters(self, embeddings, max_k=10):
        
        distortions = []
        K = range(2, min(len(embeddings), max_k))
        
        # Calculate distortion for each k
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            distortions.append(kmeans.inertia_)
        
        # Find the "elbow" point using second derivative
        k_optimal = K[np.argmin(np.diff(distortions, 2))]
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
