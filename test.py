import pandas as pd
import yake
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter
import os
import PyPDF2
import numpy as np

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

        readable_files = [".pdf", ".txt", ".docx", ".xml"]
        
        for file_path in os.listdir(path):
            if not os.path.isfile(file_path):
                file_data = {
                    "Path": file_path,
                    "Content": None
                }
            else:
                _, ext = os.path.splitext(file_path)
                if ext in readable_files:
                    content = self._read_file(os.path.join(path, file_path))
                    file_data = {
                        "Path": file_path,
                        "Content": content
                    }
                else:
                    file_data = {
                        "Path": file_path,
                        "Content": None
                    }
            files_data.append(file_data)
            
        df = pd.DataFrame(files_data)
        
        # Create weighted text combining path and content
        df["Text"] = df.apply(
            lambda row: self._create_weighted_text(
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
    
    def _read_file(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                # Optimize text length based on content analysis
                return text[:self._analyze_document_content(text)]
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    
    def _analyze_document_content(self, text):
        """
        Analyzes document content to determine optimal text extraction length.
        The length is calculated based on:
        - Number of sections (paragraph breaks)
        - Number of keywords (words > 5 characters)
        """
        if not text:
            return 1000  # Default minimum length
            
        # Count content indicators
        section_markers = text.count('\n\n') + text.count('. ') + text.count(' ')
        keywords = len([word for word in text.split() if len(word) > 5])
        
        # Calculate optimal length
        base_length = 1000
        additional_length = min(
            section_markers * 200,  # Add length for each section
            keywords * 50,          # Add length for each keyword
            3000                    # Maximum additional length
        )
        return base_length + additional_length
    
    def _create_weighted_text(self, path, content, path_weight=2):
        if pd.isna(content):
            return path * path_weight
        return (path * path_weight) + " " + content
    
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

def main():
    analyzer = DocumentAnalyzer()
    path = input("Select the desired directory to be organized: ")
    folder_structure = analyzer.analyze_directory(path)
    
    print("\nOrganized structure:")
    for folder, files in folder_structure.items():
        print(f"\n📂 {folder}:")
        for f in files:
            print(f"   - {f}")

if __name__ == "__main__":
    main()