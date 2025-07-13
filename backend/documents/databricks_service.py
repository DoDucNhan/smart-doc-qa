# import os
# import requests
# import json
# from typing import List, Dict
# from decouple import config

# class DatabricksService:
#     def __init__(self):
#         self.host = config('DATABRICKS_HOST')
#         self.token = config('DATABRICKS_TOKEN')
#         self.headers = {
#             'Authorization': f'Bearer {self.token}',
#             'Content-Type': 'application/json'
#         }
    
#     def store_embeddings(self, embeddings: List[List[float]], metadata: List[Dict]):
#         """Store embeddings and metadata in Databricks"""
#         # For simplicity, we'll use a REST API approach
#         # In production, use Databricks SDK or direct SQL warehouse connection
        
#         data = {
#             'embeddings': embeddings,
#             'metadata': metadata
#         }
        
#         # This is a placeholder - implement based on your Databricks setup
#         # You might use Unity Catalog, Delta tables, or Vector Search
#         pass
    
#     def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
#         """Search for similar embeddings"""
#         # Placeholder for vector similarity search
#         # In production, implement using Databricks Vector Search or FAISS
#         pass

# Alternative: Local FAISS implementation for development
from typing import Dict, List
import faiss
import numpy as np
import pickle
import os

class LocalVectorStore:
    def __init__(self, dimension=1536):  # OpenAI embedding dimension
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.metadata = []
        self.index_file = 'vector_index.faiss'
        self.metadata_file = 'vector_metadata.pkl'
        
        # Load existing index if available
        self.load_index()
    
    def add_embeddings(self, embeddings: List[List[float]], metadata: List[Dict]):
        """Add embeddings to the index"""
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        self.index.add(embeddings_array)
        self.metadata.extend(metadata)
        
        # Save index
        self.save_index()
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar embeddings"""
        query_array = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_array)
        
        scores, indices = self.index.search(query_array, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append({
                    'metadata': self.metadata[idx],
                    'score': float(score)
                })
        
        return results
    
    def save_index(self):
        """Save index and metadata to disk"""
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load_index(self):
        """Load index and metadata from disk"""
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)