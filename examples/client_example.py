import requests
import numpy as np

from typing import List, Optional


class VectorDBClient:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
    
    def add_document(self, id: int, embedding: List[float], metadata: str) -> bool:
        url = f"{self.server_url}/add_document"
        payload = {
            "id": id,
            "embedding": embedding,
            "metadata": metadata
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Failed to add document: {e}")
            return False

    def find_nearest(
        self, 
        query: List[float], 
        n: int, 
        metric: str = "Dot",
        metadata_filter: Optional[str] = None
    ) -> List[dict]:
        url = f"{self.server_url}/find_nearest"
        payload = {
            "query": query,
            "n": n,
            "metric": metric,
            "metadata_filter": metadata_filter
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to find nearest neighbors: {e}")
            return []
        
if __name__ == "__main__":
    client = VectorDBClient("http://127.0.0.1:8444")
    
    # Add a document
    doc_id = 2
    embedding = np.random.rand(128).tolist()  # Example 128-dim vector
    metadata = "example metadata"
    success = client.add_document(doc_id, embedding, metadata)
    if success:
        print(f"Document {doc_id} added successfully.")
    
    # Query nearest neighbors
    query_vector = np.random.rand(128).tolist()
    top_n = 5
    metric = "Dot"
    filter_str = "example"
    neighbors = client.find_nearest(query_vector, top_n, metric, filter_str)
    print("Nearest Neighbors:")
    for neighbor in neighbors:
        print(neighbor)