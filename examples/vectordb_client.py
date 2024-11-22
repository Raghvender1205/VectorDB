import requests
from typing import Optional, List


class VectorDBClient:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")

    def add_document(self, id: int, embedding: List[float], metadata: str) -> bool:
        """
        Adds a document to vectordb
        """
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
            print(f"Failed to add document ID {id}: {e}")
            return False
        

    def find_nearest(
        self, 
        query: List[float], 
        n: int, 
        metric: str = "Dot",
        metadata_filter: Optional[str] = None
    ) -> List[dict]:
        url = f"{self.server_url}/search"
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