import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader

from vectordb_client import VectorDBClient

def pdf_to_text(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    return docs
    
def get_embeddings():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    return embeddings

def main():
    pdf_path = 'document.pdf'  
    server_url = "http://127.0.0.1:8444"
    metadata_category = "pdf_document" # Example

    # Initialize client
    client = VectorDBClient(server_url)
    if not os.path.exists(pdf_path):
        logger.error(f'PDF file not found at {pdf_path}')
        return 

    logger.info(f'Loading PDF from {pdf_path}')
    try:
        docs = pdf_to_text(pdf_path)
        logger.info(f"Loaded {len(docs)} pages from PDF")
    except Exception as e:
        logger.error(f"Error: {e}")

    embeddings = get_embeddings() # get embeddings

    # Iterate through documents and add to VectorDB
    for idx, doc in enumerate(docs, start=1):
        text = doc.page_content.strip()
        if not text:
            logger.debug(f"Skipping empty page {idx}")
            continue

        # Generate embeddings
        embedding = embeddings.embed_documents([text])[0]
        if embedding is None:
            logger.error(f"Failed to generate embeddings for page {idx}")
            continue
            
        # Create unique document ID (e.g., PDF_ID_PAGE_NUMBER)
        # Assuming PDF has a unique identifier, else use idx
        doc_id = idx  # Simple unique ID; adjust as needed

        # Metadata can include more information as needed
        metadata = f"{metadata_category} - Page {idx}"

        # Add document to VectorDB
        success = client.add_document(doc_id, embedding, metadata)
        if success:
            logger.info(f"Added document ID {doc_id} to VectorDB.")
        else:
            logger.error(f"Failed to add document ID {doc_id} to VectorDB.")

    print("All documents processed.")

    # Test search functionality
    question = input("Ask: ")
    query = embeddings.embed_documents([question])[0]
    retreived_docs = client.find_nearest(query=query, n=5, metric="Cosine", metadata_filter="pdf_document")
    if retreived_docs:
        for docs in retreived_docs:
            print(f"ID: {docs['id']}, Distance: {docs['distance']}, Metadata: {docs['metadata']}")
    else:
        print('No documents retrieved')


if __name__ == "__main__":
    main()