# VectorDB
A vector database implementation in Rust. A server and client based codebase. Just run the server and call the api endpoints to
1. embed documents
2. Search through.


## Features
- VectorDB Structure: The database is represented by a struct that maps an integer ID to a vector.
- Add : Allows insertion of vectors into the database
- Find Nearest: Computes Euclidean distance between a `query` vector and all vectors in the database returning the close one.

## TODO
1. Implement more efficient distance calculations.
2. Add more complex structure for metadata handling.
3. Use advanced datastructures for fast nearest neighbor searches.
4. Adding concurrency for handling multiple queries simultaneously
5. Scaling up with a distributed system approach.
6. Add more logs in server
7. More robust structure.

### Some more options
1. Currently, your VectorDB uses Euclidean distance. You can add support for other distance metrics like Cosine similarity and Dot Product.
2. Combine vector search with metadata filtering to allow users to filter results based on additional criteria stored in the `Document` metadata.
3. Batch import and Export of Documents: you might need to import or export large numbers of documents.
4. Persistent Storage.
5. Dynamic Indexing 