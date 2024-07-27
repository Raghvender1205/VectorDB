# VectorDB
A vector database implementation in Rust.


## Features
- VectorDB Structure: The database is represented by a struct that maps an integer ID to a vector.
- Add Vector: Allows insertion of vectors into the database
- Find Nearest: Computes Euclidean distance between a `query` vector and all vectors in the database returning the close one.

## TODO
1. Implement more efficient distance calculations.
2. Add more metrics
3. Use advanced datastructures for fast nearest neighbor searches.
4. Adding concurrency for handling multiple queries simultaneously
5. Scaling up with a distributed system approach.