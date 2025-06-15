# HNSW (Hierarchial Navigable Small World) graph

It is a data structure used for `ANN` in high dimension spaces. It is particularly well-known for being both fast and memory-efficient.

`HNSW` builds a multi-layer graph where
- Each node represents a vector
- Nodes are connected to their neighbors based on distance.
- Higher levels contains `fewer` nodes and `longer-range` links, while lower levels are denser with `short-range` links

The search starts at the top layer and greedily moves towards the query vector, navigating down until the base layer.

* Follows the small world network, it's easy to move from one node to another in few steps
* Multi-Layer: Levels allows a coarse-to-fine search, improving speed
* Greedy Search: Starts from an entry point and traverses neighbors that are closer to the target

- Insert time `log(N)`
- Search time `log(N)`