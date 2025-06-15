## Move to RocksDB and HNSW Indexing
```
+----------------------+
|   REST API Layer     |
|  (Actix-Web)  |
+----------+-----------+
           |
           v
+----------------------+
|     VectorDB Layer   |
+----------------------+
| RocksDB: Key-Value DB| <-----+
|  - Key: "vec:{id}"   |       |
|  - Key: "meta:{id}"  |       |
+----------------------+       |
           |                   |
           v                   |
+----------------------+       |
| HNSW Index (In-Mem)  |-------+
| - id -> embedding    |
+----------------------+
```

- `RocksDB`:- `"vec:{id}"` for vector, `"meta:{id}"` for metadata
- `HNSW`:- Maintains in-memory or mmap'ed graph for fast search
- `Serialization`:- `Vectors` = bincode, `Metadata` = JSON
- `Metadata Filtering`:- Apply filters after/during search 
- `Persistent Index` :- Serialize HNSW graph to disk (`.dump()` + `.load()`)
- `Background re-index` :- Load vectors from RocksDB and rebuild index
- `Deletion` : `index.remove(id)` + `db.delete(key)` support.