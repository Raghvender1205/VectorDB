use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use rocksdb::{DB, Options, BlockBasedOptions, WriteBatch};
use serde::{Deserialize, Serialize};
use hnsw_rs::hnsw::{Hnsw, Neighbour};
use hnsw_rs::anndists::dist::{DistCosine, DistDot, DistL2};
use rayon::prelude::*;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

use crate::AddDocumentRequest;

/* Distance Metric */
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    Dot,
}

impl DistanceMetric {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "euclidean" => Some(Self::Euclidean),
            "cosine" => Some(Self::Cosine),
            "dot" => Some(Self::Dot),
            _ => None
        }
    }
}

/* Optimized Embedding Storage */
#[derive(Archive, RkyvDeserialize, RkyvSerialize)]
#[archive(compare(PartialEq))]
struct Embedding {
    data: Vec<f32>,
}

/* HNSW Enum */
enum MetricIndex<'a> {
    Cosine(Hnsw<'a, f32, DistCosine>),
    Dot(Hnsw<'a, f32, DistDot>),
    Euclidean(Hnsw<'a, f32, DistL2>),
}

impl<'a> MetricIndex<'a> {
    // embed to vectordb
    fn insert(&mut self, id: usize, emb: &[f32]) {
        match self {
            Self::Cosine(h) => h.insert((emb, id)),
            Self::Dot(h) => h.insert((emb, id)),
            Self::Euclidean(h) => h.insert((emb, id)), 
        }
    }

    // Search through
    fn search(&self, query: &[f32], top_k: usize) -> Vec<Neighbour> {
        let ef_search = std::cmp::max(2 * top_k, 50);
        match self {
            Self::Cosine(h) => h.search(query, top_k, ef_search),
            Self::Dot(h) => h.search(query, top_k, ef_search),
            Self::Euclidean(h) => h.search(query, top_k, ef_search),
        }
    }
}

/* Collection Meta & CollectionEntry */

#[derive(Clone, Serialize, Deserialize)]
pub struct CollectionMeta {
    pub id: u64,
    pub name: String,
    pub dim: usize,
    pub metric: DistanceMetric,
    pub doc_count: u64,
}

// Holds meta + its own index
pub struct CollectionEntry<'a> {
    pub meta: Arc<RwLock<CollectionMeta>>,
    index: Arc<RwLock<MetricIndex<'a>>>,
}

/* VectorDB */
pub struct VectorDB<'a> {
    db: Arc<DB>,
    collections: RwLock<HashMap<String, Arc<CollectionEntry<'a>>>>,
    id_counter: std::sync::atomic::AtomicU64,
}

pub type ShardDB<'a> = Arc<VectorDB<'a>>;


/* Implementations */
impl<'a> VectorDB<'a> {
    pub fn new(path: &str) -> Self  {
        // Optimize for the workload
        let mut opts = Options::default();
        
        opts.create_if_missing(true);
        opts.set_max_open_files(10000);
        opts.set_use_fsync(false);
        opts.set_bytes_per_sync(8388608);
        opts.set_write_buffer_size(128 * 1024 * 1024); // 128MB
        opts.set_max_write_buffer_number(4);
        opts.set_target_file_size_base(64 * 1024 * 1024); // 64MB
        opts.increase_parallelism(num_cpus::get() as i32);
        opts.optimize_level_style_compaction(512 * 1024 * 1024); // 512MB

        // Enable Bloom filters for faster lookups
        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_bloom_filter(10.0, false);
        block_opts.set_block_size(64 * 1024); // 64KB blocks
        opts.set_block_based_table_factory(&block_opts);

        let db = Arc::new(DB::open(&opts, path).expect("rocksdb open failed"));

        Self {
            db,
            collections: RwLock::new(HashMap::new()),
            id_counter: std::sync::atomic::AtomicU64::new(1),
        }
    }

    fn generate_id(&self) -> u64 {
        self.id_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    fn optimize_hnsw_params(&self, dim: usize) -> (usize, usize, usize) {
        // Memory consumption -> (d * 4 + M * 2 * 4) bytes per vector
        // Optimize M based on dimensionality and memory constraints
        let m = if dim > 768 { 8 } else if dim > 384 { 12 } else { 16 };
        let max_m0 = m * 2;
        let ef_construction = if dim > 768 { 100 } else { 150 };
        (m, max_m0, ef_construction)
    }

    // Create a new Collection
    pub fn create_collection(
        &self, 
        name: &str,
        metric: DistanceMetric,
        dim: usize
    ) -> Result<CollectionMeta, String> {
        if self.collections.read().unwrap().contains_key(name) {
            return Err("duplicate".into());
        }

        let id = self.generate_id();
        let meta = CollectionMeta {
            id, 
            name: name.to_string(),
            dim,
            metric: metric.clone(),
            doc_count: 0
        };

        // Optimize HNSW parameters based on dimensionality
        let (m, max_m0, ef_construction) = self.optimize_hnsw_params(dim);

        // Build HNSW index for the collection
        let hnsw = match metric {
            DistanceMetric::Cosine => {
                MetricIndex::Cosine(Hnsw::<f32, DistCosine>::new(
                    m, 100_000, max_m0, ef_construction, DistCosine {}
                ))
            }
            DistanceMetric::Dot => {
                MetricIndex::Dot(Hnsw::<f32, DistDot>::new(
                    m, 100_000, max_m0, ef_construction, DistDot {}
                ))
            }
            DistanceMetric::Euclidean => {
                MetricIndex::Euclidean(Hnsw::<f32, DistL2>::new(
                    m, 100_000, max_m0, ef_construction, DistL2 {}
                ))
            }
        };

        let entry = Arc::new(CollectionEntry { 
            meta: Arc::new(RwLock::new(meta.clone())), 
            index: Arc::new(RwLock::new(hnsw))
        });
        self.collections.write().unwrap().insert(name.to_string(), entry);

        // Persist the meta
        self.db
            .put(format!("col:{}", name), serde_json::to_vec(&meta).unwrap())
            .map_err(|e| e.to_string())?;

        Ok(meta)
    }

    // Get collection by name
    pub fn get_collection_by_name(&self, name: &str) -> Result<CollectionMeta, String> {
        self.collections
            .read()
            .unwrap()
            .get(name)
            .cloned()
            .map(|e| e.meta.read().unwrap().clone())
            .ok_or_else(|| "Collection not found".into())
    }

    pub fn list_collections(&self) -> Vec<CollectionMeta> {
        self.collections
            .read()
            .unwrap()
            .values()
            .map(|e| e.meta.read().unwrap().clone())
            .collect()
    }

    // Add a single Document
    pub fn add_document(
        &self,
        col_name: &str,
        id: Option<u64>,
        embedding: Vec<f32>,
        metadata: String,
        content: String,
    ) -> Result<u64, String> {
        let entry = self
            .collections
            .read()
            .unwrap()
            .get(col_name)
            .cloned()
            .ok_or_else(|| "Collection not found".to_string())?;

        let meta = entry.meta.read().unwrap();
        if embedding.len() != meta.dim {
            return Err("Embedding dimension mismatch".into());
        }
        drop(meta);
        let doc_id = id.unwrap_or_else(|| self.generate_id());
        let key_prefix = format!("{}:{}", entry.meta.read().unwrap().id, doc_id);

        // Serialization
        let optimized_emb = Embedding { data: embedding.clone() };
        let serialized = rkyv::to_bytes::<_, 256>(&optimized_emb)
            .map_err(|e| format!("Serialization error: {}", e))?;

        // Batch write operations
        let mut batch = WriteBatch::default();
        batch.put(format!("vec:{}", key_prefix), &serialized);
        batch.put(format!("meta:{}", key_prefix), metadata.as_bytes());
        batch.put(format!("content:{}", key_prefix), content.as_bytes());

        self.db.write(batch).map_err(|e| e.to_string())?;

        // Update index
        entry.index.write().unwrap().insert(doc_id as usize, &embedding);
        
        // Update document count
        entry.meta.write().unwrap().doc_count += 1;
        
        Ok(doc_id)
    }


    // Add documents in a batch
    pub fn add_documents(
        &self, 
        col_name: &str,
        docs: Vec<AddDocumentRequest>,
    ) -> Result<Vec<u64>, Vec<String>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }

        // Batch operations to reduce lock contention
        let _entry = self
            .collections
            .read()
            .unwrap()
            .get(col_name)
            .cloned()
            .ok_or_else(|| vec!["Collection not found".to_string()])?;

        // Pre-allocate with capacity
        let mut ok_ids = Vec::with_capacity(docs.len());
        let mut errs = Vec::with_capacity(docs.len() / 10);

        // Process in parallel chunks for better performance
        let chunk_size = std::cmp::max(docs.len() / num_cpus::get(), 1);
        let results: Vec<_> = docs
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut batch_ids = Vec::new();
                let mut batch_errs = Vec::new();
                
                for d in chunk {
                    let emb: Vec<f32> = d.embedding.iter().map(|v| *v as f32).collect();
                    match self.add_document(
                        col_name,
                        d.id.map(|x| x as u64),
                        emb,
                        d.metadata.clone(),
                        d.content.clone(),
                    ) {
                        Ok(id) => batch_ids.push(id),
                        Err(e) => batch_errs.push(format!("Doc: {:?}: {}", d.id, e)),
                    }
                }
                (batch_ids, batch_errs)
            })
            .collect();

        // Flatten results
        for (batch_ids, batch_errs) in results {
            ok_ids.extend(batch_ids);
            errs.extend(batch_errs);
        }

        if errs.is_empty() {
            Ok(ok_ids)
        } else {
            Err(errs)
        }
    }

    // Similarity Search
    pub fn search(
        &self, 
        col_name: &str,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(u64, f32, String, String)>, String> {
        let entry = self
            .collections
            .read()
            .unwrap()
            .get(col_name)
            .cloned()
            .ok_or_else(|| "Collection not found".to_string())?;
    
        let meta = entry.meta.read().unwrap();
        if query.len() != meta.dim {
            return Err("Query dimension mismatch".into());
        }
        let collection_id = meta.id;
        drop(meta);

        // Search the index
        let hits = entry.index.read().unwrap().search(query, top_k);
        let mut out = Vec::with_capacity(hits.len());

        for n in hits {
            let id = n.d_id as u64;
            let key_prefix = format!("{}:{}", collection_id, id);
            
            let meta = self
                .db
                .get(format!("meta:{}", key_prefix))
                .unwrap()
                .map(|v| String::from_utf8_lossy(&v).into_owned())
                .unwrap_or_default();

            let content = self
                .db
                .get(format!("content:{}", key_prefix))
                .unwrap()
                .map(|v| String::from_utf8_lossy(&v).into_owned())
                .unwrap_or_default();

            out.push((id, n.distance, meta, content));
        }
        Ok(out)
    }

    pub fn get_memory_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();

        // Get collection count
        let collections = self.collections.read().unwrap();
        stats.insert("collections_count".to_string(), collections.len());
        
        // Estimate memory usage
        let mut total_docs = 0;
        for entry in collections.values() {
            total_docs += entry.meta.read().unwrap().doc_count;
        }
        stats.insert("total_documents".to_string(), total_docs as usize);
        
        stats
    }

}