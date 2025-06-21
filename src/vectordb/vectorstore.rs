use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use rocksdb::DB;
use serde::{Deserialize, Serialize};
use bincode;
use hnsw_rs::hnsw::{Hnsw, Neighbour};
use hnsw_rs::anndists::dist::{DistCosine, DistDot, DistL2};

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
        let ef_search = 2 * top_k;
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
    pub metric: DistanceMetric
}

// Holds meta + its own index
pub struct CollectionEntry<'a> {
    pub meta: CollectionMeta,
    index: RwLock<MetricIndex<'a>>,
}

/* VectorDB */
pub struct VectorDB<'a> {
    db: Arc<DB>,
    collections: RwLock<HashMap<String, Arc<CollectionEntry<'a>>>>,
}

pub type ShardDB<'a> = Arc<VectorDB<'a>>;


/* Implementations */
impl<'a> VectorDB<'a> {
    pub fn new(path: &str) -> Self  {
        let db = Arc::new(DB::open_default(path).expect("rocksdb open failed"));
        Self {
            db,
            collections: RwLock::new(HashMap::new()),
        }
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

        let id = rand::random::<u64>();
        let meta = CollectionMeta {
            id, 
            name: name.to_string(),
            dim,
            metric: metric.clone(),
        };

        // Build HNSW index for the collection
        let hnsw = match metric {
            DistanceMetric::Cosine => {
                MetricIndex::Cosine(Hnsw::<f32, DistCosine>::new(16, 100_000, 10, 200, DistCosine {}))
            }
            DistanceMetric::Dot => {
                MetricIndex::Dot(Hnsw::<f32, DistDot>::new(16, 100_000, 10, 200, DistDot {}))
            }
            DistanceMetric::Euclidean => {
                MetricIndex::Euclidean(Hnsw::<f32, DistL2>::new(16, 100_000, 10, 200, DistL2 {}))
            }
        };

        let entry = Arc::new(CollectionEntry { meta: meta.clone(), index: RwLock::new(hnsw) });
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
            .map(|e| e.meta.clone())
            .ok_or_else(|| "Collection not found".into())
    }

    pub fn list_collections(&self) -> Vec<CollectionMeta> {
        self.collections
            .read()
            .unwrap()
            .values()
            .map(|e| e.meta.clone())
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
            .ok_or_else(|| "Collection not found".to_lowercase())?;

        if embedding.len() != entry.meta.dim {
            return Err("Embedding dimension mismatch".into());
        }

        let doc_id = id.unwrap_or(rand::random::<u64>());
        let key_prefix = format!("{}:{}", entry.meta.id, doc_id);

        self.db
            .put(
                format!("vec:{}", key_prefix),
                bincode::serialize(&embedding).unwrap()
            )
            .and_then(|_| self.db.put(format!("meta:{}", key_prefix), metadata.as_bytes()))
            .and_then(|_| self.db.put(format!("content:{}", key_prefix), content.as_bytes()))
            .map_err(|e| e.to_string())?;
            
        entry.index.write().unwrap().insert(doc_id as usize, &embedding);
        Ok(doc_id)
    }


    // Add documents in a batch
    pub fn add_documents(
        &self, 
        col_name: &str,
        docs: Vec<AddDocumentRequest>,
    ) -> Result<Vec<u64>, Vec<String>> {
        let mut ok_ids = Vec::new();
        let mut errs = Vec::new();

        for d in docs {
            let emb: Vec<f32> = d.embedding.iter().map(|v| *v as f32).collect();
            match self.add_document(
                col_name,
                d.id.map(|x| x as u64),
                emb,
                d.metadata.clone(),
                d.content.clone(),
            ) {
                Ok(id) => ok_ids.push(id),
                Err(e) => errs.push(format!("Doc: {:?}: {e}", d.id)),
            }
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
    
        if query.len() != entry.meta.dim {
            return Err("Query dimension mismatch".into());
        }

        let hits = entry.index.read().unwrap().search(query, top_k);
        let mut out = Vec::new();
        
        for n in hits {
            let id = n.d_id as u64;
            let key_prefix = format!("{}:{}", entry.meta.id, id);
            let meta = self
                .db
                .get(format!("meta:{}", key_prefix))
                .unwrap()
                .map(|v| String::from_utf8_lossy(&v).into())
                .unwrap_or_default();

            let content = self
                .db
                .get(format!("content:{}", key_prefix))
                .unwrap()
                .map(|v| String::from_utf8_lossy(&v).into())
                .unwrap_or_default();

            out.push((id, n.distance, meta, content));
        }
        Ok(out)
    }

}