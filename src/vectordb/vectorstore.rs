use std::sync::{Arc, RwLock};
use rocksdb::DB;
use serde::{Deserialize, Serialize};
use bincode;
use hnsw_rs::hnsw::{Hnsw, Neighbour};
use hnsw_rs::anndists::dist::{DistCosine, DistDot, DistL2};

use crate::AddDocumentRequest;


#[derive(Clone, Serialize, Deserialize)]
pub struct Collection {
    pub id: u64,
    pub name: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub enum DistanceMetric {
    Euclidean, 
    Cosine,
    DotProduct,
}

impl DistanceMetric {
    pub fn from_str(metric: &str) -> Option<Self> {
        match metric.to_lowercase().as_str() {
            "euclidean" => Some(DistanceMetric::Euclidean),
            "cosine" => Some(DistanceMetric::Cosine),
            "dot" => Some(DistanceMetric::DotProduct),
            _ => None,
        }
    }
}

enum MetricIndex<'a> {
    Cosine(Hnsw<'a, f32, DistCosine>),
    Dot(Hnsw<'a, f32, DistDot>),
    Euclidean(Hnsw<'a, f32, DistL2>),
}

impl<'a> MetricIndex<'a> {
    fn insert(&mut self, id: usize, emb: &[f32]) {
        match self {
            MetricIndex::Cosine(h) => h.insert((emb, id)),
            MetricIndex::Dot(h) => h.insert((emb, id)),
            MetricIndex::Euclidean(h) => h.insert((emb, id)),
        }
    }
    
    fn search(&self, query: &[f32], top_k: usize) -> Vec<Neighbour> {
        let ef_search = 2 * top_k;
        match self {
            MetricIndex::Cosine(h) => h.search(query, top_k, ef_search),
            MetricIndex::Dot(h) => h.search(query, top_k, ef_search),
            MetricIndex::Euclidean(h) => h.search(query, top_k, ef_search)
        }
    }
}

#[derive(Clone)]
pub struct VectorDB <'a> {
    pub db: Arc<DB>,
    hnsw: Arc<RwLock<MetricIndex<'a>>>,
    dimension: usize,
}

pub type ShardDB<'a> = Arc<RwLock<VectorDB<'a>>>;

impl<'a> VectorDB<'a> {
    pub fn new(path: &str, dimension: usize, metric: DistanceMetric) -> Self {
        let db = Arc::new(DB::open_default(path).expect("rocksdb open failed"));
        let hnsw = match metric {
            DistanceMetric::Cosine => MetricIndex::Cosine(Hnsw::<f32, DistCosine>::new(16, 100_000, 10, 200, DistCosine {})),
            DistanceMetric::DotProduct => MetricIndex::Dot(Hnsw::<f32, DistDot>::new(16, 100_000, 10, 200, DistDot {})),
            DistanceMetric::Euclidean => MetricIndex::Euclidean(Hnsw::<f32, DistL2>::new(16, 100_000, 10, 200, DistL2 {})),
        };
        Self {
            db,
            hnsw: Arc::new(RwLock::new(hnsw)),
            dimension,
        }
    }

    // Create a new collection
    pub fn create_collection(&self, name: &str) -> Result<Collection, String> {
        let id = rand::random::<u64>();
        let collection = Collection {
            id,
            name: name.to_string(),
        };
        let value = serde_json::to_vec(&collection).map_err(|e| e.to_string())?;
        self.db.put(format!("col:{}", name), value).map_err(|e| e.to_string())?;
        Ok(collection)
    }

    // get collection by name
    pub fn get_collection_by_name(&self, name: &str) -> Result<Collection, String> {
        match self.db.get(format!("col:{}", name)) {
            Ok(Some(bytes)) => {
                let col: Collection = serde_json::from_slice(&bytes).map_err(|e|e.to_string())?;
                Ok(col)
            }
            Ok(None) => Err("Collection not found".into()),
            Err(e) => Err(e.to_string()),
        }
    }   

    // List Collections
    pub fn list_collections(&self) -> Result<Vec<Collection>, String> {
        let mut collections = Vec::new();
        let prefix = b"col:";

        let iter = self.db.iterator(rocksdb::IteratorMode::Start);

        for item in iter {
            let (k, v) = item.map_err(|e| e.to_string())?;
            if k.starts_with(prefix) {
                let collection: Collection = serde_json::from_slice(&v).map_err(|e| e.to_string())?;
                collections.push(collection);
            }
        }

        Ok(collections)
    }


    // Add a single document
    pub fn add_document(
        &self, 
        id: Option<u64>,
        embedding: Vec<f32>,
        metadata: String,
        content: String,
        _collection_id: u64,
    ) -> Result<u64, String> {
        if embedding.len() != self.dimension {
            return Err("Embedding dimension mismatch".to_string());
        }

        let doc_id = id.unwrap_or(rand::random::<u64>());
        self.db.put(format!("vec:{}", doc_id), bincode::serialize(&embedding).unwrap()).map_err(|e| e.to_string())?;
        self.db.put(format!("meta:{}", doc_id), metadata.as_bytes()).map_err(|e| e.to_string())?;
        self.db.put(format!("content:{}", doc_id), content.as_bytes()).map_err(|e| e.to_string())?;

        self.hnsw.write().unwrap().insert(doc_id as usize, &embedding);
        Ok(doc_id)
    }

    // Add docs in batch
    pub fn add_documents(
        &self, 
        docs: Vec<AddDocumentRequest>,
        collection_id: u64
    ) -> Result<Vec<u64>, Vec<String>> {
        let mut inserted_ids = Vec::new();
        let mut errors = Vec::new();

        for doc in docs {
            let embedding: Vec<f32> = doc.embedding.iter().map(|v| *v as f32).collect();
            match self.add_document(
                doc.id.map(|id| id as u64),
                embedding,
                doc.metadata.clone(),
                doc.content.clone(),
                collection_id,
            ) {
                Ok(id) => inserted_ids.push(id),
                Err(e) => errors.push(format!(
                    "Doc ID {:?} failed: {}",
                    doc.id.unwrap_or(-1),
                    e 
                )),
            }
        }
    
        if errors.is_empty() {
            Ok(inserted_ids)
        } else {
            Err(errors)
        }
    }

    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        _collection_name: Option<&str>,
    ) -> Vec<(u64, f32, String, String)> {
        let results = self.hnsw.read().unwrap().search(query, top_k);
        results.into_iter().map(|n| {
            let id = n.d_id as u64;
            let dist = n.distance;
            let meta = self.db.get(format!("meta:{}", id)).unwrap().map(|v| String::from_utf8_lossy(&v).to_string()).unwrap_or_default();
            let content = self.db.get(format!("content:{}", id)).unwrap().map(|v| String::from_utf8_lossy(&v).to_string()).unwrap_or_default();
            (id, dist, meta, content)
        }).collect()
    }
}
