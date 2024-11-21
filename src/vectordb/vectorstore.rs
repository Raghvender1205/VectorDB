use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub type Vector = Array1<f64>;

#[derive(Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: i32,
    pub embedding: Vec<f64>, // Vec<f64> for better serialization
    pub metadata: String,
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


#[derive(Clone)]
pub struct VectorDB {
    documents: HashMap<i32, Document>,
}


impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            documents: HashMap::new(),
        }
    }

    pub fn add_document(&mut self, id: i32, vector: Vec<f64>, metadata: String) -> Result<(), String> {
        if self.documents.contains_key(&id) {
            return Err(format!("Document with id {} already exists.", id));
        }
        let document = Document {
            id, 
            embedding: vector,
            metadata,
        };
        self.documents.insert(id, document);
        Ok(())
    }

    // Find the top N nearest neighbors to a given vector
    pub fn find_nearest(
        &self,
        query: &[f64],
        n: usize,
        metric: DistanceMetric,
        metadata_filter: Option<&str>
    ) -> Vec<(i32, f64, String)> {
        let query_vec = Array1::from(query.to_vec());

        // Filter documents based on metadata filter if provided
        let filtered_documents: Vec<&Document> = self.documents.values()
            .filter(|doc| {
                if let Some(filter) = metadata_filter {
                    doc.metadata.contains(filter)
                } else {
                    true
                }
            })
            .collect();


        // Calculate distances for the filtered documents
        let mut distances: Vec<(i32, f64, String)> = filtered_documents.iter().filter_map(|doc| {
            if doc.embedding.len() != query.len() {
                // Skip documents with mismatched dimensions
                return None;
            }
            let doc_vec = Array1::from(doc.embedding.clone());
            let distance = match metric {
                DistanceMetric::Euclidean => (&query_vec - &doc_vec).mapv(|a| a.powi(2)).sum().sqrt(),
                DistanceMetric::Cosine => {
                    let dot_product = query_vec.dot(&doc_vec);
                    let query_norm = query_vec.mapv(|a| a.powi(2)).sum().sqrt();
                    let doc_norm = doc_vec.mapv(|a| a.powi(2)).sum().sqrt();
                    if query_norm == 0.0 || doc_norm == 0.0 {
                        // Avoid division by zero
                        1.0
                    } else {
                        1.0 - (dot_product / (query_norm * doc_norm))
                    }
                }
                DistanceMetric::DotProduct => query_vec.dot(&doc_vec),
            };
            Some((doc.id, distance, doc.metadata.clone()))
        }).collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(n);
        distances
    }
}

pub type ShardDB = Arc<Mutex<VectorDB>>;
