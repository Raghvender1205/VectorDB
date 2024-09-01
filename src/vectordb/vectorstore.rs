use ndarray::Array1;
use std::collections::HashMap;

pub type Vector = Array1<f64>;

pub struct Document {
    id: i32,
    embedding: Vector,
    metadata: String,
}

pub enum DistanceMetric {
    Euclidean,
    Cosine,
    DotProduct,
}

pub struct VectorDB {
    documents: HashMap<i32, Document>,
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            documents: HashMap::new(),
        }
    }

    pub fn add_document(&mut self, id: i32, vector: Vector, metadata: String) {
        let document = Document {
            id, 
            embedding: vector,
            metadata,
        };
        self.documents.insert(id, document);
    }

    // Find the top N nearest neighbors to a given vector
    pub fn find_nearest(&self, query: &Vector, n: usize, metric: DistanceMetric) -> Vec<(i32, f64, String)> {
        let mut distances: Vec<(i32, f64, String)> = self.documents.values().map(|doc| {
            let distance = match metric {
                DistanceMetric::Euclidean => (query - &doc.embedding).mapv(|a| a.powi(2)).sum().sqrt(),
                DistanceMetric::Cosine => {
                    let dot_product = query.dot(&doc.embedding);
                    let query_norm = query.mapv(|a| a.powi(2)).sum().sqrt();
                    let doc_norm = doc.embedding.mapv(|a| a.powi(2)).sum().sqrt();
                    1.0 - (dot_product / (query_norm * doc_norm))
                }
                DistanceMetric::DotProduct => query.dot(&doc.embedding)
            };
            (doc.id, distance, doc.metadata.clone())
        }).collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(n);
        distances
    }
}

