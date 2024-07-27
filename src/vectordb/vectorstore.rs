use ndarray::Array1;
use std::collections::HashMap;

pub type Vector = Array1<f64>;

pub struct VectorDB {
    vectors: HashMap<i32, Vector>,
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            vectors: HashMap::new(),
        }
    }

    pub fn add_vector(&mut self, id: i32, vector: Vector) {
        self.vectors.insert(id, vector);
    }
    
    // Find the nearest neighbor to a given vector
    pub fn find_nearest(&self, query: &Vector) -> Option<(i32, f64)> {
        self.vectors.iter().fold(None, |acc, (&id, vec)| {
            let distance = (query - vec).mapv(|a| a.powi(2)).sum().sqrt();
            match acc {
                None => Some((id, distance)),
                Some((_, d)) if distance < d => Some((id, distance)),
                _ => acc,
            }
        }) 
    }
}