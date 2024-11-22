use ndarray::Array1;
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use sqlx::Row;
use std::sync::{Arc, Mutex};
use std::vec;

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
    pub pool: SqlitePool,
}


impl VectorDB {
    pub async fn new(database_url: &str) -> Result<Self, sqlx::Error> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await?;

        // Create the documents table if it doesn't exist
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                embedding TEXT NOT NULL,
                metadata TEXT NOT NULL,
                content TEXT NOT NULL
            );
            "#,
        )
        .execute(&pool)
        .await?;

        Ok(VectorDB { pool })
    }  

    /// Adds a new document to the database.
    pub async fn add_document(&self, id: i32, vector: Vec<f64>, metadata: String, content: String) -> Result<(), String> {
        // Serialize the embedding vector to JSON string
        let embedding_json = serde_json::to_string(&vector).map_err(|e| e.to_string())?;

        // Insert to db
        let result = sqlx::query(
            r#"
            INSERT INTO documents (id, embedding, metadata, content)
            VALUES (?, ?, ?, ?)
            "#,
        )
        .bind(id)
        .bind(embedding_json)
        .bind(metadata)
        .bind(content)
        .execute(&self.pool)
        .await;

        match result {
            Ok(_) => Ok(()),
            Err(e) => Err(e.to_string()),
        }
    }

    /// Finds the top N nearest neighbors to a given query vector
    pub async fn search(
        &self,
        query: &[f64],
        n: usize, // top n 
        metric: DistanceMetric,
        metadata_filter: Option<&str>,
    ) -> Vec<(i32, f64, String, String)> {
        // Sql for optional metadata filtering
        let mut query_builder = String::from("SELECT id, embedding, metadata, content FROM documents");
        if let Some(_filter) = metadata_filter {
            query_builder.push_str(" WHERE metadata LIKE ?"); // TODO: Test it!!
        }

        // Execute
        let rows = if let Some(filter) = metadata_filter {
            sqlx::query(&query_builder)
                .bind(format!("%{}%", filter))
                .fetch_all(&self.pool)
                .await
                .unwrap_or_else(|_| vec![])
        } else {
            sqlx::query(&query_builder)
                .fetch_all(&self.pool)
                .await
                .unwrap_or_else(|_| vec![])
        };

        let query_vec = Array1::from(query.to_vec());

        // Calculate distance
        let mut distances = Vec::new();

        for row in rows {
            let id: i32 = row.try_get("id").unwrap_or(0);
            let embedding_str: String = row.try_get("embedding").unwrap_or_default();
            let metadata: String = row.try_get("metadata").unwrap_or_default();
            let content: String = row.try_get("content").unwrap_or_default();

            // Deserialize the embedding
            let embedding: Vec<f64> = match serde_json::from_str(&embedding_str) {
                Ok(vec) => vec,
                Err(_) => continue, // Skip malformed embeddings
            };

            if embedding.len() != query.len() {
                continue; // Skip if dimensions mismatch
            }

            let doc_vec = Array1::from(embedding.clone());
            let distance = match metric {
                DistanceMetric::Euclidean => (&query_vec - &doc_vec).mapv(|a| a.powi(2)).sum().sqrt(),
                DistanceMetric::Cosine => {
                    let dot_product = query_vec.dot(&doc_vec);
                    let query_norm = query_vec.mapv(|a| a.powi(2)).sum().sqrt();
                    let doc_norm = doc_vec.mapv(|a| a.powi(2)).sum().sqrt();
                    if query_norm == 0.0 || doc_norm == 0.0 {
                        1.0
                    } else {
                        1.0 - (dot_product / (query_norm * doc_norm))
                    }
                }
                DistanceMetric::DotProduct => query_vec.dot(&doc_vec),
            };

            distances.push((id, distance, metadata, content));
        }

        // Sort by distance and return top N
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(n);
        distances
    }
}

/// Type alias for thread-safe, shared VectorDB
pub type ShardDB = Arc<Mutex<VectorDB>>;