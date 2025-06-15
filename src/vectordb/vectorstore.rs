use ndarray::Array1;
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use sqlx::Row;
use std::sync::{Arc, Mutex};
use std::vec;

use crate::AddDocumentRequest;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Collection {
    pub id: i32,
    pub name: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: i32,
    pub embedding: Vec<f64>, // Vec<f64> for better serialization
    pub metadata: String,
    pub content: String,
    pub collection_id: i32,
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

        // Create collections table if it doesn't exist
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            );
            "#,
        )
        .execute(&pool)
        .await?;

        // Create the documents table if it doesn't exist
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                embedding TEXT NOT NULL,
                metadata TEXT NOT NULL,
                content TEXT NOT NULL,
                collection_id INTEGER NOT NULL,
                FOREIGN KEY (collection_id) REFERENCES collections(id)
            );
            "#,
        )
        .execute(&pool)
        .await?;

        Ok(VectorDB { pool })
    }  

    /// Creates a new collection
    pub async fn create_collection(&self, name: &str) -> Result<Collection, String> {
        // Insert the new collection
        let result = sqlx::query(
            r#"
            INSERT INTO collections (name)
            VALUES (?)
            "#,
        )
        .bind(name)
        .execute(&self.pool)
        .await;

        match result {
            Ok(res) => {
                let collection_id = res.last_insert_rowid() as i32;
                Ok(Collection {
                    id: collection_id,
                    name: name.to_string(),
                })
            }
            Err(e) => {
                if let sqlx::Error::Database(db_err) = &e {
                    if db_err.message().contains("UNIQUE constraint failed") {
                        return Err("Collection already exists".to_string());
                    }
                }
                Err(e.to_string())
            }
        }
    }

    /// Retrieves a collection by name.
    pub async fn get_collection_by_name(&self, name: &str) -> Result<Collection, String> {
        let row = sqlx::query(
            r#"
            SELECT id, name FROM collections
            WHERE name = ?
            "#,
        )
        .bind(name)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| e.to_string())?;

        Ok(Collection {
            id: row.try_get("id").unwrap_or(0),
            name: row.try_get("name").unwrap_or_default(),
        })
    }

    /// Check if a collection exists
    #[allow(dead_code)] // Suppress warning if not used
    pub async fn collection_exists(&self, name: &str) -> bool {
        let result = sqlx::query_scalar::<_, i64>(
            r#"
            SELECT COUNT(*) FROM collections
            WHERE name = ?
            "#,
        )
        .bind(name)
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);

        result > 0
    }

    /// Adds a new document to the database.
    pub async fn add_document(
        &self, 
        id: Option<i32>, 
        vector: Vec<f64>, 
        metadata: String, 
        content: String,
        collection_id: i32,
    ) -> Result<i32, String> {
        // Serialize the embedding vector to JSON string
        let embedding_json = serde_json::to_string(&vector).map_err(|e| e.to_string())?;

        let query = if let Some(id_val) = id {
            sqlx::query(
                r#"
                INSERT INTO documents (id, embedding, metadata, content, collection_id)
                VALUES (?, ?, ?, ?, ?)
                "#,
            )
            .bind(id_val)
            .bind(embedding_json)
            .bind(metadata)
            .bind(content)
            .bind(collection_id)
        } else {
            sqlx::query(
                r#"
                INSERT INTO documents (embedding, metadata, content, collection_id)
                VALUES (?, ?, ?, ?)
                "#,
            )
            .bind(embedding_json)
            .bind(metadata)
            .bind(content)
            .bind(collection_id)
        };

        let result = query.execute(&self.pool).await.map_err(|e| e.to_string())?;
        let inserted_id  = result.last_insert_rowid() as i32;
        
        Ok(inserted_id)
    }

    /// Batch adds multiple documents to the database within a collection
    pub async fn add_documents(
        &self,
        documents: Vec<AddDocumentRequest>,
        collection_id: i32
    ) -> Result<Vec<i32>, Vec<String>> {
        let mut errors = Vec::new();
        let mut inserted_ids = Vec::new();


        for doc in documents {
            match self
            .add_document(doc.id, doc.embedding, doc.metadata, doc.content, collection_id)
            .await
            {
                Ok(id) => inserted_ids.push(id),
                Err(e) => errors.push(format!(
                    "Failed to add document ID {}: {}",
                    doc.id.map(|id| id.to_string()).unwrap_or_else(|| "auto".to_string()),
                    e,
                )),
            }
        }

        if errors.is_empty() {
            Ok(inserted_ids)
        } else {
            Err(errors)
        }
    }

    /// Finds the top N nearest neighbors to a given query vector
    pub async fn search(
        &self,
        query: &[f64],
        n: usize, // top n 
        metric: DistanceMetric,
        collection_name: Option<&str>,
    ) -> Vec<(i32, f64, String, String)> {
        // Sql for optional metadata filtering
        let mut query_builder = String::from("SELECT id, embedding, metadata, content FROM documents");
        if let Some(_filter) = collection_name {
            query_builder.push_str(" WHERE collection_id = (SELECT id FROM collections WHERE name = ?)");
        }

        // Execute
        let rows = if let Some(filter) = collection_name {
            sqlx::query(&query_builder)
                .bind(filter)
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
                Err(_) => {
                    log::warn!("Skipping document {} due to invalid embedding JSON", id);
                    continue;
                }
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

        // Sort by distance based on metric
        match metric {
            DistanceMetric::DotProduct => {
                distances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
            }
            _ => {
                distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        distances.truncate(n);
        distances
    }
}

/// Type alias for thread-safe, shared VectorDB
pub type ShardDB = Arc<Mutex<VectorDB>>;