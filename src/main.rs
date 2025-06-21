use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use actix_web::http::StatusCode;
use actix_web::middleware::Logger;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use env_logger::Env;
use vectordb::vectorstore::{VectorDB, ShardDB, DistanceMetric};

mod vectordb;


#[derive(Deserialize, Clone)]
pub struct CreateCollectionRequest {
    pub name: String,
}

#[derive(Deserialize, Clone)]
pub struct AddDocumentRequest {
    pub id: Option<i32>,
    pub embedding: Vec<f64>,
    pub metadata: String,
    pub content: String,
    pub collection_name: String, 
}

#[derive(Serialize)]
struct AddDocumentResponse {
    id: i32,
    status: String,
}

#[derive(Deserialize)]
struct AddDocumentsRequest {
    documents: Vec<AddDocumentRequest>,
}

#[derive(Serialize)]
struct AddDocumentsResponse {
    documents: Vec<AddDocumentResponse>
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: Vec<f64>,
    pub n: usize,
    pub metric: String, // "Euclidean", "Cosine", "Dot"
    pub collection_name: String
}

#[derive(Serialize)]
pub struct NearestNeighbor {
    pub id: u64,
    pub distance: f32,
    pub metadata: String,
    pub content: String,
}

// Healthcheck
async fn health_check() -> impl Responder {
    HttpResponse::Ok().body("pong")
}


/// Handler to create a new collection
async fn create_collection(
    db: web::Data<ShardDB<'static>>,
    item: web::Json<CreateCollectionRequest>
) -> impl Responder {
    let db = db.write().unwrap();
    match db.create_collection(&item.name) {
        Ok(collection) => HttpResponse::Ok().json(collection),
        Err(err) => HttpResponse::BadRequest().body(err),
    }
}

// Handler to get collection by name
async fn get_collection_by_name(
    db: web::Data<ShardDB<'static>>,
    path: web::Path<String>,
) -> impl Responder {
    let db = db.write().unwrap();

    match db.get_collection_by_name(&path.into_inner()) {
        Ok(collection) => HttpResponse::Ok().json(collection),
        Err(_) => HttpResponse::NotFound().body("Collection not found"),
    }
}

/// Handler to list all collections
async fn list_collections(
    db: web::Data<ShardDB<'static>>,
) -> impl Responder {
    let db = db.read().unwrap(); // read lock
    match db.list_collections() {
        Ok(collections) => HttpResponse::Ok().json(collections),
        Err(e) => HttpResponse::InternalServerError().body(e),
    }
}

/// Handler to add a single document to a collection
async fn add_document(
    db: web::Data<ShardDB<'static>>,
    item: web::Json<AddDocumentRequest>,
) -> impl Responder {
    let db = db.write().unwrap();
    let collection = match db.get_collection_by_name(&item.collection_name) {
        Ok(c)  => c,
        Err(e) => return HttpResponse::BadRequest().body(format!("Collection not found: {e}")),
    };

    let embedding: Vec<f32> = item.embedding.iter().map(|&v| v as f32).collect();
    match db.add_document(
        item.id.map(|v| v as u64),                
        embedding,
        item.metadata.clone(),
        item.content.clone(),
        collection.id,
    ) {
        Ok(id) => HttpResponse::Ok().json(AddDocumentResponse {
            id: id as i32,
            status: "success".into(),
        }),
        Err(e) => HttpResponse::BadRequest().body(e),
    }
}

/// Handler to add multiple documents to a collection
async fn add_documents(
    db: web::Data<ShardDB<'static>>,
    item: web::Json<AddDocumentsRequest>,
) -> impl Responder {
    let db = db.write().unwrap();
    let collection_name = &item.documents[0].collection_name;
    let collection = match db.get_collection_by_name(collection_name) {
        Ok(c) => c,
        Err(_) => return HttpResponse::BadRequest().body("Collection not found"),
    };

    match db.add_documents(item.documents.clone(), collection.id) {
        Ok(ids) => {
            let docs = ids
                .into_iter()
                .map(|id| AddDocumentResponse {
                    id: id as i32,
                    status: "success".into(),
                })
                .collect();
            HttpResponse::Ok().json(AddDocumentsResponse { documents: docs })
        }
        Err(errors) => HttpResponse::build(StatusCode::MULTI_STATUS).json(json!({ "errors": errors })),
    }
}


/// Handler to search for relevant documents within a collection
async fn retrieve_documents(
    db: web::Data<ShardDB<'static>>,
    item: web::Json<SearchRequest>,
) -> impl Responder {
    let db = db.read().unwrap();
    let collection = match db.get_collection_by_name(&item.collection_name) {
        Ok(c)  => c,
        Err(_) => return HttpResponse::BadRequest().body("Collection not found"),
    };

    let query: Vec<f32> = item.query.iter().map(|&v| v as f32).collect();
    // Search signature is (&self, query, top_k, _collection_name)
    let hits = db.search(&query, item.n, Some(&collection.name));

    let resp: Vec<NearestNeighbor> = hits
        .into_iter()
        .map(|(id, dist, meta, content)| NearestNeighbor {
            id,
            distance: dist,
            metadata: meta,
            content,
        })
        .collect();

    HttpResponse::Ok().json(resp)
}


/// Ensures that a directory exists, creating it if necessary
fn ensure_directory(path: &PathBuf) -> std::io::Result<()> {
    if !path.exists() {
        fs::create_dir_all(path)?;
    }
    Ok(())
}


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let db_path = PathBuf::from("data/rocksdb");
    ensure_directory(&db_path)?;

    // Decide the distance metric once, when db is created 
    let metric = env::var("VECTOR_METRIC")
        .ok()
        .and_then(|m| DistanceMetric::from_str(&m))
        .unwrap_or(DistanceMetric::Cosine);

    let db: ShardDB<'static> = Arc::new(RwLock::new(VectorDB::new(
        db_path.to_str().unwrap(),
        128,                     // <-- vector dimension
        metric,
    )));

    println!("⬢ Vector DB server running on http://127.0.0.1:8444");

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(web::Data::new(db.clone()))
            .route("/ping", web::get().to(health_check))
            .route("/create_collection", web::post().to(create_collection))
            .route("/collections", web::get().to(list_collections))
            .route("/collections/{name}", web::get().to(get_collection_by_name))
            .route("/add_document", web::post().to(add_document))
            .route("/add_documents", web::post().to(add_documents))
            .route("/search", web::post().to(retrieve_documents))
    })
    .bind(("127.0.0.1", 8444))?
    .run()
    .await
}
