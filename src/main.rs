use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use actix_web::http::StatusCode;
use actix_web::middleware::Logger;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use env_logger::Env;
use vectordb::vectorstore::{VectorDB, ShardDB, DistanceMetric};

mod vectordb;

#[cfg(feature = "profiling")]
use mimalloc::MiMalloc;

#[cfg(feature = "profiling")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;



#[derive(Deserialize, Clone)]
pub struct CreateCollectionRequest {
    name: String,
    metric: String,
    dimension: usize,
}

#[derive(Deserialize, Clone)]
pub struct AddDocumentRequest {
    pub id: Option<i32>,
    pub embedding: Vec<f64>,
    pub metadata: String,
    pub content: String,
    pub collection_name: String, 
}

#[derive(Deserialize)]
struct AddDocumentsRequest {
    documents: Vec<AddDocumentRequest>,
}

#[derive(Deserialize)]
struct SearchRequest {
    query: Vec<f64>,
    n: usize, 
    collection_name: String,
}

#[derive(Serialize)]
struct CollectionResponse {
    id: u64,
    name: String,
    metric: DistanceMetric,
    dimension: usize,
    doc_count: u64
}

#[derive(Serialize)]
struct AddDocumentResponse {
    id: i32, 
    status: String,
}

#[derive(Serialize)]
struct AddDocumentsResponse {
    documents: Vec<AddDocumentResponse>,
}

#[derive(Serialize)]
struct NearestNeighbor {
    id: u64,
    distance: f32,
    metadata: String,
    content: String
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    uptime_seconds: u64,
}

#[derive(Serialize)]
struct StatsResponse {
    collections: usize,
    total_documents: usize,
    memory_usage: std::collections::HashMap<String, usize>,
}

// Health Check
async fn health_check() -> impl Responder {
    HttpResponse::Ok().json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: 0, // You can implement proper uptime tracking
    })
}

fn ensure_directory(path: &PathBuf) -> std::io::Result<()> {
    if !path.exists() {
        fs::create_dir_all(path)?;
    }
    Ok(())
}

async fn create_collection(
    db: web::Data<ShardDB<'static>>,
    req: web::Json<CreateCollectionRequest>,
) -> impl Responder {
    let metric = DistanceMetric::from_str(&req.metric)
        .unwrap_or(DistanceMetric::Cosine);

    match db.create_collection(&req.name, metric.clone(), req.dimension) {
        Ok(meta) => HttpResponse::Ok().json(CollectionResponse {
            id: meta.id,
            name: meta.name,
            metric,
            dimension: meta.dim,
            doc_count: meta.doc_count
        }),
        Err(e) if e == "duplicate" => HttpResponse::Conflict().json(json!({
            "error": "Collection already exists",
            "collection_name": req.name
        })),
        Err(e) => HttpResponse::BadRequest().json(json!({
            "error": e,
            "collection_name": req.name
        })),
    }
}

async fn list_collections(db: web::Data<ShardDB<'static>>) -> impl Responder {
    let list = db.list_collections();
    let resp: Vec<CollectionResponse> = list
        .into_iter()
        .map(|m| CollectionResponse {
            id: m.id,
            name: m.name,
            metric: m.metric,
            dimension: m.dim,
            doc_count: m.doc_count
        })
        .collect();
    HttpResponse::Ok().json(resp)
}

async fn get_collection_by_name(
    db: web::Data<ShardDB<'static>>,
    path: web::Path<String>,
) -> impl Responder {
    match db.get_collection_by_name(&path) {
        Ok(m) => HttpResponse::Ok().json(CollectionResponse {
            id: m.id,
            name: m.name,
            metric: m.metric,
            dimension: m.dim,
            doc_count: m.doc_count
        }),
        Err(_) => HttpResponse::NotFound().json(json!({
            "error": "Collection not found",
            "collection_name": path.as_str()
        })),
    }
}

async fn add_document(
    db: web::Data<ShardDB<'static>>,
    item: web::Json<AddDocumentRequest>,
) -> impl Responder {
    let emb: Vec<f32> = item.embedding.iter().map(|v| *v as f32).collect();
    match db.add_document(
        &item.collection_name,
        item.id.map(|x| x as u64),
        emb,
        item.metadata.clone(),
        item.content.clone(),
    ) {
        Ok(id) => HttpResponse::Ok().json(AddDocumentResponse {
            id: id as i32,
            status: "success".into(),
        }),
        Err(e) => HttpResponse::BadRequest().json(json!({
            "error": e,
            "collection_name": item.collection_name
        })),
    }
}

async fn add_documents(
    db: web::Data<ShardDB<'static>>,
    req: web::Json<AddDocumentsRequest>,
) -> impl Responder {
    if req.documents.is_empty() {
        return HttpResponse::BadRequest().json(json!({
            "error": "No documents provided"
        }));
    }

    let col_name = &req.documents[0].collection_name;
    match db.add_documents(col_name, req.documents.clone()) {
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
        Err(errs) => HttpResponse::build(StatusCode::MULTI_STATUS)
            .json(json!({ 
                "errors": errs,
                "collection_name": col_name
            })),
    }
}

async fn retrieve_documents(
    db: web::Data<ShardDB<'static>>,
    req: web::Json<SearchRequest>,
) -> impl Responder {
    let query: Vec<f32> = req.query.iter().map(|v| *v as f32).collect();
    match db.search(&req.collection_name, &query, req.n) {
        Ok(hits) => {
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
        Err(e) => HttpResponse::BadRequest().json(json!({
            "error": e,
            "collection_name": req.collection_name
        })),
    }
}

async fn get_stats(db: web::Data<ShardDB<'static>>) -> impl Responder {
    let collections = db.list_collections();
    let total_docs: u64 = collections.iter().map(|c| c.doc_count).sum();
    
    let memory_stats = db.get_memory_stats();
    
    HttpResponse::Ok().json(StatsResponse {
        collections: collections.len(),
        total_documents: total_docs as usize,
        memory_usage: memory_stats,
    })
}

async fn memory_profile() -> impl Responder {
    #[cfg(feature = "profiling")]
    {
        HttpResponse::Ok().json(json!({
            "message": "Memory profiling available via mimalloc",
            "allocator": "mimalloc",
            "instructions": "Compile with --features profiling to enable mimalloc allocator"
        }))
    }
    #[cfg(not(feature = "profiling"))]
    {
        HttpResponse::NotImplemented().json(json!({
            "error": "Profiling not enabled. Compile with --features profiling"
        }))
    }
}


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let db_path = PathBuf::from("data/rocksdb");
    ensure_directory(&db_path)?;

    let db: ShardDB<'static> = Arc::new(VectorDB::new(db_path.to_str().unwrap()));
    let workers = num_cpus::get();

    println!("⬢ Vector DB server running on http://127.0.0.1:8444");
    println!("⬢ Workers: {}", workers);
    println!("⬢ Database path: {}", db_path.display());

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(web::Data::new(db.clone()))
            .app_data(web::PayloadConfig::new(50 * 1024 * 1024)) // 50MB payload limit
            .service(
                web::scope("/api/v1")
                    .route("/health", web::get().to(health_check))
                    .route("/stats", web::get().to(get_stats))
                    .route("/profile", web::get().to(memory_profile))
                    .route("/collections", web::post().to(create_collection))
                    .route("/collections", web::get().to(list_collections))
                    .route("/collections/{name}", web::get().to(get_collection_by_name))
                    .route("/documents", web::post().to(add_document))
                    .route("/documents/batch", web::post().to(add_documents))
                    .route("/search", web::post().to(retrieve_documents))
            )
            // Legacy routes for backward compatibility
            .route("/ping", web::get().to(health_check))
            .route("/create_collection", web::post().to(create_collection))
            .route("/collections", web::get().to(list_collections))
            .route("/collections/{name}", web::get().to(get_collection_by_name))
            .route("/add_document", web::post().to(add_document))
            .route("/add_documents", web::post().to(add_documents))
            .route("/search", web::post().to(retrieve_documents))
    })
    .worker_max_blocking_threads(workers * 4)
    .workers(workers)
    .bind(("127.0.0.1", 8444))?
    .run()
    .await
}