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

// Health Check
async fn health_check() -> impl Responder {
    HttpResponse::Ok().body("pong")
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
        }),
        Err(e) if e == "duplicate" => HttpResponse::Conflict().body("Collection exists"),
        Err(e) => HttpResponse::BadRequest().body(e),
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
        }),
        Err(_) => HttpResponse::NotFound().body("Collection not found"),
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
        Err(e) => HttpResponse::BadRequest().body(e),
    }
}

async fn add_documents(
    db: web::Data<ShardDB<'static>>,
    req: web::Json<AddDocumentsRequest>,
) -> impl Responder {
    if req.documents.is_empty() {
        return HttpResponse::BadRequest().body("No documents");
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
            .json(json!({ "errors": errs })),
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
        Err(e) => HttpResponse::BadRequest().body(e),
    }
}


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let db_path = PathBuf::from("data/rocksdb");
    ensure_directory(&db_path)?;

    let db: ShardDB<'static> = Arc::new(VectorDB::new(db_path.to_str().unwrap()));

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