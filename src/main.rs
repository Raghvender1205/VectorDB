use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use vectordb::vectorstore::{VectorDB, ShardDB, DistanceMetric};
use vectordb::vectorstore::DistanceMetric::{Cosine, Euclidean, DotProduct};

mod vectordb;

#[derive(Deserialize)]
struct AddDocumentRequest {
    id: i32,
    embedding: Vec<f64>,
    metadata: String,
}

#[derive(Deserialize)]
struct FindNearestRequest {
    query: Vec<f64>,
    n: usize,
    metric: String, // "Euclidean", "Cosine", "Dot"
    metadata_filter: Option<String>,
}

#[derive(Serialize)]
struct NearestNeighbor {
    id: i32,
    distance: f64,
    metadata: String,
}

async fn add_document(db: web::Data<ShardDB>, item: web::Json<AddDocumentRequest>) -> impl Responder {
    let mut db = db.lock().unwrap();
    match db.add_document(item.id, item.embedding.clone(), item.metadata.clone()) {
        Ok(_) => HttpResponse::Ok().body("Document embedded successfully"),
        Err(err) => HttpResponse::BadRequest().body(err),
    }
}

async fn find_nearest(db: web::Data<ShardDB>, item: web::Json<FindNearestRequest>) -> impl Responder {
    let db = db.lock().unwrap();

    let metric = match DistanceMetric::from_str(&item.metric) {
        Some(m) => m,
        None => return HttpResponse::BadRequest().body("Invalid Distance metric"),
    };

    if item.query.is_empty() {
        return HttpResponse::BadRequest().body("Query vector is empty");
    }

    let results = db.find_nearest(
        &item.query,
        item.n,
        metric,
        item.metadata_filter.as_deref(),
    );

    let response: Vec<NearestNeighbor> = results.into_iter().map(|(id, distance, metadata)| {
        NearestNeighbor { id, distance, metadata }
    }).collect();

    HttpResponse::Ok().json(response)
}


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Init vectordb 
    let db = Arc::new(Mutex::new(VectorDB::new()));
    println!("Starting VectorDB server at http://127.0.0.1:8444");

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(db.clone()))
            .route("/add_document", web::post().to(add_document))
            .route("/find_nearest", web::post().to(find_nearest))
    })
    .bind(("127.0.0.1", 8444))?
    .run()
    .await
}