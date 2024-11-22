use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use actix_web::middleware::Logger;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use env_logger::Env;
use vectordb::vectorstore::{VectorDB, ShardDB, DistanceMetric};

mod vectordb;

#[derive(Deserialize)]
struct AddDocumentRequest {
    id: i32,
    embedding: Vec<f64>,
    metadata: String,
    content: String,
}

#[derive(Deserialize)]
struct AddDocumentsRequest {
    documents: Vec<AddDocumentRequest>,
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
    content: String,
}

async fn add_document(db: web::Data<ShardDB>, item: web::Json<AddDocumentRequest>) -> impl Responder {
    let db = db.lock().unwrap();
    match db.add_document(item.id, item.embedding.clone(), item.metadata.clone(), item.content.clone()).await {
        Ok(_) => HttpResponse::Ok().body("Document embedded successfully"),
        Err(err) => HttpResponse::BadRequest().body(err),
    }
}

async fn add_documents(db: web::Data<ShardDB>, item: web::Json<AddDocumentsRequest>) -> impl Responder {
    let db = db.lock().unwrap();
    let mut errors = Vec::new();

    for doc in &item.documents {
        if let Err(e) = db.add_document(doc.id, doc.embedding.clone(), doc.metadata.clone(), doc.content.clone()).await {
            errors.push(format!("Failed to add document ID {}: {}", doc.id, e));
        }
    }

    if errors.is_empty() {
        HttpResponse::Ok().body("All documents embedded successfully")
    } else {
        HttpResponse::BadRequest().body(errors.join("\n"))
    }
}

async fn retrieve_documents(db: web::Data<ShardDB>, item: web::Json<FindNearestRequest>) -> impl Responder {
    let db = db.lock().unwrap();

    // Parse the distance metric
    let metric = match DistanceMetric::from_str(&item.metric) {
        Some(m) => m,
        None => return HttpResponse::BadRequest().body("Invalid Distance metric"),
    };

    if item.query.is_empty() {
        return HttpResponse::BadRequest().body("Query vector is empty");
    }

    let results = db.search(
        &item.query,
        item.n,
        metric,
        item.metadata_filter.as_deref(),
    ).await;

    let response: Vec<NearestNeighbor> = results.into_iter().map(|(id, distance, metadata, content)| {
        NearestNeighbor { id, distance, metadata, content }
    }).collect();

    HttpResponse::Ok().json(response)
}


fn ensure_directory(path: &PathBuf) -> std::io::Result<()> {
    if !path.exists() {
        fs::create_dir_all(path)?;
        println!("Created directory at {:?}", path);
    } else {
        println!("Directory already exists at {:?}", path);
    }
    Ok(())
}

fn initialize_database_path() -> std::io::Result<String> {
    // Get the current working directory instead of executable path
    let current_dir = env::current_dir()?;
    let data_dir = current_dir.join("data");
    
    // Ensure the data directory exists
    ensure_directory(&data_dir)?;
    
    // Define the database path
    let db_path = data_dir.join("vectordb.db");
    
    // Ensure parent directories have correct permissions
    fs::create_dir_all(db_path.parent().unwrap())?;
    
    // Convert path to string with forward slashes
    let db_path_str = db_path.to_str()
        .expect("Invalid path")
        .replace("\\", "/");
    Ok(format!("sqlite://{}", db_path_str))
}


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logger
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    // Persistence storage
    let database_url = match initialize_database_path() {
        Ok(url) => {
            println!("Database URL: {}", url);
            url
        },
        Err(e) => {
            eprintln!("Failed to initialize database path: {}", e);
            return Err(e);
        }
    };

    // Init vectordb 
    let vector_db = match VectorDB::new(&database_url).await {
        Ok(db) => db,
        Err(e) => {
            eprintln!("Failed to initialize database: {}", e);
            // Try to create an empty database file if it doesn't exist
            if !PathBuf::from(database_url.trim_start_matches("sqlite://")).exists() {
                fs::File::create(database_url.trim_start_matches("sqlite://"))?;
                // Retry database initialization
                match VectorDB::new(&database_url).await {
                    Ok(db) => db,
                    Err(e) => {
                        eprintln!("Failed to initialize database after creating file: {}", e);
                        return Err(std::io::Error::new(std::io::ErrorKind::Other, e));
                    }
                }
            } else {
                return Err(std::io::Error::new(std::io::ErrorKind::Other, e));
            }
        }
    };
    let db = Arc::new(Mutex::new(vector_db));
    println!("Starting VectorDB server at http://127.0.0.1:8444");

    // Start server
    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default()) // Enable logging middleware
            .app_data(web::Data::new(db.clone())) // TODO: Maybe increase payload size ?
            .route("/add_document", web::post().to(add_document))
            .route("/add_documents", web::post().to(add_documents))
            .route("/search", web::post().to(retrieve_documents))
    })
    .bind(("127.0.0.1", 8444))?
    .run()
    .await
}