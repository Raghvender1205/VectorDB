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


#[derive(Deserialize, Clone)]
pub struct CreateCollectionRequest {
    pub name: String,
}

#[derive(Deserialize, Clone)]
pub struct AddDocumentRequest {
    pub id: i32,
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
pub struct SearchRequest {
    pub query: Vec<f64>,
    pub n: usize,
    pub metric: String, // "Euclidean", "Cosine", "Dot"
    pub collection_name: String
}

#[derive(Serialize)]
pub struct NearestNeighbor {
    pub id: i32,
    pub distance: f64,
    pub metadata: String,
    pub content: String,
}

/// Handler to create a new collection
async fn create_collection(
    db: web::Data<ShardDB>,
    item: web::Json<CreateCollectionRequest>
) -> impl Responder {
    let db = db.lock().unwrap();
    match db.create_collection(&item.name).await {
        Ok(colection) => HttpResponse::Ok().json(colection),
        Err(err) => {
            if err == "Collection Already exists" {
                HttpResponse::Conflict().body("Collection already exists")
            } else {
                HttpResponse::BadRequest().body(err)
            }
        }
    }
}


/// Handler to add a single document to a collection
async fn add_document(
    db: web::Data<ShardDB>,
    item: web::Json<AddDocumentRequest>,
) -> impl Responder {
    let db = db.lock().unwrap();

    // Retrieve the collection by name
    let collection = match db.get_collection_by_name(&item.collection_name).await {
        Ok(col) => col,
        Err(e) => return HttpResponse::BadRequest().body(format!("Collection not found: {}", e)),
    };

    match db
        .add_document(
            item.id,
            item.embedding.clone(),
            item.metadata.clone(),
            item.content.clone(),
            collection.id,
        )
        .await
    {
        Ok(_) => HttpResponse::Ok().body("Document embedded successfully"),
        Err(err) => HttpResponse::BadRequest().body(err),
    }
}


/// Handler to add multiple documents to a collection
async fn add_documents(
    db: web::Data<ShardDB>,
    item: web::Json<AddDocumentsRequest>,
) -> impl Responder {
    let db = db.lock().unwrap();

    if item.documents.is_empty() {
        return HttpResponse::BadRequest().body("No documents provided");
    }

    // Assuming all documents belong to the same collection
    let collection_name = &item.documents[0].collection_name;

    // Verify all documents belong to the same collection
    for doc in &item.documents {
        if doc.collection_name != *collection_name {
            return HttpResponse::BadRequest().body("All documents must belong to the same collection");
        }
    }

    // Retrieve the collection by name
    let collection = match db.get_collection_by_name(collection_name).await {
        Ok(col) => col,
        Err(_) => return HttpResponse::BadRequest().body("Collection not found"),
    };

    match db.add_documents(item.documents.clone(), collection.id).await {
        Ok(_) => HttpResponse::Ok().body("All documents embedded successfully"),
        Err(errors) => HttpResponse::BadRequest().body(errors.join("\n")),
    }
}


/// Handler to search for relevant documents within a collection
async fn retrieve_documents(
    db: web::Data<ShardDB>,
    item: web::Json<SearchRequest>,
) -> impl Responder {
    let db = db.lock().unwrap();

    // Retrieve the collection by name
    let collection = match db.get_collection_by_name(&item.collection_name).await {
        Ok(col) => col,
        Err(_) => return HttpResponse::BadRequest().body("Collection not found"),
    };

    // Parse the distance metric
    let metric = match DistanceMetric::from_str(&item.metric) {
        Some(m) => m,
        None => return HttpResponse::BadRequest().body("Invalid Distance metric"),
    };

    if item.query.is_empty() {
        return HttpResponse::BadRequest().body("Query vector is empty");
    }

    let results = db
        .search(
            &item.query,
            item.n,
            metric,
            Some(&collection.name),
        )
        .await;

    let response: Vec<NearestNeighbor> = results
        .into_iter()
        .map(|(id, distance, metadata, content)| NearestNeighbor {
            id,
            distance,
            metadata,
            content,
        })
        .collect();

    HttpResponse::Ok().json(response)
}


/// Ensures that a directory exists, creating it if necessary
fn ensure_directory(path: &PathBuf) -> std::io::Result<()> {
    if !path.exists() {
        fs::create_dir_all(path)?;
        println!("Created directory at {:?}", path);
    } else {
        println!("Directory already exists at {:?}", path);
    }
    Ok(())
}


/// Initializes the database path, ensuring correct formatting and directory structure
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
    Ok(format!("sqlite:///{}", db_path_str)) // Ensure three slashes for absolute path
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
            // Attempt to create an empty database file if it doesn't exist
            let db_path = database_url.trim_start_matches("sqlite:///").to_string();
            if !PathBuf::from(&db_path).exists() {
                fs::File::create(&db_path)?;
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
            .route("/create_collection", web::post().to(create_collection))
            .route("/add_document", web::post().to(add_document))
            .route("/add_documents", web::post().to(add_documents))
            .route("/search", web::post().to(retrieve_documents))
    })
    .bind(("127.0.0.1", 8444))?
    .run()
    .await
}