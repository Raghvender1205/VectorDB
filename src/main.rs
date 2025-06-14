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
    pub id: i32,
    pub distance: f64,
    pub metadata: String,
    pub content: String,
}

// Healthcheck
async fn health_check() -> impl Responder {
    HttpResponse::Ok().body("pong")
}


/// Handler to create a new collection
async fn create_collection(
    db: web::Data<ShardDB>,
    item: web::Json<CreateCollectionRequest>
) -> impl Responder {
    let db = db.lock().unwrap();
    match db.create_collection(&item.name).await {
        Ok(collection) => {
            log::info!("Created new collection: {}", collection.name);
            HttpResponse::Ok().json(collection)
        },
        Err(err) => {
            let err_lower = err.to_lowercase();
            if err_lower.contains("collection already exists") {
                log::info!("Collection '{}' already exists.", item.name);
                // Respond with 409 Conflict
                HttpResponse::Conflict().body("Collection already exists")
            } else {
                log::error!("Error creating collection '{}': {}", item.name, err);
                HttpResponse::BadRequest().body(err)
            }
        }
    }
}

// Handler to get collection by name
async fn get_collection_by_name(
    db: web::Data<ShardDB>,
    path: web::Path<String>,
) -> impl Responder {
    let db = db.lock().unwrap();
    let colletion_name = path.into_inner();

    match db.get_collection_by_name(&colletion_name).await {
        Ok(collection) => HttpResponse::Ok().json(collection),
        Err(_) => HttpResponse::NotFound().body("Collection not found")
    }
}


/// Handler to add a single document to a collection
async fn add_document(
    db: web::Data<ShardDB>,
    item: web::Json<AddDocumentRequest>,
) -> impl Responder {
    let db = db.lock().unwrap();
    let collection = match db.get_collection_by_name(&item.collection_name).await {
        Ok(col) => col,
        Err(e) => return HttpResponse::BadRequest().body(format!("Collection not found: {}", e)),
    };

    let result = db
        .add_document(
            item.id, // <- only if provided
            item.embedding.clone(),
            item.metadata.clone(),
            item.content.clone(),
            collection.id,
        )
        .await;

    match result {
        Ok(new_id) => HttpResponse::Ok().json(AddDocumentResponse {
            id: new_id,
            status: "Document embedded successfully".to_string(),
        }),
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
        log::warn!("No documents provided for addition.");
        return HttpResponse::BadRequest().body("No documents provided");
    }

    // Assuming all documents belong to the same collection
    let collection_name = &item.documents[0].collection_name;

    // Verify all documents belong to the same collection
    for doc in &item.documents {
        if doc.collection_name != *collection_name {
            log::warn!(
                "Document with collection mismatch: {:?} vs expected {}",
                doc.collection_name, collection_name
            );            
            return HttpResponse::BadRequest().body("All documents must belong to the same collection");
        }
    }

    // Retrieve the collection by name
    let collection = match db.get_collection_by_name(collection_name).await {
        Ok(col) => col,
        Err(_) => {
            log::warn!("Collection '{}' not found.", collection_name);
            return HttpResponse::BadRequest().body("Collection not found");
        },
    };

    match db.add_documents(item.documents.clone(), collection.id).await {
        Ok(ids) => {
            let responses = ids.into_iter().map(|id| AddDocumentResponse {
                id,
                status: "success".to_string(),
            }).collect::<Vec<_>>();
    
            HttpResponse::Ok().json(AddDocumentsResponse {
                documents: responses,
            })
        }
        Err(errors) => {
            // Handle errors with partial success
            let responses = item.documents.iter()
                .filter_map(|doc| doc.id)
                .map(|id| AddDocumentResponse {
                    id,
                    status: "failed".to_string(),
                })
                .collect::<Vec<_>>();
    
            HttpResponse::MultiStatus().json(serde_json::json!({
                "documents": responses,
                "errors": errors
            }))
        }
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
        Err(_) => {
            log::warn!("Collection '{}' not found.", item.collection_name);
            return HttpResponse::BadRequest().body("Collection not found");
        },
    };

    // Parse the distance metric
    let metric = match DistanceMetric::from_str(&item.metric) {
        Some(m) => m,
        None => {
            log::warn!("Invalid distance metric: {}", item.metric);
            return HttpResponse::BadRequest().body("Invalid Distance metric");
        },
    };

    if item.query.is_empty() {
        log::warn!("Empty query vector provided.");
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

    log::info!("Search completed in collection '{}'. Found {} results.", collection.name, response.len());
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
            .route("/ping", web::get().to(health_check))
            .route("/create_collection", web::post().to(create_collection))
            .route("/collections/{name}", web::get().to(get_collection_by_name))
            .route("/add_document", web::post().to(add_document))
            .route("/add_documents", web::post().to(add_documents))
            .route("/search", web::post().to(retrieve_documents))
    })
    .bind(("127.0.0.1", 8444))?
    .run()
    .await
}