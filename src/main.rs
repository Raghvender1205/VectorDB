mod vectordb;

use vectordb::vectorstore::VectorDB;
use vectordb::vectorstore::Vector;
use rand::Rng;

fn main() {
    let mut db = VectorDB::new();

    // Populate db with random vectors
    let mut rng = rand::thread_rng();
    for id in 0..1000 {
        let v: Vector = (0..128).map(|_| rng.gen_range(0.0..1.0)).collect();
        let metadata = format!("Document: {}", id);
        db.add_document(id, v, metadata);
    }

    // Query Vector
    let query_vector: Vector = (0..128).map(|_| rng.gen_range(0.0..1.0)).collect();

    // Find the top 5 nearest vectors
    let results = db.find_nearest(&query_vector, 5);
    for (id, distance, metadata) in results {
        println!("Document ID: {}, Distance: {}, Metadata: {}", id, distance, metadata);
    }
}