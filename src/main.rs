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
        db.add_vector(id, v);
    }

    // Query Vector
    let query_vector: Vector = (0..128).map(|_| rng.gen_range(0.0..1.0)).collect();

    // Find nearest vector
    if let Some((id, distance)) = db.find_nearest(&query_vector) {
        println!("Nearest vector ID: {} with distance: {}", id, distance);
    } else {
        println!("No vectors found");
    }
}