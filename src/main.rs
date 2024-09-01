mod vectordb;

use ndarray::array;
use vectordb::vectorstore::VectorDB;
use crate::vectordb::vectorstore::DistanceMetric;

fn main() {
    // Create a new vector database
    let mut db = VectorDB::new();

    // Add some documents with vectors and metadata
    db.add_document(1, array![1.0, 2.0, 3.0], String::from("Category: A"));
    db.add_document(2, array![4.0, 5.0, 6.0], String::from("Category: B"));
    db.add_document(3, array![7.0, 8.0, 9.0], String::from("Category: A"));

    // Query for the nearest neighbors to a vector with metadata filtering
    let query_vector = array![1.5, 2.5, 3.5];
    let nearest_neighbors = db.find_nearest(&query_vector, 2, DistanceMetric::Euclidean, Some("Category: A"));

    // Print out the nearest neighbors
    println!("Nearest neighbors (Euclidean with Category: A):");
    for (id, distance, metadata) in nearest_neighbors {
        println!("ID: {}, Distance: {:.4}, Metadata: {}", id, distance, metadata);
    }
}