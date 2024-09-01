mod vectordb;

use ndarray::array;
use vectordb::vectorstore::VectorDB;
use vectordb::vectorstore::Vector;
use crate::vectordb::vectorstore::DistanceMetric;

fn main() {
    // Create a new vector database
    let mut db = VectorDB::new();

    // Add some documents with vectors
    db.add_document(1, array![1.0, 2.0, 3.0], String::from("Document 1"));
    db.add_document(2, array![4.0, 5.0, 6.0], String::from("Document 2"));
    db.add_document(3, array![7.0, 8.0, 9.0], String::from("Document 3"));

    // Query for the nearest neighbors to a vector using Euclidean distance
    let query_vector: Vector = array![1.0, 2.0, 3.5];
    let nearest_neighbors = db.find_nearest(&query_vector, 2, DistanceMetric::Euclidean);

    // Print out the nearest neighbors
    println!("Nearest neighbors (Euclidean):");
    for (id, distance, metadata) in &nearest_neighbors {
        println!("ID: {}, Distance: {:.4}, Metadata: {}", id, distance, metadata);
    }

    // Query for the nearest neighbors to a vector using Cosine similarity
    let nearest_neighbors_cosine = db.find_nearest(&query_vector, 2, DistanceMetric::Cosine);

    // Print out the nearest neighbors
    println!("\nNearest neighbors (Cosine):");
    for (id, distance, metadata) in &nearest_neighbors_cosine {
        println!("ID: {}, Distance: {:.4}, Metadata: {}", id, distance, metadata);
    }

    // Query for the nearest neighbors to a vector using Dot Product
    let nearest_neighbors_dot = db.find_nearest(&query_vector, 2, DistanceMetric::DotProduct);

    // Print out the nearest neighbors
    println!("\nNearest neighbors (Dot Product):");
    for (id, distance, metadata) in &nearest_neighbors_dot {
        println!("ID: {}, Distance: {:.4}, Metadata: {}", id, distance, metadata);
    }
}