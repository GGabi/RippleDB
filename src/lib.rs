#![allow(non_snake_case, dead_code, unused_must_use)]

mod ripple_db;

pub use ripple_db::datastore::graph::Graph as Graph;
pub use ripple_db::rdf::query::Sparql as SparqlQuery;
pub use ripple_db::datastore::k2_tree as k2_tree;
pub use ripple_db::Triple as Triple;
pub use ripple_db::RdfNode as RdfNode;
pub use ripple_db::RdfTriple as RdfTriple;