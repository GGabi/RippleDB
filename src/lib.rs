#![allow(non_snake_case, dead_code)]
#![feature(async_closure)]

mod util;

pub use util::datastore::graph::Graph as Graph;
pub use util::rdf::query::Sparql as SparqlQuery;
pub use util::datastore::k2_tree as k2_tree;
pub use util::Triple as Triple;