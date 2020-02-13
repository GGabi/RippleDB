#![allow(non_snake_case, dead_code, unused_must_use)]

/* Exports */

pub mod datastore;
pub mod rdf;

pub use datastore::graph::Graph as Graph;
pub use rdf::query::Sparql as SparqlQuery;
pub use datastore::k2_tree as k2_tree;

/* Common Definitions */

pub type Triple = [String; 3];
pub type RdfTriple = [RdfNode; 3];

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize, Hash)]
pub enum RdfNode {
  Named{ iri: String },
  Blank{ id: String },
  RawLit{ val: String },
  LangTaggedLit{ val: String, lang: String },
  TypedLit{ val: String, datatype: String },
}
impl std::convert::From<&str> for RdfNode {
  fn from(s: &str) -> Self {
    Self::Named{ iri: s.to_string() }
  }
}