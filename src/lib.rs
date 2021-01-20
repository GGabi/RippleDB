#![allow(non_snake_case, clippy::many_single_char_names)]

/* Exports */

pub mod datastore;
pub mod rdf;
pub mod errors;

pub use datastore::graph::Graph as Graph;
// pub use datastore::k2_tree::K2Tree as K2Tree;
pub use rdf::query::Sparql as SparqlQuery;

/* Common Definitions */

pub type Triple = [String; 3];
pub type RdfTriple = [RdfNode; 3];
pub fn triple_into_rdf([s, p, o]: Triple) -> RdfTriple {
  [s.into(), p.into(), o.into()]
}

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
impl std::convert::From<String> for RdfNode {
  fn from(s: String) -> Self {
    Self::Named{ iri: s }
  }
}