#![allow(non_snake_case)]

extern crate serde;

pub mod datastore;
pub mod rdf;

use serde::{Serialize, Deserialize};

/* Common Definitions */

pub type Triple = [String; 3];
pub type RdfTriple = [RdfNode; 3];

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
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