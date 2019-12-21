/*
  Contains all the stuff needed to parse RDF data to useful
  data for the datastores to handle.
*/

extern crate rio_turtle;
extern crate rio_xml;
extern crate rio_api;

use super::super::Triple;

pub fn parse_turtle(path: String) -> Vec<Triple> {
  use rio_turtle::{TurtleParser, TurtleError};
  use rio_api::parser::TriplesParser;
  use rio_api::model::{
    NamedOrBlankNode,
    NamedNode,
    Term,
    Literal::{Simple, Typed, LanguageTaggedString}
  };
  use rio_api::model::Triple as RioTriple;
  use std::io::BufReader;
  use std::fs::File;

  let mut ret_v: Vec<Triple> = Vec::new();

  TurtleParser::new(BufReader::new(File::open(path.clone()).unwrap()), &format!("file:{}", path))
    .unwrap()
    .parse_all(&mut |t| {
      match t {
        RioTriple {
          subject: NamedOrBlankNode::NamedNode(NamedNode { iri: s }),
          predicate: NamedNode { iri: p },
          object: o,
        } => {
          match o {
            Term::Literal(Simple {
              value: o
            }) |
            Term::Literal(Typed {
              value: o,
              datatype: _,
            }) |
            Term::Literal(LanguageTaggedString {
              value: o,
              language: _,
            }) |
            Term::NamedNode(NamedNode { iri: o }) => {
              ret_v.push([s.to_string(), p.to_string(), o.to_string()]);
            },
            _ => {},
          };
        },
        _ => {},
      };
      Ok(()) as Result<(), TurtleError>
  });

  println!("Parsed Triples: {:#?}", ret_v);

  ret_v
}
pub fn parse_rdf(path: String) -> Vec<Triple> {
  use rio_xml::{RdfXmlParser, RdfXmlError};
  use rio_api::parser::TriplesParser;
  use rio_api::model::{
    NamedOrBlankNode,
    NamedNode,
    Term,
    Literal::{Simple, Typed, LanguageTaggedString}
  };
  use rio_api::model::Triple as RioTriple;
  use std::io::BufReader;
  use std::fs::File;

  let mut ret_v: Vec<Triple> = Vec::new();

  RdfXmlParser::new(BufReader::new(File::open(path.clone()).unwrap()), &format!("file:{}", path))
    .unwrap()
    .parse_all(&mut |t| {
      match t {
        RioTriple {
          subject: NamedOrBlankNode::NamedNode(NamedNode { iri: s }),
          predicate: NamedNode { iri: p },
          object: o,
        } => {
          match o {
            Term::Literal(Simple {
              value: o
            }) |
            Term::Literal(Typed {
              value: o,
              datatype: _,
            }) |
            Term::Literal(LanguageTaggedString {
              value: o,
              language: _,
            }) |
            Term::NamedNode(NamedNode { iri: o }) => {
              ret_v.push([s.to_string(), p.to_string(), o.to_string()]);
            },
            _ => {},
          };
        },
        _ => {},
      };
      Ok(()) as Result<(), RdfXmlError>
  });

  ret_v
}

#[cfg(test)]
mod unit_tests {
  use super::*;
  #[test]
  fn try_parse() {
    parse_rdf(String::from("cold-2010-complete.rdf"));
  }
}