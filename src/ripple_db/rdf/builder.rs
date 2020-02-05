extern crate rio_turtle;
extern crate rio_xml;
extern crate rio_api;
extern crate bimap;

use bimap::BiBTreeMap;

use super::super::Triple;

use rio_xml::RdfXmlFormatter;
use rio_api::formatter::TriplesFormatter;
use rio_api::model::{NamedNode, Triple as RioTriple};

fn triples_to_rdf(triples: Vec<Triple>) -> Vec<u8> {
  let mut formatter = RdfXmlFormatter::new(Vec::default()).unwrap();
  formatter.format(&RioTriple {
    subject: NamedNode { iri: "s" }.into(),
    predicate: NamedNode { iri: "p" },
    object: NamedNode { iri: "o" }.into(),
  }).unwrap();
  formatter.finish().unwrap() //Return xml
}
fn triples_to_file(triples: Vec<Triple>) -> Result<(), std::io::Error> {
  unimplemented!()
}