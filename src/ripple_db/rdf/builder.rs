extern crate rio_xml;
extern crate rio_api;

pub mod RdfBuilder {
  use {
    crate::ripple_db::{RdfNode, RdfTriple},
    rio_xml::RdfXmlFormatter,
    rio_api::{
      formatter::TriplesFormatter,
      model as Rio
    }
  };
  pub fn iter_to_rdf<T>(iter: T) -> Vec<u8>
  where T: Iterator<Item=RdfTriple> {
    let mut formatter = RdfXmlFormatter::new(Vec::default()).unwrap();
    for [s, p, o] in iter {
      let rio_s: Option<Rio::NamedOrBlankNode> = match &s {
        RdfNode::Named{iri} => Some(Rio::NamedOrBlankNode::NamedNode(Rio::NamedNode{iri: &iri})),
        RdfNode::Blank{id} => Some(Rio::NamedOrBlankNode::BlankNode(Rio::BlankNode{id:&id})),
        _ => None,
      };
      let rio_p: Option<Rio::NamedNode> = match &p {
        RdfNode::Named{iri} => Some(Rio::NamedNode{iri:&iri}),
        _ => None,
      };
      let rio_o: Option<Rio::Term> = match &o {
        RdfNode::Named{iri} => Some(Rio::Term::NamedNode(Rio::NamedNode{iri:&iri})),
        RdfNode::Blank{id} => Some(Rio::Term::BlankNode(Rio::BlankNode{id:&id})),
        RdfNode::RawLit{val} => Some(Rio::Term::Literal(Rio::Literal::Simple{value:&val})),
        RdfNode::LangTaggedLit{val, lang} => Some(Rio::Term::Literal(Rio::Literal::LanguageTaggedString{value:&val,language:&lang})),
        RdfNode::TypedLit{val, datatype} => Some(Rio::Term::Literal(Rio::Literal::Typed{value:&val,datatype:Rio::NamedNode{iri:&datatype}})),
      };
      if let (Some(s), Some(p), Some(o)) = (rio_s, rio_p, rio_o) {
        formatter.format(&Rio::Triple {
          subject: s,
          predicate: p,
          object: o,
        }).unwrap();
      }
    }
    formatter.finish().unwrap()
  }
}