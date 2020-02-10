/*
  Contains all the stuff needed to parse RDF data to useful
  data for the datastores to handle.
*/

extern crate rio_turtle;
extern crate rio_xml;
extern crate rio_api;
extern crate bimap;

use bimap::BiBTreeMap;

use super::super::{Triple, RdfTriple, RdfNode, RdfSubject, RdfObject, RdfLiteral, GraphNode, GraphTriple};

/* Pre-indexed triples to make insertion to Graph
  faster.
  Partitioned triples to make multi-threading building
  of graph easier in the future. One thread per K2Tree
  to build-up. */
pub struct ParsedTriples {
  pub dict_max: usize,
  pub dict: BiBTreeMap<GraphNode, usize>,
  pub pred_max: usize,
  pub predicates: BiBTreeMap<GraphNode, usize>,
  pub triples: Vec<[usize; 3]>,
  pub partitioned_triples: Vec<Vec<[usize; 2]>>, //Where surface index == predicate index
}
impl ParsedTriples {
  pub fn from_triples(triples: Vec<Triple>) -> Self {

    let mut dict_max: usize = 0;
    let mut dict: BiBTreeMap<String, usize> = BiBTreeMap::new();
    let mut preds_max: usize = 0;
    let mut preds: BiBTreeMap<String, usize> = BiBTreeMap::new();
    let mut indexed_trips: Vec<[usize; 3]> = Vec::new();
    let mut partitioned_trips: Vec<Vec<[usize; 2]>> = Vec::new(); //Index is the pred index
    /* Let's start aggregating shall we? */

    let mut fresh_dict = true;
    let mut fresh_pred = true;

    for [subj, pred, obj] in triples {
      let mut t: [usize; 3] = [0; 3];
      if let Some(&s) = dict.get_by_left(&subj) {
        t[0] = s;
      }
      else {
        if fresh_dict {
          fresh_dict = false;
        }
        else {
          dict_max += 1;
        }
        dict.insert(subj, dict_max);
        t[0] = dict_max;
      }
      if let Some(&p) = preds.get_by_left(&pred) {
        t[1] = p;
      }
      else {
        if fresh_pred {
          fresh_pred = false;
        }
        else {
          preds_max += 1;
        }
        preds.insert(pred, preds_max);
        t[1] = preds_max;
      }
      if let Some(&o) = dict.get_by_left(&obj) {
        t[2] = o;
      }
      else {
        dict_max += 1;
        dict.insert(obj, dict_max);
        t[2] = dict_max;
      }
      if t[1] >= partitioned_trips.len() {
        partitioned_trips.push(vec![[t[0], t[2]]]);
      }
      else {
        partitioned_trips[t[1]].push([t[0], t[2]]);
      }
      indexed_trips.push(t);
    }

    unimplemented!()
    
    // ParsedTriples {
    //   dict_max: dict_max,
    //   dict: dict,
    //   pred_max: preds_max,
    //   predicates: preds,
    //   triples: indexed_trips,
    //   partitioned_triples: partitioned_trips,
    // }
  }
  pub fn from_rdf_triples(triples: Vec<GraphTriple>) -> Self {

    let mut dict_max: usize = 0;
    let mut dict: BiBTreeMap<GraphNode, usize> = BiBTreeMap::new();
    let mut preds_max: usize = 0;
    let mut preds: BiBTreeMap<GraphNode, usize> = BiBTreeMap::new();
    let mut indexed_trips: Vec<[usize; 3]> = Vec::new();
    let mut partitioned_trips: Vec<Vec<[usize; 2]>> = Vec::new(); //Index is the pred index
    /* Let's start aggregating shall we? */

    let mut fresh_dict = true;
    let mut fresh_pred = true;

    for [subj, pred, obj] in triples {
      let mut t: [usize; 3] = [0; 3];
      if let Some(&s) = dict.get_by_left(&subj) {
        t[0] = s;
      }
      else {
        if fresh_dict {
          fresh_dict = false;
        }
        else {
          dict_max += 1;
        }
        dict.insert(subj, dict_max);
        t[0] = dict_max;
      }
      if let Some(&p) = preds.get_by_left(&pred) {
        t[1] = p;
      }
      else {
        if fresh_pred {
          fresh_pred = false;
        }
        else {
          preds_max += 1;
        }
        preds.insert(pred, preds_max);
        t[1] = preds_max;
      }
      if let Some(&o) = dict.get_by_left(&obj) {
        t[2] = o;
      }
      else {
        dict_max += 1;
        dict.insert(obj, dict_max);
        t[2] = dict_max;
      }
      if t[1] >= partitioned_trips.len() {
        partitioned_trips.push(vec![[t[0], t[2]]]);
      }
      else {
        partitioned_trips[t[1]].push([t[0], t[2]]);
      }
      indexed_trips.push(t);
    }
    
    ParsedTriples {
      dict_max: dict_max,
      dict: dict,
      pred_max: preds_max,
      predicates: preds,
      triples: indexed_trips,
      partitioned_triples: partitioned_trips,
    }
  }
  pub fn from_rdf(path: &str) -> Result<Self, rio_xml::RdfXmlError> {
    use std::io::BufReader;
    use std::fs::File;
    use rio_xml::{RdfXmlParser, RdfXmlError};
    use rio_api::{
      parser::TriplesParser,
      model::{
        Triple as RioTriple,
        NamedOrBlankNode,
        NamedNode, BlankNode,
        Term,
        Literal::{self, Simple, Typed, LanguageTaggedString}
      }
    };

    let mut triples: Vec<GraphTriple> = Vec::new();

    RdfXmlParser::new(BufReader::new(File::open(path).unwrap()), &format!("file:{}", path))
      .unwrap()
      .parse_all(&mut |t| {
        let s: GraphNode = match t.subject {
          NamedOrBlankNode::BlankNode(_) => GraphNode::Blank,
          NamedOrBlankNode::NamedNode(NamedNode{iri:s}) => GraphNode::Named{iri:s.to_string()},
        };
        let p: GraphNode = match t.predicate {
          NamedNode{iri:s} => GraphNode::Named{iri:s.to_string()},
        };
        let o: GraphNode = match t.object {
          Term::NamedNode(NamedNode{iri:o}) => GraphNode::Named{iri:o.to_string()},
          Term::BlankNode(_) => GraphNode::Blank,
          Term::Literal(lit) => match lit {
            Literal::Simple{value:o} => GraphNode::RawLit{val:o.to_string()},
            Literal::LanguageTaggedString{value:o,language:l} => GraphNode::LangTaggedLit{val:o.to_string(),lang:l.to_string()},
            Literal::Typed{value:o,datatype:NamedNode{iri:t}} => GraphNode::TypedLit{val:o.to_string(),datatype:t.to_string()},
          },
        };
        triples.push([s, p, o]);
        Ok(()) as Result<(), RdfXmlError>
    })?;
    Ok(Self::from_rdf_triples(triples))
  }
}

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

  ret_v
}

#[cfg(test)]
mod unit_tests {
  use super::*;
  #[test]
  fn try_parse() {
    use std::path::MAIN_SEPARATOR as PATH_SEP;
    ParsedTriples::from_rdf(&format!("models{}cold-2010-complete.rdf", PATH_SEP));
  }
}