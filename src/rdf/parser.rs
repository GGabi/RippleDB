
use {
  bimap::BiBTreeMap,
  crate::{RdfNode, RdfTriple, errors::ParserError}
};

type Result<T> = std::result::Result<T, ParserError>;

/* Pre-indexed triples to make insertion to Graph
  faster.
  Partitioned triples to make multi-threading building
  of graph easier in the future. One thread per K2Tree
  to build-up. */
pub struct ParsedTriples {
  pub dict_max: usize,
  pub dict: BiBTreeMap<RdfNode, usize>,
  pub pred_max: usize,
  pub predicates: BiBTreeMap<RdfNode, usize>,
  pub partitioned_triples: Vec<Vec<[usize; 2]>>, //Where surface index == predicate index
}
impl ParsedTriples {
  pub fn from_rdf_triples(triples: Vec<RdfTriple>) -> Self {

    let mut dict_max: usize = 0;
    let mut dict: BiBTreeMap<RdfNode, usize> = BiBTreeMap::new();
    let mut pred_max: usize = 0;
    let mut preds: BiBTreeMap<RdfNode, usize> = BiBTreeMap::new();
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
          pred_max += 1;
        }
        preds.insert(pred, pred_max);
        t[1] = pred_max;
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
      dict_max,
      dict,
      pred_max,
      predicates: preds,
      partitioned_triples: partitioned_trips,
    }
  }
  pub fn from_rdf(path: &str) -> Result<Self> {
    use std::io::BufReader;
    use std::fs::File;
    use rio_xml::{RdfXmlParser, RdfXmlError};
    use rio_api::{
      parser::TriplesParser,
      model::{NamedOrBlankNode, NamedNode, Term, Literal}
    };

    let mut triples: Vec<RdfTriple> = Vec::new();

    RdfXmlParser::new(BufReader::new(File::open(path).unwrap()), &format!("file:{}", path))
      .unwrap()
      .parse_all(&mut |t| {
        let s = match t.subject {
          NamedOrBlankNode::BlankNode(s) => RdfNode::Blank{id:s.to_string()},
          NamedOrBlankNode::NamedNode(NamedNode{iri:s}) => RdfNode::Named{iri:s.to_string()},
        };
        let p = match t.predicate {
          NamedNode{iri:s} => RdfNode::Named{iri:s.to_string()},
        };
        let o = match t.object {
          Term::NamedNode(NamedNode{iri:o}) => RdfNode::Named{iri:o.to_string()},
          Term::BlankNode(o) => RdfNode::Blank{id:o.to_string()},
          Term::Literal(lit) => match lit {
            Literal::Simple{value:o} => RdfNode::RawLit{val:o.to_string()},
            Literal::LanguageTaggedString{value:o,language:l} => RdfNode::LangTaggedLit{val:o.to_string(),lang:l.to_string()},
            Literal::Typed{value:o,datatype:NamedNode{iri:t}} => RdfNode::TypedLit{val:o.to_string(),datatype:t.to_string()},
          },
        };
        triples.push([s, p, o]);
        Ok(()) as std::result::Result<(), RdfXmlError>
    })?;
    Ok(Self::from_rdf_triples(triples))
  }
}

#[cfg(test)]
mod unit_tests {
  use super::*;
  #[test]
  fn try_parse() -> Result<()> {
    use std::path::MAIN_SEPARATOR as PATH_SEP;
    ParsedTriples::from_rdf(&format!("models{}cold-2010-complete.rdf", PATH_SEP))?;
    Ok(())
  }
}