#![allow(non_snake_case)]

pub mod datastore;
pub mod rdf;

/* Common Definitions */

pub type Triple = [String; 3];
pub type RdfTriple = [RdfNode; 3];
pub type GraphTriple = [GraphNode; 3];

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum GraphNode {
  Named{ iri: String },
  Blank,
  RawLit{ val: String },
  LangTaggedLit{ val: String, lang: String },
  TypedLit{ val: String, datatype: String },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RdfNode {
  Subject(RdfSubject),
  Predicate{iri: String},
  Object(RdfObject),
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RdfSubject {
  Named{iri: String},
  Blank,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RdfObject {
  Named{iri: String},
  Blank,
  Literal(RdfLiteral),
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RdfLiteral {
  Raw{val: String},
  LangTagged{val: String, lang: String},
  Typed{val: String, datatype: String},
}

// impl PartialEq for RdfNode {
//   fn eq(&self, other: &Self) -> bool {
//     match self {
//       RdfNode::Subject(s) => {
//         match s {
//           RdfSubject::Named{iri:s} => {
//             match other {
//               RdfNode::Subject(RdfSubject::Named{iri:other_s}) => {
//                 s == other_s
//               },
//               RdfNode::Object(RdfObject::Named{iri:o}) => {
//                 s == o
//               },
//               _ => false,
//             }
//           },
//           RdfSubject::Blank => {
//             match other {
//               RdfNode::Subject(RdfSubject::Blank) => true,
//               RdfNode::Object(RdfObject::Blank) => true,
//               _ => false,
//             }
//           },
//         }
//       },
//       RdfNode::Predicate{iri:p} => {
//         if let RdfNode::Predicate{iri:other_p} = other {
//           p == other_p
//         }
//         else {
//           false
//         }
//       },
//       RdfNode::Object(o) => {
//         match o {
//           RdfObject::Named{iri:o} => {},
//           RdfObject::Blank => {},
//           RdfObject::Literal(o) => {},
//         }
//       },
//     }
//   }
// }

#[derive(Clone)]
struct Nibble(u8);
impl Nibble {
  fn new(val: u8) -> Result<Self, ()> {
    if val > 15 {
      return Err(())
    }
    Ok(Nibble(val))
  }
  fn get(&self, bit_pos: u8) -> Result<bool, ()> {
    match bit_pos {
      0 => Ok((**self & 0b1000) != 0),
      1 => Ok((**self & 0b0100) != 0),
      2 => Ok((**self & 0b0010) != 0),
      3 => Ok((**self & 0b0001) != 0),
      _ => Err(())
    }
  }
  fn set(&mut self, bit_pos: u8, val: bool) -> Result<(), ()> {
    if self.get(bit_pos) != Ok(val) {
      **self += match bit_pos {
        0 => 8,
        1 => 4,
        2 => 2,
        3 => 1,
        _ => return Err(()),
      };
    }
    Ok(())
  }
}
impl std::ops::Deref for Nibble {
  type Target = u8;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}
impl std::ops::DerefMut for Nibble {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}