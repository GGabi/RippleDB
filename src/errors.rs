
type Source<E> = Box<E>;

#[derive(Debug)]
pub enum GraphError {
  NoPersistLocation,
  MissingBackup(std::path::PathBuf),
  InvalidBackup(String, std::path::PathBuf),
  FromBadJson(String, std::path::PathBuf, Source<serde_json::Error>),
  Io(Source<std::io::Error>),
  Serde(Source<serde_json::Error>),
  DeadK2Tree(String),
  K2Tree(Source<k2_tree::error::K2TreeError>),
  Parser(Source<ParserError>),
}
impl std::error::Error for GraphError {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    use GraphError::*;
    match self {
      FromBadJson(_, _, e) => Some(&*e),
      Io(e) => Some(&*e),
      Serde(e) => Some(&*e),
      K2Tree(e) => Some(&*e),
      _ => None,
    }
  }
}
impl std::fmt::Display for GraphError {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    use GraphError::*;
    match self {
      NoPersistLocation => write!(f, "Attempted call to .persist() on a Graph with no specified persistence location. Did you mean .persist_to(path)?"),
      MissingBackup(path) => write!(f, "RippleDB Graph backup does not exist at {}", path.display()),
      InvalidBackup(missing_elem, path) => write!(f, "RippleDB Graph backup at {} is invalid, missing {}", path.display(), missing_elem),
      FromBadJson(struct_type, path, e) => write!(f, "Attempted to build {} from inavlid json at {}: {}", struct_type, path.display(), *e),
      Io(e) => write!(f, "{}", *e),
      Serde(e) => write!(f, "{}", *e),
      DeadK2Tree(reason) => write!(f, "Graph's K2Tree is invalid and considered dead because {}, meaning that the Graph's integrity is most likely compromised.", reason),
      K2Tree(e) => write!(f, "{}", *e),
      Parser(e) => write!(f, "{}", *e),
    }
  }
}
impl From<std::io::Error> for GraphError {
  fn from(err: std::io::Error) -> GraphError {
    GraphError::Io(Box::new(err))
  }
}
impl From<serde_json::Error> for GraphError {
  fn from(err: serde_json::Error) -> GraphError {
    GraphError::Serde(Box::new(err))
  }
}
impl From<k2_tree::error::K2TreeError> for GraphError {
  fn from(err: k2_tree::error::K2TreeError) -> GraphError {
    GraphError::K2Tree(Box::new(err))
  }
}
impl From<ParserError> for GraphError {
  fn from(err: ParserError) -> GraphError {
    GraphError::Parser(Box::new(err))
  }
}

#[derive(Debug)]
pub enum K2TreeError {
  TraverseError(usize, usize),
  OutOfBounds([usize; 2], [usize; 2]),
  Serde(Source<serde_json::Error>),
  StemInsertionError(usize, usize),
  StemRemovalError(usize, usize),
  LeafInsertionError(usize, usize),
  LeafRemovalError(usize, usize),
  CouldNotShrink(String),
}
impl std::error::Error for K2TreeError {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    use K2TreeError::*;
    match self {
      Serde(e) => Some(&*e),
      _ => None,
    }
  }
}
impl std::fmt::Display for K2TreeError {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    use K2TreeError::*;
    match self {
      TraverseError(x, y) => write!(f, "Error encountered while traversing K2Tree for value at coordinates ({}, {})", x, y),
      OutOfBounds([x, y], [max_x, max_y]) => write!(f, "Attempt to access a bit at coordiantes ({}, {}) which are not in the range of the matrix represented by the K2Tree: ({}, {})", x, y, max_x, max_y),
      Serde(e) => write!(f, "{}", *e),
      StemInsertionError(pos, stem_len) => write!(f, "Could not insert stem of length {} to BitVec at offset {}", stem_len, pos),
      StemRemovalError(pos, stem_len) => write!(f, "Could not remove stem of length {} to BitVec at offset {}", stem_len, pos),
      LeafInsertionError(pos, leaf_len) => write!(f, "Could not insert leaf of length {} to BitVec at offset {}", leaf_len, pos),
      LeafRemovalError(pos, leaf_len) => write!(f, "Could not remove leaf of length {} to BitVec at offset {}", leaf_len, pos),
      CouldNotShrink(reason) => write!(f, "Could not shrink the matrix a K2Tree represents: {}", reason),
    }
  }
}
impl From<serde_json::Error> for K2TreeError {
  fn from(err: serde_json::Error) -> K2TreeError {
    K2TreeError::Serde(Box::new(err))
  }
}

#[derive(Debug)]
pub enum ParserError {
  Rio(Source<rio_xml::RdfXmlError>)
}
impl std::error::Error for ParserError {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    use ParserError::*;
    match self {
      Rio(e) => Some(&*e)
    }
  }
}
impl std::fmt::Display for ParserError {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    use ParserError::*;
    match self {
      Rio(e) => write!(f, "{}", e),
    }
  }
}
impl From<rio_xml::RdfXmlError> for ParserError {
  fn from(err: rio_xml::RdfXmlError) -> ParserError {
    ParserError::Rio(Box::new(err))
  }
}