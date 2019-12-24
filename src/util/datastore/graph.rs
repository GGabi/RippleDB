extern crate bimap;
extern crate bitvec;
extern crate serde;
extern crate futures;

use bimap::BiBTreeMap;
use bitvec::{prelude::bitvec, vec::BitVec};
use serde::{Serialize, Deserialize};

use super::{
  k2_tree::K2Tree,
  super::{
    Triple,
    rdf::{parser::ParsedTriples, query::Sparql},
  }
};

/* Subjects and Objects are mapped in the same
     collection to a unique int while Predicates
     are mapped seperately to unique ints.
   Each slice contains a representation of a 2-d bit matrix,
     each cell corresponding to a Subject-Object pair
     connected by a single Predicate. */
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graph {
  //Store dict_max because the max R-value in a dict is expensive to calulate on-the-fly
  //Store tombstone-indices for dict and predicates to be reused in later inserts
  //Use BiMap instead of HashMap because we want to be able to find the strings rows/columns represent
  dict_max: usize,
  dict_tombstones: Vec<usize>,
  dict: BiBTreeMap<String, usize>,
  pred_tombstones: Vec<usize>,
  predicates: BiBTreeMap<String, usize>,
  pub slices: Vec<Option<Box<K2Tree>>>,
}

/* Public */
impl Graph {
  fn new() -> Self {
    Graph {
      dict_max: 0,
      dict_tombstones: Vec::new(),
      dict: BiBTreeMap::new(),
      pred_tombstones: Vec::new(),
      predicates: BiBTreeMap::new(),
      slices: Vec::new(),
    }
  }
  fn get(&self, query: &Sparql) -> () {
    /* Initially only provide support for one variable, then scale up */
    /* Get using first triple
     * Then get the second triple, compare the results and filter
    to only contain the ones fit the pattern
     * Loop until all conditions
     * Return results */
    unimplemented!()
  }
  fn insert_triple(&mut self, val: Triple) -> Result<(), ()> {
    let col = match self.dict.get_by_left(&val[0]) {
      Some(&col) => col,
      None => {
        if self.dict_tombstones.len() > 0 {
          let col = self.dict_tombstones[0];
          self.dict.insert(val[0].clone(), col);
          col
        }
        else {
          if self.dict_max != 0 { self.dict_max += 1; }
          self.dict.insert(val[0].clone(), self.dict_max);
          if self.slices.len() > 0 {
            for slice in &mut self.slices {
              match slice {
                Some(slice) if self.dict_max > slice.matrix_width() => {
                  slice.grow();
                },
                _ => {}
              };
            }
          }
          self.dict_max
        }
      },
    };
    let row = match self.dict.get_by_left(&val[2]) {
      Some(&row) => row,
      None => {
        if self.dict_tombstones.len() > 0 {
          let row = self.dict_tombstones[0];
          self.dict.insert(val[2].clone(), row);
          row
        }
        else {
          self.dict_max += 1;
          self.dict.insert(val[2].clone(), self.dict_max);
          if self.slices.len() > 0 {
            for slice in &mut self.slices {
              match slice {
                Some(slice) if self.dict_max > slice.matrix_width() => {
                  slice.grow();
                },
                _ => {}
              };
            }
          }
          self.dict_max
        }
      },
    };
    let slice = match self.predicates.get_by_left(&val[1]) {
      Some(&slice_index) => &mut self.slices[slice_index],
      None => {
        let desired_size =
          if self.slices.len() > 0 {
            let mut desired_size = 0;
            for slice in &self.slices {
              match slice {
                Some(slice) => {
                  desired_size = slice.matrix_width();
                  break
                },
                None => {},
              }
            }
            desired_size
          }
          else {
            8 //Min size assuming k=2
          };
        let new_slice =
          if self.pred_tombstones.len() > 0 {
            let new_slice_pos = self.pred_tombstones[0];
            self.predicates.insert(val[1].clone(), new_slice_pos);
            &mut self.slices[new_slice_pos]
          }
          else {
            self.slices.push(Some(Box::new(K2Tree::new())));
            let slice_len = self.slices.len();
            self.predicates.insert(val[1].clone(), slice_len-1);
            &mut self.slices[slice_len-1]
          };
        if let Some(new_slice) = new_slice {
          while new_slice.matrix_width() < desired_size {
            new_slice.grow();
          }
        }
        new_slice
      },
    };
    match slice {
      Some(slice) => slice.set(col, row, true),
      None => Err(())
    }
  }
  fn remove_triple(&mut self, [subject, predicate, object]: &Triple) -> Result<(), ()> {
    /* TODO: Add ability to shrink matrix_width for all slices if
             needed */
    let (subject_pos, object_pos, slice_pos) = match [
      self.dict.get_by_left(subject),
      self.dict.get_by_left(object),
      self.predicates.get_by_left(predicate)] {
        [Some(&c), Some(&r), Some(&s)] => (c, r, s),
        _ => return Ok(())
    };
    let slice =
      if let Some(slice) = &mut self.slices[slice_pos] {
        slice
      }
      else {
        return Err(())
      };
    slice.set(subject_pos, object_pos, false)?;
    /* Check if we've removed all instances of a word.
    If we have: Remove from dictionaries and do other stuff */
    if slice.is_empty() {
      self.predicates.remove_by_left(predicate);
      if slice_pos == self.slices.len()-1 {
        self.slices.pop();
        while self.slices[self.slices.len()-1] == None {
          self.slices.pop();
        }
        let newly_invalid_tombstones: Vec<usize> = self.pred_tombstones
          .iter()
          .filter_map(|&tombstone|
            if tombstone > self.slices.len()-1 { Some(tombstone) }
            else { None }
          )
          .collect();
        self.pred_tombstones = self.pred_tombstones
          .iter()
          .filter_map(|tombstone|
            if newly_invalid_tombstones.contains(tombstone) { None }
            else { Some(*tombstone) }
          )
          .collect();
        self.slices = self.slices
          .iter()
          .enumerate()
          .filter_map(|(i, slice)|
            match slice {
              None if newly_invalid_tombstones.contains(&i) => None,
              _ => Some(slice.clone()),
            }
          )
          .collect();
      }
      else {
        self.pred_tombstones.push(slice_pos);
        self.slices[slice_pos] = None;
      }
    }
    let mut subject_exists = false;
    let mut object_exists = false;
    for slice in
      self.slices.iter().filter(|slice|
        match slice {
          Some(slice) => !slice.is_empty(),
          None => false,
        }) {
      if let Some(slice) = slice {
        if !subject_exists
        && (ones_in_bitvec(&slice.get_row(subject_pos)?) > 0
        || ones_in_bitvec(&slice.get_column(subject_pos)?) > 0) {
          subject_exists = true;
        }
        if !object_exists
        && (ones_in_bitvec(&slice.get_row(object_pos)?) > 0
        || ones_in_bitvec(&slice.get_column(object_pos)?) > 0) {
          object_exists = true;
        }
        if subject_exists && object_exists { break }
      }
    }
    if !subject_exists {
      self.dict.remove_by_left(subject);
      if subject_pos == self.dict_max {
        /* Find next highest valid dict_max,
        remove all newly-invalid tombstones greater than new dict_max */
        while !self.dict.contains_right(&self.dict_max) {
          self.dict_max -= 1;
        }
        self.dict_tombstones = self.dict_tombstones
          .iter()
          .filter_map(|&tombstone|
            if tombstone <= self.dict_max { Some(tombstone) }
            else { None }
          )
          .collect();
      }
      else {
        self.dict_tombstones.push(subject_pos);
      }
    }
    /* In the case of "gabe likes gabe" making "gabe" dead,
    we don't want to try to remove or declare "gabe" dead twice. */
    if !object_exists
    && subject != object {
      self.dict.remove_by_left(object);
      if object_pos == self.dict_max {
        /* Find next highest valid dict_max,
        remove all newly-invalid tombstones greater than new dict_max */
        while !self.dict.contains_right(&self.dict_max) {
          self.dict_max -= 1;
        }
        self.dict_tombstones = self.dict_tombstones
          .iter()
          .filter_map(|&tombstone|
            if tombstone < self.dict_max { Some(tombstone) }
            else { None }
          )
          .collect();
      }
      else {
        self.dict_tombstones.push(object_pos);
      }
    }
    Ok(())
  }
  fn from_rdf(path: &str) -> Result<Self, ()> {
    use std::{thread, sync::{Mutex, Arc}};
    /* Parse the .rdf file and initialise fields all
    the Graph's fields except for slices */
    let ParsedTriples {
      dict_max,
      dict,
      pred_max: _,
      predicates,
      triples: _,
      partitioned_triples,
    } = match ParsedTriples::from_rdf(path) {
      Ok(p_trips) => p_trips,
      Err(e) => return Err(()),
    };

    /* Build each K2Tree in parallel */
    let trees = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();
    for i in 0..partitioned_triples.len() {
      trees.lock().unwrap().push(Err(()));
      let doubles = partitioned_triples[i].clone();
      let trees = Arc::clone(&trees);
      handles.push(thread::spawn(move || {
        let mut tree = K2Tree::new();
        while tree.matrix_width() < dict_max {
          tree.grow();
        }
        for [x, y] in doubles {
          if let Err(_) = tree.set(x, y, true) {
            return
          }
        }
        trees.lock().unwrap()[i] = Ok(tree);
      }));
    }
    for handle in handles { handle.join().unwrap(); }

    /* Check if every slice was built successfully and 
    inserts each one into the correct location in the Graph's
    slices field */
    let mut slices: Vec<Option<Box<K2Tree>>> = Vec::new();
    for tree_result in Arc::try_unwrap(trees)
      .unwrap()
      .into_inner()
      .unwrap()
      .into_iter() {
      if let Ok(tree) = tree_result {
        slices.push(Some(Box::new(tree)));
      }
      else {
        /* One of the K2Trees failed to build so
        Graph integrity is compromised: abort */
        return Err(())
      }
    }

    Ok(Graph {
      dict_max: dict_max,
      dict_tombstones: Vec::new(),
      dict: dict,
      pred_tombstones: Vec::new(),
      predicates: predicates,
      slices: slices,
    })
  }
  async fn from_rdf_async(path: &str) -> Result<Self, ()> {
    use futures::stream::FuturesOrdered;
    use futures::StreamExt;

    /* Parse the .rdf file and initialise fields all
    the Graph's fields except for slices */
    let ParsedTriples {
      dict_max,
      dict,
      pred_max: _,
      predicates,
      triples: _,
      partitioned_triples,
    } = match ParsedTriples::from_rdf(path) {
      Ok(p_trips) => p_trips,
      Err(e) => return Err(()),
    };

    /* Build each K2Tree concurrently on one thread */
    let mut tree_futs = FuturesOrdered::new();
    for i in 0..partitioned_triples.len() {
      let doubles = partitioned_triples[i].clone();
      tree_futs.push(async move {
        let mut tree = K2Tree::new();
        while tree.matrix_width() < dict_max {
          tree.grow();
        }
        for [x, y] in doubles {
          if let Err(_) = tree.set(x, y, true) {
            return Err(())
          }
        }
        Ok(tree)
      });
    }

    /* Check if every slice was built successfully and 
    inserts each one into the correct location in the Graph's
    slices field */
    let mut trees = Vec::new();
    while let Some(fut_result) = tree_futs.next().await {
      if let Ok(tree) = fut_result {
        trees.push(Some(Box::new(tree)));
      }
      else {
        /* One of the K2Trees failed to build so
        Graph integrity is compromised: abort */
        return Err(())
      }
    }
    
    Ok(Graph {
      dict_max: dict_max,
      dict_tombstones: Vec::new(),
      dict: dict,
      pred_tombstones: Vec::new(),
      predicates: predicates,
      slices: trees,
    })
  }
  /*For even greater building performance get it to build the trees in the background and saved to files
    If the predicate isn't built yet on query, go build it, otherwise finish building the rest. */
}

/* Iterators */

/* Std Traits */

/* Private */
impl Graph {
  /* Return the triples in the compact form of their dict index */
  fn get_from_triple(&self, triple: [Option<&str>; 3]) -> Vec<[usize; 3]> {
    match triple {
      [Some(s), Some(p), Some(o)] => self.spo(s, p, o),
      [None, Some(p), Some(o)] => self._po(p, o),
      [Some(s), None, Some(o)] => self.s_o(s, o),
      [Some(s), Some(p), None] => self.sp_(s, p),
      [None, None, Some(o)] => self.__o(o),
      [None, Some(p), None] => self._p_(p),
      [Some(s), None, None] => self.s__(s),
      [None, None, None] => self.___(),
    }
  }
  fn spo(&self, s: &str, p: &str, o: &str) -> Vec<[usize; 3]> {
    match [self.dict.get_by_left(&s.to_string()),
      self.dict.get_by_left(&o.to_string()),
      self.predicates.get_by_left(&p.to_string())] {
        [Some(&x), Some(&y), Some(&slice_index)] => {
          if let Some(slice) = &self.slices[slice_index] {
            match slice.get(x, y) {
              Ok(b) if b => vec![[x, slice_index, y]],
              _ => Vec::new(),
            }
          }
          else {
            Vec::new()
          }
        },
        _ => Vec::new(),
    }
  }
  fn _po(&self, p: &str, o: &str) -> Vec<[usize; 3]> {
    match [self.dict.get_by_left(&o.to_string()),
      self.predicates.get_by_left(&p.to_string())] {
        [Some(&y), Some(&slice_index)] => {
          if let Some(slice) = &self.slices[slice_index] {
            match slice.get_row(y) {
              Ok(bitvec) => one_positions(&bitvec)
                .into_iter()
                .map(|pos| [pos, slice_index, y])
                .collect(),
              _ => Vec::new(),
            }
          }
          else {
            Vec::new()
          }
        },
        _ => Vec::new(),
    }
  }
  fn s_o(&self, s: &str, o: &str) -> Vec<[usize; 3]> {
    match [self.dict.get_by_left(&s.to_string()),
      self.predicates.get_by_left(&o.to_string())] {
        [Some(&x), Some(&y)] => {
          let mut triples: Vec<[usize; 3]> = Vec::new();
          for (i, slice) in self.slices.iter().enumerate() {
            if let Some(slice) = slice {
              match slice.get(x, y) {
                Ok(b) if b => triples.push([x, i, y]),
                _ => {},
              };
            }
          }
          triples
        },
        _ => Vec::new(),
    }
  }
  fn sp_(&self, s: &str, p: &str) -> Vec<[usize; 3]> {
    match [self.dict.get_by_left(&s.to_string()),
      self.predicates.get_by_left(&p.to_string())] {
        [Some(&x), Some(&slice_index)] => {
          if let Some(slice) = &self.slices[slice_index] {
            match slice.get_column(x) {
              Ok(bitvec) => one_positions(&bitvec)
                .into_iter()
                .map(|pos| [x, slice_index, pos])
                .collect(),
              _ => Vec::new(),
            }
          }
          else {
            Vec::new()
          }
        },
        _ => Vec::new(),
    }
  }
  fn __o(&self, o: &str) -> Vec<[usize; 3]> { unimplemented!() }
  fn _p_(&self, p: &str) -> Vec<[usize; 3]> { unimplemented!() }
  fn s__(&self, s: &str) -> Vec<[usize; 3]> { unimplemented!() }
  fn ___(&self) -> Vec<[usize; 3]> { unimplemented!() }
}

/* Utils */
impl Graph {
  pub fn heapsize(&self) -> usize {
    let mut size: usize = std::mem::size_of_val(self);
    size += std::mem::size_of::<usize>() * self.dict_tombstones.len();
    for (string, _) in self.dict.iter() {
      size += string.as_bytes().len();
      size += std::mem::size_of::<usize>();
    }
    size += std::mem::size_of::<usize>() * self.pred_tombstones.len();
    for (string, _) in self.predicates.iter() {
      size += string.as_bytes().len();
      size += std::mem::size_of::<usize>();
    }
    for slice in self.slices.iter() {
      if let Some(k2tree) = slice {
        size += k2tree.heapsize();
      }
    }
    size
  }
}
fn ones_in_bitvec(bits: &BitVec) -> usize {
  bits.iter().fold(0, |total, bit| total + bit as usize)
}
fn one_positions(bit_vec: &BitVec) -> Vec<usize> {
  bit_vec
  .iter()
  .enumerate()
  .filter_map(
    |(pos, bit)|
    if bit { Some(pos) }
    else   { None })
  .collect()
}

/* Unit Tests */
#[cfg(test)]
mod unit_tests {
  use super::*;
  #[test]
  fn manual_test() {
    let mut g = Graph::new();
    g.insert_triple(["Gabe".into(), "likes".into(), "Rust".into()]);
    g.insert_triple(["Gabe".into(), "likes".into(), "Gabe".into()]);
    g.insert_triple(["Gabe".into(), "hates".into(), "Ron".into()]);
    g.insert_triple(["Gabe".into(), "hates".into(), "Gabe".into()]);
    g.insert_triple(["".into(), "".into(), "".into()]);
    g.insert_triple(["Gabe".into(), "".into(), "".into()]);
    println!("{:#?}", g);
    g.remove_triple(&["Gabe".into(), "hates".into(), "Ron".into()]);
    g.remove_triple(&["Gabe".into(), "hates".into(), "Gabe".into()]);
    // g.remove_triple(&["".into(), "".into(), "".into()]);
    g.remove_triple(&["Gabe".into(), "".into(), "".into()]);
    println!("{:#?}", g);
  }
  #[test]
  fn from_rdf_0() {
    Graph::from_rdf("models\\www-2011-complete.rdf");
  }
  #[test]
  fn from_rdf_async_0() {
    use futures::executor;
    executor::block_on(Graph::from_rdf_async("models\\www-2011-complete.rdf"));
  }
}