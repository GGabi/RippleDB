use std::collections::HashMap;

use super::{
  k2_tree::K2Tree,
  super::{Triple, CRUD}
};

/* Subjects and Objects are mapped in the same
     collection to a unique int while Predicates
     are mapped seperately to unique ints.
   Each slice contains a 2-d bit matrix, each cell
     corresponding to a Subject-Object pair connected
     by a single Predicate. */
pub struct Graph {
  dict_max: usize, //The max value in dict, as it's non-trivial to calculate on the fly
  dict: HashMap<String, usize>,
  predicates: HashMap<String, usize>,
  slices: Vec<Box<K2Tree>>,
}

/* Public */
impl CRUD for Graph {
  type IN = Triple;
  type OUT = ();
  type QUERY = ();
  fn new() -> Self {
    Graph {
      dict_max: 0,
      dict: HashMap::new(),
      predicates: HashMap::new(),
      slices: Vec::new(),
    }
  }
  fn get(&self, query: &Self::QUERY) -> Self::OUT {
    unimplemented!()
  }
  fn insert(&mut self, val: Self::IN) -> Result<(), ()> {
    let graph_coords = (
      /* Copy out to remove lingering reference
      so the borrow-checker doesn't have a fit
      later on */
      match self.dict.get(&val[0]) {
        Some(&col) => Some(col),
        None => None,
      },
      match self.predicates.get(&val[1]) {
        Some(&slice_index) => Some(slice_index),
        None => None,
      },
      match self.dict.get(&val[2]) {
        Some(&row) => Some(row),
        None => None,
      }
    );
    match graph_coords {
      (Some(col), Some(slice_index), Some(row)) => {
        if let Some(slice) = self.slices.get_mut(slice_index) {
          slice.set(col, row, true)
        }
        else {
          Err(())
        }
      },
      (None, Some(slice_index), Some(row)) => {
        self.dict_max += 1;
        self.dict.insert(val[0].clone(), self.dict_max);
        if self.slices.len() > 0
        && self.dict_max > self.slices[0].matrix_width() {
          for slice in &mut self.slices {
            slice.grow();
          }
        }
        if let Some(slice) = self.slices.get_mut(slice_index) {
          slice.set(self.dict_max, row, true)
        }
        else {
          Err(())
        }
      },
      (Some(col), None, Some(row)) => {
        self.slices.push(Box::new(K2Tree::new()));
        let slice_len = self.slices.len();
        self.predicates.insert(val[1].clone(), slice_len-1);
        let desired_size = self.slices[0].matrix_width();
        let new_slice = &mut self.slices[slice_len-1];
        while new_slice.matrix_width() < desired_size {
          new_slice.grow();
        }
        new_slice.set(col, row, true)
      },
      (Some(col), Some(slice_index), None) => {
        self.dict_max += 1;
        self.dict.insert(val[2].clone(), self.dict_max);
        if self.slices.len() > 0
        && self.dict_max > self.slices[0].matrix_width() {
          for slice in &mut self.slices {
            slice.grow();
          }
        }
        if let Some(slice) = self.slices.get_mut(slice_index) {
          slice.set(col, self.dict_max, true)
        }
        else {
          Err(())
        }
      },
      (None, None, Some(row)) => {
        /* Add the new subject */
        self.dict_max += 1;
        self.dict.insert(val[0].clone(), self.dict_max);
        if self.slices.len() > 0
        && self.dict_max > self.slices[0].matrix_width() {
          for slice in &mut self.slices {
            slice.grow();
          }
        }
        /* Add the new slice */
        self.slices.push(Box::new(K2Tree::new()));
        let slice_len = self.slices.len();
        self.predicates.insert(val[1].clone(), slice_len-1);
        let desired_size = self.slices[0].matrix_width();
        let new_slice = &mut self.slices[slice_len-1];
        while new_slice.matrix_width() < desired_size {
          new_slice.grow();
        }
        new_slice.set(self.dict_max, row, true)
      },
      (None, Some(slice_index), None) => {
        /* Add the new object */
        self.dict_max += 1;
        let col = self.dict_max;
        self.dict.insert(val[0].clone(), self.dict_max);
        if self.slices.len() > 0
        && self.dict_max > self.slices[0].matrix_width() {
          for slice in &mut self.slices {
            slice.grow();
          }
        }
        /* Add the new subject */
        self.dict_max += 1;
        let row = self.dict_max;
        self.dict.insert(val[2].clone(), self.dict_max);
        if self.slices.len() > 0
        && self.dict_max > self.slices[0].matrix_width() {
          for slice in &mut self.slices {
            slice.grow();
          }
        }
        if let Some(slice) = self.slices.get_mut(slice_index) {
          slice.set(col, row, true)
        }
        else {
          Err(())
        }
      },
      (Some(col), None, None) => {
        /* Add the new object */
        self.dict_max += 1;
        self.dict.insert(val[2].clone(), self.dict_max);
        if self.slices.len() > 0
        && self.dict_max > self.slices[0].matrix_width() {
          for slice in &mut self.slices {
            slice.grow();
          }
        }
        /* Add the new slice */
        self.slices.push(Box::new(K2Tree::new()));
        let slice_len = self.slices.len();
        self.predicates.insert(val[1].clone(), slice_len-1);
        let desired_size = self.slices[0].matrix_width();
        let new_slice = &mut self.slices[slice_len-1];
        while new_slice.matrix_width() < desired_size {
          new_slice.grow();
        }
        new_slice.set(col, self.dict_max, true)
      },
      (None, None, None) => {
        /* Add the new object */
        self.dict_max += 1;
        let col = self.dict_max;
        self.dict.insert(val[0].clone(), self.dict_max);
        if self.slices.len() > 0
        && self.dict_max > self.slices[0].matrix_width() {
          for slice in &mut self.slices {
            slice.grow();
          }
        }
        /* Add the new subject */
        self.dict_max += 1;
        let row = self.dict_max;
        self.dict.insert(val[2].clone(), self.dict_max);
        if self.slices.len() > 0
        && self.dict_max > self.slices[0].matrix_width() {
          for slice in &mut self.slices {
            slice.grow();
          }
        }
        /* Add the new slice */
        self.slices.push(Box::new(K2Tree::new()));
        let slice_len = self.slices.len();
        self.predicates.insert(val[1].clone(), slice_len-1);
        let desired_size = self.slices[0].matrix_width();
        let new_slice = &mut self.slices[slice_len-1];
        while new_slice.matrix_width() < desired_size {
          new_slice.grow();
        }
        new_slice.set(col, row, true)
      },
    }
  }
  fn remove(&mut self, val: &Self::IN) -> Result<(), ()> {
    unimplemented!()
  }
}

/* Iterators */

/* Std Traits */

/* Private */

/* Utils */

/* Unit Tests */