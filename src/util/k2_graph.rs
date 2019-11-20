extern crate bitvec;

use bitvec::{
  vec::BitVec
};
use std::collections::HashMap;

use super::{Triple, CRUD};

/* Subjects and Objects are mapped in the same
     collection to a unique int while Predicates
     are mapped seperately to unique ints.
   Each slice contains a 2-d bit matrix, each cell
     corresponding to a Subject-Object pair connected
     by a single Predicate. */
struct Graph {
  dict: HashMap<String, usize>,
  predicates: HashMap<String, usize>,
  slices: Vec<GraphSlice>,
}
impl CRUD for Graph {
  type IN = Triple;
  type OUT = ();
  type QUERY = ();
  fn new() -> Self {
    unimplemented!()
  }
  fn insert(&mut self, val: Self::IN) -> Result<(), ()> {
    unimplemented!()
  }
  fn remove(&mut self, val: &Self::IN) -> Result<(), ()> {
    unimplemented!()
  }
  fn get(&self, query: &Self::QUERY) -> Self::OUT {
    unimplemented!()
  }
}

/* 2-d bit matrix, each column corresponding to an
     Object's uid and each row corresponding to a
     Subject's uid.
   Every Subject-Object pair is joined by single Predicate.
   K */
struct GraphSlice {
  predicate: String,
  tree: k2_tree::K2Tree,
}
impl CRUD for GraphSlice {
  type IN = Triple;
  type OUT = ();
  type QUERY = ();
  fn new() -> Self {
    GraphSlice {
      predicate: String::new(),
      tree: k2_tree::K2Tree::new(),
    }
  }
  fn insert(&mut self, val: Self::IN) -> Result<(), ()> {
    unimplemented!()
  }
  fn remove(&mut self, val: &Self::IN) -> Result<(), ()> {
    unimplemented!()
  }
  fn get(&self, query: &Self::QUERY) -> Self::OUT {
    unimplemented!()
  }
}
impl GraphSlice {
  pub fn from(triples: Vec<Triple>) -> Self {
    unimplemented!()
  }
}

pub mod k2_tree {
  use bitvec::{
    prelude::bitvec,
    vec::BitVec
  };
  #[derive(Debug)]
  pub struct K2Tree {
    matrix_width: usize,
    k: usize,
    stem_layer_starts: Vec<usize>,
    stems: BitVec,
    stem_to_leaf: Vec<usize>,
    leaves: BitVec,
  }
  /* Public Interface */
  impl K2Tree {
    pub fn new() -> Self {
      K2Tree {
        matrix_width: 0,
        k: 0,
        stem_layer_starts: Vec::new(),
        stems: BitVec::new(),
        stem_to_leaf: Vec::new(),
        leaves: BitVec::new(),
      }
    }
    pub fn test_tree() -> Self {
      K2Tree {
        matrix_width: 8,
        k: 2,
        stem_layer_starts: vec![0, 4],
        stems:  bitvec![0,1,1,1, 1,1,0,1, 1,0,0,0, 1,0,0,0],
        stem_to_leaf: vec![0, 1, 3, 4, 8],
        leaves: bitvec![0,1,1,0, 0,1,0,1, 1,1,0,0, 1,0,0,0, 0,1,1,0],
      }
    }
    pub fn get_bit(&self, x: usize, y: usize) -> bool {
      /* Assuming k=2 */
      if let DescendResult::Leaf(leaf_start, leaf_range) = self.matrix_bit(x, y, self.matrix_width) {
        if leaf_range[0][1] - leaf_range[0][0] != 1
        || leaf_range[0][1] - leaf_range[0][0] != 1 {
          /* ERROR: Final submatrix isn't a 2 by 2 so can't be a leaf */
        }
        if x == leaf_range[0][0] {
          if y == leaf_range[1][0] { return self.leaves[leaf_start] }
          else { return self.leaves[leaf_start+2] }
        }
        else {
          if y == leaf_range[1][0] { return self.leaves[leaf_start+1] }
          else { return self.leaves[leaf_start+3] }
        }
      }
      else {
        /* DescendResult::Stem means no leaf with bit at (x, y) exists,
             so bit must be 0 */
        return false
      }
    }
    pub fn set_bit(&mut self, x: usize, y: usize, state: bool) -> Result<(), ()> {
      /* Assuming k=2 */
      match self.matrix_bit(x, y, self.matrix_width) {
        DescendResult::Leaf(leaf_start, leaf_range) => {
          if leaf_range[0][1] - leaf_range[0][0] != 1
          || leaf_range[0][1] - leaf_range[0][0] != 1 {
            /* ERROR: Final submatrix isn't a 2 by 2 so can't be a leaf */
          }
          if x == leaf_range[0][0] {
            if y == leaf_range[1][0] { self.leaves.set(leaf_start, state); }
            else { self.leaves.set(leaf_start+2, state); }
          }
          else {
            if y == leaf_range[1][0] { self.leaves.set(leaf_start+1, state); }
            else { self.leaves.set(leaf_start+3, state); }
          }
          if ones_in_range(&self.leaves, leaf_start, leaf_start+3) == 0 {
            /* If leaf is now all 0's do some complex recursive stuff up the tree */
            //Remove leaf from self.leaves
            //Alter self.stem_to_leaf to reflect changes
            //Go to final layer of stems, find bit that pointed to leaf and change to 0
            //Check if stem is now all 0's, if so then remove stem and alter self.stem_layer_starts if needed
            //Go up a stem layer and change bit to 0, repeat the above checks until stem isn't all 0s
            for _ in 0..4 { self.leaves.remove(leaf_start); } //Remove the leaf
            let stem_bit_pos = self.stem_to_leaf[leaf_start/4]; //Grab location of stem bit that pointed to the leaf
            self.stem_to_leaf.remove(leaf_start/4);
            if self.stem_to_leaf.len() == 0 {
              /* If no more leaves, then remove all stems immediately and don't bother with complex stuff below */
              self.stems = bitvec![0,0,0,0];
              self.stem_layer_starts = vec![0];
              return Ok(())
            }
            let stem_layer_start = self.stem_layer_starts[self.stem_layer_starts.len()-1];
            let mut stem_start = (stem_bit_pos/4)*4 + stem_layer_start;
            self.stems.set(stem_bit_pos + stem_layer_start, false);
            let mut curr_layer = self.stem_layer_starts.len()-1;
            while ones_in_range(&self.stems, stem_start, stem_start+3) == 0 {
              for layer_start in &mut self.stem_layer_starts[curr_layer+1..] {
                *layer_start -= 1; //Adjust lower layer positions to reflect removal of leaf
              }
              let parent_bit_pos = self.parent_bit(stem_start);
              for _ in 0..4 { self.stems.remove(stem_start); }
              curr_layer -= 1;
              self.stems.set(parent_bit_pos, false);
              stem_start = self.parent_stem(stem_start);
              if curr_layer == 0 { break }
            }
          }
        },
        DescendResult::Stem(mut stem_start, mut stem_range) if state => {
          /* (None, _) means x and y are located in a submatrix of all 0's */
          /* If state = true then do some more complex stuff by adding
              stem + leaf nodes etc. and setting bit */
          println!("Stem_range: {:?}", stem_range);
          let mut layer = self.layer_from_range(stem_range);
          let stem_layer_max = self.stem_layer_starts.len()-1;
          if layer == stem_layer_max {
            let subranges = to_4_subranges(stem_range);
            for child_pos in 0..4 {
              if within_range(&subranges[child_pos], x, y) {
                self.stems.set(stem_start + child_pos, true);
                let mut i: usize = 0;
                while i < self.stem_to_leaf.len()
                && self.stem_to_leaf[i] < (stem_start - self.stem_layer_starts[self.stem_layer_starts.len()-1]) {
                  i += 1;
                }
                self.stem_to_leaf.insert(i, (stem_start - self.stem_layer_starts[self.stem_layer_starts.len()-1])+child_pos);
                for _ in 0..4 { self.leaves.insert(i*4, false); }
                let leaf_range = subranges[child_pos];
                if x == leaf_range[0][0] {
                  if y == leaf_range[1][0] { self.leaves.set(i*4, true); }
                  else { self.leaves.set((i*4)+2, true); }
                }
                else {
                  if y == leaf_range[1][0] { self.leaves.set((i*4)+1, true); }
                  else { self.leaves.set((i*4)+3, true); }
                }
                return Ok(())
              }
            }
          }
          while layer < self.stem_layer_starts.len() {
            let subranges = to_4_subranges(stem_range);
            for child_pos in 0..4 {
              if within_range(&subranges[child_pos], x, y) {
                if layer == stem_layer_max {
                  /* Add the stem the add the leaf stuff, this is the last loop iteration */
                  self.stems.set(stem_start + child_pos, true);
                  let stem_start_in_final_layer = stem_start - self.stem_layer_starts[self.stem_layer_starts.len()-1];
                  let bit_pos_in_final_layer = stem_start_in_final_layer + child_pos;
                  let mut i: usize = 0;
                  while i < self.stem_to_leaf.len()
                  && self.stem_to_leaf[i] < stem_start_in_final_layer {
                    i += 1;
                  }
                  let stem_to_leaf_len = self.stem_to_leaf.len();
                  for leaf_start in &mut self.stem_to_leaf[i..stem_to_leaf_len] {
                    *leaf_start += 4;
                  }
                  self.stem_to_leaf.insert(i, bit_pos_in_final_layer);
                  for _ in 0..4 { self.leaves.insert(i*4, false); }
                  let leaf_range = subranges[child_pos];
                  if x == leaf_range[0][0] {
                    if y == leaf_range[1][0] { self.leaves.set(i*4, true); }
                    else { self.leaves.set((i*4)+2, true); }
                  }
                  else {
                    if y == leaf_range[1][0] { self.leaves.set((i*4)+1, true); }
                    else { self.leaves.set((i*4)+3, true); }
                  }
                  break
                }
                else {
                  /* Change bit to 1 */
                  /* Get the start position of where the child stem should be, add the stem of 0s */
                  /* Update the stem_layer_starts */
                  self.stems.set(stem_start + child_pos, true);
                  stem_start = self.child_stem(layer, stem_start, child_pos)?;
                  for _ in 0..4 { self.stems.insert(stem_start, false); }
                  for layer_start in &mut self.stem_layer_starts[layer+2..stem_layer_max+1] {
                    *layer_start += 4;
                  }
                  break
                }
              }
            }
            layer += 1;
          }
        }
        _ => {},
      }
      Ok(())
    }
  }
  /* Utils */
  enum DescendResult {
    Leaf(usize, [[usize; 2]; 2]), //leaf_start, leaf_range
    Stem(usize, [[usize; 2]; 2]), //stem_start, stem_range
    None,
  }
  struct DescendEnv {
    /* Allows for descend to be recursive without parameter hell */
    x: usize,
    y: usize,
    stem_layer_max: usize,
  }
  impl K2Tree {
    fn layer_from_range(&self, range: [[usize; 2]; 2]) -> usize {
      let range_width = range[0][1]-range[0][0]+1;
      ((self.stem_layer_starts.len()+1)
      - ((range_width as f64).log(self.k as f64) as usize))
    }
    fn matrix_bit(&self, x: usize, y: usize, m_width: usize) -> DescendResult {
      let env = DescendEnv {
        x: x,
        y: y,
        stem_layer_max: self.stem_layer_starts.len()-1,
      };
      self.descend(&env, 0, 0, [[0, m_width-1], [0, m_width-1]])
    }
    fn descend(&self, env: &DescendEnv, layer: usize, stem_pos: usize, range: [[usize; 2]; 2]) -> DescendResult {
      let subranges = to_4_subranges(range);
      for (child_pos, child) in self.stems[stem_pos..stem_pos+4].iter().enumerate() {
        if within_range(&subranges[child_pos], env.x, env.y) {
          if !child { return DescendResult::Stem(stem_pos, range) } //The bit exists within a range that has all zeros
          else if layer == env.stem_layer_max {
            return DescendResult::Leaf(self.leaf_start(stem_pos + child_pos).unwrap(), subranges[child_pos])
          }
          else {
            return self.descend(env,
                                layer+1,
                                self.child_stem(layer, stem_pos, child_pos).unwrap(),
                                subranges[child_pos])
          }
        }
      }
      DescendResult::None //Should never return this but need to satisfy compiler
    }
    fn num_stems_before_child(&self, bit_pos: usize, layer: usize) -> usize {
      let layer_start = self.layer_start(layer);
      ones_in_range(&self.stems, layer_start, bit_pos)
    }
    fn layer_start(&self, l: usize) -> usize {
      self.stem_layer_starts[l]
    }
    fn layer_len(&self, l: usize) -> usize {
      if l == self.stem_layer_starts.len()-1 {
        return self.stems.len() - self.stem_layer_starts[l]
      }
      self.stem_layer_starts[l+1] - self.stem_layer_starts[l]
    }
    fn leaf_start(&self, stem_bitpos: usize) -> Result<usize, ()> {
      if !self.stems[stem_bitpos] { return Err(()) }
      Ok(self.stem_to_leaf
             .iter()
             .position(|&n| n == (stem_bitpos - self.stem_layer_starts[self.stem_layer_starts.len()-1]))
             .unwrap()
             * 4)
    }
    fn child_stem(&self, layer: usize, stem_start: usize, nth_child: usize) -> Result<usize, ()> {
      if !self.stems[stem_start+nth_child]
      || layer == self.stem_layer_starts.len()-1 {
        /* If stem_bit is 0 or final stem layer, cannot have children */
        return Err(())
      }
      Ok(self.layer_start(layer+1)
      + (self.num_stems_before_child(stem_start+nth_child, layer) * 4))
    }
    fn parent_stem(&self, stem_start: usize) -> usize {
      if stem_start >= self.stem_layer_starts[1] {
        /* If stem isn't in layer 0, look for parent */
        let mut parent_layer_start = 0;
        let mut curr_layer_start = 0;
        for (i, layer_start) in (0..self.stem_layer_starts.len()).enumerate() {
          if i == self.stem_layer_starts.len()-1 {
            if stem_start >= self.stem_layer_starts[layer_start]
            && stem_start < self.stems.len() {
              parent_layer_start = self.stem_layer_starts[layer_start-1];
              curr_layer_start = self.stem_layer_starts[layer_start];
            }
          }
          else if stem_start >= self.stem_layer_starts[layer_start]
          && stem_start < self.stem_layer_starts[layer_start+1] {
            parent_layer_start = self.stem_layer_starts[layer_start-1];
            curr_layer_start = self.stem_layer_starts[layer_start];
          }
        }
        let nth_stem_in_layer = (stem_start - curr_layer_start)/4;
        let mut i = 0;
        let mut bit_pos_in_parent_stem_layer = 0;
        for bit in &self.stems[parent_layer_start..curr_layer_start] {
          if bit {
            if i == nth_stem_in_layer { break }
            i += 1;
          }
          bit_pos_in_parent_stem_layer += 1;
        }
        return ((bit_pos_in_parent_stem_layer / 4) * 4) + parent_layer_start
      }
      std::usize::MAX
    }
    fn parent_bit(&self, stem_start: usize) -> usize {
      if stem_start >= self.stem_layer_starts[1] {
        /* If stem isn't in layer 0, look for parent */
        let mut parent_layer_start = 0;
        let mut curr_layer_start = 0;
        for (i, layer_start) in (0..self.stem_layer_starts.len()).enumerate() {
          if i == self.stem_layer_starts.len()-1 {
            if stem_start >= self.stem_layer_starts[layer_start]
            && stem_start < self.stems.len() {
              parent_layer_start = self.stem_layer_starts[layer_start-1];
              curr_layer_start = self.stem_layer_starts[layer_start];
            }
          }
          else if stem_start >= self.stem_layer_starts[layer_start]
          && stem_start < self.stem_layer_starts[layer_start+1] {
            parent_layer_start = self.stem_layer_starts[layer_start-1];
            curr_layer_start = self.stem_layer_starts[layer_start];
          }
        }
        let nth_stem_in_layer = (stem_start - curr_layer_start)/4;
        let mut i = 0;
        let mut bit_pos_in_parent_stem_layer = 0;
        for bit in &self.stems[parent_layer_start..curr_layer_start] {
          if bit {
            if i == nth_stem_in_layer { break }
            i += 1;
          }
          bit_pos_in_parent_stem_layer += 1;
        }
        return bit_pos_in_parent_stem_layer + parent_layer_start
      }
      std::usize::MAX
    }
  }
  fn to_4_subranges(r: [[usize; 2]; 2]) -> [[[usize; 2]; 2]; 4] {
    [
      [[r[0][0], r[0][0]+((r[0][1]-r[0][0])/2)],   [r[1][0], r[1][0]+((r[1][1]-r[1][0])/2)]], //Top left quadrant
      [[r[0][0]+((r[0][1]-r[0][0])/2)+1, r[0][1]], [r[1][0], r[1][0]+((r[1][1]-r[1][0])/2)]], //Top right quadrant
      [[r[0][0], r[0][0]+((r[0][1]-r[0][0])/2)],   [r[1][0]+((r[1][1]-r[1][0])/2)+1, r[1][1]]], //Bottom left quadrant
      [[r[0][0]+((r[0][1]-r[0][0])/2)+1, r[0][1]], [r[1][0]+((r[1][1]-r[1][0])/2)+1, r[1][1]]]  //Bottom right quadrant
    ]
  }
  fn within_range(r: &[[usize; 2]; 2], x: usize, y: usize) -> bool {
    x >= r[0][0] && x <= r[0][1] && y >= r[1][0] && y <= r[1][1]
  }
  fn ones_in_range(bits: &BitVec, begin: usize, end: usize) -> usize {
    bits[begin..end].iter().fold(0, |total, bit| total + bit as usize)
  }
  /* Unit Tests */
  #[cfg(test)]
  pub mod unit_tests {
    use super::*;
    #[test]
    fn to_4_subranges_0() {
      let ranges = [[[0, 7], [0, 7]], [[4, 7], [0, 3]], [[8, 15], [8, 15]]];
      let subranges = [
        [[[0, 3], [0, 3]], [[4, 7], [0, 3]], [[0, 3], [4, 7]], [[4, 7], [4, 7]]],
        [[[4, 5], [0, 1]], [[6, 7], [0, 1]], [[4, 5], [2, 3]], [[6, 7], [2, 3]]],
        [[[8, 11], [8, 11]], [[12, 15], [8, 11]], [[8, 11], [12, 15]], [[12, 15], [12, 15]]]
      ];
      for i in 0..ranges.len() {
        assert_eq!(to_4_subranges(ranges[i]), subranges[i]);
      }
    }
    #[test]
    fn within_range_0() {
      let coords = [[0, 0], [5, 6], [87, 2],[5, 5]];
      let ranges = [[[0, 3], [0, 3]], [[0, 7], [0, 7]], [[50, 99], [0, 49]], [[5, 9], [5, 9]]];
      for i in 0..coords.len() {
        assert!(within_range(&ranges[i], coords[i][0], coords[i][1]));
      }
    }
    #[test]
    fn ones_in_range_0() {
      let ranges = [
        bitvec![0,1,1,1,0,0,1,0,1,1,0,0],
        bitvec![0,0,0,0,0,0,1],
        bitvec![0,1,1,1,1,1,1,0,1,0,0,1]
      ];
      let num_ones = [6, 1, 8];
      for i in 0..ranges.len() {
        assert_eq!(ones_in_range(&ranges[i], 0, ranges[i].len()), num_ones[i]);
      }
    }
    #[test]
    fn stem_layer_start_0() {
      let tree = K2Tree::test_tree();
      assert_eq!(tree.layer_start(0), 0);
      assert_eq!(tree.layer_start(1), 4);
    }
    #[test]
    fn stem_layer_len_0() {
      let tree = K2Tree::test_tree();
      assert_eq!(tree.layer_len(0), 4);
      assert_eq!(tree.layer_len(1), 12);
    }
    #[test]
    fn leaf_start_0() {
      let tree = K2Tree::test_tree();
      assert_eq!(tree.leaf_start(4), Ok(0));
      assert_eq!(tree.leaf_start(5), Ok(4));
      assert_eq!(tree.leaf_start(7), Ok(8));
      assert_eq!(tree.leaf_start(8), Ok(12));
      assert_eq!(tree.leaf_start(12), Ok(16));
      assert_eq!(tree.leaf_start(9), Err(()));
    }
    #[test]
    fn child_stem_0() {
      let tree = K2Tree::test_tree();
      assert_eq!(tree.child_stem(0, 0, 0), Err(()));
      assert_eq!(tree.child_stem(0, 0, 1), Ok(4));
      assert_eq!(tree.child_stem(0, 0, 2), Ok(8));
      assert_eq!(tree.child_stem(0, 0, 3), Ok(12));
      assert_eq!(tree.child_stem(1, 4, 0), Err(()));
    }
    #[test]
    fn parent_stem_0() {
      let tree = K2Tree::test_tree();
      assert_eq!(tree.parent_stem(4), 0);
      assert_eq!(tree.parent_stem(8), 0);
      assert_eq!(tree.parent_stem(12), 0);
    }
    #[test]
    fn parent_bit_0() {
      let tree = K2Tree::test_tree();
      assert_eq!(tree.parent_bit(4), 1);
      assert_eq!(tree.parent_bit(8), 2);
      assert_eq!(tree.parent_bit(12), 3);
    }
    #[test]
    fn bruh() {
      let mut tree = K2Tree::test_tree();
      println!("{:#?}", tree);
      tree.set_bit(4, 5, false);
      println!("{:#?}", tree);
      tree.set_bit(5, 4, false);
      println!("{:#?}", tree);
      tree.set_bit(0, 4, false);
      println!("{:#?}", tree);
      tree.set_bit(0, 0, true);
      println!("{:#?}", tree);
      tree.set_bit(0, 1, true);
      println!("{:#?}", tree);
      tree.set_bit(7, 7, true);
      println!("{:#?}", tree);
      tree.set_bit(5, 4, true);
      println!("{:#?}", tree);
    }
  }
}