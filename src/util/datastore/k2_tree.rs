extern crate bitvec;
extern crate serde;
extern crate rand;

use rand::prelude::*;

use bitvec::{prelude::bitvec, vec::BitVec};
use serde::{
  Serialize,
  Deserialize,
  Serializer,
  ser::SerializeStruct
};

pub type BitMatrix = Vec<BitVec>;

#[derive(Debug, Clone, Hash)]
pub struct K2Tree {
  matrix_width: usize,
  k: usize, //k^2 == number of submatrices in each stem/leaf
  max_slayers: usize, //max stem layers
  slayer_starts: Vec<usize>, //stem layer starts
  stems: BitVec,
  stem_to_leaf: Vec<usize>,
  leaves: BitVec,
}

impl K2Tree {
  pub fn heapsize(&self) -> usize {
    let mut size: usize = std::mem::size_of_val(self);
    size += std::mem::size_of::<usize>() * self.slayer_starts.len();
    size += self.stems.len() / 8;
    size += std::mem::size_of::<usize>() * self.stem_to_leaf.len();
    size += self.leaves.len() / 8;
    size
  }
}

/* Public */
impl K2Tree {
  /* Creation */
  pub fn new() -> Self {
    /* For now fix k as 2, further work to make it user-defined */
    let k: usize = 2;
    let mw = k.pow(3);
    K2Tree {
      matrix_width: mw,
      k: k,
      max_slayers: (mw as f64).log(k as f64) as usize - 1,
      slayer_starts: vec![0],
      stems: bitvec![0; k*k],
      stem_to_leaf: Vec::new(),
      leaves: BitVec::new(),
    }
  }
  pub fn test_tree() -> Self {
    K2Tree {
      matrix_width: 8,
      k: 2,
      max_slayers: 2,
      slayer_starts: vec![0, 4],
      stems:  bitvec![0,1,1,1, 1,1,0,1, 1,0,0,0, 1,0,0,0],
      stem_to_leaf: vec![0, 1, 3, 4, 8],
      leaves: bitvec![0,1,1,0, 0,1,0,1, 1,1,0,0, 1,0,0,0, 0,1,1,0],
    }
  }
  /* Operation */
  pub fn is_empty(&self) -> bool {
    if ones_in_range(&self.leaves, 0, self.leaves.len()) == 0 {
      true
    }
    else {
      false
    }
  }
  pub fn get(&self, x: usize, y: usize) -> Result<bool, ()> {
    if x >= self.matrix_width || y >= self.matrix_width { return Err(()) }
    /* Assuming k=2 */
    match self.matrix_bit(x, y, self.matrix_width)? {
      DescendResult::Leaf(leaf_start, leaf_range) => {
        if leaf_range[0][1] - leaf_range[0][0] != 1
        || leaf_range[0][1] - leaf_range[0][0] != 1 {
          /* ERROR: Final submatrix isn't 2x2 so can't be a leaf */
        }
        if x == leaf_range[0][0] {
          if y == leaf_range[1][0] { Ok(self.leaves[leaf_start]) }
          else { Ok(self.leaves[leaf_start+2]) }
        }
        else {
          if y == leaf_range[1][0] { Ok(self.leaves[leaf_start+1]) }
          else { Ok(self.leaves[leaf_start+3]) }
        }
      },
      DescendResult::Stem(_, _) => Ok(false),
    }
  }
  pub fn get_row(&self, y: usize) -> Result<BitVec, ()> {
    let mut ret_v = BitVec::new();
    for x in 0..self.matrix_width-1 {
      ret_v.push(self.get(x, y)?);
    }
    Ok(ret_v)
  }
  pub fn get_column(&self, x: usize) -> Result<BitVec, ()> {
    let mut ret_v = BitVec::new();
    for y in 0..self.matrix_width-1 {
      ret_v.push(self.get(x, y)?);
    }
    Ok(ret_v)
  }
  pub fn set(&mut self, x: usize, y: usize, state: bool) -> Result<(), ()> {
    /* Assuming k=2 */
    match self.matrix_bit(x, y, self.matrix_width)? {
      DescendResult::Leaf(leaf_start, leaf_range) => {
        if leaf_range[0][1] - leaf_range[0][0] != 1
        || leaf_range[0][1] - leaf_range[0][0] != 1 {
          /* ERROR: Final submatrix isn't a 2 by 2 so can't be a leaf */
          return Err(())
        }
        /* Set the bit in the leaf to the new state */
        if x == leaf_range[0][0] {
          if y == leaf_range[1][0] { self.leaves.set(leaf_start, state); }
          else { self.leaves.set(leaf_start+2, state); }
        }
        else {
          if y == leaf_range[1][0] { self.leaves.set(leaf_start+1, state); }
          else { self.leaves.set(leaf_start+3, state); }
        }
        /* If leaf is now all 0's, remove leaf and alter rest of struct to reflect changes.
        Loop up the stems changing the parent bits to 0's and removing stems that become all 0's */
        if ones_in_range(&self.leaves, leaf_start, leaf_start+3) == 0 {
          /* - Remove the leaf
              - Use stem_to_leaf to find the dead leaf's parent bit
              - Remove the elem from stem_to_leaf that mapped to dead leaf
              - Set parent bit to 0, check if stem now all 0's
              - If all 0's:
              - - Remove stem
              - - Alter layer_starts if needed
              - - Find parent bit and set to 0
              - - Repeat until reach stem that isn't all 0's or reach stem layer 0 */
          remove_block(&mut self.leaves, leaf_start, 4)?;
          let stem_bit_pos = self.stem_to_leaf[leaf_start/4];
          self.stem_to_leaf.remove(leaf_start/4);
          if self.stem_to_leaf.len() == 0 {
            /* If no more leaves, then remove all stems immediately
            and don't bother with complex stuff below */
            self.stems = bitvec![0,0,0,0];
            self.slayer_starts = vec![0];
            return Ok(())
          }
          let layer_start = self.slayer_starts[self.slayer_starts.len()-1];
          self.stems.set(layer_start + stem_bit_pos, false); //Dead leaf parent bit = 0
          let mut curr_layer = self.slayer_starts.len()-1;
          let mut stem_start = layer_start + block_start(stem_bit_pos, 4);
          while curr_layer > 0 
          && ones_in_range(&self.stems, stem_start, stem_start+3) == 0 {
            for layer_start in &mut self.slayer_starts[curr_layer+1..] {
              *layer_start -= 1; //Adjust lower layer start positions to reflect removal of stem
            }
            let (parent_stem_start, bit_offset) = self.parent(stem_start);
            remove_block(&mut self.stems, stem_start, 4)?;
            self.stems.set(parent_stem_start + bit_offset, false);
            stem_start = parent_stem_start;
            curr_layer -= 1;
          }
        }
      },
      DescendResult::Stem(mut stem_start, mut stem_range) if state => {
        /* Descend returning Stem means no Leaf containing bit at (x, y),
        must be located in a submatrix of all 0's.
        If state = false: do nothing 
        If state = true:
          - Construct needed stems until reach final layer
          - Construct leaf corresponding to range containing (x, y)
          - Set bit at (x, y) to 1 */
        let mut layer_starts_len = self.slayer_starts.len();
        let mut layer = self.layer_from_range(stem_range);
        let mut subranges: [Range; 4];
        /* Create correct stems in layers on the way down to the final layer,
        which points to the leaves */
        while layer < self.max_slayers-1 {
          subranges = to_4_subranges(stem_range);
          for child_pos in 0..4 {
            if within_range(&subranges[child_pos], x, y) {
              /* Change bit containing (x, y) to 1 */
              self.stems.set(stem_start + child_pos, true);
              /* If we're not at max possible layer, but at the lowest
              but at the lowest existing layer: Create new layer before
              adding new stem to it.
              Otherwise: Find the correct position to add the new stem
              in the child layer. */
              if layer == layer_starts_len-1 {
                stem_start = self.stems.len();
                self.slayer_starts.push(stem_start);
                layer_starts_len += 1;
              }
              else {
                stem_start = self.child_stem(layer, stem_start, child_pos)?;
              }
              /* We're now working on the child layer */
              layer += 1;
              stem_range = subranges[child_pos];
              insert_block(&mut self.stems, stem_start, 4)?;
              /* If there are layers after the one we just insert a stem
              into: Increase the layer_starts for them by 4 to account for
              the extra stem */
              if layer+1 <= layer_starts_len {
                for layer_start in &mut self.slayer_starts[layer+1..layer_starts_len] {
                  *layer_start += 4;
                }
              }
              break //We've found the child, skip to next layer
            }
          }
        }
        /* We're at the final stem layer */
        subranges = to_4_subranges(stem_range);
        for child_pos in 0..4 {
          if within_range(&subranges[child_pos], x, y) {
            /* Keep track of whether this stem is freshly created (all 0000s) */
            let fresh_stem: bool = ones_in_range(&self.stems, stem_start, stem_start+4) == 0;
            /* Set the correct stem bit to 1 */
            self.stems.set(stem_start + child_pos, true);
            /* Get the bit position within the final stem layer,
            find the position in stem_to_leaf to insert the linking elem,
            insert linking elem */
            let layer_bit_pos = (stem_start + child_pos) - self.slayer_starts[layer_starts_len-1];
            let mut stem_to_leaf_pos: usize = 0;
            while stem_to_leaf_pos < self.stem_to_leaf.len()
            && self.stem_to_leaf[stem_to_leaf_pos] < (layer_bit_pos/4)*4 {
              //TODO: Clean up and make a note properly explaining why
              //      we need to times then divide layer_bit_pos by 4
              //      baso, we want to floor it to the nearest mult of 4
              //      to get the beginning pos of the block we just
              //      inserted into last layer of stems.
              stem_to_leaf_pos += 1;
            }
            self.stem_to_leaf.insert(stem_to_leaf_pos, layer_bit_pos);
            /* If stem is fresh, increase bit positions in stem_to_leaf
            after the new elem by 4 to account for the new stem before them */
            /*
              NOTE: THERES a fucking problem while updating the stem_to_leaf
              From what it looks like:
                When stem bit 1 is mapped to something, but then insert a stem block before
                it and map new stem bit 3 to a leaf, the algorithm thinks that the new
                3 is after the old 1, when it isnt anymore, so doesnt update it.
                <<When inserting a new stem into the final layer, update stem_to_leafs to
                match it>>
            */
            if fresh_stem {
              let stem_to_leaf_len = self.stem_to_leaf.len();
              for parent_bit_pos in &mut self.stem_to_leaf[stem_to_leaf_pos+1..stem_to_leaf_len] {
                *parent_bit_pos += 4;
              }
            }
            /* Create new leaf of all 0's */
            let leaf_start = stem_to_leaf_pos * 4;
            insert_block(&mut self.leaves, leaf_start, 4)?;
            /* Change bit at (x, y) to 1 */
            let leaf_range = subranges[child_pos];
            if x == leaf_range[0][0] {
              if y == leaf_range[1][0] { self.leaves.set(leaf_start, true); }
              else { self.leaves.set(leaf_start+2, true); }
            }
            else {
              if y == leaf_range[1][0] { self.leaves.set(leaf_start+1, true); }
              else { self.leaves.set(leaf_start+3, true); }
            }
            return Ok(())
          }
        }
      }
      _ => {},
    };
    Ok(())
  }
  /* Information */
  pub fn matrix_width(&self) -> usize {
    self.matrix_width
  }
  pub fn k(&self) -> usize {
    self.k
  }
  /* Iteration */
  pub fn stems(&self) -> Stems {
    Stems {
      tree: &self,
      pos: 0,
      layer: 0,
      stem: 0,
      bit: 0,
    }
  }
  pub fn stems_raw(&self) -> StemsRaw {
    StemsRaw {
      stems: &self.stems,
      pos: 0,
    }
  }
  pub fn leaves(&self) -> Leaves {
    Leaves {
      tree: &self,
      pos: 0,
    }
  }
  pub fn leaves_raw(&self) -> LeavesRaw {
    LeavesRaw {
      leaves: &self.leaves,
      pos: 0,
    }
  }
  /* Mutation */
  pub fn grow(&mut self) {
    self.matrix_width *= self.k;
    self.max_slayers += 1;
    if self.leaves.len() > 0  {
      /* Only insert the extra layers etc. if the
      tree isn't all 0s */
      for slayer_start in &mut self.slayer_starts {
        *slayer_start += 4;
      }
      self.slayer_starts.insert(0, 0);
      /* Insert 1000 to beginning of stems */
      for _ in 0..3 { self.stems.insert(0, false); }
      self.stems.insert(0, true);
    }
  }
  pub fn shrink_if_possible(&mut self) {
    match self.shrink() {
      _ => return,
    }
  }
  pub fn shrink(&mut self) -> Result<(), ()> {
    // TODO: Add proper errors
    if self.matrix_width <= self.k.pow(3) {
      /* Can't shrink beyond the minimum useful,
      redundant cells in a too-large matrix is negligible
      anyway cause they'll be compressed out as 0s by the
      tree */
      return Err(())
    }
    else if &self.stems[0..4] != bitvec![1,0,0,0] {
      /* Shrinking would lose information, can't have that */
      return Err(())
    }
    self.matrix_width /= self.k;
    self.max_slayers -= 1;
    self.slayer_starts.remove(0);
    for slayer_start in &mut self.slayer_starts {
      *slayer_start -= 4;
    }
    /* Remove top layer stem */
    for _ in 0..4 { self.stems.remove(0); }
    Ok(())
  }
  pub unsafe fn shrink_unchecked(&mut self) {
    self.matrix_width /= self.k;
    self.max_slayers -= 1;
    self.slayer_starts.remove(0);
    for slayer_start in &mut self.slayer_starts {
      *slayer_start -= 4;
    }
    /* Remove top layer stem */
    for _ in 0..4 { self.stems.remove(0); }
  }
  /* To / From */
  pub fn into_matrix(self) -> BitMatrix { unimplemented!() }
  pub fn to_matrix(&self) -> BitMatrix { unimplemented!() }
  pub fn from_matrix(m: BitMatrix) -> Result<Self, ()> {
    let mut tree = K2Tree::new();
    for x in 0..m.len() {
      for y in one_positions(&m[x]).into_iter() {
        tree.set(x, y, true)?;
      }
    }
    Ok(tree)
  }
  /* Serialization / Deserialization */
  pub fn to_json(&self) { unimplemented!() }
  pub fn into_json(self) { unimplemented!() }
}

/* Iterators */
pub struct StemBit {
  value: bool,
  layer: usize,
  stem: usize,
  bit: usize,
}
pub struct LeafBit {
  value: bool,
  x: usize,
  y: usize,
}
pub struct Stems<'a> {
  tree: &'a K2Tree,
  pos: usize,
  layer: usize,
  stem: usize,
  bit: usize,
}
impl<'a> Iterator for Stems<'a> {
  type Item = StemBit;
  fn next(&mut self) -> Option<Self::Item> {
    if self.pos >= self.tree.stems.len() {
      return None
    }
    let value = self.tree.stems[self.pos];
    let ret_v = Some(StemBit {
      value: value,
      layer: self.layer,
      stem: self.stem,
      bit: self.bit,
    });
    self.bit = (self.bit + 1) % 4;
    if self.bit == 0 { self.stem = (self.stem + 1) % self.tree.layer_len(self.layer); }
    if self.stem == 0 { self.layer += 1; }
    ret_v
  }
}
pub struct Leaves<'a> {
  tree: &'a K2Tree,
  pos: usize,
}
impl<'a> Iterator for Leaves<'a> {
  type Item = LeafBit;
  fn next(&mut self) -> Option<Self::Item> {
    /* Need get_coords(bit_pos) function for leaf bits for this */
    unimplemented!()
  }
}
pub struct StemsRaw<'a> {
  stems: &'a BitVec,
  pos: usize,
}
impl<'a> Iterator for StemsRaw<'a> {
  type Item = bool;
  fn next(&mut self) -> Option<Self::Item> {
    if self.pos >= self.stems.len() {
      return None
    }
    self.pos += 1;
    Some(self.stems[self.pos])
  }
}
pub struct LeavesRaw<'a> {
  leaves: &'a BitVec,
  pos: usize,
}
impl<'a> Iterator for LeavesRaw<'a> {
  type Item = bool;
  fn next(&mut self) -> Option<Self::Item> {
    if self.pos >= self.leaves.len() {
      return None
    }
    self.pos += 1;
    Some(self.leaves[self.pos])
  }
}

/* Std Traits */
impl core::fmt::Display for K2Tree {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    if self.leaves.len() == 0 { return write!(f, "[0000]") }
    let mut s = String::new();
    let mut i: usize = 1;
    for layer_num in 0..self.slayer_starts.len() {
      for bit_pos in self.layer_start(layer_num)..self.layer_start(layer_num+1) {
        if self.stems[bit_pos] {
          s.push('1');
        }
        else {
          s.push('0');
        }
        if i == self.k*self.k
        && (bit_pos - self.layer_start(layer_num)) < self.layer_len(layer_num)-1 {
          s.push(',');
          i = 1;
        } 
        else {
          i += 1;
        }
      }
      i = 1;
      s.push_str("::");
    }
    i = 1;
    for bit_pos in 0..self.leaves.len() {
      if self.leaves[bit_pos] {
        s.push('1');
      }
      else {
        s.push('0');
      }
      if i == self.k*self.k
      && bit_pos < self.leaves.len()-1 {
        s.push(',');
        i = 1;
      } 
      else {
        i += 1;
      }
    }
    write!(f, "[{}]", s)
  }
}
impl PartialEq for K2Tree {
  fn eq(&self, other: &Self) -> bool {
    self.k == other.k
    && self.matrix_width == other.matrix_width
    && self.stems == other.stems
    && self.leaves == self.leaves
  }
}
impl Eq for K2Tree {}
impl Default for K2Tree {
  fn default() -> Self {
    Self::new()
  }
}
impl std::convert::TryFrom<BitMatrix> for K2Tree {
  type Error = &'static str;
  fn try_from(bit_matrix: BitMatrix) -> Result<Self, Self::Error> {
    /* Implement error checking here */
    Ok(Self::from_matrix(bit_matrix).unwrap())
  }
}
impl std::convert::TryFrom<Vec<Vec<bool>>> for K2Tree {
  type Error = &'static str;
  fn try_from(matrix: Vec<Vec<bool>>) -> Result<Self, Self::Error> {
    /* Implement error checking here */
    Ok(Self::from_matrix(
      matrix.into_iter()
            .map(|v| {
              let mut bv = BitVec::new();
              for bit in v.into_iter() {
                bv.push(bit);
              }
              bv
            }).collect()
      ).unwrap()
    )
  }
}
impl Serialize for K2Tree {
  fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    let mut state = serializer.serialize_struct("K2Tree", 7)?;
    state.serialize_field("matrix_width", &self.matrix_width)?;
    state.serialize_field("k", &self.k)?;
    state.serialize_field("stems", &self.stems.clone().into_vec())?;
    state.serialize_field("leaves", &self.leaves.clone().into_vec())?;
    state.end()
  }
}

/* Private */
type Range = [[usize; 2]; 2];
enum DescendResult {
  Leaf(usize, Range), //leaf_start, leaf_range
  Stem(usize, Range), //stem_start, stem_range
}
struct DescendEnv {
  /* Allows for descend to be recursive without parameter hell */
  x: usize,
  y: usize,
  slayer_max: usize,
}
impl K2Tree {
  fn layer_from_range(&self, r: Range) -> usize {
    let r_width = r[0][1]-r[0][0]+1;
    ((self.matrix_width as f64).log(self.k as f64) as usize)
    - ((r_width as f64).log(self.k as f64) as usize)
  }
  fn matrix_bit(&self, x: usize, y: usize, m_width: usize) -> Result<DescendResult, ()> {
    let env = DescendEnv {
      x: x,
      y: y,
      slayer_max: self.slayer_starts.len()-1,
    };
    self.descend(&env, 0, 0, [[0, m_width-1], [0, m_width-1]])
  }
  fn descend(&self, env: &DescendEnv, layer: usize, stem_pos: usize, range: Range) -> Result<DescendResult, ()> {
    let subranges = to_4_subranges(range);
    for (child_pos, child) in self.stems[stem_pos..stem_pos+4].iter().enumerate() {
      if within_range(&subranges[child_pos], env.x, env.y) {
        if !child { return Ok(DescendResult::Stem(stem_pos, range)) } //The bit exists within a range that has all zeros
        else if layer == env.slayer_max {
          return Ok(DescendResult::Leaf(self.leaf_start(stem_pos + child_pos).unwrap(), subranges[child_pos]))
        }
        else {
          return self.descend(env,
                              layer+1,
                              self.child_stem(layer, stem_pos, child_pos).unwrap(),
                              subranges[child_pos])
        }
      }
    }
    Err(()) //Should never return this
  }
  fn num_stems_before_child(&self, bit_pos: usize, layer: usize) -> usize {
    let layer_start = self.layer_start(layer);
    ones_in_range(&self.stems, layer_start, bit_pos)
  }
  fn layer_start(&self, l: usize) -> usize {
    if l == self.slayer_starts.len() {
      self.stems.len()
    }
    else {
      self.slayer_starts[l]
    }
  }
  fn layer_len(&self, l: usize) -> usize {
    if l == self.slayer_starts.len()-1 {
      return self.stems.len() - self.slayer_starts[l]
    }
    self.slayer_starts[l+1] - self.slayer_starts[l]
  }
  fn leaf_start(&self, stem_bitpos: usize) -> Result<usize, ()> {
    if !self.stems[stem_bitpos] { return Err(()) }
    Ok(self.stem_to_leaf
            .iter()
            .position(|&n| n == (stem_bitpos - self.slayer_starts[self.slayer_starts.len()-1]))
            .unwrap()
            * 4)
  }
  fn child_stem(&self, layer: usize, stem_start: usize, nth_child: usize) -> Result<usize, ()> {
    if !self.stems[stem_start+nth_child]
    || layer == self.max_slayers-1 {
      /* If stem_bit is 0 or final stem layer, cannot have children */
      return Err(())
    }
    Ok(self.layer_start(layer+1)
    + (self.num_stems_before_child(stem_start+nth_child, layer) * 4))
  }
  fn parent(&self, stem_start: usize) -> (usize, usize) {
    /* Returns (stem_start, bit_offset) */
    if stem_start >= self.slayer_starts[1] {
      /* If stem isn't in layer 0, look for parent */
      let mut parent_layer_start = 0;
      let mut curr_layer_start = 0;
      for (i, layer_start) in (0..self.slayer_starts.len()).enumerate() {
        if i == self.slayer_starts.len()-1 {
          if stem_start >= self.slayer_starts[layer_start]
          && stem_start < self.stems.len() {
            parent_layer_start = self.slayer_starts[layer_start-1];
            curr_layer_start = self.slayer_starts[layer_start];
          }
        }
        else if stem_start >= self.slayer_starts[layer_start]
        && stem_start < self.slayer_starts[layer_start+1] {
          parent_layer_start = self.slayer_starts[layer_start-1];
          curr_layer_start = self.slayer_starts[layer_start];
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
      (((bit_pos_in_parent_stem_layer / 4) * 4) + parent_layer_start,
        bit_pos_in_parent_stem_layer % 4)
    }
    else {
      (std::usize::MAX, std::usize::MAX)
    }
  }
  fn parent_stem(&self, stem_start: usize) -> usize {
    self.parent(stem_start).0
  }
  fn parent_bit(&self, stem_start: usize) -> usize {
    let (stem_start, bit_offset) = self.parent(stem_start);
    stem_start + bit_offset
  }
}

/* Utils */
fn block_start(bit_pos: usize, block_len: usize) -> usize {
  (bit_pos / block_len) * block_len
}
fn remove_block(bit_vec: &mut BitVec, block_start: usize, block_len: usize) -> Result<(), ()> {
  if block_start >= bit_vec.len()
  || block_start % block_len != 0 {
    Err(())
  }
  else {
    Ok(for _ in 0..block_len { bit_vec.remove(block_start); })
  }
}
fn insert_block(bit_vec: &mut BitVec, block_start: usize, block_len: usize) -> Result<(), ()> {
  if block_start > bit_vec.len()
  || block_start % block_len != 0 {
    Err(())
  }
  else {
    Ok(for _ in 0..block_len { bit_vec.insert(block_start, false); })
  }
}
fn to_4_subranges(r: Range) -> [Range; 4] {
  [
    [[r[0][0], r[0][0]+((r[0][1]-r[0][0])/2)],   [r[1][0], r[1][0]+((r[1][1]-r[1][0])/2)]], //Top left quadrant
    [[r[0][0]+((r[0][1]-r[0][0])/2)+1, r[0][1]], [r[1][0], r[1][0]+((r[1][1]-r[1][0])/2)]], //Top right quadrant
    [[r[0][0], r[0][0]+((r[0][1]-r[0][0])/2)],   [r[1][0]+((r[1][1]-r[1][0])/2)+1, r[1][1]]], //Bottom left quadrant
    [[r[0][0]+((r[0][1]-r[0][0])/2)+1, r[0][1]], [r[1][0]+((r[1][1]-r[1][0])/2)+1, r[1][1]]]  //Bottom right quadrant
  ]
}
fn within_range(r: &Range, x: usize, y: usize) -> bool {
  x >= r[0][0] && x <= r[0][1] && y >= r[1][0] && y <= r[1][1]
}
fn ones_in_range(bits: &BitVec, begin: usize, end: usize) -> usize {
  bits[begin..end].iter().fold(0, |total, bit| total + bit as usize)
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
  fn flood() {
    let mut tree = K2Tree::new();
    for _ in 0..3 { tree.grow(); }
    dbg!(tree.matrix_width());
    let xs: Vec<usize> = vec![22, 22];
    let ys: Vec<usize> = vec![33, 22];
    for x in xs.into_iter() {
      for y in ys.iter() {
        tree.set(x, *y, true);
      }
    }
  }
  #[test]
  fn flood_2() {
    let mut rng = rand::thread_rng();
    let mut tree = K2Tree::new();
    for _ in 0..10 { tree.grow(); }
    dbg!(tree.matrix_width());
    let mut nums: Vec<[usize; 2]> = Vec::new();
    for _ in 0..500 {
      let x: usize = rng.gen_range(0, 512);
      let y: usize = rng.gen_range(0, 512);
      nums.push([x, y]);
      println!("{:?}", nums);
      tree.set(x, y, true);
    }
  }
  #[test]
  fn grow_0() {
    let mut tree = K2Tree::new();
    assert_eq!(tree, K2Tree {
      matrix_width: 8,
      k: 2,
      max_slayers: 2,
      slayer_starts: vec![0],
      stems: bitvec![0,0,0,0],
      stem_to_leaf: Vec::new(),
      leaves: BitVec::new(),
    });
    tree.grow();
    assert_eq!(tree, K2Tree {
      matrix_width: 16,
      k: 2,
      max_slayers: 3,
      slayer_starts: vec![0],
      stems: bitvec![0,0,0,0],
      stem_to_leaf: Vec::new(),
      leaves: BitVec::new(),
    });
  }
  #[test]
  fn grow_1() {
    let mut tree = K2Tree::new();
    assert_eq!(tree, K2Tree {
      matrix_width: 8,
      k: 2,
      max_slayers: 2,
      slayer_starts: vec![0],
      stems: bitvec![0,0,0,0],
      stem_to_leaf: Vec::new(),
      leaves: BitVec::new(),
    });
    for _ in 0..4 { tree.grow(); }
    assert_eq!(tree, K2Tree {
      matrix_width: 128,
      k: 2,
      max_slayers: 6,
      slayer_starts: vec![0],
      stems: bitvec![0,0,0,0],
      stem_to_leaf: Vec::new(),
      leaves: BitVec::new(),
    });
  }
  #[test]
  fn test_send() {
      fn assert_send<T: Send>() {}
      assert_send::<K2Tree>();
  }
  #[test]
  fn test_sync() {
      fn assert_sync<T: Sync>() {}
      assert_sync::<K2Tree>();
  }
  #[test]
  fn from_matrix_0() {
    let m = vec![
      bitvec![0,0,0,0,1,0,0,0],
      bitvec![0; 8],
      bitvec![0; 8],
      bitvec![0; 8],
      bitvec![0,1,0,0,0,1,0,0],
      bitvec![1,0,0,0,1,0,0,0],
      bitvec![0,0,1,0,0,0,0,0],
      bitvec![1,1,1,0,0,0,0,0],
    ];
    let tree = K2Tree {
      matrix_width: 8,
      k: 2,
      max_slayers: 2,
      slayer_starts: vec![0, 4],
      stems:  bitvec![0,1,1,1, 1,1,0,1, 1,0,0,0, 1,0,0,0],
      stem_to_leaf: vec![0, 1, 3, 4, 8],
      leaves: bitvec![0,1,1,0, 0,1,0,1, 1,1,0,0, 1,0,0,0, 0,1,1,0]
    };
    assert_eq!(tree, K2Tree::from_matrix(m).unwrap());
  }
  #[test]
  fn one_positions_0() {
    let bv = bitvec![0,1,0,1,0,1,0,0,0,1];
    assert_eq!(vec![1,3,5,9], one_positions(&bv));
  }
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
  fn set_0() {
    let mut tree = K2Tree::test_tree();
    assert_eq!(tree.leaves[18], true);
    tree.set(4, 5, false);
    assert_eq!(tree.leaves[18], false);
  }
  #[test]
  fn set_1() {
    let mut tree = K2Tree::test_tree();
    assert_eq!(tree.stems, bitvec![0,1,1,1,1,1,0,1,1,0,0,0,1,0,0,0]);
    assert_eq!(tree.leaves, bitvec![0,1,1,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,1,0]);
    assert_eq!(tree.stem_to_leaf, vec![0, 1, 3, 4, 8]);
    tree.set(4, 5, false);
    tree.set(5, 4, false);
    assert_eq!(tree.stems, bitvec![0,1,1,0,1,1,0,1,1,0,0,0]);
    assert_eq!(tree.leaves, bitvec![0,1,1,0,0,1,0,1,1,1,0,0,1,0,0,0]);
    assert_eq!(tree.stem_to_leaf, vec![0, 1, 3, 4]);
  }
  #[test]
  fn set_2() {
    let mut tree = K2Tree::test_tree();
    assert_eq!(tree.stems, bitvec![0,1,1,1,1,1,0,1,1,0,0,0,1,0,0,0]);
    assert_eq!(tree.leaves, bitvec![0,1,1,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,1,0]);
    assert_eq!(tree.stem_to_leaf, vec![0, 1, 3, 4, 8]);
    tree.set(4, 5, false);
    tree.set(5, 4, false);
    tree.set(0, 0, true);
    assert_eq!(tree.stems, bitvec![1,1,1,0,1,0,0,0,1,1,0,1,1,0,0,0]);
    assert_eq!(tree.leaves, bitvec![1,0,0,0,0,1,1,0,0,1,0,1,1,1,0,0,1,0,0,0]);
    assert_eq!(tree.stem_to_leaf, vec![0, 4, 5, 7, 8]);
  }
  #[test]
  fn show_me_the_changes() {
    let mut tree = K2Tree::test_tree();
    println!("{:#?}", tree);
    tree.set(4, 5, false);
    println!("{:#?}", tree);
    tree.set(5, 4, false);
    println!("{:#?}", tree);
    tree.set(0, 4, false);
    println!("{:#?}", tree);
    tree.set(0, 0, true);
    println!("{:#?}", tree);
    tree.set(0, 1, true);
    println!("{:#?}", tree);
    tree.set(7, 7, true);
    println!("{:#?}", tree);
    tree.set(5, 4, true);
    println!("{:#?}", tree);

    println!("{}", tree);

    let json = serde_json::to_string(&tree).unwrap();
    println!("{}", json);
  }
}