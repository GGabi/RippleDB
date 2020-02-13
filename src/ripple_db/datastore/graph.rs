extern crate bimap;
extern crate bitvec;
extern crate serde;
extern crate futures;
extern crate num_cpus;

use bimap::BiBTreeMap;
use bitvec::vec::BitVec;
use serde::{
  Serialize,
  Deserialize,
  ser::SerializeStruct,
  de::{self, Visitor, MapAccess}
};

use crate::ripple_db::{
  RdfNode, RdfTriple,
  datastore::k2_tree::{self, K2Tree},
  rdf::{
    query::{Sparql, QueryUnit},
    builder::RdfBuilder,
  }
};

/*
  TODO:
   x Add support for graph to store BlankNode and LangEncodedNodes rather than jus raw string
   x Add support for TriplesParser to not ignore every triple not containing basic strings
   x Add an iterator for Graph which produces Triples of rich types described above
   x x Implement Leaves iterator on K2Tree
   - Add a method to export the graph contents, produced from Graph::Iter, to RDF
*/

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
  dict: BiBTreeMap<RdfNode, usize>,
  pred_tombstones: Vec<usize>,
  predicates: BiBTreeMap<RdfNode, usize>,
  slices: Vec<Option<Box<K2Tree>>>,
  persist_location: Option<String>,
}

/* Public */
impl Graph {
  /* Constructors */
  pub fn new() -> Self {
    Graph {
      dict_max: 0,
      dict_tombstones: Vec::new(),
      dict: BiBTreeMap::new(),
      pred_tombstones: Vec::new(),
      predicates: BiBTreeMap::new(),
      slices: Vec::new(),
      persist_location: None,
    }
  }
  pub fn from_backup(path: &str) -> Result<Self, std::io::Error> {
    // unimplemented!();
    /* Private trait impl */
    impl<'de> Deserialize<'de> for Graph {
      fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "camelCase")]
        enum Field {
          DictMax,
          DictTombstones,
          Dict,
          PredTombstones,
          Predicates,
          PersistLocation
        }
        struct GraphVisitor;
        impl<'de> Visitor<'de> for GraphVisitor {
          type Value = Graph;
          fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("struct Graph")
          }
          fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<Graph, V::Error> {
            let mut dict_max = None;
            let mut dict_tombstones = None;
            let mut dict = None;
            let mut pred_tombstones = None;
            let mut predicates = None;
            let mut persist_location = None;
            while let Some(key) = map.next_key()? {
              match key {
                Field::DictMax => {
                  if dict_max.is_some() {
                      return Err(de::Error::duplicate_field("dictMax"));
                  }
                  dict_max = Some(map.next_value()?);
                }
                Field::DictTombstones => {
                  if dict_tombstones.is_some() {
                    return Err(de::Error::duplicate_field("dictTombstones"));
                  }
                  dict_tombstones = Some(map.next_value()?);
                }
                Field::Dict => {
                  if dict.is_some() {
                    return Err(de::Error::duplicate_field("dict"));
                  }
                  dict = Some(map.next_value::<Vec<(RdfNode, usize)>>()?);
                }
                Field::PredTombstones => {
                  if pred_tombstones.is_some() {
                      return Err(de::Error::duplicate_field("predTombstones"));
                  }
                  pred_tombstones = Some(map.next_value()?);
                }
                Field::Predicates => {
                  if predicates.is_some() {
                    return Err(de::Error::duplicate_field("predicates"));
                  }
                  predicates = Some(map.next_value::<Vec<(RdfNode, usize)>>()?);
                }
                Field::PersistLocation => {
                  if persist_location.is_some() {
                    return Err(de::Error::duplicate_field("persistLocation"));
                  }
                  persist_location = Some(map.next_value()?);
                }
              }
            }
            let dict_max = dict_max.ok_or_else(|| de::Error::missing_field("dictMax"))?;
            let dict_tombstones = dict_tombstones.ok_or_else(|| de::Error::missing_field("dictTombstones"))?;
            let dict = dict.ok_or_else(|| de::Error::missing_field("dict"))?;
            let pred_tombstones = pred_tombstones.ok_or_else(|| de::Error::missing_field("predTombstones"))?;
            let predicates = predicates.ok_or_else(|| de::Error::missing_field("predicates"))?;
            let persist_location = persist_location.ok_or_else(|| de::Error::missing_field("persistLocation"))?;
            
            let mut final_dict: BiBTreeMap<RdfNode, usize> = BiBTreeMap::new();
            for (key, val) in dict.into_iter() {
              final_dict.insert(key, val);
            }
            let mut final_preds: BiBTreeMap<RdfNode, usize> = BiBTreeMap::new();
            for (key, val) in predicates.into_iter() {
              final_preds.insert(key, val);
            }

            Ok(Graph {
              dict_max: dict_max,
              dict_tombstones: dict_tombstones,
              dict: final_dict,
              pred_tombstones: pred_tombstones,
              predicates: final_preds,
              slices: Vec::new(),
              persist_location: persist_location
            })
          }
        }
        const FIELDS: &'static [&'static str] = &[
          "dict_max",
          "dict_tombstones",
          "dict",
          "pred_tombstones",
          "predicates",
          "persist_location"
        ];
        deserializer.deserialize_struct("Graph", FIELDS, GraphVisitor)
      }
    }
    /* Closure definitions */
    let read_json = |path_to_file: &std::path::Path| -> Result<String, std::io::Error> {
      use std::io::Read;
      let mut buf = String::new();
      std::fs::File::open(path_to_file)?.read_to_string(&mut buf)?;
      Ok(buf)
    };
    /* Function start */
    /* Define key filesystem locations */
    let root_dir = std::path::Path::new(path);
    let trees_dir = root_dir.join("trees");
    let head_file = root_dir.join("head.json");
    let dot_file = root_dir.join(".ripplebackup");
    /* Check that all files and dirs actually exist */
    if !root_dir.is_dir()
    || !trees_dir.is_dir()
    || !head_file.is_file()
    || !dot_file.is_file() { /* Oof */ }
    /* Build surface level of the Graph from root/head.json */
    let Graph {
      dict_max,
      dict_tombstones,
      dict,
      pred_tombstones,
      predicates,
      slices: _,
      persist_location: _
    } = serde_json::from_str::<Graph>(&read_json(&head_file)?)?;

    /* Build K2Trees from json files in root/trees/ */
    let mut slices: Vec<Option<Box<K2Tree>>> = Vec::new();
    for i in 0.. {
      if let Some(_) = predicates.get_by_right(&i) {
        let tree_json = read_json(&trees_dir.join(format!("{}.json", i)))?;
        slices.push(Some(Box::new(K2Tree::from_json(&tree_json)?)));
      }
      else if pred_tombstones.contains(&i) {
        slices.push(None);
      }
      else {
        break
      }
    }

    Ok(Graph {
      dict_max: dict_max,
      dict_tombstones: dict_tombstones,
      dict: dict,
      pred_tombstones: pred_tombstones,
      predicates: predicates,
      slices: slices,
      persist_location: Some(path.to_string()),
    })
  }
  pub fn from_rdf(path: &str) -> Result<Self, &str> {
    use crate::ripple_db::rdf::parser::ParsedTriples;
    /* Parse the RDF file at path */
    let ParsedTriples {
      dict_max,
      dict,
      pred_max: _,
      predicates,
      partitioned_triples,
    } = match ParsedTriples::from_rdf(path) {
      Ok(p_trips) => p_trips,
      Err(_) => return Err("error parsing triples"),
    };
    let num_slices = partitioned_triples.len();
    /* Sort the Triples */
    let sorted_trips: Vec<TripleSet> = sort_by_size(partitioned_triples);
    let total_trips = sorted_trips.iter().fold(0, |sum, triples| sum + triples.size);
    /* Find the TripleSet that contains the median Triple */
    let median_tripleset: usize = {
      let mid_triple = total_trips / 2;
      let mut triples_before_tripleset: usize = 0;
      let mut tripleset_index: usize = 0;
      loop {
        if triples_before_tripleset >= mid_triple { break tripleset_index }
        triples_before_tripleset += sorted_trips[tripleset_index].size;
        tripleset_index += 1;
      }
    };
    /* Define two distict groups of TripleSets, lower and upper,
    where lower contains all the triplesets smaller than the median
    tripleset and upper contains all that are larger. */
    let lower_range = &sorted_trips[0..median_tripleset];
    let upper_range = &sorted_trips[median_tripleset..];
    /* Each TripleSet corresponds to a unique K2Tree that needs to be constructed.
    Designate half the system's cpu-cores to build the largest K2Trees in parallel (upper_range)
    and the other half to build all the remaining smaller K2Trees (lower_range). If there
    are less larger K2Trees to build than half the designated cores, assign all unused
    to help build the smaller K2Trees. */
    let half_threads = num_cpus::get() / 2; //Half of available cores on the system
    let (num_upper_threads, num_uppers_per_thread) = {
      if upper_range.len() < half_threads {
        (upper_range.len(), 1)
      }
      else {
        (half_threads, upper_range.len() / half_threads)
      }
    };
    let (num_lower_threads, num_lowers_per_thread) = {
      let remaining_threads = half_threads + (half_threads - num_upper_threads);
      if lower_range.len() < remaining_threads {
        (lower_range.len(), 1)
      }
      else {
        (remaining_threads, lower_range.len() / remaining_threads)
      }
    };
    /* Start building K2Trees in parallel */
    let mut handles = Vec::new();
    /* Spawn upper threads */
    for thread_num in 0..num_upper_threads {
      let triplesets = if thread_num != num_upper_threads-1 {
        sorted_trips[
            (median_tripleset + (thread_num * num_uppers_per_thread))
            ..(median_tripleset + ((thread_num + 1) * num_uppers_per_thread))
          ].to_vec()
        }
        else {
          sorted_trips[(median_tripleset + (thread_num * num_uppers_per_thread))..].to_vec()
      };
      let dict_max = dict_max;
      handles.push(std::thread::spawn(move || build_slices(triplesets, dict_max)));
    }
    /* Spawn lower threads */
    for thread_num in 0..num_lower_threads {
      let triplesets = if thread_num != num_lower_threads-1 {
          sorted_trips[
            (thread_num * num_lowers_per_thread)
            ..((thread_num + 1) * num_lowers_per_thread)
          ].to_vec()
        }
        else {
          sorted_trips[(thread_num * num_lowers_per_thread)..median_tripleset].to_vec()
      };
      let dict_max = dict_max;
      handles.push(std::thread::spawn(move || build_slices(triplesets, dict_max)));
    }
    let mut slice_sets: Vec<Vec<Slice>> = Vec::new();
    for handle in handles { slice_sets.push(handle.join().unwrap()); }
    /* Check if every K2Tree was built successfully and 
    insert each one into the correct location in the Graph's
    slices field */
    let mut slices: Vec<Option<Box<K2Tree>>> = vec![None; num_slices];
    for Slice {
      predicate_index,
      tree
    } in slice_sets.into_iter().flatten() {
        slices[predicate_index] = Some(tree);
    }
    if slices.contains(&None) {
      return Err("a tree is dead")
    }
    Ok(Graph {
      dict_max: dict_max,
      dict_tombstones: Vec::new(),
      dict: dict,
      pred_tombstones: Vec::new(),
      predicates: predicates,
      slices: slices,
      persist_location: None,
    })
  }
  /*For even greater building performance get it to build the trees in the background and saved to files
    If the predicate isn't built yet on query, go build it, otherwise finish building the rest. */
  pub fn get(&self, query: &Sparql) -> Vec<RdfNode> {
    /* Assume only one variable */
    use std::collections::HashSet;
    use QueryUnit::{Var, Val};
    /* Util closures for later */
    let cond_to_qt = |cond: &[QueryUnit; 3]| {
      let mut qt: [Option<String>; 3] = [None, None, None];
      if let Val(s) = &cond[0] { qt[0] = Some(s.to_string()); }
      if let Val(p) = &cond[1] { qt[1] = Some(p.to_string()); }
      if let Val(o) = &cond[2] { qt[2] = Some(o.to_string()); }
      qt
    };
    let var_pos = |cond: &[QueryUnit; 3]| {
      match cond {
        [Var(_), _, _] => 0,
        [_, Var(_), _] => 1,
        [_, _, Var(_)] => 2,
        _ => 3
      }
    };
    /* Gather raw results from each condition to filter later */
    let mut results: Vec<(&[QueryUnit; 3], Vec<[usize; 3]>)> = Vec::new();
    for cond in query.conds.iter() {
      let qt = cond_to_qt(cond);
      let res = self.get_from_triple(qt);
      results.push((cond, res));
    }
    /* Filter results */
    let mut final_results: Vec<usize> = results[0].1.iter().map(|[s, p, o]|
      match var_pos(results[0].0) {
        0 => s.clone(),
        1 => p.clone(),
        2 => o.clone(),
        _ => std::usize::MAX,
      }
    ).collect();
    for (query_triple, qt_results) in &results[1..] {
      let qt_var_pos = var_pos(query_triple);
      let mut used_vars_vals: HashSet<usize> = HashSet::new();
      let mut vars_vals_to_remove: Vec<usize> = Vec::new();
      for (i, final_result) in final_results.iter().enumerate() {
        if !used_vars_vals.contains(final_result) {
          used_vars_vals.insert(final_result.clone());
          let filter_t = match qt_var_pos {
            0 => [Some(final_result.clone()), None, None],
            1 => [None, Some(final_result.clone()), None],
            2 => [None, None, Some(final_result.clone())],
            _ => [None, None, None],
          };
          if self.filter_triples(qt_results.clone(), filter_t).is_empty() {
            /* There was no match for this value of the variable from final_results in
            the query triple, so mark it to be removed at the end of this cycle */
            vars_vals_to_remove.push(i);
          }
        }
      }
      final_results = final_results.into_iter().enumerate().filter_map(|(i, res)| {
        if !vars_vals_to_remove.is_empty() {
          if i == vars_vals_to_remove[0] {
            vars_vals_to_remove.remove(0);
            None
          }
          else {
            Some(res)
          }
        }
        else {
          Some(res)
        }
      }).collect();
    }
    let ret: Vec<RdfNode> = final_results.into_iter().map(|n| {
      if var_pos(results[0].0) == 1 {
        self.predicates.get_by_right(&n).unwrap().clone()
      }
      else {
        self.dict.get_by_right(&n).unwrap().clone()
      }
    }).collect();
    ret
  }
  pub fn insert_triple(&mut self, val: RdfTriple) -> Result<(), ()> {

    let col = match self.dict.get_by_left(&val[0]) {
      Some(&col) => col,
      None => {
        if !self.dict_tombstones.is_empty() {
          let col = self.dict_tombstones[0];
          self.dict.insert(val[0].clone(), col);
          col
        }
        else {
          if self.dict_max != 0 { self.dict_max += 1; }
          self.dict.insert(val[0].clone(), self.dict_max);
          if !self.slices.is_empty() {
            for slice in &mut self.slices {
              match slice {
                Some(slice) if self.dict_max >= slice.matrix_width() => {
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
        if !self.dict_tombstones.is_empty() {
          let row = self.dict_tombstones[0];
          self.dict.insert(val[2].clone(), row);
          row
        }
        else {
          self.dict_max += 1;
          self.dict.insert(val[2].clone(), self.dict_max);
          if !self.slices.is_empty() {
            for slice in &mut self.slices {
              match slice {
                Some(slice) if self.dict_max >= slice.matrix_width() => {
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
          if !self.slices.is_empty() {
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
          if !self.pred_tombstones.is_empty() {
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
  pub fn remove_triple(&mut self, [subject, predicate, object]: &RdfTriple) -> Result<(), ()> {
    /* TODO: Add ability to shrink matrix_width for all slices if
    needed */
    let (subject_pos, object_pos, slice_pos) = match [
      self.dict.get_by_left(subject),
      self.dict.get_by_left(object),
      self.predicates.get_by_left(&predicate)] {
        [Some(&c), Some(&r), Some(&s)] => (c, r, s),
        _ => return Ok(())
    };
    let slice = if let Some(slice) = &mut self.slices[slice_pos] {
        slice
      }
      else {
        return Err(())
    };
    slice.set(subject_pos, object_pos, false)?;
    /* Check if we've removed all instances of a word.
    If we have: Remove from dictionaries and do other stuff */
    if slice.is_empty() {
      self.predicates.remove_by_left(&predicate);
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
  pub fn persist_to(&mut self, path: &str) -> Result<(), std::io::Error> {
    /* Define locations to persist to */
    let root_dir = std::path::Path::new(path);
    /* Save the location this Graph is persisted to */
    self.persist_location = Some(root_dir.to_str().unwrap().to_string());
    /* Do the saving */
    self.persist()
  }
  pub fn persist_location(&self) -> &Option<String> {
    &self.persist_location
  }
  pub fn persist(&self) -> Result<(), std::io::Error> {
    /* Only want to use this trait in this func, not public as it's not really
    "serializing" the Graph and would be confusing to users if the trait was
    publicly implemented */
    impl Serialize for Graph {
      fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("Graph", 6)?;
        state.serialize_field("dictMax", &self.dict_max)?;
        state.serialize_field("dictTombstones", &self.dict_tombstones)?;
        state.serialize_field("dict", &self.dict.iter().collect() as &Vec<(&RdfNode, &usize)>)?;
        state.serialize_field("predTombstones", &self.pred_tombstones)?;
        state.serialize_field("predicates", &self.predicates.iter().collect() as &Vec<(&RdfNode, &usize)>)?;
        state.serialize_field("persistLocation", &self.persist_location)?;
        state.end()
      }
    }
    // if let None = self.persist_location { return Err("yo") }
    let path = &self.persist_location.clone().unwrap();
    /* Define locations to persist to */
    let root_dir = std::path::Path::new(path);
    let trees_dir = root_dir.join("trees");
    let head_file = root_dir.join("head.json");
    let dot_file = root_dir.join(".ripplebackup");

    if root_dir.is_dir() && dot_file.is_file() {
      /* Graph's been saved here before, wipe the head_file
      and files in root/trees/ */
      std::fs::remove_file(&head_file)?;
      for entry in std::fs::read_dir(&trees_dir)? {
        let entry_path = entry?.path();
        if entry_path.is_file() {
          std::fs::remove_file(entry_path)?;
        }
      }
    }
    else {
      std::fs::create_dir(&root_dir)?;
      std::fs::create_dir(&trees_dir)?;
      std::fs::File::create(&dot_file)?;
    }
    /* Create an serialise Graph to root/head.json */
    std::fs::File::create(&head_file)?;
    std::fs::write(head_file, serde_json::to_string(self)?)?;

    /* Serialise each K2Tree and save to a json file in root/trees/,
    Name each K2Tree's file after it's corresponding's predicate's
    rhs value in self.predicates to aid reconstruction in future */
    for (i, slice) in self.slices.iter().enumerate() {
      if let Some(k2_tree) = slice {
        let tree_file = trees_dir.join(format!("{}.json", i));
        std::fs::File::create(&tree_file)?;
        std::fs::write(tree_file, k2_tree.to_json()?)?;
      }
    }

    Ok(())
  }
  pub fn iter(&self) -> Iter {
    let iter = match &self.slices.get(0) {
      Some(Some(slice)) => Some(slice.leaves()),
      _ => None,
    };
    Iter {
      graph: self,
      slice: 0,
      slice_iter: iter,
    }
  }
  pub fn into_iter(self) -> IntoIter {
    let iter = match &self.slices.get(0) {
      Some(Some(slice)) => Some(slice.clone().into_leaves()),
      _ => None,
    };
    IntoIter {
      graph: self,
      slice: 0,
      slice_iter: iter,
    }
  }
  pub fn to_rdf(&self, path: &str) -> Result<(), std::io::Error> {
    let buf = RdfBuilder::iter_to_rdf(self.iter());
    let file = std::path::Path::new(path);
    if !file.is_file() {
      std::fs::File::create(&file)?;
      std::fs::write(file, buf)?;
    }
    Ok(())
  }
  pub fn into_rdf(self, path: &str) -> Result<(), std::io::Error> {
    let buf = RdfBuilder::iter_to_rdf(self.into_iter());
    let file = std::path::Path::new(path);
    if !file.is_file() {
      std::fs::File::create(&file)?;
      std::fs::write(file, buf)?;
    }
    Ok(())
  }
}

/* Iterators */
pub struct Iter<'a> {
  graph: &'a Graph,
  slice: usize,
  slice_iter: Option<k2_tree::Leaves<'a>>,
}
impl<'a> Iterator for Iter<'a> {
  type Item = RdfTriple;
  fn next(&mut self) -> Option<Self::Item> {

    if self.slice == self.graph.slices.len() { return None }

    loop {
      let leaf = match &mut self.slice_iter {
        Some(iter) => {
          match iter.next() {
            Some(leaf) => Some(leaf.clone()),
            None => None,
          }
        },
        None => None,
      };
      match &leaf {
        Some(leaf) => {
          if leaf.value == true {
            return Some([
              self.graph.dict.get_by_right(&leaf.x).unwrap().clone(),
              self.graph.predicates.get_by_right(&self.slice).unwrap().clone(),
              self.graph.dict.get_by_right(&leaf.y).unwrap().clone()
            ])
          }
        },
        None => {
          self.slice += 1;
          while let Some(None) = self.graph.slices.get(self.slice) {
            self.slice += 1;
          }
          if self.slice == self.graph.slices.len() { return None }
          if let Some(slice) = &self.graph.slices[self.slice] {
            self.slice_iter = Some(slice.leaves());
          }
        },
      };
    }
  }
}
pub struct IntoIter {
  graph: Graph,
  slice: usize,
  slice_iter: Option<k2_tree::IntoLeaves>
}
impl Iterator for IntoIter {
  type Item = RdfTriple;
  fn next(&mut self) -> Option<Self::Item> {
    
    if self.slice == self.graph.slices.len() { return None }

    loop {
      let leaf = match &mut self.slice_iter {
        Some(iter) => {
          match iter.next() {
            Some(leaf) => Some(leaf.clone()),
            None => None,
          }
        },
        None => None,
      };
      match &leaf {
        Some(leaf) => {
          if leaf.value == true {
            return Some([
              self.graph.dict.get_by_right(&leaf.x).unwrap().clone(),
              self.graph.predicates.get_by_right(&self.slice).unwrap().clone(),
              self.graph.dict.get_by_right(&leaf.y).unwrap().clone()
            ])
          }
        },
        None => {
          self.slice += 1;
          while let Some(None) = self.graph.slices.get(self.slice) {
            self.slice += 1;
          }
          if self.slice == self.graph.slices.len() { return None }
          let slice_num  = self.slice.clone();
          if let Some(slice) = &self.graph.slices[slice_num] {
            self.slice_iter = Some(slice.clone().into_leaves());
          }
        },
      };
    }
  }
}

/* Std Traits */

/* Private */
impl Graph {
  fn filter_triples_str(&self, triples: Vec<[String; 3]>, pattern: [Option<String>; 3]) -> Vec<[String; 3]> {
    triples.into_iter().filter(|[s, p, o]| {
      match &pattern {
        [Some(a), Some(b), Some(c)] => { s == a && p == b && o == c },
        [None, Some(b), Some(c)]    => { p == b && o == c },
        [Some(a), None, Some(c)]    => { s == a && o == c },
        [Some(a), Some(b), None]    => { s == a && p == b },
        [None, None, Some(c)]       => { o == c },
        [None, Some(b), None]       => { p == b },
        [Some(a), None, None]       => { s == a },
        [None, None, None]          => true,
      }
    }).collect()
  }
  fn filter_triples(&self, triples: Vec<[usize; 3]>, pattern: [Option<usize>; 3]) -> Vec<[usize; 3]> {
    triples.into_iter().filter(|[s, p, o]| {
      match &pattern {
        [Some(a), Some(b), Some(c)] => { s == a && p == b && o == c },
        [None, Some(b), Some(c)] => { p == b && o == c },
        [Some(a), None, Some(c)] => { s == a && o == c },
        [Some(a), Some(b), None] => { s == a && p == b },
        [None, None, Some(c)] => { o == c },
        [None, Some(b), None] => { p == b },
        [Some(a), None, None] => { s == a },
        [None, None, None] => true,
      }
    }).collect()
  }
  fn to_node(s: &str) -> RdfNode {
    RdfNode::Named{ iri: s.to_string() }
  }
  /* Return the triples in the compact form of their dict index */
  fn get_from_triple(&self, triple: [Option<String>; 3]) -> Vec<[usize; 3]> {
    match triple {
      [Some(s), Some(p), Some(o)] => self.spo(&s, &p, &o),
      [None, Some(p), Some(o)]    => self._po(&p, &o),
      [Some(s), None, Some(o)]    => self.s_o(&s, &o),
      [Some(s), Some(p), None]    => self.sp_(&s, &p),
      [None, None, Some(o)]       => self.__o(&o),
      [None, Some(p), None]       => self._p_(&p),
      [Some(s), None, None]       => self.s__(&s),
      [None, None, None]          => self.___(),
    }
  }
  fn spo(&self, s: &str, p: &str, o: &str) -> Vec<[usize; 3]> {
    match [self.dict.get_by_left(&RdfNode::Named{iri:s.to_string()}),
      self.dict.get_by_left(&Self::to_node(o)),
      self.predicates.get_by_left(&Self::to_node(p))] {
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
    match [self.dict.get_by_left(&Self::to_node(o)),
      self.predicates.get_by_left(&Self::to_node(p))] {
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
    match [self.dict.get_by_left(&Self::to_node(s)),
      self.dict.get_by_left(&Self::to_node(o))] {
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
    match [self.dict.get_by_left(&Self::to_node(s)),
      self.predicates.get_by_left(&Self::to_node(p))] {
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
  fn __o(&self, o: &str) -> Vec<[usize; 3]> {
    match self.dict.get_by_left(&Self::to_node(o)) {
        Some(&y) => {
          let mut ret_v = Vec::new();
          for (index, slice) in self.slices.iter().enumerate() {
            if let Some(slice) = slice {
              ret_v.append(&mut match slice.get_row(y) {
                Ok(bitvec) => one_positions(&bitvec)
                  .into_iter()
                  .map(|pos| [pos, index, y])
                  .collect(),
                _ => Vec::new(),
              });
            }
          }
          ret_v
        },
        _ => Vec::new(),
    }
  }
  fn _p_(&self, p: &str) -> Vec<[usize; 3]> {
    match self.predicates.get_by_left(&Self::to_node(p)) {
      Some(&slice_index) => {
        if let Some(slice) = &self.slices[slice_index] {
          let mut ret_v = Vec::new();
          for x in 0..slice.matrix_width() {
            ret_v.append(&mut match slice.get_column(x) {
              Ok(bitvec) => one_positions(&bitvec)
                .into_iter()
                .map(|pos| [x, slice_index, pos])
                .collect(),
              _ => Vec::new(),
            });
          }
          ret_v
        }
        else {
          Vec::new()
        }
      },
      _ => Vec::new(),
    }
  }
  fn s__(&self, s: &str) -> Vec<[usize; 3]> {
    match self.dict.get_by_left(&Self::to_node(s)) {
      Some(&x) => {
        let mut ret_v = Vec::new();
        for (index, slice) in self.slices.iter().enumerate() {
          if let Some(slice) = slice {
            ret_v.append(&mut match slice.get_column(x) {
              Ok(bitvec) => one_positions(&bitvec)
                .into_iter()
                .map(|pos| [x, index, pos])
                .collect(),
              _ => Vec::new(),
            });
          }
        }
        ret_v
      },
      _ => Vec::new(),
    }
  }
  fn ___(&self) -> Vec<[usize; 3]> {
    let mut ret_v = Vec::new();
    for (index, slice) in self.slices.iter().enumerate() {
      if let Some(slice) = slice {
        for x in 0..slice.matrix_width() {
          ret_v.append(&mut match slice.get_column(x) {
            Ok(bitvec) => one_positions(&bitvec)
              .into_iter()
              .map(|pos| [x, index, pos])
              .collect(),
            _ => Vec::new(),
          });
        }
      }
    };
    ret_v
  }
}

/* Utils */
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
struct Slice {
  pub predicate_index: usize,
  pub tree: Box<K2Tree>,
}
#[derive(Clone, Debug)]
struct TripleSet {
  pub size: usize,
  pub predicate_index: usize,
  pub doubles: Vec<[usize; 2]>,
}
type PartitionedTriples = Vec<Vec<[usize; 2]>>;
async fn build_tree(pred_index: usize, doubles: &[[usize; 2]], dict_max: usize) -> Option<Slice> {
  let mut tree = K2Tree::new();
  while tree.matrix_width() < dict_max {
    tree.grow();
  }
  for &[x, y] in doubles {
    if tree.set(x, y, true).is_err() {
      return None
    }
  }
  Some(Slice{
    predicate_index: pred_index,
    tree: Box::new(tree),
  })
}
fn build_slices(triple_sets: Vec<TripleSet>, dict_max: usize) -> Vec<Slice> {
  use futures::{executor, StreamExt, stream::FuturesUnordered};
  executor::block_on(async {
    let mut futs = FuturesUnordered::new();
    for TripleSet {
      size: _,
      predicate_index: pi,
      doubles: ds
    } in &triple_sets {
      futs.push(build_tree(*pi, ds, dict_max));
    }
    let mut ret_vals = Vec::new();
    while let Some(Some(tree)) = futs.next().await {
      ret_vals.push(tree);
    };
    ret_vals
  })
}
fn sort_by_size(triples: PartitionedTriples) -> Vec<TripleSet> {
  let mut sorted_triples = triples.into_iter()
    .enumerate()
    .map(|(i, doubles)| TripleSet{
      size: doubles.len(),
      predicate_index: i,
      doubles: doubles
    })
    .collect::<Vec<TripleSet>>();
  sorted_triples.sort_by(|a, b| a.size.cmp(&b.size));
  sorted_triples
}

/* Unit Tests */
// #[cfg(test)]
// mod unit_tests {
//   use super::*;
//   use std::path::MAIN_SEPARATOR as PATH_SEP;
//   #[test]
//   fn get_0() {
//     let mut g = Graph::new();
//     g.insert_triple(["Gabe".into(), "likes".into(), "Rust".into()]);
//     g.insert_triple(["Gabe".into(), "likes".into(), "Js".into()]);
//     g.insert_triple(["Gabe".into(), "likes".into(), "Harry".into()]);
//     g.insert_triple(["Scala".into(), "is".into(), "male".into()]);
//     g.insert_triple(["Gabe".into(), "likes".into(), "Ron".into()]);
//     g.insert_triple(["Gabe".into(), "likes".into(), "Chris".into()]);
//     g.insert_triple(["Ron".into(), "is".into(), "male".into()]);
//     g.insert_triple(["Chris".into(), "is".into(), "male".into()]);
//     g.insert_triple(["Ron".into(), "isnt".into(), "rude".into()]);
//     g.insert_triple(["Chris".into(), "isnt".into(), "rude".into()]);
//     g.insert_triple(["Harry".into(), "isnt".into(), "rude".into()]);
//     let query = Sparql::new()
//       .select(vec!["$name".into()])
//       .filter(vec![["Gabe".into(), "likes".into(), "$name".into()],
//         ["$name".into(), "is".into(), "male".into()],
//         ["$name".into(), "isnt".into(), "rude".into()]]
//     );
//     assert_eq!(g.get(&query), vec![String::from("Ron"), "Chris".into()]);
//   }
//   #[test]
//   fn get_from_rdf_0() {
//     let g = Graph::from_rdf(&format!("models{}www-2011-complete.rdf", PATH_SEP)).unwrap();
//     let query = Sparql::new()
//       .select(vec!["$name".into()])
//       .filter(vec![[
//         "http://data.semanticweb.org/conference/www/2011/proceedings".into(),
//         "http://data.semanticweb.org/ns/swc/ontology#hasPart".into(),
//         "$name".into()],
//         ["http://data.semanticweb.org/person/iasonas-polakis".into(),
//         "http://xmlns.com/foaf/0.1/made".into(),
//         "$name".into()]]
//     );
//     let paper = g.get(&query);
//     assert_eq!(paper, vec![String::from("http://data.semanticweb.org/conference/www/2011/paper/we-b-the-web-of-short-urls")]);
//   }
//   #[test]
//   fn from_rdf_0() {
//     assert!(Graph::from_rdf(&format!("models{}www-2011-complete.rdf", PATH_SEP)).is_ok());
//   }
//   #[test]
//   fn from_rdf_async_0() {
//     use futures::executor;
//     assert!(executor::block_on(
//       Graph::from_rdf_async(&format!("models{}pref_labels.rdf", PATH_SEP))
//       ).is_ok()
//     );
//   }
//   #[test]
//   fn persist_to_0() {
//     let mut g = Graph::new();
//     g.insert_triple(["Gabe".into(), "likes".into(), "Rust".into()]);
//     g.insert_triple(["Gabe".into(), "likes".into(), "Js".into()]);
//     g.insert_triple(["Gabe".into(), "likes".into(), "Harry".into()]);
//     g.insert_triple(["Scala".into(), "is".into(), "male".into()]);
//     g.insert_triple(["Gabe".into(), "likes".into(), "Ron".into()]);
//     g.insert_triple(["Gabe".into(), "likes".into(), "Chris".into()]);
//     g.insert_triple(["Ron".into(), "is".into(), "male".into()]);
//     g.insert_triple(["Chris".into(), "is".into(), "male".into()]);
//     g.insert_triple(["Ron".into(), "isnt".into(), "rude".into()]);
//     g.insert_triple(["Chris".into(), "isnt".into(), "rude".into()]);
//     g.insert_triple(["Harry".into(), "isnt".into(), "rude".into()]);
//     assert!(g.persist_to(&format!("C:{}temp{}persist-test", PATH_SEP, PATH_SEP)).is_ok());
//   }
//   #[test]
//   fn persist_to_1() {
//     let mut g = Graph::from_rdf(&format!("models{}www-2011-complete.rdf", PATH_SEP)).unwrap();
//     assert!(g.persist_to(&format!("C:{}temp{}lrec-2008-backup", PATH_SEP, PATH_SEP)).is_ok());
//   }
//   #[test]
//   fn persist_0() {
//     let mut g = Graph::new();
//     g.insert_triple(["Gabe".into(), "likes".into(), "Rust".into()]);
//     g.insert_triple(["Gabe".into(), "likes".into(), "Js".into()]);
//     g.insert_triple(["Gabe".into(), "likes".into(), "Harry".into()]);
//     g.insert_triple(["Scala".into(), "is".into(), "male".into()]);
//     g.insert_triple(["Gabe".into(), "likes".into(), "Ron".into()]);
//     g.insert_triple(["Gabe".into(), "likes".into(), "Chris".into()]);
//     g.insert_triple(["Ron".into(), "is".into(), "male".into()]);
//     g.persist_to(&format!("C:{}temp{}persist-test", PATH_SEP, PATH_SEP));
//     g.insert_triple(["Chris".into(), "isnt".into(), "rude".into()]);
//     g.insert_triple(["Harry".into(), "isnt".into(), "rude".into()]);
//     assert!(g.persist().is_ok());
//   }
//   #[test]
//   fn from_backup_0() {
//     let mut g = Graph::new();
//     g.insert_triple(["Gabe".into(), "likes".into(), "Rust".into()]);
//     g.insert_triple(["Gabe".into(), "likes".into(), "Js".into()]);
//     g.insert_triple(["Gabe".into(), "likes".into(), "Harry".into()]);
//     g.insert_triple(["Scala".into(), "is".into(), "male".into()]);
//     g.insert_triple(["Gabe".into(), "likes".into(), "Ron".into()]);
//     g.insert_triple(["Gabe".into(), "likes".into(), "Chris".into()]);
//     g.insert_triple(["Ron".into(), "is".into(), "male".into()]);
//     g.insert_triple(["Chris".into(), "is".into(), "male".into()]);
//     g.insert_triple(["Ron".into(), "isnt".into(), "rude".into()]);
//     g.insert_triple(["Chris".into(), "isnt".into(), "rude".into()]);
//     g.insert_triple(["Harry".into(), "isnt".into(), "rude".into()]);
//     let path = "C:\\temp\\persist_test";
//     g.persist_to(path);
//     assert_eq!(Graph::from_backup(path).unwrap(), g);
//   }
// }