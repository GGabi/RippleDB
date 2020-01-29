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

use crate::util::{
  Triple,
  datastore::k2_tree::K2Tree,
  rdf::query::{Sparql, QueryUnit}
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
  slices: Vec<Option<Box<K2Tree>>>,
  persist_location: Option<String>,
}

/* Public */
impl Graph {
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
    /* Private trait impl */
    impl<'de> Deserialize<'de> for Graph {
      fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
          Dict_Max,
          Dict_Tombstones,
          Dict,
          Pred_Tombstones,
          Predicates,
          Persist_Location
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
                Field::Dict_Max => {
                  if dict_max.is_some() {
                      return Err(de::Error::duplicate_field("dict_max"));
                  }
                  dict_max = Some(map.next_value()?);
                }
                Field::Dict_Tombstones => {
                  if dict_tombstones.is_some() {
                    return Err(de::Error::duplicate_field("dict_tombstones"));
                  }
                  dict_tombstones = Some(map.next_value()?);
                }
                Field::Dict => {
                  if dict.is_some() {
                    return Err(de::Error::duplicate_field("dict"));
                  }
                  dict = Some(map.next_value::<Vec<(String, usize)>>()?);
                }
                Field::Pred_Tombstones => {
                  if pred_tombstones.is_some() {
                      return Err(de::Error::duplicate_field("pred_tombstones"));
                  }
                  pred_tombstones = Some(map.next_value()?);
                }
                Field::Predicates => {
                  if predicates.is_some() {
                    return Err(de::Error::duplicate_field("predicates"));
                  }
                  predicates = Some(map.next_value::<Vec<(String, usize)>>()?);
                }
                Field::Persist_Location => {
                  if persist_location.is_some() {
                    return Err(de::Error::duplicate_field("persist_location"));
                  }
                  persist_location = Some(map.next_value()?);
                }
              }
            }
            let dict_max = dict_max.ok_or_else(|| de::Error::missing_field("dict_max"))?;
            let dict_tombstones = dict_tombstones.ok_or_else(|| de::Error::missing_field("dict_tombstones"))?;
            let dict = dict.ok_or_else(|| de::Error::missing_field("dict"))?;
            let pred_tombstones = pred_tombstones.ok_or_else(|| de::Error::missing_field("pred_tombstones"))?;
            let predicates = predicates.ok_or_else(|| de::Error::missing_field("predicates"))?;
            let persist_location = persist_location.ok_or_else(|| de::Error::missing_field("persist_location"))?;
            
            let mut final_dict: BiBTreeMap<String, usize> = BiBTreeMap::new();
            for (key, val) in dict.into_iter() {
              final_dict.insert(key, val);
            }
            let mut final_preds: BiBTreeMap<String, usize> = BiBTreeMap::new();
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
      dict_max: dict_max,
      dict_tombstones: dict_tombstones,
      dict: dict,
      pred_tombstones: pred_tombstones,
      predicates: predicates,
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
  pub fn from_rdf_thread_per_tree(path: &str) -> Result<Self, ()> {
    use crate::util::rdf::parser::ParsedTriples;
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
      persist_location: None,
    })
  }
  pub async fn from_rdf_async(path: &str) -> Result<Self, ()> {
    use crate::util::rdf::parser::ParsedTriples;
    use futures::{StreamExt, stream::FuturesOrdered};
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
      Err(_) => return Err(()),
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
      persist_location: None,
    })
  }
  pub fn from_rdf_atomicly_synced(path: &str) -> Result<Self, &str> {
    use {
      crate::util::rdf::parser::ParsedTriples,
      std::{
        thread,
        sync::{Arc, atomic::{AtomicPtr, AtomicUsize, Ordering}}
      },
      futures::{executor, StreamExt, stream::FuturesUnordered}
    };
    async unsafe fn build_tree(pos: usize,
      triples: &ShareVec,
      dict_max: usize) -> Option<(usize, Box<K2Tree>)> {
      let doubles = &(*triples.0)[pos];
      let mut tree = K2Tree::new();
      while tree.matrix_width() < dict_max {
        tree.grow();
      }
      for &[x, y] in doubles {
        if let Err(_) = tree.set(x, y, true) {
          return None
        }
      }
      Some((pos, Box::new(tree)))
    }
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
      Err(e) => return Err("error parsing triples"),
    };
    /* Build each K2Tree in parallel */
    let triples_len = partitioned_triples.len();
    let triples = partitioned_triples;
    let counter = Arc::new(AtomicUsize::new(0));
    /* God this is trash rewrite the whole thing lmaooooo */
    let mut handles = Vec::new();
    dbg!(num_cpus::get());
    struct ShareVec(*const Vec<Vec<[usize; 2]>>);
    unsafe impl Send for ShareVec {}
    for _ in 0..num_cpus::get() {
      let counter = Arc::clone(&counter);
      let triples = ShareVec(&triples);
      let dict_max = dict_max;
      handles.push(thread::spawn(move || executor::block_on(async {
          let mut futs = FuturesUnordered::new();
          let mut counter_val = counter.fetch_add(1, Ordering::Relaxed);
          while counter_val < triples_len {
            unsafe { futs.push(build_tree(counter_val, &triples, dict_max)); }
            counter_val = counter.fetch_add(1, Ordering::Relaxed)
          }
          let mut ret_vals = Vec::new();
          while let Some(Some(hello)) = futs.next().await {
            ret_vals.push(hello);
          };
          ret_vals
      })));
    }
    let mut vals: Vec<Vec<(usize, Box<K2Tree>)>> = Vec::new();
    for handle in handles { vals.push(handle.join().unwrap()); }
    /* Check if every slice was built successfully and 
    inserts each one into the correct location in the Graph's
    slices field */
    let mut slices: Vec<Option<Box<K2Tree>>> = vec![None; triples_len];
    for (pos, tree) in vals.into_iter().flat_map(|tuple| tuple) {
        slices[pos] = Some(tree);
    }
    if slices.iter().filter_map(|slice|
      if let Some(_) = slice {
        Some(0)
      }
      else {
        None
      }
    ).collect::<Vec<u8>>().len() != triples_len { return Err("tree is dead") }
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
  pub fn from_rdf(path: &str) -> Result<Self, &str> {
    use crate::util::rdf::parser::ParsedTriples;
    /* Parse the RDF file at path */
    let ParsedTriples {
      dict_max,
      dict,
      pred_max: _,
      predicates,
      triples: _,
      partitioned_triples,
    } = match ParsedTriples::from_rdf(path) {
      Ok(p_trips) => p_trips,
      Err(e) => return Err("error parsing triples"),
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
      let end_of_lowers = num_lower_threads * num_lowers_per_thread;
      let triplesets = sorted_trips[
        (end_of_lowers + (thread_num * num_uppers_per_thread))
        ..(end_of_lowers + ((thread_num + 1) * num_uppers_per_thread))
      ].to_vec();
      let dict_max = dict_max;
      handles.push(std::thread::spawn(move || build_slices(triplesets, dict_max)));
    }
    /* Spawn lower threads */
    for thread_num in 0..num_lower_threads {
      let triplesets = sorted_trips[
        (thread_num * num_lowers_per_thread)
        ..((thread_num + 1) * num_lowers_per_thread)
      ].to_vec();
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
      predicate_index: pos,
      tree: tree
    } in slice_sets.into_iter().flatten() {
        slices[pos] = Some(tree);
    }
    if slices.iter().fold(0, |sum, slice| 
      if let Some(tree) = slice { sum + 1 }
      else { sum }
    ) != num_slices {
      return Err("tree is dead")
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
  pub fn get(&self, query: &Sparql) -> Vec<String> {
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
    let num_to_str = |[s, p, o]: &[usize; 3]| {
      [
        self.dict.get_by_right(&s).unwrap().to_string(),
        self.predicates.get_by_right(&p).unwrap().to_string(),
        self.dict.get_by_right(&o).unwrap().to_string()
      ]
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
    let mut results: Vec<(&[QueryUnit; 3], Vec<[String; 3]>)> = Vec::new();
    for cond in query.conds.iter() {
      let qt = cond_to_qt(cond);
      let res = self.get_from_triple(qt);
      results.push((cond, res.into_iter().map(|t| num_to_str(&t)).collect()));
    }
    /* Filter results */
    let mut final_results: Vec<String> = results[0].1.iter().map(|[s, p, o]|
      match var_pos(results[0].0) {
        0 => s.clone(),
        1 => p.clone(),
        2 => o.clone(),
        _ => String::new(),
      }
    ).collect();
    for (query_triple, qt_results) in &results[1..] {
      let qt_var_pos = var_pos(query_triple);
      let mut used_vars_vals: HashSet<String> = HashSet::new();
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
          if self.filter_triples_str(qt_results.clone(), filter_t).len() == 0 {
            /* There was no match for this value of the variable from final_results in
            the query triple, so mark it to be removed at the end of this cycle */
            vars_vals_to_remove.push(i);
          }
        }
      }
      final_results = final_results.into_iter().enumerate().filter_map(|(i, res)| {
        if vars_vals_to_remove.len() > 0 {
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
    final_results
  }
  pub fn insert_triple(&mut self, val: Triple) -> Result<(), ()> {
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
  pub fn remove_triple(&mut self, [subject, predicate, object]: &Triple) -> Result<(), ()> {
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
  pub fn persist_to(&mut self, path: &str) -> Result<(), std::io::Error> {
    /* Only want to use this trait in this func, not public as it's not really
    "serializing" the Graph and would be confusing to users if the trait was
    publicly implemented */
    impl Serialize for Graph {
      fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("Graph", 6)?;
        state.serialize_field("dict_max", &self.dict_max)?;
        state.serialize_field("dict_tombstones", &self.dict_tombstones)?;
        state.serialize_field("dict", &self.dict.iter().collect() as &Vec<(&String, &usize)>)?;
        state.serialize_field("pred_tombstones", &self.pred_tombstones)?;
        state.serialize_field("predicates", &self.predicates.iter().collect() as &Vec<(&String, &usize)>)?;
        state.serialize_field("persist_location", &self.persist_location)?;
        state.end()
      }
    }
    /* Define locations to persist to */
    let root_dir = std::path::Path::new(path);
    let trees_dir = root_dir.join("trees");
    let head_file = root_dir.join("head.json");
    let dot_file = root_dir.join(".ripplebackup");

    if root_dir.is_dir() {
      /* If the folder already exists then either:
       - It's being used by another process
       - This Graph has already been saved here
      In both cases we don't wanna disturb this location. */
      // return Err("Dir already exists")
      return Ok(())
    }
    
    /* Save the location this Graph is persisted to */
    self.persist_location = Some(root_dir.to_str().unwrap().to_string());

    /* If we good then create root/, root/trees/ and root/head.json which
    containing surface info on Graph. (Dict contents etc.) */
    std::fs::create_dir(&root_dir)?;
    std::fs::create_dir(&trees_dir)?;
    std::fs::File::create(&dot_file)?;
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
  pub fn persist_location(&self) -> Option<String> {
    self.persist_location.clone()
  }
}

/* Iterators */

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
      match pattern {
        [Some(a), Some(b), Some(c)] => { *s == a && *p == b && *o == c },
        [None, Some(b), Some(c)] => { *p == b && *o == c },
        [Some(a), None, Some(c)] => { *s == a && *o == c },
        [Some(a), Some(b), None] => { *s == a && *p == b },
        [None, None, Some(c)] => { *o == c },
        [None, Some(b), None] => { *p == b },
        [Some(a), None, None] => { *s == a },
        [None, None, None] => true,
      }
    }).collect()
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
  fn __o(&self, o: &str) -> Vec<[usize; 3]> {
    match self.dict.get_by_left(&o.to_string()) {
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
    match self.predicates.get_by_left(&p.to_string()) {
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
    match self.dict.get_by_left(&s.to_string()) {
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
async fn build_tree(pred_index: usize, doubles: &Vec<[usize; 2]>, dict_max: usize) -> Option<Slice> {
  let mut tree = K2Tree::new();
  while tree.matrix_width() < dict_max {
    tree.grow();
  }
  for &[x, y] in doubles {
    if let Err(_) = tree.set(x, y, true) {
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
  sorted_triples.sort_by(|a, b| b.size.cmp(&a.size));
  sorted_triples
}

/* Unit Tests */
#[cfg(test)]
mod unit_tests {
  use super::*;
  use std::path::MAIN_SEPARATOR as PATH_SEP;
  #[test]
  fn get_0() {
    let mut g = Graph::new();
    g.insert_triple(["Gabe".into(), "likes".into(), "Rust".into()]);
    g.insert_triple(["Gabe".into(), "likes".into(), "Js".into()]);
    g.insert_triple(["Gabe".into(), "likes".into(), "Harry".into()]);
    g.insert_triple(["Scala".into(), "is".into(), "male".into()]);
    g.insert_triple(["Gabe".into(), "likes".into(), "Ron".into()]);
    g.insert_triple(["Gabe".into(), "likes".into(), "Chris".into()]);
    g.insert_triple(["Ron".into(), "is".into(), "male".into()]);
    g.insert_triple(["Chris".into(), "is".into(), "male".into()]);
    g.insert_triple(["Ron".into(), "isnt".into(), "rude".into()]);
    g.insert_triple(["Chris".into(), "isnt".into(), "rude".into()]);
    g.insert_triple(["Harry".into(), "isnt".into(), "rude".into()]);
    let query = Sparql::new()
      .select(vec!["$name".into()])
      .filter(vec![["Gabe".into(), "likes".into(), "$name".into()],
        ["$name".into(), "is".into(), "male".into()],
        ["$name".into(), "isnt".into(), "rude".into()]]
    );
    assert_eq!(g.get(&query), vec![String::from("Ron"), "Chris".into()]);
  }
  #[test]
  fn get_from_rdf_0() {
    let g = Graph::from_rdf(&format!("models{}www-2011-complete.rdf", PATH_SEP)).unwrap();
    let query = Sparql::new()
      .select(vec!["$name".into()])
      .filter(vec![[
        "http://data.semanticweb.org/conference/www/2011/proceedings".into(),
        "http://data.semanticweb.org/ns/swc/ontology#hasPart".into(),
        "$name".into()],
        ["http://data.semanticweb.org/person/iasonas-polakis".into(),
        "http://xmlns.com/foaf/0.1/made".into(),
        "$name".into()]]
    );
    let paper = g.get(&query);
    assert_eq!(paper, vec![String::from("http://data.semanticweb.org/conference/www/2011/paper/we-b-the-web-of-short-urls")]);
  }
  #[test]
  fn from_rdf_0() {
    assert!(Graph::from_rdf(&format!("models{}www-2011-complete.rdf", PATH_SEP)).is_ok());
  }
  #[test]
  fn from_rdf_async_0() {
    use futures::executor;
    assert!(executor::block_on(
      Graph::from_rdf_async(&format!("models{}pref_labels.rdf", PATH_SEP))
      ).is_ok()
    );
  }
  #[test]
  fn persist_to_0() {
    let mut g = Graph::new();
    g.insert_triple(["Gabe".into(), "likes".into(), "Rust".into()]);
    g.insert_triple(["Gabe".into(), "likes".into(), "Js".into()]);
    g.insert_triple(["Gabe".into(), "likes".into(), "Harry".into()]);
    g.insert_triple(["Scala".into(), "is".into(), "male".into()]);
    g.insert_triple(["Gabe".into(), "likes".into(), "Ron".into()]);
    g.insert_triple(["Gabe".into(), "likes".into(), "Chris".into()]);
    g.insert_triple(["Ron".into(), "is".into(), "male".into()]);
    g.insert_triple(["Chris".into(), "is".into(), "male".into()]);
    g.insert_triple(["Ron".into(), "isnt".into(), "rude".into()]);
    g.insert_triple(["Chris".into(), "isnt".into(), "rude".into()]);
    g.insert_triple(["Harry".into(), "isnt".into(), "rude".into()]);
    dbg!(g.persist_to("C:\\temp\\persist_test"));
  }
  #[test]
  fn persist_to_1() {
    let mut g = Graph::from_rdf("models\\lrec-2008-complete.rdf").unwrap();
    dbg!(g.persist_to("C:\\temp\\lrec-2008-backup"));
  }
  #[test]
  fn from_backup_0() {
    let mut g = Graph::new();
    g.insert_triple(["Gabe".into(), "likes".into(), "Rust".into()]);
    g.insert_triple(["Gabe".into(), "likes".into(), "Js".into()]);
    g.insert_triple(["Gabe".into(), "likes".into(), "Harry".into()]);
    g.insert_triple(["Scala".into(), "is".into(), "male".into()]);
    g.insert_triple(["Gabe".into(), "likes".into(), "Ron".into()]);
    g.insert_triple(["Gabe".into(), "likes".into(), "Chris".into()]);
    g.insert_triple(["Ron".into(), "is".into(), "male".into()]);
    g.insert_triple(["Chris".into(), "is".into(), "male".into()]);
    g.insert_triple(["Ron".into(), "isnt".into(), "rude".into()]);
    g.insert_triple(["Chris".into(), "isnt".into(), "rude".into()]);
    g.insert_triple(["Harry".into(), "isnt".into(), "rude".into()]);
    let path = "C:\\temp\\persist_test";
    g.persist_to(path);
    assert_eq!(Graph::from_backup(path).unwrap(), g);
  }
}