
#[derive(Clone, Debug, PartialEq)]
pub enum QueryUnit {
  Val(String),
  Var(String),
  None,
}
impl<'a> From<&'a str> for QueryUnit {
  fn from(s: &str) -> Self {
    match s.chars().next() {
      Some('$') => QueryUnit::Var(s[1..].into()),
      Some(_)   => QueryUnit::Val(s.into()),
      None      => QueryUnit::None,
    }
  }
}
impl From<String> for QueryUnit {
  fn from(s: String) -> Self {
    match s.chars().next() {
      Some('$') => QueryUnit::Var(s[1..].into()),
      Some(_)   => QueryUnit::Val(s),
      None      => QueryUnit::None,
    }
  }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Sparql {
  pub vars: Vec<QueryUnit>,
  pub conds: Vec<[QueryUnit; 3]>,
}
impl Sparql {
  pub fn new() -> Self {
    Sparql {
      vars: Vec::new(),
      conds: Vec::new(),
    }
  }
  pub fn select(mut self, vars: Vec<String>) -> Self {
    self.vars = vars.to_vec()
      .into_iter()
      .map(QueryUnit::from)
      .collect();
    self
  }
  pub fn filter(mut self, conds: Vec<[String; 3]>) -> Self {
    self.conds = conds.to_vec()
      .into_iter()
      .map(|[s, p, o]| [QueryUnit::from(s), QueryUnit::from(p), QueryUnit::from(o)])
      .filter(|[s, p, o]| {
        for qunit in [s, p, o].iter() {
          if let QueryUnit::Var(_) = qunit {
            if !self.vars.contains(qunit) {
              panic!("Undeclared variable in query!");
            }
          }
        }
        true
      })
      .collect();
    self
  }
}

/* Create a sparql qeuery using familiar syntax:
let query = sparql!(
  select {
    $name
  }
  where {
    [$name, "is", "male"],
    ["Gabe", "likes", $name]
}); */
#[macro_export]
macro_rules! sparql {
  () => { unimplemented!() }
}