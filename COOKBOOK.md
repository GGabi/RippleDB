# Cookbook

Please keep in mind that this was developed as part of my CompSci dissertation, some features are ommitted because of the time constraints uni life has placed on me :sweat_smile:

## Outline

- What is a Graph Database?
- What is Ripple?
- Creating a New Graph
- Inserting and Removing Data
- Querying a Graph
- Persisting a Graph
- Restoring from Backup
- What is RDF?
- Building a Graph from an Existing RDF Store
- Producing an RDF Store
- The Graph Exposed: Beautiful and Documented

## 1. What is a Graph Database?

A graph DB is a database that models data using vertices and edges rather than the record-table approach of relational DBs. This leads to a few things:

- A lack of predefined tables eliminates the need for rigid schemas, allowing graph databases to easily deal with different, and possibly unexpected, types of data.
- Relations, or edges, are first class entities which makes them perfect for use-cases where the structure of data is just as, or maybe even more, important than the contents itself.
- Many-to-many relationships are a feature, not a bug, of data structure.

## 2. What is Ripple?

RippleDB is an embedded graph database specifically designed to be very good at storing Semantic Web data without sacrificing simplicity. That's a few potentially new words, however, so let's define them:

- Embedded database: A database that exists as an entity within, not adjacent to, the applications that utilise it. These usually come in the form of libraries, like SQLite.
- Semantic Web: A field dedicated to describing the structure of data on the web in a way that makes it as useful to machines as it is to humans. Think social media analysis etc.
-- RDF is a data format used extensively within this field and is the primary format supported by Ripple.
- Simplicity: Having a small learning-curve (our definition).

## 3. Creating a new Graph

Believe us when we say we've kept it simple:

```rust
use ripple_db::Graph;
let g = Graph::new(); //Done(!!)
```

## 4. Inserting and Removing Data

RippleDB is actually a specific type of graph database called a triple-store, which works exclusively with triples. Triples aren't complicated though, they're simply arrays of strings of length three, and are of the form [Subject, Predicate, Object].
The datatype that RippleDB actually deals with is called an RdfTriple, which is the primary unit of data for the Semantic Web. As such, this crate defines two distinct types of triple:

```rust
pub type Triple = [String; 3];
pub type RdfTriple = [RdfNode; 3]
```

We'll cover what an RdfNode is in a later section, all you need to know for now is that in order to convert a Triple to an RdfTriple just do the following:

```rust
use ripple_db::{Triple, RdfTriple, triple_into_rdf};
let triple: Triple = ["Ripple".into(), "is".into(), "infallible".into()];
let rdf_triple: RdfTriple = ["Ripple".into(), "loves".into(), "rdf".into()];
let another_rdf_triple = triple_into_rdf(triple);
```

Notice any difficulties? Of course not!
(In the future we hope to implement some helpful macros to make this even easier. Alas, this will have to do for now.)
Once you've got your data ready, inserting and removing is as easy as:

```rust
use ripple_db::{Graph, RdfTriple};
let mut g = Graph::new();
let t: RdfTriple = ["s".into(), "p".into(), "o".into()];
g.insert_triple(t.clone())?;
g.remove_triple(&t)?;
```

(Remove-queries coming soon! Sorry!)

## 5. Querying a Graph

RippleDB implements a subset of SPARQL for all your query-y needs! (It'll become fully compliant if there's demand)
Here's an example that asks for the name of everyone that is cool and loves Ripple:

```rust
use ripple_db::{SparqlQuery};
let q = SparqlQuery::new()
  .select(vec!["$name".into()])
  .filter(vec![
    ["$name".into(), "is".into(), "cool".into()],
    ["$name".into(), "loves".into(), "Ripple".into()]
]);
```

And using putting it into practise:

```rust
/* Imports */
let mut g = Graph::new();
/* Insert data etc. */
let results: Vec<RdfNode> = g.get(&q);
```

## 6. Persisting a Graph

Would use would a database be if you couldn't save it to disk? After pondering this question we decided to implement the following methods for ripple_db::Graph:

```rust
/* Sets the backup location and makes an initial save to it */
fn persist_to(&mut self, path: &str) -> Result<()>;
/* Gets the backup location */
fn persist_location(&self) -> &Option<String>;
/* Syncs a graph's backup with it's current in-memory state */
fn persist(&self) -> Result<()>;
```

Example of use:

```rust
use ripple_db::Graph;
let mut g = Graph::new();
g.persist_to("/temp/MyBackup")?;
let location = g.backup_location().unwrap();
for triple in &triples {
  g.insert_triple(*triple)?;
}
g.persist()?;
```

## 7. Restoring from Backup

How many times have you wanted to restore from a backup? Hopefully more than zero, otherwise we've wasted our time with this next feature!
As is with creating a new graph instance, building one from a backup is just a single method call:

```rust
use ripple_db::Graph;
let g = Graph::from_backup("/temp/MyBackup")?;
```

The on-disk representation of our graphs are sufficiently close to the in-memory structure to make the process of backup and restore blindingly fast, we're talking in the order of milleseconds per megabyte! Filesystem IO is almost guaranteed to be the only bottleneck you'll ever encounter, so don't feel the need to minimise the number of calls to backup/restores you do for the sake of performance. (Unless you really need those ms!)

## 8. What is RDF?

We touched on this earlier, but the time has finally come for a full explenation. RDF stands for Resource Description Framework and is a file-format for representing datasets consisting on triples. While there are other formats, like N-Triple or Turtle, in practise RDF is the most widespread. RDF datasets are of the form `filename.rdf`.

RDF datasets are made up of RDFTriples which are collections of RDFNodes. Because RippleDB is a RDF triple-store, we thought you'd be dealing with these things quite a bit, so we defined an enum to make this a little easier:

```rust
pub enum RdfNode {
  Named{ iri: String },
  Blank{ id: String },
  RawLit{ val: String },
  LangTaggedLit{ val: String, lang: String },
  TypedLit{ val: String, datatype: String },
}
```

Now that you know what an `RdfNode` actually is, allow us to pull back the curtains a little more: all those .into() calls made on strings earlier in this document produce `RdfNode::Named`.
For the vast majority of your Ripple journeys you'll only really need to focus on the `Named` variant, but let's quickly explain them all:

- `Named` is a named-node and references an entity that can appear in multiple triples
- `Blank` is a node with no name, simply an id, which is useful for when the identity of an entity isn't known or is irrelevant to the dataset
- `RawLit` is a simple string literal, it does not refer to an entity
- `LangTaggedLit` is a string literal with a specified language
- `TypedLit` is a literal with a specified type, such as Integer or Date

Phew! Now that's out the way, let's ignore it and move on to the juicy stuff.

## 9. Building a Graph from an Existing RDF Store

We highly expect you to do this at least once if you're really involved with RDF or the Semantic Web. Luckily, it's not much more difficult than creating a new graph or building one from a backup:

```rust
use ripple_db::Graph;
let g = Graph::from_rdf("/path/to/data.rdf")?;
```

Brilliant, right? We think so, but there is one caveat to using this constructor... RippleDB uses a datastructure called BitVec as a big part of compressing its contents as much as possible, which is perfect in every case except one, it's terribly slow when used in a program compiled without the `--release` flag. We don't know how or why, but building a graph from a 1MB dataset goes up from 7 seconds to almost 5 minutes when this flag is ommited! To offset this slowdown, we've written `from_rdf()` to build its graphs concurrently across parallel threads but there's only so much we can do. So please, for the love of Rust, don't forget to compile with `--release`!

## 10. Producing an RDF Store

So, we've covered building a graph from an RDF file, what if we want to produce our own? It shouldn't be a surprise to you by now that we've made it a single method call:

```rust
let buf: Vec<u8> = graph.to_rdf()?;
let stringed = buf as String; //If you like
```

We were originally gonna make it produce a .rdf file at a specified location for you, but giving you the freedom to do or don't this yourself seemed like the better option in the long run. Send this over the wire or write it to a file!

## 11. The Graph Exposed: Beautiful and Documented

Apologies for the title, but it's not changing.
Here's the full public API for `ripple_db::Graph` as it stands:

```rust
/* Constructors */
fn new() -> Graph;
fn from_backup(path: &str) -> Result<Self>;
fn from_rdf(path: &str) -> Result<Self>;
/* Get, Insert and Remove */
fn get(&self, query: &SparqlQuery) -> Vec<RdfNode>;
fn insert_triple(&mut self, triple: RdfTriple) -> Result<()>;
fn remove_triple(&mut self, triple: &RdfTriple) -> Result<()>;
/* Persistence */
fn persist_to(&mut self, path: &str) -> Result<()>;
fn persist_location(&self) -> &Option<String>;
fn persist(&self) -> Result<()>;
/* Iterators */
fn iter(&self) -> Graph::Iter;
fn into_iter(self) -> Graph::IntoIter;
/* Export to RDF (as a buffer of bytes) */
fn to_rdf(&self) -> Result<Vec<u8>>;
fn into_rdf(self) -> Result<Vec<u8>>;
```

At this point, there shoudn't be any explaining to do. Ta!
-- GGabi :ophiuchus:
