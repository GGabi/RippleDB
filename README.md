
# RippleDB
## An Intuitive GraphDB for the Performance-Oriented
RippleDB is an embedded Graph Database implemented in Rust so it can have complete control over how data is managed in order to provide increased performance for the end-user. It is a simple, no-nonsense solution to storing and querying Semantic Web (RDF) data which focuses on providing an interface with a learning-curve so shallow don't realize you're climbing it.
### The aim of this project is to: 
 - **Import and Go**: Allow users to set-up a fully-functional graph database with just a few lines of code.
 - **Seems Functional**: Provide all of the features a developer would come to expect from a database.
 - **Seems Practical**: Be developed to a standard that users expect of libraries intended for tackling real problems in production.
 - **Lightning Fast**: Bless you with all the performance benefits that come with implementing something in Rust.
### What this project will not do:
 - **We're not Google**: My aim is not to provide an Enterprise-level database solution.
 - **I'm Only One Man**: I have no guarantees of RippleDB working seamlessly with obscenely huge datasets.
### What we do have: 
 - **Import and Go**: To set up a database simply import RippleDB and call `Graph::new()`.
 - **Bit-Level Compression**: `Graph`s vertically-partition their data into slices, represented as bit-matrices, which are then compressed using a data-structure specifically designed for compressing sparse bit-matrices: the `K2Tree`.
 -  - K2Tree proposal: 
 - **All the K2Trees!**: RippleDB's implementation is completely standalone, so `use ripple_db::K2Tree` in other projects to your heart's content!
 - **Comprehensive Interface**: All the database-operations you would expect are present: `insert_triple`, `remove_triple`, `get`, `persist_to (the filesystem)`, `from_backup`, `from_rdf`, `to_rdf` and `iter (through its contents)`. That's all of 'em right?
 -  **Fancy Types**: RDF nodes can be complex, which is why we made them easy. Graphs accept `RdfTriple`s composed of easily definable `RdfNode`s. Need a named-node? `RdfNode::Named` has got you covered. Fancy a blankey boy? `RdfNode::Blank`'s here for you. Feeling German? `RdfNode::LangTaggedLit` sagt hallo! I think you get the picture.
 - **Even Fancier Queries**: `SparqlQuery`s can be created thusly: 
> `let q = SparqlQuery::new().select(["$name"]).filter([["$name", "likes","Janet"]]);`
 - **Parallel Concurrency**: When Graph's are built from existing RDF datasets they are done so concurrently over multiple threads for that sweet, sweet speed. Don't forget to build with `--release`! 
### What we don't have: (yet)
 - **SPARQL Compliant**: Our queries are not yet fully compliant to the SPARQL standard.
 - **Faster Documents!**: A persistence model using Amazon ION. 
   - Using C and Rust's FFI.
 - **Sneaky Sorting**: Sorting of data to take place while saving to files.
   - Maximizing the sparseness of our slices increases data-compression" 
   - Periodic purging of tombstones can be a good thing, right?
 - **Embracing the Future**: WebAssembly bindings with support for JS Promises.
 - **RippleJS**: A NPM package as a wrapper to allow NodeJS to interact with Ripple as if it's Javascript.
