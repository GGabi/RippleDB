use criterion::*;
use ripple_db::Graph;

pub fn from_rdf(c: &mut Criterion) {
  let mut group = c.benchmark_group("graph_builder");
  group.sample_size(10);
  // group.bench_function("Graph::from_rdf() with ~1.5MB file",
  //   |b| b.iter(||
  //     // Graph::from_rdf_atomicly_synced(black_box("models\\www-2011-complete.rdf"))
  //   )
  // );
  group.finish();
}

pub fn persist_to(c: &mut Criterion) {
  let mut group = c.benchmark_group("graph_builder");
  group.sample_size(10);
  let mut g = Graph::from_rdf("models\\lrec-2008-complete.rdf").unwrap();
  // group.bench_function("Graph::from_rdf() with ~500KB file",
  //   |b| b.iter(|| {
  //     // g.persist_to(black_box("C:\\temp\\bench_test"));
  //     // std::fs::remove_dir_all("C:\\temp\\bench_test");
  //   })
  // );
  group.finish();
}

pub fn from_backup(c: &mut Criterion) {
  let mut group = c.benchmark_group("graph_builder");
  group.sample_size(10);
  let mut g = Graph::from_rdf("models\\lrec-2008-complete.rdf").unwrap();
  // g.persist_to("C:\\temp\\bench_test");
  // group.bench_function("Graph::from_rdf() with ~500KB file",
  //   |b| b.iter(|| {
  //     Graph::from_backup(black_box("C:\\temp\\bench_test"));
  //   })
  // );
  // std::fs::remove_dir_all("C:\\temp\\bench_test");
  group.finish();
}

criterion_group!(benches, from_rdf);
criterion_main!(benches);

//old on www-2011-complete.rdf: 7.40 secs (88% peak cpu)
//new on www-2011-complete.rdf: 4.55 secs (40% peak cpu)

//old on lrec-2008-complete.rdf: 31.85 secs
//new on lrec-2008-complete.rdf: 34.94 secs