use criterion::*;
use ripple_db::Graph;

pub fn from_rdf(c: &mut Criterion) {
  use futures::executor;
  let mut group = c.benchmark_group("graph_builder");
  group.sample_size(10);
  // group.bench_function("Graph::from_rdf() with ~500KB file",
  //   |b| b.iter(|| Graph::from_rdf(black_box("models\\www-2011-complete.rdf")))
  // );
  group.bench_function("Graph::from_rdf_better() with ~500KB file",
    |b| b.iter(|| executor::block_on(
      Graph::from_rdf_better(black_box("models\\www-2011-complete.rdf"))
    ))
  );
  group.finish();
}

pub fn persist_to(c: &mut Criterion) {
  let mut group = c.benchmark_group("graph_builder");
  group.sample_size(10);
  let mut g = Graph::from_rdf("models\\lrec-2008-complete.rdf").unwrap();
  group.bench_function("Graph::from_rdf() with ~500KB file",
    |b| b.iter(|| {
      g.persist_to(black_box("C:\\temp\\bench_test"));
      std::fs::remove_dir_all("C:\\temp\\bench_test");
    })
  );
  group.finish();
}

pub fn from_backup(c: &mut Criterion) {
  let mut group = c.benchmark_group("graph_builder");
  group.sample_size(10);
  let mut g = Graph::from_rdf("models\\lrec-2008-complete.rdf").unwrap();
  g.persist_to("C:\\temp\\bench_test");
  group.bench_function("Graph::from_rdf() with ~500KB file",
    |b| b.iter(|| {
      Graph::from_backup(black_box("C:\\temp\\bench_test"));
    })
  );
  std::fs::remove_dir_all("C:\\temp\\bench_test");
  group.finish();
}

criterion_group!(benches, from_rdf);
criterion_main!(benches);

//old: 381 secs
//new without concurrency: 954 secs
//new with conc: 853 secs