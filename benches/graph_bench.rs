use criterion::*;
use ripple_db::Graph;

pub fn from_rdf(c: &mut Criterion) {
  use futures::executor;
  let mut group = c.benchmark_group("graph_builder");
  group.sample_size(10);
  group.bench_function("Graph::from_rdf() with ~500KB file",
    |b| b.iter(|| Graph::from_rdf(black_box("models\\www-2007-complete.rdf")))
  );
  group.bench_function("Graph::from_rdf_async() with ~500KB file",
    |b| b.iter(|| executor::block_on(
      Graph::from_rdf_async(black_box("models\\www-2007-complete.rdf"))
    ))
  );
  group.finish();
}

criterion_group!(benches, from_rdf);
criterion_main!(benches);