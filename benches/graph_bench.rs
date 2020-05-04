use criterion::*;
use ripple_db::Graph;

use std::path::MAIN_SEPARATOR as PATH_SEP;

pub fn from_rdf(c: &mut Criterion) {
  let mut group = c.benchmark_group("graph_builder");
  group.sample_size(10);
  group.bench_function("Graph::from_rdf() with ~1MB file",
    |b| b.iter(||
      Graph::from_rdf(black_box(&format!("models{}www-2011-complete.rdf", PATH_SEP)))
    )
  );
  group.finish();
}

pub fn persist_to(c: &mut Criterion) {
  let mut group = c.benchmark_group("graph_builder");
  group.sample_size(10);
  let mut g = Graph::from_rdf(&format!("models{}www-2008-complete.rdf", PATH_SEP)).unwrap();
  group.bench_function("Creating Graph backup",
    |b| b.iter(|| {
      g.persist_to(black_box("bench_test"));
      std::fs::remove_dir_all("bench_test");
    })
  );
  group.finish();
}

pub fn from_backup(c: &mut Criterion) {
  let mut group = c.benchmark_group("graph_builder");
  group.sample_size(10);
  let mut g = Graph::from_rdf(&format!("models{}www-2008-complete.rdf", PATH_SEP)).unwrap();
  g.persist_to("bench_test");
  group.bench_function("Build Graph from backup",
    |b| b.iter(|| {
      Graph::from_backup(black_box("bench_test"));
    })
  );
  std::fs::remove_dir_all("bench_test");
  group.finish();
}

criterion_group!(benches, from_rdf, persist_to, from_backup);
criterion_main!(benches);

//old on www-2011-complete.rdf: 7.40 secs (88% peak cpu)
//new on www-2011-complete.rdf: 4.55 secs (40% peak cpu)

//old on lrec-2008-complete.rdf: 31.85 secs
//new on lrec-2008-complete.rdf: 34.94 secs