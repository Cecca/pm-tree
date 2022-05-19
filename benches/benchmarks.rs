use std::path::PathBuf;
use std::io::prelude::*;
use std::str::FromStr;
use criterion::{criterion_group, criterion_main, Criterion};
use hdf5::File;
use pm_tree::{Euclidean, PMTree};

macro_rules! bench_builder {
    ($group: ident, $B:literal, $P:literal, $dataset:ident) => {
        $group.bench_function(stringify!(B=$B P=$P), |b| b.iter(|| PMTree::<_, Euclidean, $B, $P>::for_dataset(&$dataset, 1234)));
    };
}

macro_rules! bench_range_query {
    ($group: ident, $B:literal, $P:literal, $dataset:ident, $from: literal, $range: literal) => {
        let tree = PMTree::<_, Euclidean, $B, $P>::for_dataset(&$dataset, 1234);
        let query = &$dataset[$from];
        $group.bench_function(stringify!(B=$B P=$P r=$range), |b| b.iter(|| {
            let mut res = Vec::new();
            tree.range_query($range, query, &$dataset, |i| res.push(i));
        }));
    };
}

fn bench_glove25(c: &mut Criterion) {
    use ndarray::s;
    let path = ensure_glove25();
    let f = hdf5::File::open(path).unwrap();
    let data = f.dataset("/train").unwrap();
    let array = data.read_slice_2d::<f64, _>(s![..1000000, ..]).unwrap();
    let mut dataset = Vec::new();
    for row in array.rows() {
        let r = row.as_slice().unwrap();
        dataset.push(Vec::from(r));
    }

    {
        let mut group = c.benchmark_group("tree construction");
        group.sample_size(10);
        bench_builder!(group, 32, 8, dataset);
        bench_builder!(group, 64, 8, dataset);
    }

    {
        let mut group = c.benchmark_group("range query");
        group.sample_size(10);
        bench_range_query!(group, 32, 4, dataset, 0, 2.0);
        bench_range_query!(group, 64, 4, dataset, 0, 2.0);
        bench_range_query!(group, 32, 8, dataset, 0, 2.0);
        bench_range_query!(group, 64, 8, dataset, 0, 2.0);

        bench_range_query!(group, 32, 4, dataset, 0, 4.0);
        bench_range_query!(group, 64, 4, dataset, 0, 4.0);
        bench_range_query!(group, 32, 8, dataset, 0, 4.0);
        bench_range_query!(group, 64, 8, dataset, 0, 4.0);
    }
}

fn ensure_glove25() -> std::path::PathBuf {
    let local = PathBuf::from_str(".glove-25.hdf5").unwrap();
    if local.is_file() {
        return local;
    }
    eprintln!("Downloading the dataset for tests");
    let mut f = std::fs::File::create(&local).unwrap();
    std::io::copy(
        // the file is, very simply, stored in the public folder of my personal dropbox
        &mut ureq::get("https://dl.dropboxusercontent.com/s/kdh02vg1lb3qm5j/glove-25.hdf5?dl=0")
            .call()
            .unwrap()
            .into_reader(),
        &mut f,
    )
    .unwrap();
    local
}

criterion_group!(benches, bench_glove25);
criterion_main!(benches);
