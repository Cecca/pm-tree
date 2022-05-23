use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hdf5::File;
use pm_tree::{Distance, Euclidean, OrdF64, PMTree};
use std::io::prelude::*;
use std::str::FromStr;
use std::{collections::BinaryHeap, path::PathBuf};

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

macro_rules! bench_closest_pair {
    ($group: ident, $B:literal, $P:literal, $dataset:ident, $k: literal) => {
        let tree = PMTree::<_, Euclidean, $B, $P>::for_dataset(&$dataset, 1234);
        $group.bench_function(stringify!(B=$B P=$P k=$k), |b| b.iter(|| {
            tree.closest_pairs($k, &$dataset);
        }));
    };
}


fn linear_scan<T, D: Distance<T>, F: FnMut(usize)>(
    range: f64,
    query: &T,
    dataset: &[T],
    mut callback: F,
) {
    for (i, v) in dataset.iter().enumerate() {
        if D::distance(query, v) <= range {
            callback(i);
        }
    }
}

fn linear_scan_pairs<T, D: Distance<T>>(k: usize, dataset: &[T]) -> Vec<(f64, usize, usize)> {
    let mut expected = BinaryHeap::new();

    for i in 0..dataset.len() {
        for j in (i + 1)..dataset.len() {
            let a = &dataset[i];
            let b = &dataset[j];
            let d = OrdF64(D::distance(a, b));
            expected.push((d, i, j));
            if expected.len() > k {
                expected.pop();
            }
        }
    }
    let expected: Vec<(f64, usize, usize)> = expected
        .into_sorted_vec()
        .into_iter()
        .map(|(d, a, b)| (d.0, a, b))
        .collect();

    expected
}

fn bench_glove25(c: &mut Criterion) {
    use ndarray::s;
    let path = ensure_glove25();
    let f = hdf5::File::open(path).unwrap();
    let data = f.dataset("/train").unwrap();
    let array = data.read_slice_2d::<f64, _>(s![..100000, ..]).unwrap();
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
        let mut group = c.benchmark_group("closest pair");
        group.sample_size(10);
        // group.bench_function("linear scan k = 1", |b| {
        //     b.iter(|| {
        //         linear_scan_pairs::<_, Euclidean>(1, &dataset);
        //     })
        // });
        bench_closest_pair!(group, 32, 4, dataset, 1);
    }

    {
        let mut group = c.benchmark_group("range query");
        group.sample_size(10);
        group.bench_function("linear scan r = 2", |b| {
            b.iter(|| {
                let mut res = Vec::new();
                linear_scan::<_, Euclidean, _>(2.0, &dataset[0], &dataset, |i| res.push(i));
            })
        });
        group.bench_function("linear scan r = 4", |b| {
            b.iter(|| {
                let mut res = Vec::new();
                linear_scan::<_, Euclidean, _>(4.0, &dataset[0], &dataset, |i| res.push(i));
            })
        });

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
