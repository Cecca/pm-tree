use anyhow::Result;
use argh::FromArgs;
use pm_tree::*;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256StarStar;
use std::time::Instant;
use std::{ops::Div, path::PathBuf};

#[derive(FromArgs)]
/// an implementation of the PM-LSH index for the closest pair problem
struct Args {
    #[argh(option, default = "1")]
    /// the number of pairs to retrieve
    k: usize,

    #[argh(option, default = "16")]
    /// the number of dimensions onto which to project,
    m: usize,

    #[argh(option, default = "16")]
    /// the node capacity
    capacity: usize,

    #[argh(positional)]
    /// the input file
    path: PathBuf,
}

fn load_dataset(path: &PathBuf) -> Vec<Vec<f64>> {
    let f = hdf5::File::open(path).unwrap();
    let data = f.dataset("/train").unwrap();
    let array = data.read_2d::<f64>().unwrap();
    // let array = data.read_slice_2d::<f64, _>(s![..100000, ..]).unwrap();
    let mut dataset = Vec::new();
    for row in array.rows() {
        let norm = row.dot(&row).sqrt();
        let normalized = row.div(norm);
        let r = normalized.as_slice().unwrap();
        dataset.push(Vec::from(r));
    }
    dataset
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn project(dataset: &[Vec<f64>], m: usize, seed: u64) -> Vec<Vec<f64>> {
    let dim = dataset[0].len();
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let projs: Vec<Vec<f64>> = (0..m)
        .map(|_| {
            let v: Vec<f64> = normal.sample_iter(&mut rng).take(dim).collect();
            v
        })
        .collect();

    dataset
        .iter()
        .map(|v| {
            let projected: Vec<f64> = projs.iter().map(|p| dot(&v, &p)).collect();
            projected
        })
        .collect()
}

fn run_tree<const B: usize>(data: &[Vec<f64>], k: usize) -> Vec<(f64, usize, usize)>{
    let t_index = Instant::now();
    let tree = PMTree::<_, Euclidean, B, 5>::for_dataset(&data, 1234);
    eprintln!("({:?}) index built", t_index.elapsed());

    let t_query = Instant::now();
    let res = tree.closest_pairs(k, &data);
    eprintln!("({:?}) query completed", t_query.elapsed());
    res
}

fn main() -> Result<()> {
    debug_assert!(false, "compile this binary in release mode!");
    env_logger::init();
    let args: Args = argh::from_env();
    let t_overall = Instant::now();

    let t_load = Instant::now();
    let dataset = load_dataset(&args.path);
    eprintln!(
        "({:?}) Loaded dataset with {} vectors",
        t_load.elapsed(),
        dataset.len()
    );

    let t_project = Instant::now();
    let projected = project(&dataset, args.m, 1234);
    eprintln!("({:?}) random projection completed", t_project.elapsed());

    let res = match args.capacity {
        2 => run_tree::<2>(&projected, args.k),
        4 => run_tree::<4>(&projected, args.k),
        8 => run_tree::<8>(&projected, args.k),
        16 => run_tree::<16>(&projected, args.k),
        32 => run_tree::<32>(&projected, args.k),
        c => panic!("Unsupported capacity {}", c),
    };

    eprintln!("Overall time {:?}", t_overall.elapsed());

    for (d, a, b) in res {
        println!("{},{},{}", a, b, d);
    }

    Ok(())
}
