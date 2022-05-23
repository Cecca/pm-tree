use anyhow::Result;
use argh::FromArgs;
use pm_tree::*;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256StarStar;
use statrs::distribution::{ChiSquared, Continuous, ContinuousCDF};
use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::fs::File;
use std::str::FromStr;
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

fn estimate_gamma<T: Clone + Debug, D: Distance<T> + Debug, const B: usize, const P: usize>(
    dataset: &[T],
    sample_size: usize,
    seed: u64,
) -> f64 {
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    let sample: Vec<T> = dataset
        .choose_multiple(&mut rng, sample_size)
        .cloned()
        .collect();
    eprintln!("Sampled {} items", sample.len());
    let tree = PMTree::<T, D, B, P>::for_dataset(&sample, seed);

    let mut distribution = Vec::new();
    tree.for_each_lca_pair(|r, a, b| {
        let d = D::distance(&sample[a], &sample[b]);
        let gamma = r / d;
        distribution.push(gamma);
    });

    distribution.sort_by(|a, b| a.partial_cmp(&b).unwrap());

    distribution[(0.85 * distribution.len() as f64) as usize]
}

// the original paper states: in the radius filtering method, we set
// T = \beta n (n-1)/2 + k
//
// the parameter beta can be determined along with \alpha_2, using Eq. 10 (page 11 of the paper)
fn run_pm_lsh<const B: usize, const P: usize>(
    data: &[Vec<f64>],
    k: usize,
    m: usize,
) -> Vec<(f64, usize, usize)> {
    let alpha_1: f64 = 1.0 / std::f64::consts::E;
    let alpha_2: f64 = 0.0024;
    let n = data.len();
    let npairs = n * (n - 1) / 2;
    let max_candidates = (alpha_2 * ((n * (n - 1)) / 2) as f64 + k as f64) as usize;
    let gamma: f64 = estimate_gamma::<Vec<f64>, Euclidean, B, P>(data, 10000, 1234);
    // see Lemma 3 and Eq. 10 in the paper
    let t: f64 = ChiSquared::new(m as f64)
        .unwrap()
        .inverse_cdf(1.0 - alpha_1)
        .sqrt();
    eprintln!("n={n} npairs={npairs} T={max_candidates} gamma={gamma} t={t}");
    assert!(t >= 1.0);

    let t_project = Instant::now();
    let projected = project(&data, m, 1234);
    eprintln!("({:?}) random projection completed", t_project.elapsed());

    let t_index = Instant::now();
    let tree = PMTree::<_, Euclidean, B, 5>::for_dataset(&projected, 1234);
    eprintln!("({:?}) index built", t_index.elapsed());

    let t_query = Instant::now();
    let mut evaluated_candidates = 0;
    let mut candidates = BinaryHeap::new();
    tree.for_each_leaf_node_pair(|a, b| {
        let d = Euclidean::distance(&data[a], &data[b]);
        candidates.push((OrdF64(d), a, b));
        evaluated_candidates += 1;
        if candidates.len() > k {
            candidates.pop();
        }
    });

    let upper_bound = (candidates.peek().unwrap().0).0;
    let range = gamma * t * upper_bound;
    eprintln!("upper_bound={upper_bound} range={range}");
    assert!(range > upper_bound);

    tree.for_each_subtree_cluster(range, |a, b| {
        let proj_d = Euclidean::distance(&projected[a], &projected[b]);
        if proj_d <= t * upper_bound {
            evaluated_candidates += 1;
            let d = Euclidean::distance(&data[a], &data[b]);
            candidates.push((OrdF64(d), a, b));
            if candidates.len() > k {
                candidates.pop();
            }
        }
        evaluated_candidates > max_candidates // this is the stopping condition: when it is met, the for_each stops
    });
    eprintln!(
        " ({:?}) query evaluated (evaluated candidates = {})",
        t_query.elapsed(),
        evaluated_candidates
    );

    assert_eq!(candidates.len(), k);

    candidates
        .into_sorted_vec()
        .into_iter()
        .map(|(d, a, b)| (d.0, a, b))
        .collect()
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

    let res = match args.capacity {
        2 => run_pm_lsh::<2, 5>(&dataset, args.k, args.m),
        4 => run_pm_lsh::<4, 5>(&dataset, args.k, args.m),
        8 => run_pm_lsh::<8, 5>(&dataset, args.k, args.m),
        16 => run_pm_lsh::<16, 5>(&dataset, args.k, args.m),
        32 => run_pm_lsh::<32, 5>(&dataset, args.k, args.m),
        c => panic!("Unsupported capacity {}", c),
    };

    eprintln!("Overall time {:?}", t_overall.elapsed());

    for (d, a, b) in res {
        println!("{},{},{}", a, b, d);
    }

    Ok(())
}

#[test]
fn test_mnist() {
    let dataset = load_dataset(&ensure_mnist());

    let k = 10;

    let t_baseline = Instant::now();
    let mut expected = BinaryHeap::new();
    for a in 0..dataset.len() {
        for b in (a + 1)..dataset.len() {
            let d = OrdF64(Euclidean::distance(&dataset[a], &dataset[b]));
            expected.push((d, a, b));
            if expected.len() > k {
                expected.pop();
            }
        }
    }
    eprintln!("Time for linear scan {:?}", t_baseline.elapsed());
    let expected: Vec<(f64, usize, usize)> = expected
        .into_sorted_vec()
        .into_iter()
        .map(|(d, a, b)| (d.0, a, b))
        .collect();

    let t_tree = Instant::now();
    let res: Vec<(f64, usize, usize)> = run_pm_lsh::<16, 5>(&dataset, k, 16);
    eprintln!("Time for tree {:?}", t_tree.elapsed());

    assert_eq!(expected, res);
}

#[cfg(test)]
fn ensure_mnist() -> std::path::PathBuf {
    let local = PathBuf::from_str(".mnist-784-euclidean.hdf5").unwrap();
    if local.is_file() {
        return local;
    }
    eprintln!("Downloading the dataset for tests");
    let mut f = File::create(&local).unwrap();
    std::io::copy(
        &mut ureq::get("http://ann-benchmarks.com/mnist-784-euclidean.hdf5")
            .call()
            .unwrap()
            .into_reader(),
        &mut f,
    )
    .unwrap();
    local
}
