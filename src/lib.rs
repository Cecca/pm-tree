#![feature(drain_filter, new_uninit, maybe_uninit_uninit_array)]

use std::fmt::Debug;
use std::marker::PhantomData;

pub trait Distance<T> {
    fn distance(a: &T, b: &T) -> f64;
}

// #[derive(Serialize, Deserialize)]
#[derive(Debug)]
pub struct PMTree<T, D: Distance<T>, const B: usize, const P: usize> {
    root: Box<Node<T, D, B, P>>,
    pivots: [usize; P],
    _markert: PhantomData<T>,
    _markerd: PhantomData<D>,
}

impl<T, D: Distance<T>, const B: usize, const P: usize> PMTree<T, D, B, P> {
    pub fn new(pivots: [usize; P]) -> Self {
        Self {
            root: Box::new(Node::Leaf(LeafNode::new())),
            pivots,
            _markert: PhantomData,
            _markerd: PhantomData,
        }
    }

    pub fn for_dataset(dataset: &[T], seed: u64) -> Self {
        use rand::distributions::Uniform;
        use rand::prelude::*;
        use rand_xoshiro::Xoshiro256PlusPlus;
        // Pivot selection: As in the original paper, we take several random samples, and
        // select the one with the maximum sum of pairwise distances
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let distr = Uniform::new(0usize, dataset.len());
        let (_diversity, pivots) = (0..100)
            .map(|_| {
                let mut cands = [0; P];
                for i in 0..P {
                    cands[i] = distr.sample(&mut rng);
                }
                let mut diversity = 0.0;
                for i in 0..P {
                    for j in (i + 1)..P {
                        diversity += D::distance(&dataset[cands[i]], &dataset[cands[j]]);
                    }
                }
                (diversity, cands)
            })
            .max_by(|p1, p2| p1.0.partial_cmp(&p2.0).unwrap())
            .unwrap();

        let mut tree = Self::new(pivots);
        for i in 0..dataset.len() {
            tree.insert(i, dataset);
        }
        tree
    }

    pub fn insert(&mut self, o: usize, dataset: &[T]) {
        let possible_split = self.root.insert(o, None, self.pivots, dataset);
        if let Some((o1, n1, o2, n2)) = possible_split {
            // replace the current root with a new one
            let mut new_root = InnerNode::new();
            new_root.do_insert(o1, Box::new(n1), None, self.pivots, dataset);
            new_root.do_insert(o2, Box::new(n2), None, self.pivots, dataset);
            self.root = Box::new(Node::Inner(new_root));
        }
    }

    pub fn size(&self) -> usize {
        self.root.size()
    }

    pub fn height(&self) -> usize {
        self.root.height()
    }

    /// Run the given range query and return the count of distance computations
    pub fn range_query<F: FnMut(usize)>(
        &self,
        range: f64,
        q: &T,
        dataset: &[T],
        mut callback: F,
    ) -> usize {
        let mut cnt_dists = P;
        let mut q_pivot_dists = [0.0; P];
        for i in 0..P {
            q_pivot_dists[i] = D::distance(q, &dataset[self.pivots[i]]);
        }
        cnt_dists += self
            .root
            .range_query(range, q, q_pivot_dists, dataset, &mut callback);
        cnt_dists
    }

    #[cfg(test)]
    fn for_each_leaf<F: FnMut(&LeafNode<T, D, B, P>)>(&self, mut callback: F) {
        self.root.for_each_leaf(&mut callback);
    }

    #[cfg(test)]
    fn for_node_in_path<F: FnMut(&Node<T, D, B, P>)>(&self, id: usize, mut callback: F) {
        self.root.for_node_in_path(id, &mut callback);
    }
}

// #[derive(Serialize, Deserialize)]
#[derive(Debug, Copy, Clone)]
struct Hyperring {
    min: f64,
    max: f64,
}

impl Default for Hyperring {
    fn default() -> Self {
        Self {
            min: std::f64::INFINITY,
            max: 0.0,
        }
    }
}

// #[derive(Serialize, Deserialize)]
#[derive(Debug)]
enum Node<T, D: Distance<T>, const B: usize, const P: usize> {
    Inner(InnerNode<T, D, B, P>),
    Leaf(LeafNode<T, D, B, P>),
}

impl<T, D: Distance<T>, const B: usize, const P: usize> Node<T, D, B, P> {
    fn insert(
        &mut self,
        o: usize,
        parent: Option<usize>,
        pivots: [usize; P],
        dataset: &[T],
    ) -> Option<(usize, Self, usize, Self)> {
        match self {
            Node::Inner(inner) => inner
                .insert(o, parent, pivots, dataset)
                .map(|(o1, n1, o2, n2)| (o1, Self::Inner(n1), o2, Self::Inner(n2))),
            Node::Leaf(leaf) => leaf
                .insert(o, parent, pivots, dataset)
                .map(|(o1, n1, o2, n2)| (o1, Self::Leaf(n1), o2, Self::Leaf(n2))),
        }
    }

    fn height(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Inner(inner) => {
                1 + inner
                    .children
                    .iter()
                    .take(inner.len)
                    .map(|c| c.as_ref().unwrap().height())
                    .max()
                    .unwrap()
            }
        }
    }

    fn update_hyperrings_and_radius(
        &self,
        router: usize,
        pivots: [usize; P],
        hyperrings: &mut [Hyperring; P],
        dataset: &[T],
    ) -> f64 {
        match self {
            Self::Inner(inner) => inner
                .children
                .iter()
                .take(inner.len)
                .map(|c| {
                    c.as_ref()
                        .unwrap()
                        .update_hyperrings_and_radius(router, pivots, hyperrings, dataset)
                })
                .max_by(|r1, r2| r1.partial_cmp(&r2).unwrap())
                .unwrap(),
            Self::Leaf(leaf) => {
                let mut r = 0.0;
                for &o in leaf.elements.iter().take(leaf.len) {
                    let d = D::distance(&dataset[o], &dataset[router]);
                    if d > r {
                        r = d;
                    }
                    for p in 0..P {
                        let d = D::distance(&dataset[o], &dataset[pivots[p]]);
                        if d < hyperrings[p].min {
                            hyperrings[p].min = d;
                        }
                        if d > hyperrings[p].max {
                            hyperrings[p].max = d;
                        }
                    }
                }
                r
            }
        }
    }

    #[cfg(test)]
    fn for_each_leaf<F: FnMut(&LeafNode<T, D, B, P>)>(&self, callback: &mut F) {
        match self {
            Self::Leaf(leaf) => callback(leaf),
            Self::Inner(inner) => inner
                .children
                .iter()
                .take(inner.len)
                .for_each(|c| c.as_ref().unwrap().for_each_leaf(callback)),
        }
    }

    /// compute the size of this subtree
    fn size(&self) -> usize {
        match self {
            Self::Inner(inner) => inner
                .children
                .iter()
                .take(inner.len)
                .map(|c| c.as_ref().unwrap().size())
                .sum(),
            Self::Leaf(leaf) => leaf.len,
        }
    }

    /// performs a range query, and return the number of distances computed
    fn range_query<F: FnMut(usize)>(
        &self,
        range: f64,
        q: &T,
        q_pivot_dists: [f64; P],
        dataset: &[T],
        callback: &mut F,
    ) -> usize {
        match self {
            Self::Inner(inner) => {
                let mut dists = 0;
                for i in 0..inner.len {
                    dists += 1;
                    if inner.hyperrings[i]
                        .iter()
                        .zip(&q_pivot_dists)
                        .all(|(hr, qd)| qd - range <= hr.max && qd + range >= hr.min)
                        && D::distance(q, &dataset[inner.routers[i]]) <= inner.radius[i] + range
                    {
                        dists += inner.children[i].as_ref().unwrap().range_query(
                            range,
                            q,
                            q_pivot_dists,
                            dataset,
                            callback,
                        );
                    }
                }
                dists
            }
            Self::Leaf(leaf) => {
                let mut dists = 0;
                for i in 0..leaf.len {
                    // use the triangle inequality with the pivots to compute lower bounds to the
                    // distance between the query and the point, discarding the ones that are too far away
                    if leaf.pivot_distances[i]
                        .iter()
                        .zip(q_pivot_dists.iter())
                        .all(|(pd, qd)| (pd - qd).abs() <= range)
                    {
                        dists += 1;
                        if D::distance(q, &dataset[leaf.elements[i]]) <= range {
                            callback(leaf.elements[i]);
                        }
                    }
                }
                dists
            }
        }
    }

    #[cfg(test)]
    fn for_node_in_path<F: FnMut(&Node<T, D, B, P>)>(&self, id: usize, callback: &mut F) -> bool {
        match self {
            Self::Inner(inner) => {
                for child in inner.children[..inner.len].iter() {
                    let child = child.as_ref().unwrap();
                    if child.for_node_in_path(id, callback) {
                        callback(&self);
                        return true;
                    }
                }
                false
            }
            Self::Leaf(leaf) => {
                if let Some(_) = leaf.elements[..leaf.len].iter().find(|o| **o == id) {
                    callback(&self);
                    true
                } else {
                    false
                }
            }
        }
    }
}

fn promote<T, D: Distance<T>>(o: usize, os: &[usize], dataset: &[T]) -> (usize, usize) {
    // assert!(os.len() <= B);
    // // Pre-compute the pairwise distances
    // let mut distmatrix = [[std::f64::INFINITY; B + 1]; B + 1];
    // for i in 0..os.len() {
    //     for j in (i + 1)..os.len() {
    //         let d = D::distance(&dataset[os[i]], &dataset[os[j]]);
    //         distmatrix[i][j] = d;
    //         distmatrix[j][i] = d;
    //     }
    // }
    // for j in 0..os.len() {
    //     let d = D::distance(&dataset[o], &dataset[os[j]]);
    //     distmatrix[j][B] = d;
    //     distmatrix[B][j] = d;
    // }

    // // Compute how clusters would split
    // let mut radii_sums = [[std::f64::INFINITY; B+1]; B+1];
    // for i in 0..os.len() {
    //     for j in 0..os.len() {
    //         // os[i], os[j] is a pair
    //     }
    // }
    (o, os[0])
}

// #[derive(Serialize, Deserialize)]
// #[derive(Debug)]
struct InnerNode<T, D: Distance<T>, const B: usize, const P: usize> {
    len: usize,
    parent_distance: [Option<f64>; B],
    routers: [usize; B],
    radius: [f64; B],
    hyperrings: [[Hyperring; P]; B],
    children: [Option<Box<Node<T, D, B, P>>>; B],
    _markert: PhantomData<T>,
    _markerd: PhantomData<D>,
}

impl<T, D: Distance<T>, const B: usize, const P: usize> InnerNode<T, D, B, P> {
    fn new() -> Self {
        let children: [Option<Box<Node<T, D, B, P>>>; B] =
            unsafe { std::mem::transmute_copy(&[0usize; B]) };
        Self {
            len: 0,
            parent_distance: [None; B],
            routers: [0; B],
            radius: [0.0; B],
            hyperrings: [[Default::default(); P]; B],
            children,
            _markert: PhantomData,
            _markerd: PhantomData,
        }
    }

    fn insert(
        &mut self,
        o: usize,
        parent: Option<usize>,
        pivots: [usize; P],
        dataset: &[T],
    ) -> Option<(usize, Self, usize, Self)> {
        // find the routing element closest
        let closest = (0..self.len)
            .map(|i| {
                let d = D::distance(&dataset[self.routers[i]], &dataset[o]);
                (i, d)
            })
            .min_by(|p1, p2| p1.1.partial_cmp(&p2.1).unwrap());

        if let Some((closest, distance)) = closest {
            let possible_split =
                self.children[closest]
                    .as_mut()
                    .unwrap()
                    .insert(o, Some(closest), pivots, dataset);

            if let Some((o1, n1, o2, n2)) = possible_split {
                // replace node `closest` with o1
                self.replace_at(closest, o1, Box::new(n1), parent, pivots, dataset);

                // now, insert the new node directly, if possible, otherwise split the current node
                if self.len < B {
                    // FIXME: in case of split, update the radius and hyperrings without recomputing from scratch
                    self.do_insert(o2, Box::new(n2), parent, pivots, dataset);
                    None
                } else {
                    Some(self.split(o2, Box::new(n2), pivots, dataset))
                }
            } else {
                // no split, we simply update the radius and hyperrings
                self.radius[closest] = distance;
                for j in 0..P {
                    if distance > self.hyperrings[closest][j].max {
                        self.hyperrings[closest][j].max = distance;
                    }
                    if distance < self.hyperrings[closest][j].min {
                        self.hyperrings[closest][j].min = distance;
                    }
                }
                None
            }
        } else {
            panic!("Empty inner node?");
        }
    }

    fn do_insert(
        &mut self,
        o: usize,
        child: Box<Node<T, D, B, P>>,
        parent: Option<usize>,
        pivots: [usize; P],
        dataset: &[T],
    ) {
        assert!(self.len < B);
        self.replace_at(self.len, o, child, parent, pivots, dataset);
        self.len += 1;
    }

    fn replace_at(
        &mut self,
        i: usize,
        o: usize,
        child: Box<Node<T, D, B, P>>,
        parent: Option<usize>,
        pivots: [usize; P],
        dataset: &[T],
    ) {
        assert!(i < B);
        self.routers[i] = o;
        self.parent_distance[i] = parent.map(|parent| D::distance(&dataset[o], &dataset[parent]));
        self.hyperrings[i] = [Default::default(); P];
        self.children[i].replace(child);
        let r = self.children[i].as_mut().unwrap().update_hyperrings_and_radius(o, pivots, &mut self.hyperrings[i], dataset);
        self.radius[i] = r;
    }

    fn split(
        &mut self,
        o: usize,
        child: Box<Node<T, D, B, P>>,
        pivots: [usize; P],
        dataset: &[T],
    ) -> (usize, Self, usize, Self) {
        assert!(self.len == B);
        let (o1, o2) = promote::<_, D>(o, &self.routers[..self.len], dataset);
        let data1 = &dataset[o1];
        let data2 = &dataset[o2];
        let mut n1 = Self::new();
        let mut n2 = Self::new();

        if D::distance(&dataset[o], data1) < D::distance(&dataset[o], data2) {
            n1.do_insert(o, child, Some(o1), pivots, dataset);
        } else {
            n2.do_insert(o, child, Some(o2), pivots, dataset);
        }
        for i in 0..self.len {
            if D::distance(&dataset[self.routers[i]], data1)
                < D::distance(&dataset[self.routers[i]], data2)
            {
                // add to first node
                assert!(n1.len < B);
                n1.do_insert(
                    self.routers[i],
                    self.children[i].take().unwrap(),
                    Some(o1),
                    pivots,
                    dataset,
                );
            } else {
                // add to second node
                assert!(n2.len < B);
                n2.do_insert(
                    self.routers[i],
                    self.children[i].take().unwrap(),
                    Some(o2),
                    pivots,
                    dataset,
                );
            }
        }
        self.len = 0;
        (o1, n1, o2, n2)
    }
}

impl<T, D, const B: usize, const P: usize> std::fmt::Debug for InnerNode<T, D, B, P>
where
    T: Debug,
    D: Distance<T> + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InnerNode")
            .field("len", &self.len)
            .field(
                "parent_distance",
                &format_args!("{:?}", &self.parent_distance[..self.len]),
            )
            .field("routers", &format_args!("{:?}", &self.routers[..self.len]))
            .field("radius", &format_args!("{:?}", &self.radius[..self.len]))
            .field(
                "hyperrings",
                &format_args!("{:?}", &self.hyperrings[..self.len]),
            )
            // .field("children", &format_args!("{:?}", &self.children[..self.len]))
            .finish()
    }
}

struct LeafNode<T, D: Distance<T>, const B: usize, const P: usize> {
    len: usize,
    parent_distance: [Option<f64>; B],
    elements: [usize; B],
    pivot_distances: [[f64; P]; B],
    _markert: PhantomData<T>,
    _markerd: PhantomData<D>,
}

impl<T, D: Distance<T>, const B: usize, const P: usize> LeafNode<T, D, B, P> {
    fn new() -> Self {
        Self {
            len: 0,
            parent_distance: [None; B],
            pivot_distances: [[0.0; P]; B],
            elements: [0; B],
            _markert: PhantomData,
            _markerd: PhantomData,
        }
    }

    fn insert(
        &mut self,
        o: usize,
        parent: Option<usize>,
        pivots: [usize; P],
        dataset: &[T],
    ) -> Option<(usize, Self, usize, Self)> {
        if self.len == B {
            Some(self.split(o, pivots, dataset))
        } else {
            self.do_insert(o, parent, pivots, dataset);
            None
        }
    }

    fn do_insert(&mut self, o: usize, parent: Option<usize>, pivots: [usize; P], dataset: &[T]) {
        assert!(self.len < B);
        self.parent_distance[self.len] =
            parent.map(|parent| D::distance(&dataset[o], &dataset[parent]));
        self.elements[self.len] = o;
        for i in 0..P {
            self.pivot_distances[self.len][i] = D::distance(&dataset[o], &dataset[pivots[i]]);
        }
        self.len += 1;
    }

    /// Split the current node, leaving it empty, and returns two new nodes with the corresponding routing points
    fn split(&mut self, o: usize, pivots: [usize; P], dataset: &[T]) -> (usize, Self, usize, Self) {
        assert!(self.len == B);
        let (o1, o2) = promote::<_, D>(o, &self.elements[..self.len], dataset);
        let data1 = &dataset[o1];
        let data2 = &dataset[o2];
        let mut n1 = Self::new();
        let mut n2 = Self::new();

        if D::distance(&dataset[o], data1) < D::distance(&dataset[o], data2) {
            n1.do_insert(o, Some(o1), pivots, dataset);
        } else {
            n2.do_insert(o, Some(o2), pivots, dataset);
        }
        for i in 0..self.len {
            let d1 = D::distance(&dataset[self.elements[i]], data1);
            let d2 = D::distance(&dataset[self.elements[i]], data2);
            if d1 < d2 {
                // add to first node
                assert!(n1.len < B);
                n1.do_insert(self.elements[i], Some(o1), pivots, dataset);
            } else {
                // add to second node
                assert!(n2.len < B);
                n2.do_insert(self.elements[i], Some(o2), pivots, dataset);
            }
        }
        self.len = 0;

        (o1, n1, o2, n2)
    }
}

impl<T, D, const B: usize, const P: usize> std::fmt::Debug for LeafNode<T, D, B, P>
where
    T: Debug,
    D: Distance<T> + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeafNode")
            .field("len", &self.len)
            .field(
                "parent_distance",
                &format_args!("{:?}", &self.parent_distance[..self.len]),
            )
            .field(
                "pivot_distances",
                &format_args!("{:?}", &self.pivot_distances[..self.len]),
            )
            .field(
                "elements",
                &format_args!("{:?}", &self.elements[..self.len]),
            )
            .finish()
    }
}

// #[derive(Serialize, Deserialize)]
#[derive(Debug)]
pub struct Euclidean;

impl Distance<Vec<f64>> for Euclidean {
    fn distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        assert!(a.len() == b.len());
        a.iter()
            .zip(b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Distance, Euclidean, PMTree};
    use pretty_assertions::assert_eq;
    use std::collections::BTreeSet;
    use std::fs::File;
    use std::io::prelude::*;
    use std::path::PathBuf;
    use std::str::FromStr;
    use std::time::Instant;

    fn ensure_glove25() -> std::path::PathBuf {
        let local = PathBuf::from_str(".glove-25.hdf5").unwrap();
        if local.is_file() {
            return local;
        }
        eprintln!("Downloading the dataset for tests");
        let mut f = File::create(&local).unwrap();
        std::io::copy(
            // the file is, very simply, stored in the public folder of my personal dropbox
            &mut ureq::get(
                "https://dl.dropboxusercontent.com/s/kdh02vg1lb3qm5j/glove-25.hdf5?dl=0",
            )
            .call()
            .unwrap()
            .into_reader(),
            &mut f,
        )
        .unwrap();
        local
    }

    #[test]
    fn construction() {
        let dataset = vec![
            vec![-0.479525, -0.0900315],
            vec![1.77065, -2.03216],
            vec![-0.709144, -0.469802],
            vec![1.90905, -1.91834],
            vec![-0.355722, 1.64757],
            vec![-0.658588, 0.490459],
            vec![0.738724, -0.432818],
            vec![2.21156, 1.1524],
            vec![0.959106, -1.03304],
            vec![-0.543183, 0.201419],
        ];

        let mut pm_tree = PMTree::<_, Euclidean, 2, 3>::new([1, 3, 8]);
        for i in 0..dataset.len() {
            pm_tree.insert(i, &dataset);
        }

        assert_eq!(pm_tree.size(), dataset.len());
    }

    #[test]
    fn pivot_distances() {
        let dataset = vec![
            vec![-0.479525, -0.0900315],
            vec![1.77065, -2.03216],
            vec![-0.709144, -0.469802],
            vec![1.90905, -1.91834],
            vec![-0.355722, 1.64757],
            vec![-0.658588, 0.490459],
            vec![0.738724, -0.432818],
            vec![2.21156, 1.1524],
            vec![0.959106, -1.03304],
            vec![-0.543183, 0.201419],
        ];

        const P: usize = 3;
        let pivots = [1, 3, 8];
        let mut pm_tree = PMTree::<_, Euclidean, 2, P>::new(pivots);
        for i in 0..dataset.len() {
            pm_tree.insert(i, &dataset);
        }

        pm_tree.for_each_leaf(|leaf| {
            for i in 0..leaf.len {
                let x = &dataset[leaf.elements[i]];
                let mut actual_dists = [0.0; P];
                for j in 0..P {
                    let pivot = &dataset[pivots[j]];
                    actual_dists[j] = Euclidean::distance(x, pivot);
                }
                assert_eq!(actual_dists, leaf.pivot_distances[i]);
            }
        })
    }

    macro_rules! test_range_query {
        ($dataset:ident, $B:literal, $P:literal, $from:literal, $range:literal) => {
            let t_build = Instant::now();
            let pm_tree = PMTree::<Vec<f64>, Euclidean, $B, $P>::for_dataset(&$dataset, 1234);
            assert_eq!(pm_tree.size(), $dataset.len());

            eprintln!(
                "tree built in {:?}, with height {}",
                t_build.elapsed(),
                pm_tree.height()
            );
            let mut f = File::create("/tmp/tree.txt").unwrap();
            writeln!(f, "{:#?}", pm_tree).unwrap();

            let query = &$dataset[0];
            let mut q_pivot_dists = [0.0; $P];
            for i in 0..$P {
                q_pivot_dists[i] = Euclidean::distance(query, &$dataset[pm_tree.pivots[i]]);
            }

            pm_tree.for_node_in_path(623, |node| {
                pm_tree.pivots;
                match node {
                    Node::Inner(inner) => {
                        let mut conditions = Vec::new();
                        let mut tests = Vec::new();
                        let mut distances = Vec::new();
                        let mut thresholds = Vec::new();
                        for i in 0..inner.len {
                            conditions.push(
                                inner.hyperrings[i]
                                    .iter()
                                    .zip(&q_pivot_dists)
                                    .all(|(hr, qd)| qd - $range <= hr.max && qd + $range >= hr.min),
                            );
                            tests.push(
                                Euclidean::distance(query, &$dataset[inner.routers[i]])
                                    <= inner.radius[i] + $range,
                            );
                            distances.push(Euclidean::distance(query, &$dataset[inner.routers[i]]));
                            thresholds.push(inner.radius[i] + $range);
                        }
                        println!("inner distances {:?}", distances);
                        println!("inner threshold {:?}", thresholds);
                        println!("inner tests {:?}", tests);
                        println!("inner {:?} {:?}", conditions, &inner.routers[..inner.len]);
                    }
                    Node::Leaf(leaf) => {
                        let mut conditions = Vec::new();
                        let mut tests = Vec::new();
                        for i in 0..leaf.len {
                            conditions.push(
                                leaf.pivot_distances[i]
                                    .iter()
                                    .zip(q_pivot_dists.iter())
                                    .all(|(pd, qd)| (pd - qd).abs() <= $range),
                            );
                            tests.push(
                                Euclidean::distance(query, &$dataset[leaf.elements[i]]) <= $range,
                            )
                        }
                        println!("leaf elements {:?}", &leaf.elements[..leaf.len]);
                        println!("leaf tests {:?}", tests);
                        println!("leaf {:?}", conditions)
                    }
                }
            });

            let t_baseline = Instant::now();
            let mut expected = BTreeSet::new();
            for (i, v) in $dataset.iter().enumerate() {
                if Euclidean::distance(v, query) <= $range {
                    expected.insert(i);
                }
            }
            eprintln!("Time for linear scan {:?}", t_baseline.elapsed());

            let t_tree = Instant::now();
            let mut res = BTreeSet::new();
            let cnt_dists = pm_tree.range_query($range, query, &$dataset, |i| {
                res.insert(i);
            });
            eprintln!("Time for tree {:?}", t_tree.elapsed());
            eprintln!(
                "computed {} distances, solution for range {} has {} elements, and should have {}",
                cnt_dists,
                $range,
                res.len(),
                expected.len()
            );

            assert_eq!(expected, res);
        };
    }

    #[test]
    fn range_query_glove25() {
        use ndarray::s;
        let path = ensure_glove25();
        let f = hdf5::File::open(path).unwrap();
        let data = f.dataset("/train").unwrap();
        let array = data.read_slice_2d::<f64, _>(s![..624, ..]).unwrap();
        let mut dataset = Vec::new();
        for row in array.rows() {
            let r = row.as_slice().unwrap();
            dataset.push(Vec::from(r));
        }

        test_range_query!(dataset, 32, 8, 0, 2.0);
        test_range_query!(dataset, 32, 8, 0, 4.0);
    }

    #[test]
    fn range_query() {
        let dataset = vec![
            vec![-0.479525, -0.0900315],
            vec![1.77065, -2.03216],
            vec![-0.709144, -0.469802],
            vec![1.90905, -1.91834],
            vec![-0.355722, 1.64757],
            vec![-0.658588, 0.490459],
            vec![0.738724, -0.432818],
            vec![2.21156, 1.1524],
            vec![0.959106, -1.03304],
            vec![-0.543183, 0.201419],
        ];

        let mut pm_tree = PMTree::<_, Euclidean, 2, 3>::new([1, 3, 8]);
        for i in 0..dataset.len() {
            pm_tree.insert(i, &dataset);
        }

        let query = &dataset[0];
        let range = 0.2;

        let mut expected = Vec::new();
        for (i, v) in dataset.iter().enumerate() {
            if Euclidean::distance(v, query) <= range {
                expected.push(i);
            }
        }

        let mut res = Vec::new();
        let cnt_dists = pm_tree.range_query(range, query, &dataset, |i| res.push(i));
        res.sort();
        eprintln!("computed {cnt_dists} distances");

        assert_eq!(expected, res);
    }
}
