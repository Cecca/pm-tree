#![feature(drain_filter, new_uninit, maybe_uninit_uninit_array)]

use std::marker::PhantomData;

// use serde_derive::{Deserialize, Serialize};

const B: usize = 2;

pub trait Distance<T> {
    fn distance(a: &T, b: &T) -> f64;
}

// #[derive(Serialize, Deserialize)]
#[derive(Debug)]
pub struct PMTree<T, D: Distance<T>, const P: usize> {
    root: Box<Node<T, D, P>>,
    pivots: [usize; P],
    _markert: PhantomData<T>,
    _markerd: PhantomData<D>,
}

impl<T, D: Distance<T>, const P: usize> PMTree<T, D, P> {
    pub fn new(pivots: [usize; P]) -> Self {
        Self {
            root: Box::new(Node::Leaf(LeafNode::new())),
            pivots,
            _markert: PhantomData,
            _markerd: PhantomData,
        }
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
    fn for_each_leaf<F: FnMut(&LeafNode<T, D, P>)>(&self, mut callback: F) {
        self.root.for_each_leaf(&mut callback);
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
enum Node<T, D: Distance<T>, const P: usize> {
    Inner(InnerNode<T, D, P>),
    Leaf(LeafNode<T, D, P>),
}

impl<T, D: Distance<T>, const P: usize> Node<T, D, P> {
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
    fn for_each_leaf<F: FnMut(&LeafNode<T, D, P>)>(&self, callback: &mut F) {
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
}

// #[derive(Serialize, Deserialize)]
#[derive(Debug)]
struct InnerNode<T, D: Distance<T>, const P: usize> {
    len: usize,
    parent_distance: [Option<f64>; B],
    routers: [usize; B],
    radius: [f64; B],
    hyperrings: [[Hyperring; P]; B],
    children: [Option<Box<Node<T, D, P>>>; B],
    _markert: PhantomData<T>,
    _markerd: PhantomData<D>,
}

impl<T, D: Distance<T>, const P: usize> InnerNode<T, D, P> {
    fn new() -> Self {
        let children: [Option<Box<Node<T, D, P>>>; B] = unsafe { std::mem::transmute([0usize; B]) };
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
                let d = D::distance(&dataset[i], &dataset[o]);
                (i, d)
            })
            .min_by(|p1, p2| p1.1.partial_cmp(&p2.1).unwrap());

        if let Some((closest, _distance)) = closest {
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
                    self.do_insert(o2, Box::new(n2), parent, pivots, dataset);
                    None
                } else {
                    Some(self.split(o2, Box::new(n2), pivots, dataset))
                }
            } else {
                None
            }
        } else {
            panic!("Empty inner node?");
        }
    }

    fn do_insert(
        &mut self,
        o: usize,
        child: Box<Node<T, D, P>>,
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
        child: Box<Node<T, D, P>>,
        parent: Option<usize>,
        pivots: [usize; P],
        dataset: &[T],
    ) {
        assert!(i < B);
        self.routers[i] = o;
        self.parent_distance[i] = parent.map(|parent| D::distance(&dataset[o], &dataset[parent]));
        self.hyperrings[i] = [Default::default(); P];
        let r = child.update_hyperrings_and_radius(o, pivots, &mut self.hyperrings[i], dataset);
        self.radius[i] = r;
        self.children[i].replace(child);
    }

    fn split(
        &mut self,
        o2: usize,
        child: Box<Node<T, D, P>>,
        pivots: [usize; P],
        dataset: &[T],
    ) -> (usize, Self, usize, Self) {
        assert!(self.len == B);
        let o1 = self.routers[0];
        let data1 = &dataset[o1];
        let data2 = &dataset[o2];
        let mut n1 = Self::new();
        let mut n2 = Self::new();

        n2.do_insert(o2, child, Some(o2), pivots, dataset);
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

// #[derive(Serialize, Deserialize)]
#[derive(Debug)]
struct LeafNode<T, D: Distance<T>, const P: usize> {
    len: usize,
    parent_distance: [Option<f64>; B],
    elements: [usize; B],
    pivot_distances: [[f64; P]; B],
    _markert: PhantomData<T>,
    _markerd: PhantomData<D>,
}

impl<T, D: Distance<T>, const P: usize> LeafNode<T, D, P> {
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
    fn split(
        &mut self,
        o2: usize,
        pivots: [usize; P],
        dataset: &[T],
    ) -> (usize, Self, usize, Self) {
        assert!(self.len == B);
        let o1 = self.elements[0];
        let data1 = &dataset[o1];
        let data2 = &dataset[o2];
        let mut n1 = Self::new();
        let mut n2 = Self::new();

        n2.insert(o2, Some(o2), pivots, dataset);
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
    use crate::{Distance, Euclidean, PMTree};
    use std::fs::File;
    use std::io::prelude::*;

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

        let mut pm_tree = PMTree::<_, Euclidean, 3>::new([1, 3, 8]);
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
        let mut pm_tree = PMTree::<_, Euclidean, P>::new(pivots);
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

        let mut pm_tree = PMTree::<_, Euclidean, 3>::new([1, 3, 8]);
        for i in 0..dataset.len() {
            pm_tree.insert(i, &dataset);
        }

        let mut f = File::create("tree.txt").unwrap();
        writeln!(f, "{:#?}", pm_tree).unwrap();
        drop(f);

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
