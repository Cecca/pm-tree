#![feature(drain_filter, new_uninit, maybe_uninit_uninit_array)]

use progress_logger::ProgressLogger;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::time::Instant;

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

impl<T: Debug, D: Distance<T> + Debug, const B: usize, const P: usize> PMTree<T, D, B, P> {
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

    pub fn closest_pairs(&self, k: usize, dataset: &[T]) -> Vec<(f64, usize, usize)> {
        let mut result: BinaryHeap<(OrdF64, usize, usize)> = BinaryHeap::new();

        let t_init_result = Instant::now();
        // init the result with self-joins of the leaves
        self.for_each_leaf(|leaf: &LeafNode<T, D, B, P>| {
            for i in 0..leaf.len {
                for j in (i + 1)..leaf.len {
                    let a = leaf.elements[i];
                    let b = leaf.elements[j];
                    let d = D::distance(&dataset[a], &dataset[b]);
                    result.push((OrdF64(d), a, b));
                    if result.len() > k {
                        result.pop();
                    }
                }
            }
        });
        eprintln!(
            " ({:?}) initialized result set, distance upper bound is {}",
            t_init_result.elapsed(),
            (result.peek().unwrap().0).0
        );

        assert!(result.len() == k);

        match self.root.as_ref() {
            Node::Leaf(_leaf) => (), // do nothing in this case
            Node::Inner(inner) => {
                let mut visited_leaves = 0;
                let mut pq: BinaryHeap<NodePair<T, D, B, P>> = BinaryHeap::new();
                let mut pl = ProgressLogger::builder()
                    .with_items_name("distances")
                    .start();
                // initialize with the direct children of the root
                for i in 0..inner.len {
                    for j in i..inner.len {
                        let a = inner.routers[i];
                        let b = inner.routers[j];
                        let mindist = inner.mindist(&inner, i, j, self.pivots, dataset);
                        pq.push(NodePair::new(
                            1,
                            mindist,
                            a,
                            &inner.children[i].as_ref().unwrap(),
                            b,
                            &inner.children[j].as_ref().unwrap(),
                        ));
                    }
                }

                while let Some(node_pair) = pq.pop() {
                    assert!(result.len() == k);
                    // eprintln!(" Popped pair at mindist {}", node_pair.mindist);
                    if OrdF64(node_pair.mindist) > result.peek().unwrap().0 {
                        // early stop
                        break;
                    }

                    if node_pair.a == node_pair.b {
                        // self join case
                        match node_pair.anode.as_ref() {
                            Node::Leaf(_leaf) => {
                                // the self join of a leaf can be skipped, since we already did it in the initialization phase
                            }
                            Node::Inner(inner) => {
                                for i in 0..inner.len {
                                    for j in i..inner.len {
                                        let a = inner.routers[i];
                                        let b = inner.routers[j];
                                        let mindist =
                                            inner.mindist(&inner, i, j, self.pivots, dataset);
                                        // eprintln!(" Enqueuing at mindist {}", mindist);
                                        pq.push(NodePair::new(
                                            node_pair.level + 1,
                                            mindist,
                                            a,
                                            &inner.children[i].as_ref().unwrap(),
                                            b,
                                            &inner.children[j].as_ref().unwrap(),
                                        ));
                                    }
                                }
                            }
                        }
                    } else {
                        // join of two different nodes
                        match (node_pair.anode.as_ref(), node_pair.bnode.as_ref()) {
                            (Node::Leaf(l1), Node::Leaf(l2)) => {
                                visited_leaves += 1;
                                let t_leaf = Instant::now();
                                for &a in &l1.elements[..l1.len] {
                                    for &b in &l2.elements[..l2.len] {
                                        let d = D::distance(&dataset[a], &dataset[b]);
                                        result.push((OrdF64(d), a, b));
                                        if result.len() > k {
                                            result.pop();
                                        }
                                    }
                                }
                                pl.update_light((l1.len * l2.len) as u64);
                                // eprintln!(" Solved leaf in {:?}", t_leaf.elapsed());
                            }
                            (Node::Inner(n1), Node::Inner(n2)) => {
                                for i in 0..n1.len {
                                    for j in 0..n2.len {
                                        let a = n1.routers[i];
                                        let b = n2.routers[j];
                                        let mindist = n1.mindist(n2, i, j, self.pivots, dataset);
                                        // eprintln!(" Enqueuing at mindist {}", mindist);
                                        pq.push(NodePair::new(
                                            node_pair.level + 1,
                                            mindist,
                                            a,
                                            &n1.children[i].as_ref().unwrap(),
                                            b,
                                            &n2.children[j].as_ref().unwrap(),
                                        ));
                                    }
                                }
                            }
                            _ => panic!("leaves should all be at the same level!"),
                        }
                    }
                }
                pl.stop();
                let nleaves = self.count_leaves();
                eprintln!(
                    " visited {} leaf pairs over {}",
                    visited_leaves,
                    nleaves * nleaves
                );
            }
        }

        result
            .into_sorted_vec()
            .into_iter()
            .take(k)
            .map(|(d, a, b)| (d.0, std::cmp::min(a, b), std::cmp::max(a, b)))
            .collect()
    }

    pub fn for_each_subtree_cluster<'tree, F: FnMut(usize, usize) -> bool>(
        &'tree self,
        radius: f64,
        mut callback: F,
    ) {
        let mut subtrees = Vec::new();
        let mut stack = Vec::new();

        stack.push((&self.root, 0));
        while let Some((node, depth)) = stack.pop() {
            match node.as_ref() {
                Node::Leaf(_leaf) => (), // do nothing
                Node::Inner(inner) => {
                    for i in 0..inner.len {
                        if inner.radius[i] < radius {
                            subtrees.push((inner.radius[i], inner.children[i].as_ref().unwrap()));
                        } else {
                            stack.push((&inner.children[i].as_ref().unwrap(), depth + 1));
                        }
                    }
                }
            }
        }

        // sort the subtrees by increasing radius
        subtrees.sort_by(|p1, p2| p1.0.partial_cmp(&p2.0).unwrap());

        for (_radius, node) in subtrees {
            node.for_each_id(&mut |a| {
                node.for_each_id(&mut|b| {
                    if a < b { // don't evaluate pairs twice, and skip self-pairs
                        let should_stop = callback(a, b);
                        if should_stop {
                            return;
                        }
                    }
                });
            });
        }
    }

    pub fn for_each_leaf_node_pair<F: FnMut(usize, usize)>(&self, mut callback: F) {
        self.for_each_leaf(|leaf| {
            for i in 0..leaf.len {
                for j in (i + 1)..leaf.len {
                    let a = leaf.elements[i];
                    let b = leaf.elements[j];
                    callback(a, b)
                }
            }
        });
    }

    fn for_each_leaf<F: FnMut(&LeafNode<T, D, B, P>)>(&self, mut callback: F) {
        self.root.for_each_leaf(&mut callback);
    }

    // the callback takes two element IDs, along with the radius of their lowest common ancestor
    pub fn for_each_lca_pair<F>(&self, mut callback: F)
    where
        F: FnMut(f64, usize, usize),
    {
        let mut stack = Vec::new();
        match self.root.as_ref() {
            Node::Leaf(_) => (), // nothing to do
            Node::Inner(inner) => {
                for i in 0..inner.len {
                    stack.push((inner.radius[i], inner.children[i].as_ref().unwrap()))
                }
            }
        }

        while let Some((radius, node)) = stack.pop() {
            match node.as_ref() {
                Node::Leaf(_) => (), // nothing to do
                Node::Inner(inner) => {
                    for i in 0..inner.len {
                        for j in (i + 1)..inner.len {
                            inner.children[i].as_ref().unwrap().for_each_id(&mut |a| {
                                inner.children[j].as_ref().unwrap().for_each_id(&mut |b| {
                                    callback(radius, a, b);
                                })
                            });
                        }
                    }
                }
            }
        }
    }

    fn count_leaves(&self) -> usize {
        let mut cnt = 0;
        self.for_each_leaf(|_| cnt += 1);
        cnt
    }

    #[cfg(test)]
    fn for_node_in_path<F: FnMut(&Node<T, D, B, P>)>(&self, id: usize, mut callback: F) {
        self.root.for_node_in_path(id, &mut callback);
    }
}

struct NodePair<'tree, T, D: Distance<T>, const B: usize, const P: usize> {
    level: usize,
    mindist: f64,
    a: usize,
    anode: &'tree Box<Node<T, D, B, P>>,
    b: usize,
    bnode: &'tree Box<Node<T, D, B, P>>,
}

impl<'tree, T, D: Distance<T>, const B: usize, const P: usize> NodePair<'tree, T, D, B, P> {
    fn new(
        level: usize,
        mindist: f64,
        a: usize,
        anode: &'tree Box<Node<T, D, B, P>>,
        b: usize,
        bnode: &'tree Box<Node<T, D, B, P>>,
    ) -> Self {
        let mindist = if mindist < 0.0 { 0.0 } else { mindist };
        if a < b {
            Self {
                level,
                mindist,
                a,
                anode,
                b,
                bnode,
            }
        } else {
            Self {
                level,
                mindist,
                a: b,
                anode: bnode,
                b: a,
                bnode: anode,
            }
        }
    }
}

impl<'tree, T, D: Distance<T>, const B: usize, const P: usize> Eq for NodePair<'tree, T, D, B, P> {}

impl<'tree, T, D: Distance<T>, const B: usize, const P: usize> PartialEq
    for NodePair<'tree, T, D, B, P>
{
    fn eq(&self, other: &Self) -> bool {
        self.mindist == other.mindist && self.level == other.level
    }
}

impl<'tree, T, D: Distance<T>, const B: usize, const P: usize> PartialOrd
    for NodePair<'tree, T, D, B, P>
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.mindist
            .partial_cmp(&other.mindist)
            .map(|o| o.reverse().then(self.level.cmp(&other.level)))
    }
}

impl<'tree, T, D: Distance<T>, const B: usize, const P: usize> Ord for NodePair<'tree, T, D, B, P> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

// #[derive(Serialize, Deserialize)]
#[derive(Debug, Copy, Clone, PartialEq)]
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

impl Hyperring {
    /// Return a lower bound to the distance between the nodes belonging to the two given
    /// hyperrings, assuming that they both refer to the same pivot.
    /// It is up to the caller to ensure that this holds
    fn distance(&self, other: &Self) -> f64 {
        if self.max < other.min {
            other.min - self.max
        } else if other.max < self.min {
            self.min - other.max
        } else {
            // the two hyperrings overlap
            0.0
        }
    }
}

// #[derive(Serialize, Deserialize)]
#[derive(Debug)]
enum Node<T, D: Distance<T>, const B: usize, const P: usize> {
    Inner(InnerNode<T, D, B, P>),
    Leaf(LeafNode<T, D, B, P>),
}

impl<T: Debug, D: Distance<T> + Debug, const B: usize, const P: usize> Node<T, D, B, P> {
    fn insert(
        &mut self,
        o: usize,
        parent: Option<usize>,
        pivots: [usize; P],
        dataset: &[T],
    ) -> Option<(usize, Self, usize, Self)> {
        let mut pivot_dists = [0.0; P];
        for p_idx in 0..P {
            pivot_dists[p_idx] = D::distance(&dataset[o], &dataset[pivots[p_idx]]);
        }
        match self {
            Node::Inner(inner) => inner
                .insert(o, parent, pivot_dists, pivots, dataset)
                .map(|(o1, n1, o2, n2)| (o1, Self::Inner(n1), o2, Self::Inner(n2))),
            Node::Leaf(leaf) => leaf
                .insert(o, parent, pivot_dists, pivots, dataset)
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

    fn for_each_id<F: FnMut(usize)>(&self, callback: &mut F) {
        self.for_each_leaf(&mut |leaf: &LeafNode<T, D, B, P>| {
            for id in &leaf.elements[..leaf.len] {
                callback(*id);
            }
        });
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
    let mut distances = vec![vec![0.0; os.len() + 1]; os.len() + 1];
    // compute cluster sizes
    for (i, &i_idx) in os.iter().enumerate() {
        for (j, &j_idx) in os[(i + 1)..].iter().enumerate() {
            let d = D::distance(&dataset[i_idx], &dataset[j_idx]);
            distances[i][j] = d;
            distances[j][i] = d;
        }
        let d = D::distance(&dataset[i_idx], &dataset[o]);
        distances[i][os.len()] = d;
        distances[os.len()][i] = d;
    }

    // find the best pair
    let mut best_pair = (0, 0);
    let mut best_radius = std::f64::INFINITY;
    for (i, &i_idx) in os.iter().enumerate() {
        for (j, &j_idx) in os[(i + 1)..].iter().enumerate() {
            // accumulate the maximum radius
            for h in 0..distances.len() {
                let d1 = distances[i][h];
                let d2 = distances[j][h];
                let r = std::cmp::min_by(d1, d2, |a, b| a.partial_cmp(&b).unwrap());
                if r < best_radius {
                    best_radius = r;
                    best_pair = (i_idx, j_idx);
                }
            }
        }
        // do it also for `o`
        for h in 0..distances.len() {
            let d1 = distances[i][h];
            let d2 = distances[os.len()][h];
            let r = std::cmp::min_by(d1, d2, |a, b| a.partial_cmp(&b).unwrap());
            if r < best_radius {
                best_radius = r;
                best_pair = (i_idx, o);
            }
        }
    }

    best_pair
}

macro_rules! update_max {
    ($reference:expr, $candidate:expr) => {
        if $candidate > $reference {
            $reference = $candidate
        }
    };
    ($reference:expr, $candidate:ident) => {
        if $candidate > $reference {
            $reference = $candidate
        }
    };
}

macro_rules! update_min {
    ($reference:expr, $candidate:expr) => {
        if $candidate < $reference {
            $reference = $candidate
        }
    };
    ($reference:expr, $candidate:ident) => {
        if $candidate < $reference {
            $reference = $candidate
        }
    };
}

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

impl<T: Debug, D: Distance<T> + Debug, const B: usize, const P: usize> InnerNode<T, D, B, P> {
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
        pivots_dists: [f64; P],
        pivots: [usize; P],
        dataset: &[T],
    ) -> Option<(usize, Self, usize, Self)> {
        #[cfg(test)]
        assert!(
            self.check_hyperrings(pivots, dataset),
            "precondition on hyperrings"
        );

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

            let ret = if let Some((o1, n1, o2, n2)) = possible_split {
                // replace node `closest` with o1
                self.replace_at(closest, o1, Box::new(n1), parent, pivots, dataset);

                // now, insert the new node directly, if possible, otherwise split the current node
                if self.len < B {
                    // FIXME: in case of split, update the radius and hyperrings without recomputing from scratch
                    self.do_insert(o2, Box::new(n2), parent, pivots, dataset);
                    #[cfg(test)]
                    assert!(
                        self.check_hyperrings(pivots, dataset),
                        "fail hyperrings check direct replacement"
                    );
                    None
                } else {
                    let splitresult = self.split(o2, Box::new(n2), pivots, dataset);
                    #[cfg(test)]
                    assert!(
                        splitresult.1.check_radius(dataset),
                        "failed radius check after split (left)"
                    );
                    #[cfg(test)]
                    assert!(
                        splitresult.3.check_radius(dataset),
                        "failed radius check after split (right)"
                    );
                    #[cfg(test)]
                    assert!(
                        splitresult.1.check_hyperrings(pivots, dataset),
                        "fail hyperrings check after split (left)"
                    );
                    #[cfg(test)]
                    assert!(
                        splitresult.3.check_hyperrings(pivots, dataset),
                        "fail hyperrings check after split (right)"
                    );
                    Some(splitresult)
                }
            } else {
                // no split, we simply update the radius and hyperrings
                update_max!(self.radius[closest], distance);
                for j in 0..P {
                    update_max!(self.hyperrings[closest][j].max, pivots_dists[j]);
                    update_min!(self.hyperrings[closest][j].min, pivots_dists[j]);
                }

                #[cfg(test)]
                assert!(
                    self.check_radius(dataset),
                    "fail radius check after insertion with no child split (child index {})",
                    closest
                );
                #[cfg(test)]
                assert!(
                    self.check_hyperrings(pivots, dataset),
                    "fail hyperrings check after insertion with no child split (child index {})",
                    closest
                );
                None
            };
            #[cfg(test)]
            assert!(
                self.check_radius(dataset),
                "fail radius check after insertion"
            );
            ret
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
        let r = self.children[i]
            .as_mut()
            .unwrap()
            .update_hyperrings_and_radius(o, pivots, &mut self.hyperrings[i], dataset);
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

    /// compute the radius (i.e. don't use the cached one) for the `i`-th child
    #[cfg(test)]
    fn compute_radius(&self, i: usize, dataset: &[T]) -> f64 {
        assert!(i < self.len);
        let mut radius = 0.0;
        let center = &dataset[self.routers[i]];
        self.children[i].as_ref().unwrap().for_each_id(&mut |id| {
            let v = &dataset[id];
            let d = D::distance(v, center);
            if d > radius {
                radius = d;
            }
        });
        radius
    }

    #[cfg(test)]
    fn compute_hyperrings(&self, i: usize, pivots: [usize; P], dataset: &[T]) -> [Hyperring; P] {
        assert!(i < self.len);
        let mut rings = [Hyperring::default(); P];
        self.children[i].as_ref().unwrap().for_each_id(&mut |id| {
            let v = &dataset[id];
            for p_idx in 0..P {
                let d = D::distance(v, &dataset[pivots[p_idx]]);
                update_max!(rings[p_idx].max, d);
                update_min!(rings[p_idx].min, d);
            }
        });
        rings
    }

    /// Checks that the all the radii of the children are correct
    #[cfg(test)]
    fn check_radius(&self, dataset: &[T]) -> bool {
        for i in 0..self.len {
            let expected = self.radius[i];
            let actual = self.compute_radius(i, dataset);
            if expected != actual {
                eprintln!("child {} expected radius {} actual {}", i, expected, actual);
                return false;
            }
        }
        true
    }

    #[cfg(test)]
    fn check_hyperrings(&self, pivots: [usize; P], dataset: &[T]) -> bool {
        for i in 0..self.len {
            let expected = self.hyperrings[i];
            let actual = self.compute_hyperrings(i, pivots, dataset);
            if expected != actual {
                eprintln!(
                    "child {} expected hyperrings\n{:#?}\nactual\n{:#?}",
                    i, expected, actual
                );
                return false;
            }
        }
        true
    }

    #[cfg(test)]
    fn compute_mindist(&self, other: &Self, i: usize, j: usize, dataset: &[T]) -> f64 {
        let mut actual_mindist = std::f64::INFINITY;
        let mut ids_a = Vec::new();
        let mut ids_b = Vec::new();
        self.children[i]
            .as_ref()
            .unwrap()
            .for_each_id(&mut |id| ids_a.push(id));
        other.children[j]
            .as_ref()
            .unwrap()
            .for_each_id(&mut |id| ids_b.push(id));
        for &id_a in &ids_a {
            for &id_b in &ids_b {
                let d = D::distance(&dataset[id_a], &dataset[id_b]);
                if d < actual_mindist {
                    actual_mindist = d;
                }
            }
        }
        actual_mindist
    }

    /// Given `self` and `other`, return the minimum distance between child `i` of `self` and
    /// child `j` of `other`.
    fn mindist(&self, other: &Self, i: usize, j: usize, pivots: [usize; P], dataset: &[T]) -> f64 {
        #[cfg(test)]
        assert!(self.check_radius(dataset));
        #[cfg(test)]
        assert!(other.check_radius(dataset));

        #[cfg(test)]
        assert!(self.check_hyperrings(pivots, dataset));
        #[cfg(test)]
        assert!(other.check_hyperrings(pivots, dataset));

        assert!(i < self.len);
        assert!(j < other.len);
        let a = self.routers[i];
        let b = other.routers[j];
        let radius_a = self.radius[i];
        let radius_b = other.radius[j];
        let d = D::distance(&dataset[a], &dataset[b]);

        #[cfg(test)]
        let actual_mindist = self.compute_mindist(other, i, j, dataset);

        let mindist = d - radius_a - radius_b;
        let mut mindist = if mindist < 0.0 { 0.0 } else { mindist };

        #[cfg(test)]
        assert!(
            mindist <= actual_mindist,
            "radius mindist {}, actual {}",
            mindist,
            actual_mindist
        );

        for (hr_a, hr_b) in self.hyperrings[i].iter().zip(&other.hyperrings[j]) {
            let d = hr_a.distance(hr_b);
            if d > mindist {
                mindist = d;
            }
        }

        #[cfg(test)]
        assert!(
            mindist <= actual_mindist,
            "computed mindist {}, actual {}",
            mindist,
            actual_mindist
        );

        mindist
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
            .field("routers", &format_args!("{:?}", &self.routers[..self.len]))
            // .field(
            //     "parent_distance",
            //     &format_args!("{:?}", &self.parent_distance[..self.len]),
            // )
            // .field("radius", &format_args!("{:?}", &self.radius[..self.len]))
            // .field(
            //     "hyperrings",
            //     &format_args!("{:?}", &self.hyperrings[..self.len]),
            // )
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
        pivot_dists: [f64; P],
        pivots: [usize; P],
        dataset: &[T],
    ) -> Option<(usize, Self, usize, Self)> {
        if self.len == B {
            Some(self.split(o, pivot_dists, pivots, dataset))
        } else {
            self.do_insert(o, parent, pivot_dists, pivots, dataset);
            None
        }
    }

    fn do_insert(
        &mut self,
        o: usize,
        parent: Option<usize>,
        pivot_dists: [f64; P],
        pivots: [usize; P],
        dataset: &[T],
    ) {
        assert!(self.len < B);
        self.parent_distance[self.len] =
            parent.map(|parent| D::distance(&dataset[o], &dataset[parent]));
        self.elements[self.len] = o;
        self.pivot_distances[self.len] = pivot_dists;
        self.len += 1;
    }

    /// Split the current node, leaving it empty, and returns two new nodes with the corresponding routing points
    fn split(
        &mut self,
        o: usize,
        pivot_dists: [f64; P],
        pivots: [usize; P],
        dataset: &[T],
    ) -> (usize, Self, usize, Self) {
        assert!(self.len == B);
        let (o1, o2) = promote::<_, D>(o, &self.elements[..self.len], dataset);
        let data1 = &dataset[o1];
        let data2 = &dataset[o2];
        let mut n1 = Self::new();
        let mut n2 = Self::new();

        if D::distance(&dataset[o], data1) < D::distance(&dataset[o], data2) {
            n1.do_insert(o, Some(o1), pivot_dists, pivots, dataset);
        } else {
            n2.do_insert(o, Some(o2), pivot_dists, pivots, dataset);
        }
        for i in 0..self.len {
            let d1 = D::distance(&dataset[self.elements[i]], data1);
            let d2 = D::distance(&dataset[self.elements[i]], data2);
            if d1 < d2 {
                // add to first node
                assert!(n1.len < B);
                n1.do_insert(
                    self.elements[i],
                    Some(o1),
                    self.pivot_distances[i],
                    pivots,
                    dataset,
                );
            } else {
                // add to second node
                assert!(n2.len < B);
                n2.do_insert(
                    self.elements[i],
                    Some(o2),
                    self.pivot_distances[i],
                    pivots,
                    dataset,
                );
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
                "elements",
                &format_args!("{:?}", &self.elements[..self.len]),
            )
            // .field(
            //     "parent_distance",
            //     &format_args!("{:?}", &self.parent_distance[..self.len]),
            // )
            // .field(
            //     "pivot_distances",
            //     &format_args!("{:?}", &self.pivot_distances[..self.len]),
            // )
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OrdF64(pub f64);

impl PartialEq for OrdF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

// #[derive(Serialize, Deserialize)]
#[derive(Debug)]
pub struct Euclidean;

impl Distance<Vec<f64>> for Euclidean {
    fn distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        use packed_simd::f64x4;
        assert!(a.len() == b.len());

        let achunks = a.chunks_exact(4);
        let bchunks = b.chunks_exact(4);

        let srem = achunks
            .remainder()
            .iter()
            .zip(bchunks.remainder())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        let s = achunks
            .map(f64x4::from_slice_unaligned)
            .zip(bchunks.map(f64x4::from_slice_unaligned))
            .map(|(a, b)| (a - b))
            .map(|p| p * p)
            .sum::<f64x4>()
            .sum();

        s + srem
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

    macro_rules! test_closest_pairs {
        ($dataset:ident, $B:literal, $P:literal, $k:literal) => {
            let t_build = Instant::now();
            let pm_tree = PMTree::<Vec<f64>, Euclidean, $B, $P>::for_dataset(&$dataset, 1234);
            assert_eq!(pm_tree.size(), $dataset.len());

            eprintln!(
                "tree built in {:?}, with height {}",
                t_build.elapsed(),
                pm_tree.height()
            );
            // let mut f = File::create("/tmp/tree.txt").unwrap();
            // writeln!(f, "{:#?}", pm_tree).unwrap();

            let t_baseline = Instant::now();
            let mut expected = BinaryHeap::new();
            for a in 0..$dataset.len() {
                for b in (a + 1)..$dataset.len() {
                    let d = OrdF64(Euclidean::distance(&$dataset[a], &$dataset[b]));
                    expected.push((d, a, b));
                    if expected.len() > $k {
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
            let res: Vec<(f64, usize, usize)> = pm_tree.closest_pairs($k, &$dataset);
            eprintln!("Time for tree {:?}", t_tree.elapsed());

            assert_eq!(expected, res);
        };
    }

    #[test]
    fn range_query_glove25() {
        use ndarray::s;
        let path = ensure_glove25();
        let f = hdf5::File::open(path).unwrap();
        let data = f.dataset("/train").unwrap();
        let array = data.read_slice_2d::<f64, _>(s![..10000, ..]).unwrap();
        let mut dataset = Vec::new();
        for row in array.rows() {
            let r = row.as_slice().unwrap();
            dataset.push(Vec::from(r));
        }

        test_range_query!(dataset, 32, 8, 0, 2.0);
        test_range_query!(dataset, 32, 8, 0, 4.0);
    }

    #[test]
    fn closest_pairs_glove25() {
        use ndarray::s;
        let path = ensure_glove25();
        let f = hdf5::File::open(path).unwrap();
        let data = f.dataset("/train").unwrap();
        let array = data.read_slice_2d::<f64, _>(s![..10000, ..]).unwrap();
        let mut dataset = Vec::new();
        for row in array.rows() {
            let r = row.as_slice().unwrap();
            dataset.push(Vec::from(r));
        }

        // test_closest_pairs!(dataset, 32, 8, 5);
        // test_closest_pairs!(dataset, 32, 8, 10);
        // test_closest_pairs!(dataset, 32, 8, 100);
        test_closest_pairs!(dataset, 64, 8, 2);
        // test_closest_pairs!(dataset, 64, 8, 10);
        // test_closest_pairs!(dataset, 64, 8, 100);
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

    #[test]
    fn range_closest_pairs() {
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

        let k = 4;
        let mut expected = BinaryHeap::new();

        for i in 0..dataset.len() {
            for j in (i + 1)..dataset.len() {
                let a = &dataset[i];
                let b = &dataset[j];
                let d = OrdF64(Euclidean::distance(a, b));
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

        let actual = pm_tree.closest_pairs(k, &dataset);

        assert_eq!(expected, actual);
    }
}
