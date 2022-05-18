#![feature(drain_filter, new_uninit, maybe_uninit_uninit_array)]

use std::marker::PhantomData;

use serde_derive::{Deserialize, Serialize};

const B: usize = 4;

pub trait Distance<T> {
    fn distance(a: &T, b: &T) -> f64;
}

#[derive(Serialize, Deserialize)]
pub struct PMTree<T, D: Distance<T>, const B: usize, const P: usize>
where
    for<'d> [usize; P]: serde::Serialize + serde::Deserialize<'d>,
{
    root: Box<Node<T, D>>,
    pivots: [usize; P],
    _markert: PhantomData<T>,
    _markerd: PhantomData<D>,
}

impl<T, D: Distance<T>, const B: usize, const P: usize> PMTree<T, D, B, P>
where
    for<'d> [usize; P]: serde::Serialize + serde::Deserialize<'d>,
{
    pub fn new(pivots: [usize; P]) -> Self {
        Self {
            root: Box::new(Node::Leaf(LeafNode::new())),
            pivots,
            _markert: PhantomData,
            _markerd: PhantomData,
        }
    }

    pub fn insert(&mut self, o: usize, dataset: &[T]) {
        let possible_split = self.root.insert(o, None, dataset);
        if let Some((o1, n1, o2, n2)) = possible_split {
            // replace the current root with a new one
            let mut new_root = InnerNode::new(None);
            new_root.do_insert(o1, Box::new(n1), None, dataset);
            new_root.do_insert(o2, Box::new(n2), None, dataset);
            self.root = Box::new(Node::Inner(new_root));
        }
    }

    pub fn size(&self) -> usize {
        self.root.size()
    }
}

#[derive(Serialize, Deserialize)]
struct Hyperring {
    min: f64,
    max: f64,
}

#[derive(Serialize, Deserialize)]
enum Node<T, D: Distance<T>> {
    Inner(InnerNode<T, D>),
    Leaf(LeafNode<T, D>),
}

impl<T, D: Distance<T>> Node<T, D> {
    fn insert(
        &mut self,
        o: usize,
        parent: Option<usize>,
        dataset: &[T],
    ) -> Option<(usize, Self, usize, Self)> {
        match self {
            Node::Inner(inner) => inner
                .insert(o, parent, dataset)
                .map(|(o1, n1, o2, n2)| (o1, Self::Inner(n1), o2, Self::Inner(n2))),
            Node::Leaf(leaf) => leaf
                .insert(o, parent, dataset)
                .map(|(o1, n1, o2, n2)| (o1, Self::Leaf(n1), o2, Self::Leaf(n2))),
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

    fn radii(&self) -> [f64; B] {
        let mut r = [0.0; B];
        match self {
            Node::Inner(inner) => {
                for i in 0..inner.len {
                    r[i] = inner.radius[i];
                }
            }
            Node::Leaf(leaf) => {
                for i in 0..leaf.len {
                    r[i] = leaf.parent_distance[i].unwrap()
                }
            }
        }
        r
    }

    /// Computes an upper bound to the radius of the ball
    /// centered at the given router, containing all the nodes of the subtree
    /// rooted at the given node.
    fn new_radius(&self, router: usize, dataset: &[T]) -> f64 {
        match self {
            Node::Inner(inner) => {
                let mut r = 0.0;
                for i in 0..inner.len {
                    let d =
                        D::distance(&dataset[inner.routers[i]], &dataset[router]) + inner.radius[i];
                    if d > r {
                        r = d;
                    }
                }
                r
            }
            Node::Leaf(leaf) => {
                let mut r = 0.0;
                for i in 0..leaf.len {
                    let d = D::distance(&dataset[leaf.elements[i]], &dataset[router]);
                    if d > r {
                        r = d;
                    }
                }
                r
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
struct InnerNode<T, D: Distance<T>> {
    len: usize,
    parent: Option<usize>,
    parent_distance: [Option<f64>; B],
    routers: [usize; B],
    radius: [f64; B],
    children: [Option<Box<Node<T, D>>>; B],
    _markert: PhantomData<T>,
    _markerd: PhantomData<D>,
}

impl<T, D: Distance<T>> InnerNode<T, D> {
    fn new(parent: Option<usize>) -> Self {
        let children: [Option<Box<Node<T, D>>>; B] = unsafe { std::mem::transmute([0usize; B]) };
        Self {
            len: 0,
            parent,
            parent_distance: [None; B],
            routers: [0; B],
            radius: [0.0; B],
            children,
            _markert: PhantomData,
            _markerd: PhantomData,
        }
    }

    fn insert(
        &mut self,
        o: usize,
        parent: Option<usize>,
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
                    .insert(o, Some(closest), dataset);

            if let Some((o1, n1, o2, n2)) = possible_split {
                // replace node `closest` with o1
                if let Some(parent) = self.parent {
                    self.parent_distance[closest]
                        .replace(D::distance(&dataset[o1], &dataset[parent]));
                }
                self.routers[closest] = o1;
                self.children[closest].replace(Box::new(n1));
                // TODO: update radius

                // now, insert the new node directly, if possible, otherwise split the current node
                if self.len < B {
                    self.do_insert(o2, Box::new(n2), self.parent, dataset);
                    None
                } else {
                    Some(self.split(o2, Box::new(n2), dataset))
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
        child: Box<Node<T, D>>,
        parent: Option<usize>,
        dataset: &[T],
    ) {
        assert!(self.len < B);
        self.routers[self.len] = o;
        self.children[self.len].replace(child);
        self.parent_distance[self.len] =
            parent.map(|parent| D::distance(&dataset[o], &dataset[parent]));
        // TODO: update radius
        self.len += 1;
    }

    fn split(
        &mut self,
        o2: usize,
        child: Box<Node<T, D>>,
        dataset: &[T],
    ) -> (usize, Self, usize, Self) {
        assert!(self.len == B);
        let o1 = self.routers[0];
        let data1 = &dataset[o1];
        let data2 = &dataset[o2];
        let mut n1 = Self::new(Some(o1));
        let mut n2 = Self::new(Some(o2));

        n2.do_insert(o2, child, Some(o2), dataset);
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
                    dataset,
                );
            } else {
                // add to second node
                assert!(n2.len < B);
                n2.do_insert(
                    self.routers[i],
                    self.children[i].take().unwrap(),
                    Some(o2),
                    dataset,
                );
            }
        }
        self.len = 0;
        (o1, n1, o2, n2)
    }
}

#[derive(Serialize, Deserialize)]
struct LeafNode<T, D: Distance<T>> {
    len: usize,
    parent_distance: [Option<f64>; B],
    elements: [usize; B],
    _markert: PhantomData<T>,
    _markerd: PhantomData<D>,
}

impl<T, D: Distance<T>> LeafNode<T, D> {
    fn new() -> Self {
        Self {
            len: 0,
            parent_distance: [None; B],
            elements: [0; B],
            _markert: PhantomData,
            _markerd: PhantomData,
        }
    }

    fn insert(
        &mut self,
        o: usize,
        parent: Option<usize>,
        dataset: &[T],
    ) -> Option<(usize, Self, usize, Self)> {
        if self.len == B {
            Some(self.split(o, dataset))
        } else {
            self.do_insert(o, parent, dataset);
            None
        }
    }

    fn do_insert(&mut self, o: usize, parent: Option<usize>, dataset: &[T]) {
        assert!(self.len < B);
        self.parent_distance[self.len] =
            parent.map(|parent| D::distance(&dataset[o], &dataset[parent]));
        self.elements[self.len] = o;
        self.len += 1;
    }

    /// Split the current node, leaving it empty, and returns two new nodes with the corresponding routing points
    fn split(&mut self, o2: usize, dataset: &[T]) -> (usize, Self, usize, Self) {
        assert!(self.len == B);
        let o1 = self.elements[0];
        let data1 = &dataset[o1];
        let data2 = &dataset[o2];
        let mut n1 = Self::new();
        let mut n2 = Self::new();

        n2.insert(o2, Some(o2), dataset);
        for i in 0..self.len {
            let d1 = D::distance(&dataset[self.elements[i]], data1);
            let d2 = D::distance(&dataset[self.elements[i]], data2);
            if d1 < d2 {
                // add to first node
                assert!(n1.len < B);
                n1.do_insert(self.elements[i], Some(o1), dataset);
            } else {
                // add to second node
                assert!(n2.len < B);
                n2.do_insert(self.elements[i], Some(o2), dataset);
            }
        }
        self.len = 0;

        (o1, n1, o2, n2)
    }
}

#[derive(Serialize, Deserialize)]
pub struct Euclidean;

impl Distance<Vec<f64>> for Euclidean {
    fn distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        assert!(a.len() == b.len());
        a.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Euclidean, PMTree};
    use std::fs::File;
    use std::io::prelude::*;

    #[test]
    fn it_works() {
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

        let mut f = File::create("tree.json").unwrap();
        writeln!(f, "{}", serde_json::to_string_pretty(&pm_tree).unwrap()).unwrap();

        assert_eq!(pm_tree.size(), dataset.len());
    }
}
