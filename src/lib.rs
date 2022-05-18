#![feature(drain_filter, new_uninit, maybe_uninit_uninit_array)]

use serde_derive::{Deserialize, Serialize};

const B: usize = 32;

#[derive(Serialize, Deserialize)]
pub struct PMTree {
    root: Box<Node>,
}

impl PMTree {
    pub fn new() -> Self {
        Self {
            root: Box::new(Node::Leaf(LeafNode::new())),
        }
    }

    pub fn insert<T>(&mut self, o: usize, dataset: &[T],  dist: impl Fn(&T, &T) -> f64)
    {
        let possible_split = self.root.insert(o, None, dataset, &dist);
        if let Some((o1, n1, o2, n2)) = possible_split {
            // replace the current root with a new one
            let mut new_root = InnerNode::new(None);
            new_root.do_insert(o1, Box::new(n1), None, dataset, &dist);
            new_root.do_insert(o2, Box::new(n2), None, dataset, &dist);
            self.root = Box::new(Node::Inner(new_root));
        }
    }
}

#[derive(Serialize, Deserialize)]
enum Node {
    Inner(InnerNode),
    Leaf(LeafNode),
}

impl Node {
    fn insert<T, D>(
        &mut self,
        o: usize,
        parent: Option<usize>,
        dataset: &[T],
        dist: D,
    ) -> Option<(usize, Self, usize, Self)>
    where
        D: Fn(&T, &T) -> f64,
    {
        match self {
            Node::Inner(inner) => inner
                .insert(o, parent, dataset, dist)
                .map(|(o1, n1, o2, n2)| (o1, Self::Inner(n1), o2, Self::Inner(n2))),
            Node::Leaf(leaf) => leaf
                .insert(o, parent, dataset, dist)
                .map(|(o1, n1, o2, n2)| (o1, Self::Leaf(n1), o2, Self::Leaf(n2))),
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
    fn new_radius<T, D>(&self, router: usize, dataset: &[T], dist: D) -> f64
    where
        D: Fn(&T, &T) -> f64,
    {
        match self {
            Node::Inner(inner) => {
                let mut r = 0.0;
                for i in 0..inner.len {
                    let d = dist(&dataset[inner.routers[i]], &dataset[router]) + inner.radius[i];
                    if d > r {
                        r = d;
                    }
                }
                r
            }
            Node::Leaf(leaf) => {
                let mut r = 0.0;
                for i in 0..leaf.len {
                    let d = dist(&dataset[leaf.elements[i]], &dataset[router]);
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
struct InnerNode {
    len: usize,
    parent: Option<usize>,
    parent_distance: [Option<f64>; B],
    routers: [usize; B],
    radius: [f64; B],
    children: [Option<Box<Node>>; B],
}

impl InnerNode {
    fn new(parent: Option<usize>) -> Self {
        // let children : [ Option<Box<Node<B>>>; B] = unsafe {
        //     std::mem::transmute([0usize; B])
        // };
        Self {
            len: 0,
            parent,
            parent_distance: [None; B],
            routers: [0; B],
            radius: [0.0; B],
            children: todo!(),
        }
    }

    fn insert<T, D>(
        &mut self,
        o: usize,
        parent: Option<usize>,
        dataset: &[T],
        dist: D,
    ) -> Option<(usize, Self, usize, Self)>
    where
        D: Fn(&T, &T) -> f64,
    {
        // find the routing element closest
        let closest = (0..self.len)
            .map(|i| {
                let d = dist(&dataset[i], &dataset[o]);
                (i, d)
            })
            .min_by(|p1, p2| p1.1.partial_cmp(&p2.1).unwrap());

        if let Some((closest, _distance)) = closest {
            let possible_split =
                self.children[closest]
                    .as_mut()
                    .unwrap()
                    .insert(o, Some(closest), dataset, &dist);

            if let Some((o1, n1, o2, n2)) = possible_split {
                // replace node `closest` with o1
                if let Some(parent) = self.parent {
                    self.parent_distance[closest].replace(dist(&dataset[o1], &dataset[parent]));
                }
                self.routers[closest] = o1;
                self.children[closest].replace(Box::new(n1));
                // TODO: update radius

                // now, insert the new node directly, if possible, otherwise split the current node
                if self.len < B {
                    self.do_insert(o2, Box::new(n2), self.parent, dataset, &dist);
                    None
                } else {
                    Some(self.split(o2, Box::new(n2), dataset, &dist))
                }
            } else {
                None
            }
        } else {
            panic!("Empty inner node?");
        }
    }

    fn do_insert<T, D>(
        &mut self,
        o: usize,
        child: Box<Node>,
        parent: Option<usize>,
        dataset: &[T],
        dist: D,
    ) where
        D: Fn(&T, &T) -> f64,
    {
        assert!(self.len < B);
        self.routers[self.len] = o;
        self.children[self.len].replace(child);
        self.parent_distance[self.len] = parent.map(|parent| dist(&dataset[o], &dataset[parent]));
        // TODO: update radius
        self.len += 1;
    }

    fn split<T, D>(
        &mut self,
        o2: usize,
        child: Box<Node>,
        dataset: &[T],
        dist: D,
    ) -> (usize, Self, usize, Self)
    where
        D: Fn(&T, &T) -> f64,
    {
        assert!(self.len == B);
        let o1 = self.routers[0];
        let data1 = &dataset[o1];
        let data2 = &dataset[o2];
        let mut n1 = Self::new(Some(o1));
        let mut n2 = Self::new(Some(o2));

        n2.do_insert(o2, child, Some(o2), dataset, &dist);
        for i in 0..self.len {
            if dist(&dataset[self.routers[i]], data1) < dist(&dataset[self.routers[i]], data2) {
                // add to first node
                assert!(n1.len < B);
                n1.do_insert(
                    self.routers[i],
                    self.children[i].take().unwrap(),
                    Some(o1),
                    dataset,
                    &dist,
                );
            } else {
                // add to second node
                assert!(n2.len < B);
                n2.do_insert(
                    self.routers[i],
                    self.children[i].take().unwrap(),
                    Some(o2),
                    dataset,
                    &dist,
                );
            }
        }
        self.len = 0;
        (o1, n1, o2, n2)
    }
}

#[derive(Serialize, Deserialize)]
struct LeafNode {
    len: usize,
    parent_distance: [Option<f64>; B],
    elements: [usize; B],
}

impl LeafNode {
    fn new() -> Self {
        Self {
            len: 0,
            parent_distance: [None; B],
            elements: [0; B],
        }
    }

    fn insert<T, D>(
        &mut self,
        o: usize,
        parent: Option<usize>,
        dataset: &[T],
        dist: D,
    ) -> Option<(usize, Self, usize, Self)>
    where
        D: Fn(&T, &T) -> f64,
    {
        if self.len == B {
            Some(self.split(o, dataset, dist))
        } else {
            self.do_insert(o, parent, dataset, dist);
            None
        }
    }

    fn do_insert<T, D>(&mut self, o: usize, parent: Option<usize>, dataset: &[T], dist: D)
    where
        D: Fn(&T, &T) -> f64,
    {
        assert!(self.len < B);
        self.parent_distance[self.len] = parent.map(|parent| dist(&dataset[o], &dataset[parent]));
        self.elements[self.len] = o;
        self.len += 1;
    }

    /// Split the current node, leaving it empty, and returns two new nodes with the corresponding routing points
    fn split<T, D>(&mut self, o2: usize, dataset: &[T], dist: D) -> (usize, Self, usize, Self)
    where
        D: Fn(&T, &T) -> f64,
    {
        assert!(self.len == B);
        let o1 = self.elements[0];
        let data1 = &dataset[o1];
        let data2 = &dataset[o2];
        let mut n1 = Self::new();
        let mut n2 = Self::new();

        n2.insert(o2, Some(o2), dataset, &dist);
        for i in 0..self.len {
            let d1 = dist(&dataset[self.elements[i]], data1);
            let d2 = dist(&dataset[self.elements[i]], data2);
            if d1 < d2 {
                // add to first node
                assert!(n1.len < B);
                n1.do_insert(self.elements[i], Some(o1), dataset, &dist);
            } else {
                // add to second node
                assert!(n2.len < B);
                n2.do_insert(self.elements[i], Some(o2), dataset, &dist);
            }
        }
        self.len = 0;

        (o1, n1, o2, n2)
    }
}

pub fn euclidean(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    assert!(a.len() == b.len());
    a.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum()
}

#[cfg(test)]
mod tests {
    use crate::{PMTree, euclidean};

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

        let mut pm_tree = PMTree::new();
        pm_tree.insert(0, &dataset, euclidean);

        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
