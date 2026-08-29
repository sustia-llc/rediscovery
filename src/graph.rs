//! Graph construction for Tier 0 of the POC.
//!
//! [`Graph`] stores an undirected, loop-free, unweighted graph as a dense
//! `DMatrix<f64>` adjacency matrix plus a degree vector; the POC's topologies
//! stay at n ≤ ~50, where dense storage costs nothing. Constructors cover the
//! four tiny graphs the paper studies ([`Graph::path_star`], [`Graph::grid`],
//! [`Graph::cycle`], [`Graph::irregular`]), the tree-star family of Appendix
//! C.2 ([`Graph::tree_star`]), and [`Graph::complete`] as a test aid. The
//! `spectral` module and the `Node2Vec` dynamics of later tiers consume these.

use nalgebra::{DMatrix, DVector};

use crate::error::{Error, GraphParameter, Result};

/// An undirected, loop-free, unweighted graph over vertices `0..order`.
///
/// The adjacency matrix is symmetric with a zero diagonal and entries in
/// {0.0, 1.0}; the degree vector holds each vertex's adjacency row sum. Both
/// invariants are established at construction and the fields are private, so
/// every `Graph` value satisfies them.
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    adjacency: DMatrix<f64>,
    degrees: DVector<f64>,
}

impl Graph {
    /// Builds a graph on `order` vertices from an undirected edge list,
    /// tolerating repeated edges.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidGraphParameter`] if `order` is zero,
    /// [`Error::SelfLoop`] for an edge with equal endpoints, and
    /// [`Error::EdgeOutOfBounds`] for an endpoint at or beyond `order`.
    pub fn from_edges(order: usize, edges: &[(usize, usize)]) -> Result<Self> {
        require_at_least(GraphParameter::Order, 1, order)?;

        let mut adjacency = DMatrix::<f64>::zeros(order, order);
        for &(u, v) in edges {
            if u == v {
                return Err(Error::SelfLoop { vertex: u });
            }
            if u >= order || v >= order {
                return Err(Error::EdgeOutOfBounds { u, v, order });
            }
            adjacency[(u, v)] = 1.0;
            adjacency[(v, u)] = 1.0;
        }

        let degrees = DVector::from_iterator(order, adjacency.row_iter().map(|row| row.sum()));
        Ok(Self { adjacency, degrees })
    }

    /// Builds a path-star: a central root with `arms` disjoint paths, each
    /// carrying `arm_len` vertices beyond the root.
    ///
    /// The result has `1 + arms * arm_len` vertices and `arms * arm_len`
    /// edges; the root is vertex 0 and arm `a` occupies the contiguous block
    /// `1 + a * arm_len .. 1 + (a + 1) * arm_len`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidGraphParameter`] if `arms` or `arm_len` is
    /// zero, and [`Error::GraphTooLarge`] if the vertex count overflows
    /// `usize`.
    pub fn path_star(arms: usize, arm_len: usize) -> Result<Self> {
        require_at_least(GraphParameter::Arms, 1, arms)?;
        require_at_least(GraphParameter::ArmLength, 1, arm_len)?;

        let order = arms
            .checked_mul(arm_len)
            .and_then(|leaves| leaves.checked_add(1))
            .ok_or(Error::GraphTooLarge)?;

        let mut edges = Vec::with_capacity(order - 1);
        for arm in 0..arms {
            let base = 1 + arm * arm_len;
            edges.push((0, base));
            for step in 1..arm_len {
                edges.push((base + step - 1, base + step));
            }
        }

        Self::from_edges(order, &edges)
    }

    /// Builds a `rows` × `cols` four-neighbour lattice.
    ///
    /// The result has `rows * cols` vertices and
    /// `rows * (cols - 1) + cols * (rows - 1)` edges; vertex `(r, c)` is
    /// index `r * cols + c`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidGraphParameter`] if `rows` or `cols` is zero,
    /// and [`Error::GraphTooLarge`] if the vertex count overflows `usize`.
    pub fn grid(rows: usize, cols: usize) -> Result<Self> {
        require_at_least(GraphParameter::Rows, 1, rows)?;
        require_at_least(GraphParameter::Columns, 1, cols)?;

        let order = rows.checked_mul(cols).ok_or(Error::GraphTooLarge)?;

        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let here = r * cols + c;
                if c + 1 < cols {
                    edges.push((here, here + 1));
                }
                if r + 1 < rows {
                    edges.push((here, here + cols));
                }
            }
        }

        Self::from_edges(order, &edges)
    }

    /// Builds the `n`-vertex cycle, with vertex `i` adjacent to
    /// `(i + 1) % n`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidGraphParameter`] if `n < 3`, the smallest
    /// cycle expressible without repeated edges.
    pub fn cycle(n: usize) -> Result<Self> {
        require_at_least(GraphParameter::CycleOrder, 3, n)?;

        let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
        Self::from_edges(n, &edges)
    }

    /// Builds the 15-vertex, two-component irregular graph of decision D4.
    ///
    /// Component A is an 11-cycle on vertices 0–10 with chords (0, 2) and
    /// (5, 7); component B is a kite — a triangle on 11, 12, 13 with vertex
    /// 14 adjacent to 12 and 13. The two components share no edge. 18 edges
    /// total. This approximates the paper's Figure 21, which publishes no
    /// adjacency list.
    ///
    /// # Errors
    ///
    /// Returns an error only if the fixed edge list violates a [`Graph`]
    /// invariant, which construction of this fixed topology does not.
    pub fn irregular() -> Result<Self> {
        let mut edges: Vec<(usize, usize)> = (0..11).map(|i| (i, (i + 1) % 11)).collect();
        edges.extend_from_slice(&[
            (0, 2),
            (5, 7),
            (11, 12),
            (12, 13),
            (11, 13),
            (12, 14),
            (13, 14),
        ]);
        Self::from_edges(15, &edges)
    }

    /// Builds the tree-star `T_{d,ell}`: a root of degree `d` whose every
    /// non-leaf descendant has two children, with `ell` edges on each
    /// root-to-leaf path.
    ///
    /// The result has `1 + d * (2^ell - 1)` vertices and one fewer edge.
    /// Vertices are numbered breadth-first from the root.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidGraphParameter`] if `d` or `ell` is zero, and
    /// [`Error::GraphTooLarge`] if the vertex count overflows `usize`.
    pub fn tree_star(d: usize, ell: usize) -> Result<Self> {
        require_at_least(GraphParameter::RootDegree, 1, d)?;
        require_at_least(GraphParameter::PathLength, 1, ell)?;

        // Per-arm vertex count 2^ell - 1, summed level by level so that the
        // loop is linear in `ell` and overflow is caught before allocating.
        let mut per_arm: usize = 0;
        let mut level: usize = 1;
        for _ in 0..ell {
            per_arm = per_arm.checked_add(level).ok_or(Error::GraphTooLarge)?;
            level = level.saturating_mul(2);
        }
        let order = d
            .checked_mul(per_arm)
            .and_then(|descendants| descendants.checked_add(1))
            .ok_or(Error::GraphTooLarge)?;

        let mut edges = Vec::with_capacity(order - 1);
        let mut next_index = 1;
        let mut frontier: Vec<usize> = Vec::new();
        for _ in 0..d {
            edges.push((0, next_index));
            frontier.push(next_index);
            next_index += 1;
        }
        for _ in 1..ell {
            let mut children = Vec::with_capacity(frontier.len() * 2);
            for &parent in &frontier {
                for _ in 0..2 {
                    edges.push((parent, next_index));
                    children.push(next_index);
                    next_index += 1;
                }
            }
            frontier = children;
        }

        Self::from_edges(order, &edges)
    }

    /// Builds the complete graph on `n` vertices, in which every distinct
    /// pair is adjacent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidGraphParameter`] if `n < 2`, below which the
    /// graph has no edges and its random-walk Laplacian is undefined.
    pub fn complete(n: usize) -> Result<Self> {
        require_at_least(GraphParameter::Order, 2, n)?;

        let mut edges = Vec::with_capacity(n * (n - 1) / 2);
        for u in 0..n {
            for v in (u + 1)..n {
                edges.push((u, v));
            }
        }

        Self::from_edges(n, &edges)
    }

    /// The number of vertices.
    #[must_use]
    pub fn order(&self) -> usize {
        self.degrees.len()
    }

    /// The number of undirected edges, counted from the adjacency matrix's
    /// strict upper triangle.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        let order = self.order();
        let mut count = 0;
        for u in 0..order {
            for v in (u + 1)..order {
                if self.adjacency[(u, v)] > 0.0 {
                    count += 1;
                }
            }
        }
        count
    }

    /// The symmetric, zero-diagonal adjacency matrix `A`.
    #[must_use]
    pub fn adjacency(&self) -> &DMatrix<f64> {
        &self.adjacency
    }

    /// The degree vector, entry `i` holding row `i`'s adjacency sum.
    #[must_use]
    pub fn degrees(&self) -> &DVector<f64> {
        &self.degrees
    }
}

/// Rejects `value` below `minimum`, naming `parameter` in the error.
fn require_at_least(parameter: GraphParameter, minimum: usize, value: usize) -> Result<()> {
    if value < minimum {
        return Err(Error::InvalidGraphParameter {
            parameter,
            minimum,
            value,
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "adjacency entries and degrees are exact small integers in f64"
)]
mod tests {
    use super::*;

    /// Exact `usize` → `f64` for the small counts these tests compare.
    fn exact(n: usize) -> f64 {
        f64::from(u32::try_from(n).expect("test fixture counts fit in u32"))
    }

    /// Every constructor under test, with the label used in failure messages.
    fn fixtures() -> Vec<(&'static str, Graph)> {
        vec![
            (
                "path_star(4,4)",
                Graph::path_star(4, 4).expect("path_star(4,4)"),
            ),
            ("grid(4,4)", Graph::grid(4, 4).expect("grid(4,4)")),
            ("cycle(15)", Graph::cycle(15).expect("cycle(15)")),
            ("irregular()", Graph::irregular().expect("irregular()")),
            (
                "tree_star(3,3)",
                Graph::tree_star(3, 3).expect("tree_star(3,3)"),
            ),
            ("complete(7)", Graph::complete(7).expect("complete(7)")),
        ]
    }

    /// D1–D4 vertex counts, plus the edge count each topology implies.
    #[test]
    fn constructors_have_the_specified_node_and_edge_counts() {
        let cases: [(&str, Graph, usize, usize); 6] = [
            (
                "path_star(4,4)",
                Graph::path_star(4, 4).expect("path_star(4,4)"),
                17,
                16,
            ),
            ("grid(4,4)", Graph::grid(4, 4).expect("grid(4,4)"), 16, 24),
            ("cycle(15)", Graph::cycle(15).expect("cycle(15)"), 15, 15),
            (
                "irregular()",
                Graph::irregular().expect("irregular()"),
                15,
                18,
            ),
            (
                "tree_star(3,3)",
                Graph::tree_star(3, 3).expect("tree_star(3,3)"),
                22,
                21,
            ),
            (
                "complete(7)",
                Graph::complete(7).expect("complete(7)"),
                7,
                21,
            ),
        ];

        for (name, graph, order, edges) in cases {
            assert_eq!(
                graph.order(),
                order,
                "{name}: order is {}, expected {order}",
                graph.order()
            );
            assert_eq!(
                graph.edge_count(),
                edges,
                "{name}: edge count is {}, expected {edges}",
                graph.edge_count()
            );
        }
    }

    /// `tree_star` realizes 1 + d(2^ell − 1) vertices and one fewer edge
    /// across the family. The vertex count checks the constructor's level-wise
    /// summation against the closed form; the edge count is the half that
    /// reaches the built edge list.
    #[test]
    fn tree_star_matches_its_node_count_formula() {
        for d in 1..5_usize {
            for ell in 1..6_u32 {
                let graph =
                    Graph::tree_star(d, ell as usize).expect("tree_star with small parameters");
                let expected = 1 + d * (2_usize.pow(ell) - 1);
                assert_eq!(
                    graph.order(),
                    expected,
                    "tree_star({d},{ell}): order is {}, formula gives {expected}",
                    graph.order()
                );
                assert_eq!(
                    graph.edge_count(),
                    expected - 1,
                    "tree_star({d},{ell}): edge count is {}, a tree needs {}",
                    graph.edge_count(),
                    expected - 1
                );
            }
        }
    }

    /// `tree_star` realizes the shape its docs claim: a root of degree `d`,
    /// `d · 2^(ell−1)` leaves of degree 1, and `d · (2^(ell−1) − 1)` internal
    /// vertices of degree 3 — one parent and two children each.
    #[test]
    fn tree_star_degree_profile() {
        for (d, ell) in [(4_usize, 3_u32), (3, 2), (2, 4)] {
            let graph = Graph::tree_star(d, ell as usize).expect("tree_star");
            let label = format!("tree_star({d},{ell})");

            assert!(
                (graph.degrees()[0] - exact(d)).abs() < 1e-12,
                "{label}: root degree is {}, expected {d}",
                graph.degrees()[0]
            );

            let mut leaves = 0_usize;
            let mut internal = 0_usize;
            for vertex in 1..graph.order() {
                let degree = graph.degrees()[vertex];
                if (degree - 1.0).abs() < 1e-12 {
                    leaves += 1;
                } else if (degree - 3.0).abs() < 1e-12 {
                    internal += 1;
                } else {
                    panic!("{label}: vertex {vertex} has degree {degree}, expected 1 or 3");
                }
            }

            let expected_leaves = d * 2_usize.pow(ell - 1);
            let expected_internal = d * (2_usize.pow(ell - 1) - 1);
            assert_eq!(
                leaves, expected_leaves,
                "{label}: {leaves} leaves, expected {expected_leaves}"
            );
            assert_eq!(
                internal, expected_internal,
                "{label}: {internal} internal vertices, expected {expected_internal}"
            );
        }
    }

    /// Adjacency is symmetric, zero on the diagonal, and 0/1 valued.
    #[test]
    fn constructors_are_undirected_and_loop_free() {
        for (name, graph) in fixtures() {
            let asymmetry = (graph.adjacency() - graph.adjacency().transpose()).amax();
            assert!(
                asymmetry < 1e-15,
                "{name}: max |A − Aᵀ| = {asymmetry:.3e}, expected 0"
            );

            let loop_weight = graph.adjacency().diagonal().amax();
            assert!(
                loop_weight < 1e-15,
                "{name}: max diagonal entry = {loop_weight:.3e}, expected 0"
            );

            for (index, &entry) in graph.adjacency().iter().enumerate() {
                assert!(
                    entry == 0.0 || entry == 1.0,
                    "{name}: adjacency entry {index} is {entry}, expected 0 or 1"
                );
            }
        }
    }

    /// The stored degree vector agrees with the adjacency rows, and the
    /// degree sum is twice the independently counted edge total.
    #[test]
    fn degrees_agree_with_adjacency_and_edge_count() {
        for (name, graph) in fixtures() {
            for (vertex, row) in graph.adjacency().row_iter().enumerate() {
                let row_sum = row.sum();
                assert!(
                    (graph.degrees()[vertex] - row_sum).abs() < 1e-12,
                    "{name}: degrees[{vertex}] = {}, adjacency row sum = {row_sum}",
                    graph.degrees()[vertex]
                );
            }

            let degree_sum = graph.degrees().sum();
            let expected = 2.0 * exact(graph.edge_count());
            assert!(
                (degree_sum - expected).abs() < 1e-12,
                "{name}: degree sum {degree_sum}, expected 2 × {} = {expected}",
                graph.edge_count()
            );
        }
    }

    /// The path-star's root carries the arm count and each arm ends in a leaf.
    #[test]
    fn path_star_degree_profile() {
        let graph = Graph::path_star(4, 4).expect("path_star(4,4)");
        assert!(
            (graph.degrees()[0] - 4.0).abs() < 1e-12,
            "root degree is {}, expected 4",
            graph.degrees()[0]
        );
        for arm in 0..4 {
            let leaf = 1 + arm * 4 + 3;
            assert!(
                (graph.degrees()[leaf] - 1.0).abs() < 1e-12,
                "leaf {leaf} degree is {}, expected 1",
                graph.degrees()[leaf]
            );
        }
    }

    /// The D4 topology is exactly the specified edge set, with no edge
    /// crossing between its two components.
    #[test]
    fn irregular_matches_the_d4_topology() {
        let graph = Graph::irregular().expect("irregular()");

        // Checked before the edge-set equality below, which would otherwise
        // absorb every cross-component edge and leave this assertion dead.
        for u in 0..11 {
            for v in 11..15 {
                assert!(
                    graph.adjacency()[(u, v)] == 0.0,
                    "irregular(): edge ({u}, {v}) joins the two components, which D4 keeps disjoint"
                );
            }
        }

        let mut observed: Vec<(usize, usize)> = Vec::new();
        for u in 0..graph.order() {
            for v in (u + 1)..graph.order() {
                if graph.adjacency()[(u, v)] > 0.0 {
                    observed.push((u, v));
                }
            }
        }

        let mut expected: Vec<(usize, usize)> = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (0, 10),
            (0, 2),
            (5, 7),
            (11, 12),
            (11, 13),
            (12, 13),
            (12, 14),
            (13, 14),
        ];
        expected.sort_unstable();

        assert_eq!(
            observed, expected,
            "irregular(): edge set is {observed:?}, D4 specifies {expected:?}"
        );
    }

    /// The cycle is 2-regular.
    #[test]
    fn cycle_is_two_regular() {
        let graph = Graph::cycle(15).expect("cycle(15)");
        for vertex in 0..graph.order() {
            assert!(
                (graph.degrees()[vertex] - 2.0).abs() < 1e-12,
                "cycle(15): degrees[{vertex}] = {}, expected 2",
                graph.degrees()[vertex]
            );
        }
    }

    /// Degenerate parameters are rejected with a typed error naming them.
    #[test]
    fn degenerate_parameters_are_rejected() {
        let cases: [(&str, Result<Graph>, GraphParameter, usize); 8] = [
            (
                "path_star(0,4)",
                Graph::path_star(0, 4),
                GraphParameter::Arms,
                1,
            ),
            (
                "path_star(4,0)",
                Graph::path_star(4, 0),
                GraphParameter::ArmLength,
                1,
            ),
            ("grid(0,4)", Graph::grid(0, 4), GraphParameter::Rows, 1),
            ("grid(4,0)", Graph::grid(4, 0), GraphParameter::Columns, 1),
            ("cycle(2)", Graph::cycle(2), GraphParameter::CycleOrder, 3),
            (
                "tree_star(0,2)",
                Graph::tree_star(0, 2),
                GraphParameter::RootDegree,
                1,
            ),
            (
                "tree_star(2,0)",
                Graph::tree_star(2, 0),
                GraphParameter::PathLength,
                1,
            ),
            ("complete(1)", Graph::complete(1), GraphParameter::Order, 2),
        ];

        for (name, result, parameter, minimum) in cases {
            match result {
                Err(Error::InvalidGraphParameter {
                    parameter: observed,
                    minimum: observed_min,
                    ..
                }) => {
                    assert_eq!(
                        observed, parameter,
                        "{name}: rejected parameter {observed:?}, expected {parameter:?}"
                    );
                    assert_eq!(
                        observed_min, minimum,
                        "{name}: reported minimum {observed_min}, expected {minimum}"
                    );
                }
                Err(other) => panic!("{name}: expected InvalidGraphParameter, got {other:?}"),
                Ok(graph) => panic!(
                    "{name}: expected rejection, built a {}-vertex graph",
                    graph.order()
                ),
            }
        }
    }

    /// `from_edges` refuses self-loops and out-of-range endpoints.
    #[test]
    fn from_edges_rejects_invalid_edges() {
        match Graph::from_edges(4, &[(1, 1)]) {
            Err(Error::SelfLoop { vertex }) => assert_eq!(vertex, 1, "reported vertex {vertex}"),
            other => panic!("expected SelfLoop, got {other:?}"),
        }
        match Graph::from_edges(4, &[(1, 4)]) {
            Err(Error::EdgeOutOfBounds { u, v, order }) => {
                assert_eq!((u, v, order), (1, 4, 4), "reported ({u}, {v}, {order})");
            }
            other => panic!("expected EdgeOutOfBounds, got {other:?}"),
        }
    }
}
