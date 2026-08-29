//! Spectral infrastructure for Tier 0 of the POC.
//!
//! [`laplacian`] builds the asymmetrically normalized random-walk Laplacian
//! L = (I − D⁻¹A) + (I − D⁻¹A)ᵀ of Appendix F from a [`Graph`], and
//! [`Spectrum`] holds the symmetric eigendecomposition of −L with the
//! deterministic ordering and sign convention of decision D5: eigenpairs
//! descending by eigenvalue, and each eigenvector scaled so that its first
//! component of magnitude above [`SIGN_PIVOT_TOLERANCE`] is positive. All
//! arithmetic is f64. Everything downstream that reasons about eigenvector
//! alignment (Tier 1's `Node2Vec` dynamics) consumes this.

use nalgebra::{DMatrix, DVector, linalg::SymmetricEigen};

use crate::error::{Error, Result};
use crate::graph::Graph;

/// Magnitude above which an eigenvector component may fix that
/// eigenvector's sign (decision D5).
pub const SIGN_PIVOT_TOLERANCE: f64 = 1e-9;

/// Computes L = (I − D⁻¹A) + (I − D⁻¹A)ᵀ for `graph`.
///
/// The summand I − D⁻¹A is not symmetric on an irregular graph; the returned
/// L is, being of the form X + Xᵀ.
///
/// # Errors
///
/// Returns [`Error::IsolatedVertex`] for a degree-zero vertex, where D⁻¹
/// does not exist.
pub fn laplacian(graph: &Graph) -> Result<DMatrix<f64>> {
    let order = graph.order();

    if let Some((vertex, _)) = graph
        .degrees()
        .iter()
        .enumerate()
        .find(|(_, degree)| **degree <= 0.0)
    {
        return Err(Error::IsolatedVertex { vertex });
    }

    let mut transition = graph.adjacency().clone();
    for (vertex, mut row) in transition.row_iter_mut().enumerate() {
        row /= graph.degrees()[vertex];
    }

    let deviation = DMatrix::<f64>::identity(order, order) - transition;
    Ok(&deviation + deviation.transpose())
}

/// The eigendecomposition of a symmetric matrix, ordered and sign-fixed per
/// decision D5.
///
/// Eigenvalues descend; eigenvector `j` is column `j` of
/// [`Spectrum::eigenvectors`], is unit-norm, and has a positive first
/// component of magnitude above [`SIGN_PIVOT_TOLERANCE`].
#[derive(Debug, Clone)]
pub struct Spectrum {
    eigenvalues: DVector<f64>,
    eigenvectors: DMatrix<f64>,
}

impl Spectrum {
    /// Decomposes `symmetric`, then applies the D5 ordering and sign
    /// convention.
    ///
    /// Only the lower triangle of `symmetric` (including its diagonal) is
    /// read, so a non-symmetric argument is decomposed as though its upper
    /// triangle mirrored its lower one; callers supply matrices of the form
    /// X + Xᵀ.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotSquare`] for a non-square argument and
    /// [`Error::EigenNotConverged`] if the symmetric QR iteration fails to
    /// converge.
    pub fn new(symmetric: DMatrix<f64>) -> Result<Self> {
        let (rows, columns) = symmetric.shape();
        if rows != columns {
            return Err(Error::NotSquare { rows, columns });
        }

        // `max_niter == 0` means "iterate until convergence" in nalgebra.
        let eigen = SymmetricEigen::try_new(symmetric, f64::EPSILON, 0)
            .ok_or(Error::EigenNotConverged { order: rows })?;

        let mut order: Vec<usize> = (0..rows).collect();
        order.sort_by(|&a, &b| eigen.eigenvalues[b].total_cmp(&eigen.eigenvalues[a]));

        let eigenvalues = DVector::from_iterator(rows, order.iter().map(|&i| eigen.eigenvalues[i]));
        let mut eigenvectors = DMatrix::<f64>::zeros(rows, rows);
        for (target, &source) in order.iter().enumerate() {
            eigenvectors
                .column_mut(target)
                .copy_from(&eigen.eigenvectors.column(source));
        }

        for mut column in eigenvectors.column_iter_mut() {
            let pivot = column
                .iter()
                .copied()
                .find(|component| component.abs() > SIGN_PIVOT_TOLERANCE);
            if let Some(pivot) = pivot
                && pivot < 0.0
            {
                column.neg_mut();
            }
        }

        Ok(Self {
            eigenvalues,
            eigenvectors,
        })
    }

    /// Decomposes −L for `graph`, where L is [`laplacian`]'s output.
    ///
    /// # Errors
    ///
    /// Propagates [`laplacian`]'s [`Error::IsolatedVertex`] and
    /// [`Spectrum::new`]'s [`Error::EigenNotConverged`].
    pub fn of_negative_laplacian(graph: &Graph) -> Result<Self> {
        Self::new(-laplacian(graph)?)
    }

    /// The eigenvalues, descending.
    #[must_use]
    pub fn eigenvalues(&self) -> &DVector<f64> {
        &self.eigenvalues
    }

    /// The eigenvectors, column `j` matching eigenvalue `j`.
    #[must_use]
    pub fn eigenvectors(&self) -> &DMatrix<f64> {
        &self.eigenvectors
    }

    /// The number of eigenpairs, equal to the decomposed matrix's dimension.
    #[must_use]
    pub fn len(&self) -> usize {
        self.eigenvalues.len()
    }

    /// Whether the decomposed matrix was 0×0.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.eigenvalues.is_empty()
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    reason = "test indices are small and exact in f64"
)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Every constructor, with the label used in failure messages.
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

    /// −L of the 15-cycle has the closed-form eigenvalues
    /// 2cos(2πk/15) − 2, k = 0..14.
    #[test]
    fn cycle15_eigenvalues_match_the_closed_form() {
        let graph = Graph::cycle(15).expect("cycle(15)");
        let spectrum = Spectrum::of_negative_laplacian(&graph).expect("spectrum of cycle(15)");

        let mut expected: Vec<f64> = (0..15)
            .map(|k| 2.0 * (2.0 * PI * f64::from(k) / 15.0).cos() - 2.0)
            .collect();
        expected.sort_by(|a, b| b.total_cmp(a));

        for (k, &want) in expected.iter().enumerate() {
            let got = spectrum.eigenvalues()[k];
            assert!(
                (got - want).abs() < 1e-9,
                "cycle(15): eigenvalue {k} is {got:.15}, closed form gives {want:.15} \
                 (|Δ| = {:.3e}, tolerance 1e-9)",
                (got - want).abs()
            );
        }
    }

    /// The 15-cycle's top eigenvector is constant, its eigenvalue being the
    /// simple 0 of a connected graph.
    #[test]
    fn cycle15_top_eigenvector_is_constant() {
        let graph = Graph::cycle(15).expect("cycle(15)");
        let spectrum = Spectrum::of_negative_laplacian(&graph).expect("spectrum of cycle(15)");

        let top = spectrum.eigenvectors().column(0);
        let uniform = 1.0 / 15.0_f64.sqrt();
        for (vertex, &component) in top.iter().enumerate() {
            assert!(
                (component - uniform).abs() < 1e-9,
                "cycle(15): top eigenvector component {vertex} is {component:.15}, \
                 expected {uniform:.15} (|Δ| = {:.3e}, tolerance 1e-9)",
                (component - uniform).abs()
            );
        }
    }

    /// The 15-cycle's degenerate second eigenvalue carries a 2-D eigenspace
    /// equal to the span of the k = 1 Fourier modes. Asserted as a subspace
    /// projection residual, never per-vector: the pair is degenerate, so the
    /// individual eigenvectors are only defined up to a rotation within it.
    #[test]
    fn cycle15_fiedler_pair_spans_the_fourier_modes() {
        let graph = Graph::cycle(15).expect("cycle(15)");
        let spectrum = Spectrum::of_negative_laplacian(&graph).expect("spectrum of cycle(15)");

        let gap = (spectrum.eigenvalues()[1] - spectrum.eigenvalues()[2]).abs();
        assert!(
            gap < 1e-9,
            "cycle(15): eigenvalues 1 and 2 differ by {gap:.3e}; the Fiedler pair should be \
             degenerate before a subspace test is meaningful"
        );

        let mut fourier = DMatrix::<f64>::zeros(15, 2);
        for i in 0..15 {
            let theta = 2.0 * PI * (i as f64) / 15.0;
            fourier[(i, 0)] = theta.cos();
            fourier[(i, 1)] = theta.sin();
        }
        for mut column in fourier.column_iter_mut() {
            let norm = column.norm();
            column /= norm;
        }

        let pair = spectrum.eigenvectors().columns(1, 2).into_owned();
        let residual = (&fourier - &pair * (pair.transpose() * &fourier)).norm();
        assert!(
            residual < 1e-9,
            "cycle(15): ‖F − QQᵀF‖_F = {residual:.3e} for the Fourier modes F and the \
             computed Fiedler pair Q, tolerance 1e-9"
        );
    }

    /// −L of the complete graph has eigenvalue 0 once and −2n/(n−1) with
    /// multiplicity n−1.
    #[test]
    fn complete_graph_eigenvalues_match_the_closed_form() {
        for n in [2_usize, 3, 5, 8, 15] {
            let graph = Graph::complete(n).expect("complete(n)");
            let spectrum =
                Spectrum::of_negative_laplacian(&graph).expect("spectrum of complete(n)");

            let top = spectrum.eigenvalues()[0];
            assert!(
                top.abs() < 1e-9,
                "complete({n}): top eigenvalue is {top:.3e}, expected 0 (tolerance 1e-9)"
            );

            let bulk = -2.0 * (n as f64) / ((n - 1) as f64);
            for k in 1..n {
                let got = spectrum.eigenvalues()[k];
                assert!(
                    (got - bulk).abs() < 1e-9,
                    "complete({n}): eigenvalue {k} is {got:.15}, closed form gives \
                     {bulk:.15} (|Δ| = {:.3e}, tolerance 1e-9)",
                    (got - bulk).abs()
                );
            }
        }
    }

    /// The complete graph's top eigenvector is constant.
    #[test]
    fn complete_graph_top_eigenvector_is_constant() {
        let n = 8_usize;
        let graph = Graph::complete(n).expect("complete(8)");
        let spectrum = Spectrum::of_negative_laplacian(&graph).expect("spectrum of complete(8)");

        let top = spectrum.eigenvectors().column(0);
        let uniform = 1.0 / (n as f64).sqrt();
        for (vertex, &component) in top.iter().enumerate() {
            assert!(
                (component - uniform).abs() < 1e-9,
                "complete({n}): top eigenvector component {vertex} is {component:.15}, \
                 expected {uniform:.15} (|Δ| = {:.3e}, tolerance 1e-9)",
                (component - uniform).abs()
            );
        }
    }

    /// L = X + Xᵀ is symmetric on every constructor, including the
    /// irregular-degree ones where I − D⁻¹A is not.
    #[test]
    fn laplacian_is_symmetric_on_every_constructor() {
        for (name, graph) in fixtures() {
            let l = laplacian(&graph).expect("laplacian");
            let asymmetry = (&l - l.transpose()).norm();
            assert!(
                asymmetry < 1e-10,
                "{name}: ‖L − Lᵀ‖_F = {asymmetry:.3e}, tolerance 1e-10"
            );
        }
    }

    /// E Λ Eᵀ reconstructs −L on every constructor.
    #[test]
    fn spectrum_reconstructs_negative_laplacian_on_every_constructor() {
        for (name, graph) in fixtures() {
            let target = -laplacian(&graph).expect("laplacian");
            let spectrum = Spectrum::new(target.clone()).expect("spectrum");
            let e = spectrum.eigenvectors();
            let reconstruction = e * DMatrix::from_diagonal(spectrum.eigenvalues()) * e.transpose();
            let residual = (&reconstruction - &target).norm();
            assert!(
                residual < 1e-10,
                "{name}: ‖EΛEᵀ − (−L)‖_F = {residual:.3e}, tolerance 1e-10"
            );
        }
    }

    /// The eigenvectors are orthonormal on every constructor.
    #[test]
    fn spectrum_eigenvectors_are_orthonormal_on_every_constructor() {
        for (name, graph) in fixtures() {
            let spectrum = Spectrum::of_negative_laplacian(&graph).expect("spectrum");
            let e = spectrum.eigenvectors();
            let deviation = (e.transpose() * e
                - DMatrix::<f64>::identity(spectrum.len(), spectrum.len()))
            .norm();
            assert!(
                deviation < 1e-10,
                "{name}: ‖EᵀE − I‖_F = {deviation:.3e}, tolerance 1e-10"
            );
        }
    }

    /// D5's ordering: eigenvalues descend on every constructor.
    #[test]
    fn spectrum_eigenvalues_descend_on_every_constructor() {
        for (name, graph) in fixtures() {
            let spectrum = Spectrum::of_negative_laplacian(&graph).expect("spectrum");
            for (k, window) in spectrum.eigenvalues().as_slice().windows(2).enumerate() {
                assert!(
                    window[0] >= window[1],
                    "{name}: eigenvalue {k} = {} is below eigenvalue {} = {}, breaking D5's \
                     descending order",
                    window[0],
                    k + 1,
                    window[1]
                );
            }
        }
    }

    /// D5's sign convention: each eigenvector's first component above
    /// `SIGN_PIVOT_TOLERANCE` is positive, on every constructor.
    #[test]
    fn spectrum_eigenvector_signs_are_deterministic_on_every_constructor() {
        for (name, graph) in fixtures() {
            let spectrum = Spectrum::of_negative_laplacian(&graph).expect("spectrum");
            for (k, column) in spectrum.eigenvectors().column_iter().enumerate() {
                let Some(pivot) = column
                    .iter()
                    .copied()
                    .find(|component| component.abs() > SIGN_PIVOT_TOLERANCE)
                else {
                    panic!(
                        "{name}: eigenvector {k} has no component above {SIGN_PIVOT_TOLERANCE:e}, \
                         so D5's sign rule has nothing to fix"
                    )
                };
                assert!(
                    pivot > 0.0,
                    "{name}: eigenvector {k} has leading significant component {pivot:.6e}; \
                     D5 requires it positive"
                );
            }
        }
    }

    /// A degree-zero vertex has no random-walk Laplacian.
    #[test]
    fn laplacian_rejects_an_isolated_vertex() {
        let graph = Graph::grid(1, 1).expect("grid(1,1)");
        match laplacian(&graph) {
            Err(Error::IsolatedVertex { vertex }) => {
                assert_eq!(vertex, 0, "reported vertex {vertex}, expected 0");
            }
            other => panic!("expected IsolatedVertex, got {other:?}"),
        }
    }

    /// A non-square matrix is rejected before nalgebra's own assertion.
    #[test]
    fn spectrum_rejects_a_non_square_matrix() {
        match Spectrum::new(DMatrix::<f64>::zeros(2, 3)) {
            Err(Error::NotSquare { rows, columns }) => {
                assert_eq!((rows, columns), (2, 3), "reported {rows}×{columns}");
            }
            other => panic!("expected NotSquare, got {other:?}"),
        }
    }
}
