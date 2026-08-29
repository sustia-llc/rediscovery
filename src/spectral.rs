//! Spectral infrastructure for Tier 0 of the POC.
//!
//! [`transition`] builds the row-normalized walk matrix D⁻¹A, [`laplacian`]
//! the asymmetrically normalized random-walk Laplacian
//! L = (I − D⁻¹A) + (I − D⁻¹A)ᵀ of Appendix F, and [`Spectrum`] holds the
//! symmetric eigendecomposition of −L with the deterministic ordering and
//! sign convention of decision D5 (`docs/2510.26745v2-poc-analysis.md` §8).
//! All arithmetic is f64. Everything downstream that reasons about
//! eigenvector alignment (Tier 1's `Node2Vec` dynamics) consumes this.

use nalgebra::{DMatrix, DVector, linalg::SymmetricEigen};

use crate::error::{Error, Result};
use crate::graph::Graph;

/// Magnitude above which an eigenvector component may fix that
/// eigenvector's sign (decision D5).
pub const SIGN_PIVOT_TOLERANCE: f64 = 1e-9;

/// Computes the row-normalized transition matrix W = D⁻¹A for `graph`.
///
/// # Errors
///
/// Returns [`Error::IsolatedVertex`] for a degree-zero vertex, where D⁻¹
/// does not exist.
pub fn transition(graph: &Graph) -> Result<DMatrix<f64>> {
    if let Some((vertex, _)) = graph
        .degrees()
        .iter()
        .enumerate()
        .find(|(_, degree)| **degree <= 0.0)
    {
        return Err(Error::IsolatedVertex { vertex });
    }

    let mut walk = graph.adjacency().clone();
    for (vertex, mut row) in walk.row_iter_mut().enumerate() {
        row /= graph.degrees()[vertex];
    }
    Ok(walk)
}

/// Computes X + Xᵀ, the symmetric part of `matrix` scaled by two.
#[must_use]
pub fn symmetrize(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    matrix + matrix.transpose()
}

/// Computes L = (I − D⁻¹A) + (I − D⁻¹A)ᵀ for `graph`.
///
/// The summand I − D⁻¹A is not symmetric on an irregular graph; the returned
/// L is, being of the form X + Xᵀ.
///
/// # Errors
///
/// Propagates [`transition`]'s [`Error::IsolatedVertex`].
pub fn laplacian(graph: &Graph) -> Result<DMatrix<f64>> {
    let order = graph.order();
    let deviation = DMatrix::<f64>::identity(order, order) - transition(graph)?;
    Ok(symmetrize(&deviation))
}

/// The eigendecomposition of a symmetric matrix, ordered and sign-fixed per
/// decision D5.
///
/// Eigenvalues descend; eigenvector `j` is column `j` of
/// [`Spectrum::eigenvectors`], is unit-norm, and has a positive first
/// component of magnitude above [`SIGN_PIVOT_TOLERANCE`]. Within a group of
/// equal eigenvalues (see [`Spectrum::degenerate_groups`]) the columns are
/// one orthonormal basis of the group's eigenspace; only the span is
/// determined, so cross-run comparisons over such groups must compare
/// subspaces, not columns.
#[derive(Debug, Clone)]
pub struct Spectrum {
    eigenvalues: DVector<f64>,
    eigenvectors: DMatrix<f64>,
}

impl Spectrum {
    /// Decomposes `symmetric`, then applies the D5 ordering and sign
    /// convention.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmptyMatrix`] for a 0×0 argument,
    /// [`Error::NotSquare`] for a non-square one, [`Error::NonFinite`] for a
    /// NaN or infinite entry, and [`Error::NotSymmetric`] for entries whose
    /// mirror differs.
    ///
    /// # Panics
    ///
    /// Panics if `SymmetricEigen::try_new` with `max_niter = 0` returns
    /// `None`: nalgebra 0.35's iteration returns `None` only on reaching a
    /// positive `max_niter` (`symmetric_eigen.rs:245-247`), and its own
    /// `SymmetricEigen::new` unwraps the same call.
    #[allow(
        clippy::float_cmp,
        reason = "callers build X + Xᵀ, which is bitwise symmetric; a tolerance would be an invented knob"
    )]
    pub fn new(symmetric: DMatrix<f64>) -> Result<Self> {
        let (rows, columns) = symmetric.shape();
        if rows != columns {
            return Err(Error::NotSquare { rows, columns });
        }
        if rows == 0 {
            return Err(Error::EmptyMatrix);
        }
        for row in 0..rows {
            for column in row..columns {
                let entry = symmetric[(row, column)];
                if !entry.is_finite() {
                    return Err(Error::NonFinite { row, column });
                }
                let mirror = symmetric[(column, row)];
                if !mirror.is_finite() {
                    return Err(Error::NonFinite {
                        row: column,
                        column: row,
                    });
                }
                if entry != mirror {
                    return Err(Error::NotSymmetric { row, column });
                }
            }
        }

        let eigen = SymmetricEigen::try_new(symmetric, f64::EPSILON, 0).expect(
            "invariant: max_niter = 0 iterates until convergence and never yields None \
             (nalgebra 0.35 symmetric_eigen.rs:245-247)",
        );

        let mut order: Vec<usize> = (0..rows).collect();
        order.sort_by(|&a, &b| eigen.eigenvalues[b].total_cmp(&eigen.eigenvalues[a]));

        let eigenvalues = eigen.eigenvalues.select_rows(order.iter());
        let mut eigenvectors = eigen.eigenvectors.select_columns(order.iter());

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
    /// [`Spectrum::new`]'s errors.
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
    pub fn order(&self) -> usize {
        self.eigenvalues.len()
    }

    /// Splits `0..order` into maximal runs of consecutive indices whose
    /// adjacent eigenvalues differ by at most `gap_tolerance`, in the stored
    /// descending order. Within a returned group the eigenvector columns are
    /// one orthonormal basis of the group's eigenspace; assertions across
    /// runs or labelings must compare the spans.
    #[must_use]
    pub fn degenerate_groups(&self, gap_tolerance: f64) -> Vec<std::ops::Range<usize>> {
        let order = self.order();
        let mut groups = Vec::new();
        let mut start = 0;
        for end in 1..=order {
            if end == order
                || (self.eigenvalues[end - 1] - self.eigenvalues[end]).abs() > gap_tolerance
            {
                groups.push(start..end);
                start = end;
            }
        }
        groups
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    reason = "test indices are small and exact in f64"
)]
mod tests {
    use super::*;
    use crate::graph::test_fixtures;
    use std::f64::consts::PI;

    /// Tolerance for eigenvalues checked against closed forms.
    const EIGENVALUE_TOLERANCE: f64 = 1e-9;
    /// Tolerance for Frobenius residuals of exact matrix identities.
    const RESIDUAL_TOLERANCE: f64 = 1e-10;

    /// ‖`EᵀE` − I‖_F for a spectrum's eigenvector matrix.
    fn orthonormality_residual(spectrum: &Spectrum) -> f64 {
        let e = spectrum.eigenvectors();
        let n = spectrum.order();
        (e.transpose() * e - DMatrix::<f64>::identity(n, n)).norm()
    }

    /// ‖`EΛEᵀ` − target‖_F for a spectrum against the matrix it decomposed.
    fn reconstruction_residual(spectrum: &Spectrum, target: &DMatrix<f64>) -> f64 {
        let e = spectrum.eigenvectors();
        let reconstruction = e * DMatrix::from_diagonal(spectrum.eigenvalues()) * e.transpose();
        (&reconstruction - target).norm()
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
                (got - want).abs() < EIGENVALUE_TOLERANCE,
                "cycle(15): eigenvalue {k} is {got:.15}, closed form gives {want:.15} \
                 (|Δ| = {:.3e}, tolerance {EIGENVALUE_TOLERANCE:e})",
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
                (component - uniform).abs() < EIGENVALUE_TOLERANCE,
                "cycle(15): top eigenvector component {vertex} is {component:.15}, \
                 expected {uniform:.15} (|Δ| = {:.3e}, tolerance {EIGENVALUE_TOLERANCE:e})",
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
            gap < EIGENVALUE_TOLERANCE,
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
            residual < EIGENVALUE_TOLERANCE,
            "cycle(15): ‖F − QQᵀF‖_F = {residual:.3e} for the Fourier modes F and the \
             computed Fiedler pair Q, tolerance {EIGENVALUE_TOLERANCE:e}"
        );
    }

    /// The 15-cycle's spectrum groups as the simple 0 followed by seven
    /// degenerate pairs — the closed form's cos(2πk/15) symmetry.
    #[test]
    fn cycle15_degenerate_groups_match_the_closed_form() {
        let graph = Graph::cycle(15).expect("cycle(15)");
        let spectrum = Spectrum::of_negative_laplacian(&graph).expect("spectrum of cycle(15)");

        let groups = spectrum.degenerate_groups(EIGENVALUE_TOLERANCE);
        let expected: Vec<std::ops::Range<usize>> =
            vec![0..1, 1..3, 3..5, 5..7, 7..9, 9..11, 11..13, 13..15];
        assert_eq!(
            groups, expected,
            "cycle(15): degenerate groups are {groups:?}, closed form gives {expected:?}"
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
                top.abs() < EIGENVALUE_TOLERANCE,
                "complete({n}): top eigenvalue is {top:.3e}, expected 0 \
                 (tolerance {EIGENVALUE_TOLERANCE:e})"
            );

            let bulk = -2.0 * (n as f64) / ((n - 1) as f64);
            for k in 1..n {
                let got = spectrum.eigenvalues()[k];
                assert!(
                    (got - bulk).abs() < EIGENVALUE_TOLERANCE,
                    "complete({n}): eigenvalue {k} is {got:.15}, closed form gives \
                     {bulk:.15} (|Δ| = {:.3e}, tolerance {EIGENVALUE_TOLERANCE:e})",
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
                (component - uniform).abs() < EIGENVALUE_TOLERANCE,
                "complete({n}): top eigenvector component {vertex} is {component:.15}, \
                 expected {uniform:.15} (|Δ| = {:.3e}, tolerance {EIGENVALUE_TOLERANCE:e})",
                (component - uniform).abs()
            );
        }
    }

    /// The rows of W = D⁻¹A each sum to one on every constructor.
    #[test]
    fn transition_rows_sum_to_one_on_every_constructor() {
        for (name, graph) in test_fixtures() {
            let walk = transition(&graph).expect("transition");
            for (vertex, row) in walk.row_iter().enumerate() {
                let sum = row.sum();
                assert!(
                    (sum - 1.0).abs() < RESIDUAL_TOLERANCE,
                    "{name}: transition row {vertex} sums to {sum:.15}, expected 1"
                );
            }
        }
    }

    /// L = X + Xᵀ is symmetric on every constructor, including the
    /// irregular-degree ones where I − D⁻¹A is not.
    #[test]
    fn laplacian_is_symmetric_on_every_constructor() {
        for (name, graph) in test_fixtures() {
            let l = laplacian(&graph).expect("laplacian");
            let asymmetry = (&l - l.transpose()).norm();
            assert!(
                asymmetry < RESIDUAL_TOLERANCE,
                "{name}: ‖L − Lᵀ‖_F = {asymmetry:.3e}, tolerance {RESIDUAL_TOLERANCE:e}"
            );
        }
    }

    /// E Λ Eᵀ reconstructs −L on every constructor.
    #[test]
    fn spectrum_reconstructs_negative_laplacian_on_every_constructor() {
        for (name, graph) in test_fixtures() {
            let target = -laplacian(&graph).expect("laplacian");
            let spectrum = Spectrum::new(target.clone()).expect("spectrum");
            let residual = reconstruction_residual(&spectrum, &target);
            assert!(
                residual < RESIDUAL_TOLERANCE,
                "{name}: ‖EΛEᵀ − (−L)‖_F = {residual:.3e}, tolerance {RESIDUAL_TOLERANCE:e}"
            );
        }
    }

    /// The eigenvectors are orthonormal on every constructor.
    #[test]
    fn spectrum_eigenvectors_are_orthonormal_on_every_constructor() {
        for (name, graph) in test_fixtures() {
            let spectrum = Spectrum::of_negative_laplacian(&graph).expect("spectrum");
            let deviation = orthonormality_residual(&spectrum);
            assert!(
                deviation < RESIDUAL_TOLERANCE,
                "{name}: ‖EᵀE − I‖_F = {deviation:.3e}, tolerance {RESIDUAL_TOLERANCE:e}"
            );
        }
    }

    /// D5's ordering: eigenvalues descend on every constructor.
    #[test]
    fn spectrum_eigenvalues_descend_on_every_constructor() {
        for (name, graph) in test_fixtures() {
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
        for (name, graph) in test_fixtures() {
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

    /// A 0×0 matrix is square but rejected before nalgebra's empty-matrix
    /// assertion.
    #[test]
    fn spectrum_rejects_an_empty_matrix() {
        match Spectrum::new(DMatrix::<f64>::zeros(0, 0)) {
            Err(Error::EmptyMatrix) => {}
            other => panic!("expected EmptyMatrix, got {other:?}"),
        }
    }

    /// NaN and infinite entries are rejected with the offending index rather
    /// than decomposed into a NaN spectrum.
    #[test]
    fn spectrum_rejects_non_finite_entries() {
        let mut with_nan = DMatrix::<f64>::identity(3, 3);
        with_nan[(0, 2)] = f64::NAN;
        with_nan[(2, 0)] = f64::NAN;
        match Spectrum::new(with_nan) {
            Err(Error::NonFinite { row: 0, column: 2 }) => {}
            other => panic!("expected NonFinite at (0, 2), got {other:?}"),
        }

        let mut with_inf = DMatrix::<f64>::identity(3, 3);
        with_inf[(1, 1)] = f64::INFINITY;
        match Spectrum::new(with_inf) {
            Err(Error::NonFinite { row: 1, column: 1 }) => {}
            other => panic!("expected NonFinite at (1, 1), got {other:?}"),
        }
    }

    /// An asymmetric matrix is rejected instead of being silently decomposed
    /// as its lower-triangle mirror.
    #[test]
    fn spectrum_rejects_an_asymmetric_matrix() {
        let mut asymmetric = DMatrix::<f64>::identity(3, 3);
        asymmetric[(0, 1)] = 99.0;
        match Spectrum::new(asymmetric) {
            Err(Error::NotSymmetric { row: 0, column: 1 }) => {}
            other => panic!("expected NotSymmetric at (0, 1), got {other:?}"),
        }
    }
}
