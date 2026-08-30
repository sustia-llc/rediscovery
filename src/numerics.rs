//! Numeric primitives shared by the tier modules.
//!
//! [`gaussian_matrix`] draws a seeded matrix of scaled standard normals,
//! [`row_softmax`] and [`log_sum_exp`] carry the row-maximum shift the softmax
//! paths use, and [`weighted_log_likelihood`] evaluates
//! Σ_ij W_ij (Z_ij − log Σ_k exp Z_ik) for a weight matrix and a matrix of
//! logits. `node2vec` (Tier 1) and `tinynn` (Tier 2) share these, so each of
//! the four formulas has one source.

#![allow(
    clippy::doc_markdown,
    reason = "the docs carry matrix notation with subscripts — W_ij, Z_ij — that the lint reads as unbackticked identifiers"
)]

use nalgebra::DMatrix;
use rand::SeedableRng;
use rand::distr::{Distribution, OpenClosed01};
use rand_chacha::ChaCha20Rng;

use crate::error::{Error, Result};

/// Draws a `rows`×`columns` matrix whose entries are `sigma`-scaled standard
/// normals from a `ChaCha20` stream keyed by `seed`.
///
/// Entries are filled row by row from a fixed number of draws each, so a given
/// `(seed, rows, columns, sigma)` yields bit-identical values.
///
/// # Errors
///
/// Returns [`Error::MatrixTooLarge`] when `rows * columns` overflows `usize`.
pub fn gaussian_matrix(rows: usize, columns: usize, sigma: f64, seed: u64) -> Result<DMatrix<f64>> {
    let entries = rows
        .checked_mul(columns)
        .ok_or(Error::MatrixTooLarge { rows, columns })?;

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut values = Vec::with_capacity(entries);
    for _ in 0..entries {
        values.push(sigma * standard_normal(&mut rng));
    }
    Ok(DMatrix::from_row_iterator(rows, columns, values))
}

/// One draw from N(0, 1) by the Box–Muller transform, consuming two uniforms
/// from `rng` on the half-open interval (0, 1].
fn standard_normal<R: rand::Rng + ?Sized>(rng: &mut R) -> f64 {
    let radial: f64 = OpenClosed01.sample(rng);
    let angular: f64 = OpenClosed01.sample(rng);
    (-2.0 * radial.ln()).sqrt() * (std::f64::consts::TAU * angular).cos()
}

/// Row `i`'s log Σ_k exp, shifted by that row's maximum before exponentiating.
#[must_use]
pub fn log_sum_exp(logits: &DMatrix<f64>, i: usize) -> f64 {
    let row = logits.row(i);
    let peak = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let shifted: f64 = row.iter().map(|entry| (entry - peak).exp()).sum();
    peak + shifted.ln()
}

/// Applies a row-wise softmax to `logits`, shifting each row by its maximum
/// before exponentiating.
#[must_use]
pub fn row_softmax(logits: &DMatrix<f64>) -> DMatrix<f64> {
    let mut probabilities = logits.clone();
    for mut row in probabilities.row_iter_mut() {
        let peak = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        row.apply(|entry| *entry = (*entry - peak).exp());
        let total = row.sum();
        row /= total;
    }
    probabilities
}

/// Evaluates Σ_i Σ_j W_ij (Z_ij − log Σ_k exp Z_ik) over the positive entries
/// of `weights`, each row's log-partition shifted by that row's maximum.
///
/// With `weights` row-stochastic this is the log-likelihood the row
/// distributions assign under `logits`; its negation is the cross-entropy.
/// Entries of `weights` are read over `logits`'s row and column range.
#[must_use]
pub fn weighted_log_likelihood(weights: &DMatrix<f64>, logits: &DMatrix<f64>) -> f64 {
    let mut total = 0.0;
    for i in 0..logits.nrows() {
        let log_partition = log_sum_exp(logits, i);
        for j in 0..logits.ncols() {
            let weight = weights[(i, j)];
            if weight > 0.0 {
                total += weight * (logits[(i, j)] - log_partition);
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same seed redraws the same matrix bit for bit, and a different seed
    /// does not — so the identity above is not a property of the draw being
    /// degenerate.
    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the claim is bit-identity of a seeded draw, not numeric closeness"
    )]
    fn a_seeded_draw_is_reproducible_and_seed_dependent() {
        let first = gaussian_matrix(5, 7, 0.5, 20_260_829).expect("gaussian_matrix");
        let second = gaussian_matrix(5, 7, 0.5, 20_260_829).expect("gaussian_matrix");
        let other = gaussian_matrix(5, 7, 0.5, 20_260_830).expect("gaussian_matrix");

        for (index, (left, right)) in first.iter().zip(second.iter()).enumerate() {
            assert!(
                left == right,
                "entry {index} is {left:e} on the first draw and {right:e} on the second, \
                 at the same seed"
            );
        }
        let deviation = (&first - &other).amax();
        assert!(
            deviation > 0.0,
            "seeds 20260829 and 20260830 drew an identical matrix (max |Δ| = {deviation:e})"
        );
    }

    /// A row-softmax row sums to one and matches the log-partition path:
    /// ln P_ij equals Z_ij − `log_sum_exp`(Z, i).
    #[test]
    fn the_softmax_and_the_log_partition_agree() {
        let logits = gaussian_matrix(6, 6, 3.0, 11).expect("gaussian_matrix");
        let probabilities = row_softmax(&logits);

        for i in 0..logits.nrows() {
            let sum = probabilities.row(i).sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "row {i} sums to {sum:.15}, expected 1"
            );
            let partition = log_sum_exp(&logits, i);
            for j in 0..logits.ncols() {
                let expected = logits[(i, j)] - partition;
                let observed = probabilities[(i, j)].ln();
                assert!(
                    (observed - expected).abs() < 1e-12,
                    "ln P({i}, {j}) is {observed:.15}, Z − log Σ exp is {expected:.15}"
                );
            }
        }
    }

    /// `weighted_log_likelihood` equals Σ W_ij ln P_ij computed from the
    /// separately built probability matrix.
    #[test]
    fn the_log_likelihood_agrees_with_the_probability_matrix() {
        let logits = gaussian_matrix(6, 6, 2.0, 13).expect("gaussian_matrix");
        let weights = DMatrix::from_fn(6, 6, |i, j| if i == j { 0.0 } else { 0.2 });
        let probabilities = row_softmax(&logits);

        let mut expected = 0.0;
        for i in 0..6 {
            for j in 0..6 {
                if weights[(i, j)] > 0.0 {
                    expected += weights[(i, j)] * probabilities[(i, j)].ln();
                }
            }
        }

        let observed = weighted_log_likelihood(&weights, &logits);
        assert!(
            (observed - expected).abs() < 1e-12,
            "log-likelihood is {observed:.15}, Σ W_ij ln P_ij is {expected:.15}"
        );
    }

    /// A draw whose entry count overflows `usize` is a typed error rather than
    /// an allocation attempt.
    #[test]
    fn an_unrepresentable_draw_is_a_typed_error() {
        match gaussian_matrix(usize::MAX, 2, 1.0, 0) {
            Err(Error::MatrixTooLarge { rows, columns }) => {
                assert_eq!(
                    (rows, columns),
                    (usize::MAX, 2),
                    "reported {rows}×{columns}"
                );
            }
            other => panic!("expected MatrixTooLarge, got {other:?}"),
        }
    }
}
