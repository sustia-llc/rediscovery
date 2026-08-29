//! Public-API seam for Tier 0: `Graph` → `laplacian` → `Spectrum`.
//!
//! Compiles against the exported surface only, so it also pins the API that
//! Tier 1 consumes. The algebraic pins and their falsification live in the
//! in-module unit tests; these assertions guard the seam and the drift
//! between the two.

use std::f64::consts::PI;
use std::time::Instant;

use nalgebra::DMatrix;
use rediscovery::error::Result;
use rediscovery::graph::Graph;
use rediscovery::spectral::{Spectrum, laplacian};

/// Tolerance for eigenvalues checked against closed forms.
const EIGENVALUE_TOLERANCE: f64 = 1e-9;
/// Tolerance for Frobenius residuals of exact matrix identities.
const RESIDUAL_TOLERANCE: f64 = 1e-10;
/// Tolerance for two computations of the same matrix that differ only in
/// floating-point association order.
const DRIFT_TOLERANCE: f64 = 1e-12;

/// A constructor callable twice to produce independent, identical graphs.
type Builder = fn() -> Result<Graph>;

/// Every Tier 0 constructor, with the label used in failure messages.
fn builders() -> Vec<(&'static str, Builder)> {
    vec![
        ("path_star(4,4)", || Graph::path_star(4, 4)),
        ("grid(4,4)", || Graph::grid(4, 4)),
        ("cycle(15)", || Graph::cycle(15)),
        ("irregular()", Graph::irregular),
        ("tree_star(4,3)", || Graph::tree_star(4, 3)),
        ("complete(8)", || Graph::complete(8)),
    ]
}

/// The full flow runs through the public surface for every constructor:
/// `laplacian` matches (I − D⁻¹A) + (I − D⁻¹A)ᵀ recomputed from the public
/// adjacency and degree accessors, and the `Spectrum` it feeds reconstructs
/// that same L with orthonormal eigenvectors.
#[test]
fn graph_to_spectrum_flow_for_every_constructor() {
    let started = Instant::now();

    for (name, build) in builders() {
        let graph = build().expect("constructor");
        let n = graph.order();

        let l = laplacian(&graph).expect("laplacian");
        assert_eq!(
            l.shape(),
            (n, n),
            "{name}: laplacian shape {:?}, expected ({n}, {n})",
            l.shape()
        );

        // Entrywise reference from the Appendix F formula's closed form:
        // L has diagonal 2 and off-diagonal −(1/d_u + 1/d_v) exactly where
        // A has an edge.
        let reference = DMatrix::<f64>::from_fn(n, n, |u, v| {
            if u == v {
                2.0
            } else if graph.adjacency()[(u, v)] > 0.0 {
                -(1.0 / graph.degrees()[u] + 1.0 / graph.degrees()[v])
            } else {
                0.0
            }
        });
        let drift = (&l - &reference).norm();
        assert!(
            drift < DRIFT_TOLERANCE,
            "{name}: ‖laplacian() − closed form‖_F = {drift:.3e}, tolerance {DRIFT_TOLERANCE:e}"
        );

        let spectrum = Spectrum::of_negative_laplacian(&graph).expect("spectrum");
        assert_eq!(
            spectrum.order(),
            n,
            "{name}: spectrum holds {} eigenpairs, expected {n}",
            spectrum.order()
        );
        assert_eq!(
            spectrum.eigenvectors().shape(),
            (n, n),
            "{name}: eigenvector matrix shape {:?}, expected ({n}, {n})",
            spectrum.eigenvectors().shape()
        );

        let e = spectrum.eigenvectors();
        let orthonormality = (e.transpose() * e - DMatrix::<f64>::identity(n, n)).norm();
        assert!(
            orthonormality < RESIDUAL_TOLERANCE,
            "{name}: ‖EᵀE − I‖_F = {orthonormality:.3e}, tolerance {RESIDUAL_TOLERANCE:e}"
        );

        let reconstruction = e * DMatrix::from_diagonal(spectrum.eigenvalues()) * e.transpose();
        let residual = (&reconstruction + &l).norm();
        assert!(
            residual < RESIDUAL_TOLERANCE,
            "{name}: ‖EΛEᵀ + L‖_F = {residual:.3e}, tolerance {RESIDUAL_TOLERANCE:e}"
        );
    }

    println!(
        "graph_to_spectrum_flow_for_every_constructor: {:?}",
        started.elapsed()
    );
}

/// Two independent identical calls — fresh graph, fresh decomposition —
/// produce bit-identical eigenvalues and eigenvectors.
#[test]
fn spectrum_is_deterministic_across_identical_calls() {
    let started = Instant::now();

    for (name, build) in builders() {
        let first = Spectrum::of_negative_laplacian(&build().expect("constructor"))
            .expect("first spectrum");
        let second = Spectrum::of_negative_laplacian(&build().expect("constructor"))
            .expect("second spectrum");

        assert_eq!(
            first.eigenvalues(),
            second.eigenvalues(),
            "{name}: eigenvalues differ between two identical calls"
        );
        assert_eq!(
            first.eigenvectors(),
            second.eigenvectors(),
            "{name}: eigenvectors differ between two identical calls"
        );
    }

    println!(
        "spectrum_is_deterministic_across_identical_calls: {:?}",
        started.elapsed()
    );
}

/// The 15-cycle closed form re-checked through the public API only, guarding
/// against drift between the in-module pin and the exported surface.
#[test]
fn cycle15_closed_form_through_the_public_api() {
    let graph = Graph::cycle(15).expect("cycle(15)");
    let spectrum = Spectrum::of_negative_laplacian(&graph).expect("spectrum");

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
