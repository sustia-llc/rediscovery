//! Spectral infrastructure for Tier 0 of the POC.
//!
//! Computes the asymmetric random-walk Laplacian L = (I − D⁻¹A) + (I − D⁻¹A)ᵀ
//! from a graph's adjacency/degree data, and its symmetric eigendecomposition
//! with a deterministic eigenvalue ordering. Everything downstream that
//! reasons about eigenvector alignment (Tier 1's `Node2Vec` dynamics)
//! consumes this.
