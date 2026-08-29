# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Tier 0 numerics: `Graph` over dense adjacency with
  `path_star`/`grid`/`cycle`/`irregular`/`tree_star`/`complete` constructors
  and a public `from_edges` builder; `spectral::transition` (D⁻¹A),
  `spectral::laplacian` ((I − D⁻¹A) + (I − D⁻¹A)ᵀ), `spectral::symmetrize`,
  and `Spectrum` — eigendecomposition of −L with descending order, a
  deterministic sign convention, and `degenerate_groups` for
  subspace-safe comparisons. Input validation covers empty, non-square,
  non-finite, and asymmetric matrices, isolated vertices, and
  unrepresentable vertex counts. Closed-form and property pins in-module;
  public-seam integration tests in `tests/tier0_spectral.rs` and
  `tests/lifecycle.rs`.
- Library-first crate scaffold mirroring `surrealdb-live-message` conventions:
  `error` (thiserror enum + `Result` alias), `logger` (tracing), `settings`
  (config crate + `config/*.toml`; experiment output dir + global RNG seed),
  `graph`/`spectral` Tier-0 module stubs, and `subsystems::runner::Runner` —
  the async lifecycle core owning a `TaskTracker` + `CancellationToken`, with
  `spawn` (jobs receive a `child_token()`), `cancellation_token()`, and
  `shutdown()` (cancel → close → drain). Lifecycle tests each verified to
  fail when the behavior they pin is reverted; a review sweep then hardened
  the drain pin against single-yield vacuity, added a per-job token-isolation
  test, and bounded every lifecycle await. Thin ctrl-c daemon in
  `src/main.rs`.
- `docs/2510.26745v2-poc-analysis.md` — validation report for the markdown
  conversion of arXiv 2510.26745v2, claims inventory, reproducibility
  assessment, tiered POC design space, verification pins, risks, and the
  recorded decisions (nalgebra; structure mirroring `surrealdb-live-message`).
