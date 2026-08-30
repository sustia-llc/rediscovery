# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Tier 2 `TinyNn`: the §B.2.2 model as Z = E W Eᵀ with a tied 512-wide
  embedding and one trainable weight matrix, in frozen-E (associative) and
  learnable-E (geometric) regimes over the same degree-normalized
  cross-entropy Tier 1 uses. Hand-derived gradients FD-checked per parameter
  block, a GELU variant, per-step CSV instrumentation, cosine and adjacency
  heatmaps, and `examples/tier2_tinynn.rs`. New `numerics` and `output`
  modules hold the softmax, log-sum-exp, cross-entropy, seeded Gaussian draw
  and matrix-CSV writer that Tiers 1 and 2 share.

### Findings (Tier 2)

- **Refutation 3c reproduces.** The frozen regime reaches its maximum
  top-d(u) neighbour score at **step 1** on all four graphs and both seeds,
  inside the paper's two steps, while a geometry needs 554–743 steps at
  η = 0.001 — a ratio of 743 on the 15-cycle. The same asymmetry holds
  inside a single learnable run (memorization at step 15–29, geometry at
  605–743, a ratio of 49.5 on the cycle).
- **The ≤2-step result depends on an initializer the paper does not state.**
  At `weight_sigma = 1/√m` it takes 5–10 steps; at `embedding_sigma = 1.0` it
  does not arrive within 20. The committed default (E ~ N(0,1/m),
  W ~ N(0,1/m²)) is the near-orthogonal-embedding setting the one-step
  argument is derived at, and is a documented guess.
- **Figure 22's "η = 0.1 is too aggressive to create the geometry" did not
  reproduce.** At η = 0.1 the criterion is met earliest (7–9 steps) with an
  equal-or-larger peak margin than the smaller rates reach.
- **Figure 23's "gradual decrease in similarity" does not hold for this
  architecture.** The learned shell profile is non-monotone on all four
  graphs — distance-2 pairs carry the highest mean cosine (cycle at η = 0.01:
  −0.102 / +0.389 / −0.198) — which is why the geometry criterion measures
  the deepest shell's distance from zero rather than monotone decay.
- The geometry criterion is this POC's own definition; the paper states none.
  An adjacency-row embedding scores exactly zero on it.

### Added

- Tier 1 `Node2Vec` dynamics: the 1-hop weight-tied system of Appendix F —
  the Eq-1 objective with the self term in the softmax denominator,
  P = row_softmax(VVᵀ), and Lemma 6's step ΔV = ηCV with
  C = (W − P) + (W − P)ᵀ — plus the weight-untied variant, seeded ChaCha
  initialization, cancellation-aware runs, per-step CSV instrumentation
  (eigenvector projections, coefficient norms, objective, Observation-8
  subspace residual), node-node cosine dumps, and `examples/tier1_fig9.rs`.
  Both gradients are pinned by central finite differences.

### Findings (Tier 1)

- **Lemma 6's sign is correct and the text's Proposition-7 restatement is
  not.** Flipping C to +(P + Pᵀ) takes the finite-difference deviation to
  7.27 against a 1e-7 tolerance.
- **The 15-cycle reproduces Figure 9 in full**: converged at step 15 855,
  projection separation ≈ 4.1e4, ‖Ce_i‖₂ 5.0e-7 on the Fiedler pair,
  Observation-8 residual 4.9e-6.
- **On the path-star and grid the coefficient norm does not reach zero on
  the Fiedler-like set, and the implementation matches the paper's own
  plotted panels while doing so.** Measured 1.587e-1 (path-star, set 1..4)
  and 6.406e-2 (grid, set 1..3) at 10 000 steps, rising with training to a
  fixed point (path-star 1.790e-1 at 1e6 steps, byte-stable at 2e6) and
  unchanged across η ∈ {0.001, 0.01, 0.1}, σ ∈ {1, 4}, m ∈ {100, 400}, and a
  different initialization draw. The paper's Figure 9 path-star panel plots
  the same plateau (peak ≈0.3 near epoch 100, settling ≈0.15), so the
  caption's "converges to 0" over-claims relative to its own figure. Two
  `#[ignore]`d tests carry the measurements.
- **Observation 8 holds only on the circulant graph.** Residual 4.9e-6 on
  the 15-cycle against 0.25 (grid), 0.42 (path-star) and 0.22 (irregular) at
  10 000 steps, all three growing with further training. Degree-regularity
  alone does not explain the split: a 3-regular non-vertex-transitive graph
  also misses the condition (5.5e-2 at 1e5 steps, rising).
- **Proposition 7's shared-eigenvector claim fails numerically off the
  circulant graph.** On the irregular graph 99.6% of ‖Ce_0‖₂ is C's
  eigenvector rotating away from −L's, not its eigenvalue approaching zero —
  a different failure from the one Appendix F.2's narrative assumes. Both
  quantities are exactly zero at initialization, confirming Fact 1 to 4.8e-15.
- **The disconnected graph needs the projection claim read per component**:
  its two components contribute two near-null eigenvalues whose eigenvectors
  are the component indicators, followed by the two components' Fiedler
  vectors, so the single-degenerate-group definition of "Fiedler-like" does
  not apply to it.

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
