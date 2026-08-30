# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Tier 2 `TinyNn`: the §B.2.2 model as Z = E W Eᵀ with a tied 512-wide
  embedding and one trainable weight matrix, in frozen-E and learnable-E
  regimes over the same degree-normalized cross-entropy Tier 1 uses. Hand-derived gradients FD-checked per parameter
  block, a GELU variant, per-step CSV instrumentation, cosine and adjacency
  heatmaps, and `examples/tier2_tinynn.rs`. New `numerics` and `output`
  modules hold the softmax, log-sum-exp, cross-entropy, seeded Gaussian draw
  and matrix-CSV writer that Tiers 1 and 2 share.

### Findings (Tier 2)

> **Retraction, 2026-08-29.** An adversarial review of the shell-based
> geometry criterion found it unsound, and every claim below that used the
> word "geometry" is withdrawn pending the replacement measure. What the
> criterion actually scores is the deepest read shell's mean cosine, and it
> reads only shells 2 and 3. Measured: a cosine matrix with neighbours
> near-antipodal (d1 = −0.9, d2 = +0.6, d3 = +0.2) scores 0.200 and passes on
> all four graphs; a near-collapsed cone (d1 = d2 = 0.95, d3 = 0.80) scores
> 0.150 and passes. Raising the read depth to four shells makes the same runs
> fail (path-star peaks at −0.000439 over 1200 steps). Worse, the certified
> embeddings load on the **bottom** of the spectrum — top singular direction
> at eigenvector index 16/15/14 of −L with Fiedler-mass **0.00**, against
> Node2Vec's 0.98–1.00 on the same graphs — so the quantity measured is not
> the spectral geometry §4.1 describes. The 0.05 threshold also rejects the
> paper's own reference geometry on the path-star (0.0313) while accepting a
> rank-1 embedding on the grid and cycle.

- **Refutation 3c's associative half reproduces.** The frozen regime reaches
  its maximum top-d(u) neighbour score at **step 1** on all four graphs and
  both seeds, inside the paper's two steps. Initial scores 0.089–0.221 and
  0.078–0.167; correlation between the model's distribution and D⁻¹A is
  0.974–0.981 at the hit step. The timing *ratio* against the geometric side
  is withdrawn with the criterion, and the pin that carried it mixed regimes
  and rates: at matched η within one learnable run the largest ratio measured
  is **49.5** (cycle, η = 0.001), below the 50 it asserted, and 13.6 at
  seed 42.
- **The ≤2-step result depends on an initializer the paper does not state.**
  Measured across a σ grid: at `weight_sigma = 1/√m` it takes 4–10 steps
  (seed-dependent); at the PyTorch `nn.Linear` default it fails on three of
  four graphs; at `embedding_sigma = 1.0` it never arrives within 200 steps.
  It is also width-dependent — never within 500 steps at m = 8, and 1–2 steps
  only from m ≈ 128. The committed default (E ~ N(0,1/√m), W ~ N(0,1/m)) sits
  where ‖EEᵀ − I‖_max = 0.156, the near-orthogonality the one-step argument
  assumes, and is a documented guess.
- **Figure 22's "η = 0.1 is too aggressive" did not reproduce, and cannot
  under the captions' optimizer.** The three swept rates trace one
  gradient-flow trajectory — max|E(η=0.1, 10 steps) − E(η=0.01, 100 steps)|
  is 1.3e-2 against max|E| ≈ 2.4e-1, and the step count scales as 1/η — so
  under the constant-rate full-batch GD Figures 7/8/22 describe, no learning
  rate can be "too aggressive to create" what a smaller one creates later.
  §B.3 instead specifies AdamW with weight decay and a cosine schedule; the
  implementation follows the captions. The η = 0.1 embedding is not
  degenerate (numeric rank = n, effective rank 9.9–12.3, row-norm spread
  ≤ 1.34). This finding survives the retraction because it rests on the
  trajectory, not on the criterion.
- **Figure 23's "gradual decrease in similarity" does not hold for this
  architecture, and the deviation is larger than first reported.** Over the
  full diameter the distance-2 mean is the global maximum on every graph
  (cycle η = 0.001: −0.169, **+0.379**, −0.098, −0.088, −0.083, −0.184,
  −0.136), not a decay. Node2Vec on the same graphs is monotone throughout
  (cycle +0.962 → +0.121 across d1..d7), which is the contrast the paper
  draws.
- **A diverged run could report a geometry.** At η = 10 the irregular graph
  scored 0.098 at loss 9.99e27, cosines being scale-invariant; once the
  embedding is wholly non-finite the score is 0.0000 rather than NaN. No
  reported number is affected (all swept rates stay finite, final losses
  10.18–20.57), but the measure needs a finiteness guard.

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
