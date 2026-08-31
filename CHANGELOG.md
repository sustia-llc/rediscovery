# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Findings (Tier 2)

- **The Gaussian-initializer null is scoped to the D-graphs; beyond them
  the boundary is jointly graph- and seed-dependent.** The committed
  finding — no Gaussian-initialized learnable run crosses the 0.75
  criterion in 20 000 steps at any ρ — holds on the four D-graphs at both
  committed seeds. On `grid(6, 8)` at the committed knobs (width 512,
  η = 0.01, ρ = 1, geometry stop at the criterion, 20 000-step budget) the
  outcome splits by seed: at 20260829 the Gaussian run **crosses at
  step 77**, and run past the stop to a 2 000-step budget it peaks at
  alignment **0.8160584913387996**; at seed 42 it does not cross — peak
  0.004342691178681286 within 2 000 steps, and none within 20 000 (that
  deeper budget measured in the PR #9 review, recorded on the PR). Both
  grid behaviors are pinned in `tests/tier2_flip_scope.rs`. First observed
  downstream in spatial-priors (finding F15); which graphs and seeds the
  dichotomy covers is its own open question.

## [0.1.0] - 2026-08-31

### Changed

- The Tier-2 sweep loops — the 24-configuration timing sweep, the
  12-configuration learnable sweep, the frozen-run test and the example's
  learnable runs — fan out across runs on 6-worker scoped rayon pools
  (dev-dependency; `src/` stays rayon-free) (#5). Per-run arithmetic is
  untouched: the masked sweep output diffs empty against the sequential
  baseline and all 36 example CSVs are byte-identical. Measured: lib binary
  32.4 → 17.7 s, tier-2 integration binary 60.6 → 28.3 s, example
  126 → 53 s; the sweep itself parallelizes 2.09× against the 3.57× the
  pre-#6 arithmetic gave, the per-step compute having halved against the
  same memory traffic. A panicking configuration re-raises after the
  surviving configurations' measurement lines print.

- Tier 2 `TinyNn::gradients_of` computes its two transposed-side products as
  `A.transpose() * B` instead of `A.tr_mul(B)`, routing them through
  nalgebra's gemm path rather than its per-entry column-dot loop
  (`xx_mul_to_uninit`) (#6). Max entrywise difference between the two forms
  ≤ 2.3e-16 over the committed budgets (measured in the #6 review, recorded
  in its closing comment); a 74-run two-arm A/B over the
  committed budgets moved no discrete event (memorization steps, alignment
  steps, outcomes) and no quoted Findings value at its quoted precision,
  with drift above 1e-11 confined to the η = 0.1 × 2000-step example runs.
  Measured on one machine: 13.3 → 7.0 ms per learnable step; under
  `cargo test`, the tier-2 integration binary 140.8 → 69.5 s and the lib
  binary 93.2 → 38.7 s. Three Findings values below are restated from new
  test-printed instruments where the original quote lacked a reproducing
  one: the hit-step correlation, the η = 0.1 non-degeneracy figures, and
  the Figure-23 shell profile.

### Added

- The W-initialization flip demonstration (#5): `examples/w_init_flip.rs` —
  two runs differing only in W(0), the identity run verified bit-identical
  to the committed transition sweep (all ten recorded steps; crossing at
  step 9, final alignment 0.7569668681021637) — and `examples/w_init_flip.py`,
  an independent stdlib-plus-numpy reimplementation with the Rust crate as
  the pinned reference: `fiedler_alignment` agrees to 9.4e-16 on shared
  deterministic inputs, one gradient step agrees to 2.1e-17 entrywise, and
  the flip reproduces at four numpy seeds (identity crossing at steps 9–15,
  Gaussian peaks 0.100–0.501, all below the criterion).
- The MIT license and a rewritten public-facing README, ahead of the
  repository going public.
- The issue-#5 transition machinery (#5): additive `Params` fields —
  `weight_init` (identity or the committed Gaussian), `weight_rate_ratio`
  (ρ = η_W/η_E), an `optimizer` carrying a hand-rolled decoupled AdamW with
  a linear-warmup-then-cosine schedule, and an `alignment_stop` geometry
  stop reported through `StopReason`/`Run::stop_reason` — plus
  `examples/tier2_transition.rs`, which runs the 64-run W-sweep and the
  24-run §B.3 arm and writes per-run and summary CSVs. The AdamW
  implementation matches a two-step hand reference bit-for-bit; the ρ
  placement, schedule closed form and boundaries, stop mappings, threshold
  validation and same-seed bit-identity are pinned and falsified; a
  falsifying review's 6 important and 5 minor findings are applied.
- A sampled central-differences pin at the production width m = 512 (#5):
  64 seeded entries per block, both blocks and both activations on all four
  D-graphs, measured max deviation 2.147e-9 against a 1e-7 tolerance with a
  1e-4 non-vacuity floor on the sampled |analytic|. Falsified by an operand
  swap (6.06e-4 red) and a sign flip (1.25e-3 red), and a defect on a
  sampled column went red at 1.0e-3 — while a +1e-3 defect confined to an
  unsampled column measured red nowhere in the lib binary, the coverage
  being 64 of the 262 144 W entries, which the pin's doc states as a seeded
  sample.
- Tier 2 `TinyNn`: the §B.2.2 model as Z = E W Eᵀ with a tied 512-wide
  embedding and one trainable weight matrix, in frozen-E and learnable-E
  regimes over the same degree-normalized cross-entropy Tier 1 uses.
  Hand-derived gradients FD-checked per parameter block, a GELU variant,
  per-step CSV instrumentation, cosine and adjacency heatmaps, and
  `examples/tier2_tinynn.rs`. Two measures: `fiedler_alignment` (the
  spectral criterion) and `deepest_shell_separation` over the full graph
  diameter (reported, not thresholded). New `numerics` and `output` modules
  hold the softmax, log-sum-exp, cross-entropy, seeded Gaussian draw and
  matrix-CSV writer that Tiers 1 and 2 share.

### Findings (Tier 2)

- **The spectral geometry is decided by W's initialization, not by its
  trainability (#5).** Sweeping the learnable regime at η = 0.01 over
  W(0) ∈ {identity, the committed N(0, 1/m)} × ρ = η_W/η_E ∈ {0, 1/8, 1/2,
  1} on the four D-graphs at both seeds (`examples/tier2_transition.rs`):
  all 32 identity-initialized runs cross the 0.75 criterion within 7–35
  steps — including ρ = 1, where W trains as freely as in the committed
  regime — with top-d scores still 0.40–0.65 at the crossing (the geometry
  stop truncates those runs there, so their memorization columns describe
  truncated runs). None of the 32 Gaussian-initialized runs crosses within
  20,000 steps at any ρ, ρ = 0 included; all reach top-d 1.0 (steps 2–664)
  and peak at alignment 0.032–0.657. §4.4's weight-tying reading and issue
  #5's trainable-middle-layer suspect both miss at these knobs; the
  initializer — which §B.2.2 does not state — is the variable.
- **§B.3's optimizer does not recover the geometry (#5).** Decoupled AdamW
  (wd 0.01 per §B.3; β₁ = 0.9, β₂ = 0.999, ε = 1e-8 and a 5 % linear warmup
  into cosine decay as documented knobs §B.3 leaves unstated) on the
  committed initializer at the committed budgets {(0.001, 1200),
  (0.01, 200), (0.1, 50)}: all 24 runs end at their step limit below the
  criterion (peaks 0.068–0.508) while memorizing at steps 2–9. Issue #5's
  optimizer suspect is eliminated at these budgets.

> **History.** The first Tier-2 geometry measure was a shell-based cosine
> criterion. An adversarial review found it unsound — it read only shells 2
> and 3, certified a cosine matrix whose neighbours were near-antipodal
> (score 0.200), and its certified embeddings loaded on the **bottom** of the
> spectrum (Fiedler-mass 0.00). Those claims were retracted and the measure
> replaced by the spectral one below, under which the geometric result
> reverses. The shell profile survives as a reported instrument, not a
> criterion.

- **Under a spectral measure calibrated so the paper's own references pass,
  no TinyNN run forms the geometry.** `fiedler_alignment` deflates one
  eigenvector of −L per connected component and averages the squared
  Fiedler-eigenspace projection of the leading principal directions of the
  remainder. Calibration on all four graphs: the Laplacian Fiedler
  eigenvectors score **1.000000** and Tier-1 Node2Vec **0.980–1.000**, while a
  rank-1 Fiedler-sign embedding scores 0.289–0.409, all-rows-identical
  0.000–0.031, 200 Gaussian draws peak at 0.380–0.491, and an embedding built
  from the **bottom** eigenvectors of −L — adjacent vertices near-antipodal,
  the structure the retracted criterion certified — scores **0.000000** on
  every graph. A clean gap, with the 0.75 threshold inside it. Against that scale the learnable TinyNN runs
  peak at **0.031–0.513** and the alignment step is `None` in all 24 runs
  (four graphs × three rates × two seeds) and all 12 example runs. The
  architecture memorizes the edges and does not develop the spectral geometry
  §4.1 describes, which is a non-reproduction of the geometric half of
  Refutation 3b/3c under our knob choices.
- **Refutation 3c's associative half reproduces.** The frozen regime reaches
  its maximum top-d(u) neighbour score at **step 1** on all four graphs and
  both seeds, inside the paper's two steps. Initial scores 0.089–0.221 and
  0.078–0.167; the Pearson correlation over the off-diagonal entries between
  the model's distribution and D⁻¹A at the hit step is 0.9419–0.9756, printed
  per graph and seed by the frozen-run test. Memorization steps in the learnable regime run
  1–51 across the sweep. No timing *ratio* is claimed: only one of the two
  events occurs, so the pin asserts the memorization step, the alignment null,
  and a budget floor of 10× the memorization step against a measured minimum
  of 23.5.
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
  degenerate: numeric rank = n on every sweep run; at η = 0.1 the
  singular-value participation ratio is 12.6–14.7 and the row-norm spread
  ≤ 1.22 — printed per run by the learnable-sweep test. Restated under the spectral measure: on the
  timing-sweep budgets at seed 42, η = 0.1 gives the highest peak alignment
  on three of four graphs and sits 0.019 below η = 0.01 on the cycle; on the
  2000-step example sweep it wins two of four — so it is not the rate that
  fails to form structure. This finding survived the retraction: it rests on the
  trajectory, not on the criterion.
- **Figure 23's "gradual decrease in similarity" does not hold for this
  architecture, and the deviation is larger than first reported.** Over the
  full diameter the distance-2 mean is the global maximum on every graph
  (cycle η = 0.001 at 1200 steps: −0.163, **+0.348**, −0.096, −0.071,
  −0.081, −0.159, −0.125, printed by the learnable-sweep test), not a decay.
  Node2Vec on the same graphs is monotone throughout
  (cycle +0.962 → +0.121 across d1..d7), which is the contrast the paper
  draws.
- **A diverged run could report a geometry** under the retracted measure: at
  η = 10 the irregular graph scored 0.098 at loss 9.99e27, cosines being
  scale-invariant. Both instruments now reject a non-finite embedding — the
  shell profile returns NaN and `fiedler_alignment` returns
  `Error::NonFinite`. A diverged-but-finite run can still score, which the
  scale-invariance the measure needs makes unavoidable.

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
