# rediscovery

CPU-only numerics POC replicating results from arXiv 2510.26745v2 ("Deep
sequence models tend to memorize geometrically; it is unclear why" —
Node2Vec spectral-bias dynamics). Design analysis:
[`docs/2510.26745v2-poc-analysis.md`](docs/2510.26745v2-poc-analysis.md).

![Associative vs. geometric memory across tiny graphs](docs/assets/fig1-associative-vs-geometric.jpeg)

*Figure 1 of [arXiv 2510.26745v2](https://arxiv.org/abs/2510.26745)
(Noroozizadeh, Nagarajan, Rosenfeld, Kumar): the same graphs memorized as an
associative lookup over arbitrary embeddings (left) vs. the geometric
embeddings a Transformer learns (middle) vs. the cleaner geometry of a
Node2Vec model (right). The right two columns are what this POC replicates.*

## Status

All three POC tiers are implemented: `graph`/`spectral` (Tier 0),
`node2vec` (Tier 1, Appendix F dynamics), and `tinynn` (Tier 2, the §B.2.2
associative-vs-geometric competition), with `numerics`/`output` holding what
the tiers share. Measured findings — including where the implementation
reproduces the paper's figures but not their captions — are recorded under
Findings in [`CHANGELOG.md`](CHANGELOG.md).

### Tier 0

Tier 0 is implemented: `graph` builds the paper's tiny topologies (path-star,
grid, cycle, irregular, tree-star, complete) over dense nalgebra adjacency,
and `spectral` provides the row-normalized transition matrix D⁻¹A, the
random-walk Laplacian L = (I − D⁻¹A) + (I − D⁻¹A)ᵀ, and `Spectrum` — the
eigendecomposition of −L with descending order, a deterministic sign
convention, and degenerate-group detection. Closed-form pins (15-cycle,
complete graphs) and property pins run in-module; public-seam integration
tests live in `tests/`. Tiers 1–2 (Node2Vec dynamics, TinyNN competition)
are next; decision labels D1–D10 are recorded in
[`docs/2510.26745v2-poc-analysis.md`](docs/2510.26745v2-poc-analysis.md) §8.

`subsystems::runner` is the async task lifecycle core: a `Runner` owns a
`TaskTracker` + `CancellationToken`, exposes `cancellation_token()`, and
spawns cancellation-aware jobs that each receive a `child_token()`.
`shutdown()` implements the standard cancel → close → drain sequence. It
takes no `SubsystemHandle` and does no signal handling internally — that is
a binary-level concern, wired up in `src/main.rs` as a bare ctrl-c wait
around `Runner::shutdown()`.

## Running tests

```sh
cargo test
```

`Runner` lifecycle tests (drain on shutdown, cancellation reaching a
spawned job's child token, shutdown with no jobs) live in
`src/subsystems/runner.rs`.
