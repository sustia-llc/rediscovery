# rediscovery

CPU-only numerics POC replicating results from arXiv 2510.26745v2 ("Deep
sequence models tend to memorize geometrically; it is unclear why" —
Node2Vec spectral-bias dynamics). Design analysis:
[`docs/2510.26745v2-poc-analysis.md`](docs/2510.26745v2-poc-analysis.md).

## Status

Library-first scaffold. `graph` and `spectral` are module stubs for Tier 0
(graph construction, Laplacian/eigendecomposition) — no numerics implemented
yet.

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
