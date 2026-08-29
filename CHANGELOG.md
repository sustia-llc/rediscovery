# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Library-first crate scaffold mirroring `surrealdb-live-message` conventions:
  `error` (thiserror enum + `Result` alias), `logger` (tracing), `settings`
  (config crate + `config/*.toml`; experiment output dir + global RNG seed),
  `graph`/`spectral` Tier-0 module stubs, and `subsystems::runner::Runner` —
  the async lifecycle core owning a `TaskTracker` + `CancellationToken`, with
  `spawn` (jobs receive a `child_token()`), `cancellation_token()`, and
  `shutdown()` (cancel → close → drain). Lifecycle tests each verified to
  fail when the behavior they pin is reverted. Thin ctrl-c daemon in
  `src/main.rs`.
- `docs/2510.26745v2-poc-analysis.md` — validation report for the markdown
  conversion of arXiv 2510.26745v2, claims inventory, reproducibility
  assessment, tiered POC design space, verification pins, risks, and the
  recorded decisions (nalgebra; structure mirroring `surrealdb-live-message`).
