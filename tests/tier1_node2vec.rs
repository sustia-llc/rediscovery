//! Public-surface seam for Tier 1: the `run_tied` experiment API's
//! determinism, its CSV, and its cancellation behaviour under a `Runner`.
//!
//! Every run writes to a uniquely named path under the system temp directory,
//! removed when its guard drops. Awaits that a lifecycle regression could hang
//! are bounded by [`RUN_BOUND`], turning such a regression into a named
//! failure instead of a stuck binary.

#![allow(
    clippy::doc_markdown,
    reason = "the docs carry matrix notation with subscripts — ‖Vᵀe_i‖₂, ‖Ce_i‖₂ — that the lint reads as unbackticked identifiers"
)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering, Ordering as AtomicOrdering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rediscovery::graph::Graph;
use rediscovery::node2vec::{self, Outcome, Params, StepRecord, TiedRun};
use rediscovery::subsystems::runner::Runner;

/// Upper bound on awaits a cancellation regression could hang; the passing
/// paths resolve well inside it.
const RUN_BOUND: Duration = Duration::from_secs(5);

/// Steps the determinism pin compares, enough for the softmax non-linearity
/// to make a seed difference visible.
const TRAJECTORY_STEPS: usize = 25;

/// Applied updates before the cancellation test signals its token.
const POLLS_BEFORE_CANCEL: usize = 5;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A unique path under the system temp directory, removed on drop.
struct TempPath(PathBuf);

impl TempPath {
    fn new(label: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let name = format!(
            "rediscovery-tier1-{label}-{}-{nanos}-{counter}.csv",
            std::process::id()
        );
        Self(std::env::temp_dir().join(name))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// The four graphs of decisions D1–D4.
fn d_graphs() -> Vec<(&'static str, Graph)> {
    vec![
        (
            "path_star(4,4)",
            Graph::path_star(4, 4).expect("path_star(4,4)"),
        ),
        ("grid(4,4)", Graph::grid(4, 4).expect("grid(4,4)")),
        ("cycle(15)", Graph::cycle(15).expect("cycle(15)")),
        ("irregular()", Graph::irregular().expect("irregular()")),
    ]
}

/// Parameters that run exactly `TRAJECTORY_STEPS` updates: the tolerance sits
/// below any relative update the loop produces, so the stop is the step limit.
fn trajectory_params() -> Params {
    Params {
        dimension: 20,
        max_steps: TRAJECTORY_STEPS,
        tolerance: 1e-300,
        ..Params::default()
    }
}

/// Runs the weight-tied system into `path` with no cancellation.
fn run_into(graph: &Graph, params: &Params, seed: u64, path: &Path) -> TiedRun {
    node2vec::run_tied(graph, params, seed, path, || false).expect("run_tied")
}

/// The comma-separated fields of every line of `path`.
fn read_rows(path: &Path) -> Vec<Vec<String>> {
    let text = std::fs::read_to_string(path).expect("read history CSV");
    text.lines()
        .map(|line| line.split(',').map(str::to_string).collect())
        .collect()
}

/// A run repeated at the same seed reproduces its trajectory bit for bit —
/// full-batch ascent has no sampling and the initializer draws from a seeded
/// `ChaCha20` stream. A run at a different seed does not, so the comparison
/// is not satisfied by a degenerate initializer.
#[test]
#[allow(
    clippy::float_cmp,
    reason = "the claim is bit-identity of a deterministic trajectory, not numeric closeness"
)]
fn the_same_seed_reproduces_a_trajectory_bit_for_bit() {
    let params = trajectory_params();

    for (name, graph) in d_graphs() {
        let first_path = TempPath::new("determinism-a");
        let second_path = TempPath::new("determinism-b");
        let other_path = TempPath::new("determinism-c");

        let first = run_into(&graph, &params, 20_260_829, first_path.path());
        let second = run_into(&graph, &params, 20_260_829, second_path.path());
        let other = run_into(&graph, &params, 20_260_830, other_path.path());

        assert_eq!(
            first.steps(),
            TRAJECTORY_STEPS,
            "{name}: ran {} steps, expected the {TRAJECTORY_STEPS}-step limit",
            first.steps()
        );

        for (index, (left, right)) in first
            .embedding()
            .iter()
            .zip(second.embedding().iter())
            .enumerate()
        {
            assert!(
                left == right,
                "{name}: entry {index} of V after {TRAJECTORY_STEPS} steps is {left:e} on the \
                 first run and {right:e} on the second, at the same seed"
            );
        }
        assert_eq!(
            first.records(),
            second.records(),
            "{name}: the two same-seed runs recorded different histories"
        );

        let deviation = (first.embedding() - other.embedding()).amax();
        assert!(
            deviation > 0.0,
            "{name}: seeds 20260829 and 20260830 produced an identical V after \
             {TRAJECTORY_STEPS} steps (max |Δ| = {deviation:e}), so the determinism \
             comparison above would hold for any seed"
        );

        assert_eq!(
            read_rows(first_path.path()),
            read_rows(second_path.path()),
            "{name}: the two same-seed runs wrote different CSVs"
        );
    }
}

/// The history CSV round-trips: its header names one column per recorded
/// field, and re-parsing every row reproduces the in-memory records exactly.
#[test]
#[allow(
    clippy::float_cmp,
    reason = "Rust's shortest float formatting round-trips exactly; the pin is that identity"
)]
fn the_history_csv_round_trips_the_in_memory_records() {
    let params = trajectory_params();

    for (name, graph) in d_graphs() {
        let temp = TempPath::new("round-trip");
        let run = run_into(&graph, &params, 4_242, temp.path());
        let rows = read_rows(temp.path());
        let order = graph.order();
        let expected_fields = 4 + 2 * order;

        assert_eq!(
            rows.len(),
            run.records().len() + 1,
            "{name}: the CSV has {} lines, expected a header plus {} records",
            rows.len(),
            run.records().len()
        );

        let mut header = vec![
            "step".to_string(),
            "objective".to_string(),
            "relative_update".to_string(),
            "observation8_residual".to_string(),
        ];
        header.extend((0..order).map(|i| format!("projection_{i}")));
        header.extend((0..order).map(|i| format!("coefficient_norm_{i}")));
        assert_eq!(rows[0], header, "{name}: unexpected CSV header");

        for (record, row) in run.records().iter().zip(&rows[1..]) {
            assert_eq!(
                row.len(),
                expected_fields,
                "{name}: row for step {} has {} fields, expected {expected_fields}",
                record.step(),
                row.len()
            );
            assert_eq!(
                row[0]
                    .parse::<usize>()
                    .unwrap_or_else(|error| panic!("{name}: step column {:?}: {error}", row[0])),
                record.step(),
                "{name}: the step column disagrees with the record"
            );
            assert_field(
                name,
                record,
                "objective",
                parse(row, 1, name),
                record.objective(),
            );
            assert_field(
                name,
                record,
                "relative_update",
                parse(row, 2, name),
                record.relative_update(),
            );
            assert_field(
                name,
                record,
                "observation8_residual",
                parse(row, 3, name),
                record.observation8_residual(),
            );
            for (i, &value) in record.projections().iter().enumerate() {
                assert_field(name, record, "projection", parse(row, 4 + i, name), value);
            }
            for (i, &value) in record.coefficient_norms().iter().enumerate() {
                assert_field(
                    name,
                    record,
                    "coefficient_norm",
                    parse(row, 4 + order + i, name),
                    value,
                );
            }
        }
    }
}

/// Field `index` of `row`, parsed as `f64`.
fn parse(row: &[String], index: usize, name: &str) -> f64 {
    row[index]
        .parse::<f64>()
        .unwrap_or_else(|error| panic!("{name}: field {index} is {:?}: {error}", row[index]))
}

/// Asserts a parsed CSV field equals the in-memory value bit for bit.
#[allow(
    clippy::float_cmp,
    reason = "Rust's shortest float formatting round-trips exactly; the pin is that identity"
)]
fn assert_field(name: &str, record: &StepRecord, column: &str, parsed: f64, stored: f64) {
    assert!(
        parsed == stored,
        "{name}: {column} at step {} re-parses as {parsed:e}, stored as {stored:e}",
        record.step()
    );
}

/// The Fig-9 projection signature through the public experiment API on the
/// 15-cycle: ‖Vᵀe_i‖₂ stays large on the Fiedler-like pair and falls away for
/// every later eigenvector, while ‖Ce_i‖₂ falls to zero on that pair.
#[test]
fn the_cycle_signature_holds_through_the_public_api() {
    let graph = Graph::cycle(15).expect("cycle(15)");
    let params = Params {
        max_steps: 20_000,
        ..Params::default()
    };
    let temp = TempPath::new("signature");
    let started = Instant::now();
    let run = run_into(&graph, &params, 20_260_829, temp.path());
    let record = run.last().expect("a run records its initial state");
    let fiedler =
        node2vec::fiedler_like_range(run.spectrum(), node2vec::fiedler_spread(run.spectrum()));

    println!(
        "the_cycle_signature_holds_through_the_public_api: {:?}, outcome {:?}, {} steps, \
         Fiedler-like set {fiedler:?}",
        started.elapsed(),
        run.outcome(),
        run.steps()
    );

    assert_eq!(fiedler, 1..3, "cycle(15): Fiedler-like set is {fiedler:?}");
    for i in fiedler.clone() {
        assert!(
            record.projections()[i] > 1.0,
            "cycle(15): ‖Vᵀe_{i}‖₂ = {:.6} on the Fiedler-like set, threshold 1.0",
            record.projections()[i]
        );
        assert!(
            record.coefficient_norms()[i] < 1e-3,
            "cycle(15): ‖Ce_{i}‖₂ = {:.6e} on the Fiedler-like set, tolerance 1e-3",
            record.coefficient_norms()[i]
        );
    }
    for i in fiedler.end..run.spectrum().order() {
        assert!(
            record.projections()[i] < 1.0,
            "cycle(15): ‖Vᵀe_{i}‖₂ = {:.6} beyond the Fiedler-like set, threshold 1.0",
            record.projections()[i]
        );
    }
}

/// A `Runner`-driven run stops mid-sweep when its child token is cancelled
/// and leaves a CSV of a header plus complete rows whose last row describes
/// the state the run returns. The token is cancelled once the run reports
/// [`POLLS_BEFORE_CANCEL`] polls, so the stop lands strictly inside the run;
/// `max_steps` is set far above what [`RUN_BOUND`] allows, so a run that
/// never polled the token would exhaust the bound instead of returning.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cancelled_run_stops_and_leaves_a_well_formed_csv() {
    let temp = TempPath::new("cancellation");
    let path = temp.path().to_path_buf();
    let params = Params {
        dimension: 16,
        max_steps: 200_000,
        ..Params::default()
    };

    let runner = Runner::new();
    let handle = runner.spawn(move |token| async move {
        let graph = Graph::cycle(15).expect("cycle(15)");
        let polls = AtomicUsize::new(0);
        node2vec::run_tied(&graph, &params, 20_260_829, &path, || {
            let seen = polls.fetch_add(1, AtomicOrdering::SeqCst);
            if seen == POLLS_BEFORE_CANCEL {
                token.cancel();
            }
            token.is_cancelled()
        })
    });

    let run = tokio::time::timeout(RUN_BOUND, handle)
        .await
        .expect("the cancelled run did not return within RUN_BOUND")
        .expect("the run task panicked")
        .expect("run_tied");

    assert_eq!(
        run.outcome(),
        Outcome::Stopped,
        "outcome is {:?} after {} steps, expected Stopped",
        run.outcome(),
        run.steps()
    );
    assert_eq!(
        run.steps(),
        POLLS_BEFORE_CANCEL,
        "the run applied {} updates, expected {POLLS_BEFORE_CANCEL} before the cancel",
        run.steps()
    );

    // The returned state is the one the last record describes: the stop is
    // taken before the pending update is applied.
    let graph = Graph::cycle(15).expect("cycle(15)");
    let system = node2vec::Node2Vec::new(&graph).expect("system");
    let objective = system.objective(run.embedding()).expect("objective");
    let recorded = run
        .last()
        .expect("a run records its initial state")
        .objective();
    assert!(
        (objective - recorded).abs() < 1e-12,
        "the returned embedding scores {objective:.12} but the last record says \
         {recorded:.12}; a stopped run must return the state it recorded"
    );

    let rows = read_rows(temp.path());
    let expected_fields = 4 + 2 * 15;
    assert_eq!(
        rows.len(),
        run.records().len() + 1,
        "the partial CSV has {} lines, expected a header plus the {} recorded steps",
        rows.len(),
        run.records().len()
    );
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(
            row.len(),
            expected_fields,
            "line {index} of the partial CSV has {} fields, expected {expected_fields}",
            row.len()
        );
    }

    tokio::time::timeout(RUN_BOUND, runner.shutdown())
        .await
        .expect("shutdown() did not complete within RUN_BOUND");
}
