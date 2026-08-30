//! Public-surface seam for Tier 2: the `tinynn::run` experiment API's
//! associative timing, its determinism, the associative-versus-geometric step
//! ratio, and its cancellation behaviour under a `Runner`.
//!
//! Every run writes to uniquely named paths under the system temp directory,
//! removed when their guards drop. Awaits that a lifecycle regression could
//! hang are bounded by [`RUN_BOUND`], turning such a regression into a named
//! failure instead of a stuck binary.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering, Ordering as AtomicOrdering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rediscovery::graph::Graph;
use rediscovery::node2vec::Outcome;
use rediscovery::subsystems::runner::Runner;
use rediscovery::tinynn::{self, GEOMETRY_MARGIN, GEOMETRY_SHELLS, Outputs, Params, Regime, Run};

/// Upper bound on awaits a cancellation regression could hang; the passing
/// paths resolve well inside it.
const RUN_BOUND: Duration = Duration::from_secs(30);

/// Seed shared by the pins.
const SEED: u64 = 20_260_829;

/// Applied updates the frozen-embedding runs are allowed. Figure 7's claim is
/// about the first two; the rest show the plateau after them.
const ASSOCIATIVE_BUDGET: usize = 10;

/// Steps within which Refutation 3c (md 266) puts the associative fit.
const ASSOCIATIVE_LIMIT: usize = 2;

/// Applied updates the learnable-embedding run of the ratio pin is allowed,
/// above the step at which the criterion is met on the 15-cycle at η = 0.001.
const GEOMETRIC_BUDGET: usize = 1_200;

/// Ratio of geometric to associative steps decision D10 pins, in place of the
/// paper's own inconsistent 100 (md 256) and ~200 (md 302).
const TIMING_RATIO: f64 = 50.0;

/// Steps the determinism pin compares, enough for both parameter blocks to
/// move well away from their draw.
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
            "rediscovery-tier2-{label}-{}-{nanos}-{counter}.csv",
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

/// The two temp paths one run writes.
struct RunPaths {
    history: TempPath,
    cosines: TempPath,
}

impl RunPaths {
    fn new(label: &str) -> Self {
        Self {
            history: TempPath::new(&format!("{label}-history")),
            cosines: TempPath::new(&format!("{label}-cosines")),
        }
    }

    fn outputs(&self) -> Outputs<'_> {
        Outputs {
            history: self.history.path(),
            cosines: self.cosines.path(),
        }
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

/// The comma-separated fields of every line of `path`.
fn read_rows(path: &Path) -> Vec<Vec<String>> {
    let text = std::fs::read_to_string(path).expect("read CSV");
    text.lines()
        .map(|line| line.split(',').map(str::to_string).collect())
        .collect()
}

/// The frozen-embedding parameters of Figure 7: decision D10's η = 0.1 and a
/// budget just past the claim.
fn frozen_params() -> Params {
    Params {
        max_steps: ASSOCIATIVE_BUDGET,
        ..Params::default()
    }
}

/// Runs `params` on `graph` into `paths`, with no cancellation.
fn run_into(graph: &Graph, params: &Params, seed: u64, paths: &RunPaths) -> Run {
    tinynn::run(graph, params, seed, &paths.outputs(), || false).expect("run")
}

/// Figure 7's claim through the public API: with the embedding frozen and
/// η = 0.1, the fraction of each vertex's d(u) neighbours inside its top-d(u)
/// next-token probabilities reaches its maximum within two full-batch steps on
/// every D-graph. The score before the first update is asserted below that
/// maximum, so the pin measures the descent rather than the draw.
#[test]
fn the_frozen_run_memorizes_the_edges_within_two_steps() {
    let params = frozen_params();

    for (name, graph) in d_graphs() {
        let paths = RunPaths::new("frozen");
        let started = Instant::now();
        let run = run_into(&graph, &params, SEED, &paths);
        let initial = run.records()[0].associative_score();

        println!(
            "{name}: {:?}, {} steps, associative step {:?}, peak {:.6}, initial {:.6}",
            started.elapsed(),
            run.steps(),
            run.associative_step(),
            run.peak_associative_score(),
            initial
        );

        assert!(
            initial < 1.0,
            "{name}: the top-d score is already {initial:.15} before the first update, so the \
             step count below would be met by the draw alone"
        );
        let step = run.associative_step().unwrap_or_else(|| {
            panic!(
                "{name}: the top-d score never reached 1 in {ASSOCIATIVE_BUDGET} steps; it \
                 peaked at {:.6}",
                run.peak_associative_score()
            )
        });
        assert!(
            step <= ASSOCIATIVE_LIMIT,
            "{name}: the top-d score first reached 1 at step {step}, above Refutation 3c's \
             {ASSOCIATIVE_LIMIT} (initial score {initial:.6})"
        );

        let cosines = read_rows(paths.cosines.path());
        assert_eq!(
            cosines.len(),
            graph.order() + 1,
            "{name}: the cosine CSV has {} lines, expected a header plus {} rows",
            cosines.len(),
            graph.order()
        );
    }
}

/// A run repeated at the same seed reproduces its trajectory bit for bit —
/// full-batch descent has no sampling and both blocks are drawn from a seeded
/// `ChaCha20` stream. A run at a different seed does not, so the comparison is
/// not satisfied by a degenerate initializer.
#[test]
#[allow(
    clippy::float_cmp,
    reason = "the claim is bit-identity of a deterministic trajectory, not numeric closeness"
)]
fn the_same_seed_reproduces_a_trajectory_bit_for_bit() {
    let params = Params {
        max_steps: TRAJECTORY_STEPS,
        tolerance: 1e-300,
        regime: Regime::LearnableEmbedding,
        ..Params::default()
    };

    for (name, graph) in d_graphs() {
        let first_paths = RunPaths::new("determinism-a");
        let second_paths = RunPaths::new("determinism-b");
        let other_paths = RunPaths::new("determinism-c");

        let first = run_into(&graph, &params, SEED, &first_paths);
        let second = run_into(&graph, &params, SEED, &second_paths);
        let other = run_into(&graph, &params, SEED + 1, &other_paths);

        assert_eq!(
            first.steps(),
            TRAJECTORY_STEPS,
            "{name}: ran {} steps, expected the {TRAJECTORY_STEPS}-step limit",
            first.steps()
        );

        for (block, left, right) in [
            (
                "E",
                first.parameters().embedding(),
                second.parameters().embedding(),
            ),
            (
                "W",
                first.parameters().weight(),
                second.parameters().weight(),
            ),
        ] {
            for (index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
                assert!(
                    left == right,
                    "{name}: entry {index} of {block} after {TRAJECTORY_STEPS} steps is \
                     {left:e} on the first run and {right:e} on the second, at the same seed"
                );
            }
        }
        assert_eq!(
            first.records(),
            second.records(),
            "{name}: the two same-seed runs recorded different histories"
        );

        let deviation = (first.parameters().embedding() - other.parameters().embedding()).amax();
        assert!(
            deviation > 0.0,
            "{name}: seeds {SEED} and {} produced an identical E after {TRAJECTORY_STEPS} \
             steps (max |Δ| = {deviation:e}), so the determinism comparison above would hold \
             for any seed",
            SEED + 1
        );

        assert_eq!(
            read_rows(first_paths.history.path()),
            read_rows(second_paths.history.path()),
            "{name}: the two same-seed runs wrote different history CSVs"
        );
        assert_eq!(
            read_rows(first_paths.cosines.path()),
            read_rows(second_paths.cosines.path()),
            "{name}: the two same-seed runs wrote different cosine CSVs"
        );
    }
}

/// Refutation 3c's timing asymmetry on the 15-cycle, as the ratio decision D10
/// pins rather than the paper's own inconsistent step counts: the frozen run
/// reaches full edge memorization in a small multiple of one step, while the
/// learnable run at η = 0.001 takes at least [`TIMING_RATIO`] times as many
/// steps to meet the geometry criterion.
#[test]
fn the_associative_fit_outruns_the_geometry_by_fifty_times() {
    let graph = Graph::cycle(15).expect("cycle(15)");

    let associative_paths = RunPaths::new("ratio-frozen");
    let associative = run_into(&graph, &frozen_params(), SEED, &associative_paths);
    let associative_step = associative.associative_step().unwrap_or_else(|| {
        panic!(
            "cycle(15): the top-d score never reached 1 in {ASSOCIATIVE_BUDGET} steps; it \
             peaked at {:.6}",
            associative.peak_associative_score()
        )
    });
    assert!(
        associative_step >= 1,
        "cycle(15): the top-d score was already at its maximum before the first update, so the \
         ratio below would measure nothing"
    );

    let geometric_params = Params {
        learning_rate: 0.001,
        max_steps: GEOMETRIC_BUDGET,
        regime: Regime::LearnableEmbedding,
        ..Params::default()
    };
    let geometric_paths = RunPaths::new("ratio-learnable");
    let started = Instant::now();
    let geometric = run_into(&graph, &geometric_params, SEED, &geometric_paths);
    let geometric_step = geometric.geometry_step(GEOMETRY_MARGIN).unwrap_or_else(|| {
        panic!(
            "cycle(15): the geometry margin never reached {GEOMETRY_MARGIN} in \
             {GEOMETRIC_BUDGET} steps; it peaked at {:.6}",
            geometric.peak_geometry_margin()
        )
    });

    // Reported alongside: the same asymmetry inside the one learnable run,
    // where both events happen under identical parameters.
    let same_run_associative = geometric.associative_step().unwrap_or_else(|| {
        panic!(
            "cycle(15): the learnable run never memorized the edges in {GEOMETRIC_BUDGET} \
             steps; its top-d score peaked at {:.6}",
            geometric.peak_associative_score()
        )
    });

    #[allow(
        clippy::cast_precision_loss,
        reason = "step counts here are below 2^53 and exact in f64"
    )]
    let ratio = geometric_step as f64 / associative_step as f64;
    #[allow(
        clippy::cast_precision_loss,
        reason = "step counts here are below 2^53 and exact in f64"
    )]
    let same_run_ratio = geometric_step as f64 / same_run_associative as f64;
    println!(
        "cycle(15): frozen associative step {associative_step}, geometry step {geometric_step} \
         (peak margin {:.6}, {:?}), ratio {ratio:.1}; within the learnable run alone, \
         memorization at step {same_run_associative} gives {same_run_ratio:.1}",
        geometric.peak_geometry_margin(),
        started.elapsed()
    );

    // The denominator is bounded by Refutation 3c's own claim rather than by
    // whatever the frozen run happened to reach, so the ratio below cannot
    // become the geometry step alone through an incidental step of 1.
    assert!(
        associative_step <= 2,
        "cycle(15): the frozen run memorized at step {associative_step}, above Refutation 3c's 2"
    );
    assert!(
        ratio >= TIMING_RATIO,
        "cycle(15): the geometry criterion is met at step {geometric_step} against the \
         associative step {associative_step}, a ratio of {ratio:.1} below the pinned \
         {TIMING_RATIO}"
    );
}

/// A `Runner`-driven run stops mid-sweep when its child token is cancelled and
/// leaves a history CSV of a header plus complete rows. The token is cancelled
/// once the run reports [`POLLS_BEFORE_CANCEL`] polls, so the stop lands
/// strictly inside the run; `max_steps` is set far above what [`RUN_BOUND`]
/// allows, so a run that never polled the token would exhaust the bound
/// instead of returning.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cancelled_run_stops_and_leaves_a_well_formed_csv() {
    let paths = RunPaths::new("cancellation");
    let history = paths.history.path().to_path_buf();
    let cosines = paths.cosines.path().to_path_buf();
    let params = Params {
        max_steps: 200_000,
        regime: Regime::LearnableEmbedding,
        ..Params::default()
    };

    let runner = Runner::new();
    let handle = runner.spawn(move |token| async move {
        tokio::task::spawn_blocking(move || {
            let graph = Graph::cycle(15).expect("cycle(15)");
            let outputs = Outputs {
                history: &history,
                cosines: &cosines,
            };
            let polls = AtomicUsize::new(0);
            tinynn::run(&graph, &params, SEED, &outputs, || {
                let seen = polls.fetch_add(1, AtomicOrdering::SeqCst);
                if seen == POLLS_BEFORE_CANCEL {
                    token.cancel();
                }
                token.is_cancelled()
            })
        })
        .await
    });

    let run = tokio::time::timeout(RUN_BOUND, handle)
        .await
        .expect("the cancelled run did not return within RUN_BOUND")
        .expect("the run task panicked")
        .expect("the blocking task panicked")
        .expect("run");

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

    let rows = read_rows(paths.history.path());
    let expected_fields = 5 + GEOMETRY_SHELLS;
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
