//! Issue #8: the graph classes Observation 8 is measured over, circulant and
//! not.
//!
//! The Node2Vec arm runs weight-tied gradient ascent at the Fig-9 knobs over
//! fifteen graphs and two seeds, recording each run's outcome, its final
//! Observation-8 residual, and — on the Fiedler-like set
//! `node2vec::fiedler_like_range` gives at `node2vec::fiedler_spread` — the
//! final ‖Ce_i‖₂ with its Rayleigh and rotation components per member. The
//! retention arm runs the identity-initialized learnable TinyNN regime over the
//! twelve graphs not already in the committed record, recording each run's
//! crossing of the 0.75 Fiedler criterion, its peak, and where it ends.
//!
//! The sweep set spans three construction classes: circulants — the cycles,
//! `C_12(1,2)`, `C_12(1,3)`, `C_15(1,4)`, `C_20(1,5)` and `complete(7)`, all
//! built as C_n(offsets) here; the Petersen graph, vertex-transitive and not
//! circulant; the Frucht graph, whose automorphism group is trivial; and four
//! graphs that are neither, three of them the D-graphs the committed record
//! already carries. The two symmetry classifications are literature-known
//! properties of those graphs, not measurements this example takes.
//!
//! Each run writes its per-step history under the `tier1_circulance_` stem and
//! each arm writes one summary CSV, with the Fiedler-set members and the
//! per-graph degenerate-group structure in two further CSVs, into
//! `SETTINGS.output.dir`. The configurations of an arm go across a rayon pool of
//! `SWEEP_THREADS` workers, each run internally sequential; a configuration that
//! panics is re-raised after the surviving rows are written. The runs are
//! synchronous numerics, so the `Runner` job hands them to `spawn_blocking` and
//! each polls its child token between steps; ctrl-c drains through
//! `Runner::shutdown` and leaves complete CSV rows. The output directory comes
//! from `SETTINGS` here and reaches the library as an explicit argument
//! (decision D8).

#![allow(
    clippy::doc_markdown,
    reason = "the docs carry matrix notation with subscripts — ‖Ce_i‖₂, C_n(1,k) — that the lint reads as unbackticked identifiers"
)]

use std::any::Any;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use rayon::prelude::*;
use tokio_util::sync::CancellationToken;

use rediscovery::error::Error;
use rediscovery::graph::Graph;
use rediscovery::logger;
use rediscovery::node2vec::{self, DEGENERACY_TOLERANCE, fiedler_like_range, fiedler_spread};
use rediscovery::settings::SETTINGS;
use rediscovery::spectral::Spectrum;
use rediscovery::subsystems::runner::Runner;
use rediscovery::tinynn::{self, FIEDLER_ALIGNMENT, Outputs, Regime, StepRecord, WeightInit};

/// Seeds both arms run: `config/default.toml`'s `rng.seed` and the second seed
/// the committed Tier-2 sweeps carry.
const SEEDS: [u64; 2] = [42, 20_260_829];

/// Worker threads each sweep pool runs, bounding how many runs of one arm are
/// in flight at once.
const SWEEP_THREADS: usize = 6;

/// Applied updates the retention arm allows before the step limit binds.
const RETENTION_STEPS: usize = 2_000;

/// Descent rate the retention arm runs at.
const RETENTION_RATE: f64 = 0.01;

/// Relative-update tolerance at or below which a retention run stops as
/// converged.
const RETENTION_TOLERANCE: f64 = 1e-10;

/// Chord offsets of the Frucht graph's LCF notation `[-5,-2,-4,2,5,-2,2,5,-2,
/// -5,4,2]`, each entry naming the chord from vertex `i` to `i + entry` modulo
/// twelve. The list names every chord from both of its ends.
const FRUCHT_LCF: [i32; 12] = [-5, -2, -4, 2, 5, -2, 2, 5, -2, -5, 4, 2];

/// Vertices the Frucht graph carries.
const FRUCHT_ORDER: usize = FRUCHT_LCF.len();

/// Built here as C_n(offsets): every cycle, `Graph::complete`, and
/// [`circulant`]'s output.
const CIRCULANT: &str = "circulant";

/// Vertex-transitive and not circulant — the Petersen graph's literature-known
/// classification.
const VERTEX_TRANSITIVE: &str = "vertex_transitive";

/// Trivial automorphism group — the Frucht graph's literature-known defining
/// property.
const ASYMMETRIC: &str = "asymmetric";

/// Neither circulant nor vertex-transitive.
const NEITHER: &str = "neither";

/// One graph of the sweep: the file-name stem it writes under, its symmetry
/// class, whether the retention arm runs it, and the graph.
struct Subject {
    stem: &'static str,
    class: &'static str,
    retention: bool,
    graph: Graph,
}

/// One member of a run's Fiedler-like set, measured at the final step.
struct FiedlerMember {
    index: usize,
    norm: f64,
    rayleigh: f64,
    rotation: f64,
    share: f64,
}

/// One Node2Vec run's row of the arm's summary CSV, with the Fiedler-set
/// members that expand into the companion CSV.
struct TiedSummary {
    stem: &'static str,
    class: &'static str,
    order: usize,
    edges: usize,
    seed: u64,
    outcome: node2vec::Outcome,
    steps: usize,
    residual: f64,
    fiedler: Range<usize>,
    members: Vec<FiedlerMember>,
    elapsed: Duration,
}

/// What a completed retention run measured.
struct Measured {
    stop: tinynn::StopReason,
    steps: usize,
    crossing: Option<usize>,
    peak: f64,
    peak_step: Option<usize>,
    final_alignment: f64,
}

/// One retention run's row of that arm's summary CSV. `measured` is `None` for
/// a graph the TinyNN metrics do not accept.
struct RetentionSummary {
    stem: &'static str,
    class: &'static str,
    order: usize,
    seed: u64,
    measured: Option<Measured>,
    elapsed: Duration,
}

/// One configuration of either arm: a graph and a seed.
struct Config<'a> {
    subject: &'a Subject,
    seed: u64,
}

/// What ended a sweep short of every configuration completing.
enum Failure {
    Panicked(Box<dyn Any + Send>),
    Errored(Error),
}

/// Builds the circulant graph C_n(offsets): vertex `i` adjacent to
/// `(i + k) mod n` for every offset `k`, each edge kept once.
///
/// # Errors
///
/// Rejects an offset outside `1..n`, and propagates [`Graph::from_edges`]'s
/// errors.
fn circulant(n: usize, offsets: &[usize]) -> Result<Graph> {
    for &offset in offsets {
        ensure!(
            offset > 0 && offset < n,
            "circulant({n}): offset {offset} is outside 1..{n}"
        );
    }

    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(n * offsets.len());
    for &offset in offsets {
        for vertex in 0..n {
            let other = (vertex + offset) % n;
            edges.push((vertex.min(other), vertex.max(other)));
        }
    }
    edges.sort_unstable();
    edges.dedup();

    Ok(Graph::from_edges(n, &edges)?)
}

/// Builds the `n`-vertex path, vertex `i` adjacent to `i + 1`.
///
/// # Errors
///
/// Propagates [`Graph::from_edges`]'s errors.
fn path(n: usize) -> Result<Graph> {
    let edges: Vec<(usize, usize)> = (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect();
    Ok(Graph::from_edges(n, &edges)?)
}

/// Builds the Petersen graph: an outer 5-cycle, five spokes, and an inner
/// pentagram, checked for 3-regularity over 15 edges.
///
/// # Errors
///
/// Propagates [`Graph::from_edges`]'s errors and rejects a built graph that is
/// not 3-regular on 15 edges.
fn petersen() -> Result<Graph> {
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(15);
    for i in 0..5 {
        edges.push((i, (i + 1) % 5));
        edges.push((i, i + 5));
        edges.push((5 + i, 5 + ((i + 2) % 5)));
    }

    let graph = Graph::from_edges(10, &edges)?;
    check_shape("petersen", &graph, 3.0, 15)?;
    Ok(graph)
}

/// Builds the Frucht graph: the 12-vertex Hamiltonian cycle plus the
/// [`FRUCHT_LCF`] chords, checked for 3-regularity over 18 edges.
///
/// # Errors
///
/// Propagates [`Graph::from_edges`]'s errors, an offset arithmetic conversion
/// failure, and rejects a built graph that is not 3-regular on 18 edges.
fn frucht() -> Result<Graph> {
    let order = i32::try_from(FRUCHT_ORDER)?;
    let mut edges: Vec<(usize, usize)> = (0..FRUCHT_ORDER)
        .map(|i| (i.min((i + 1) % FRUCHT_ORDER), i.max((i + 1) % FRUCHT_ORDER)))
        .collect();
    for (index, &chord) in FRUCHT_LCF.iter().enumerate() {
        let from = i32::try_from(index)?;
        let to = usize::try_from((from + chord).rem_euclid(order))?;
        edges.push((index.min(to), index.max(to)));
    }
    edges.sort_unstable();
    edges.dedup();

    let graph = Graph::from_edges(FRUCHT_ORDER, &edges)?;
    check_shape("frucht", &graph, 3.0, 18)?;
    Ok(graph)
}

/// Builds the `n`-vertex cycle with the extra edge (`from`, `to`).
///
/// # Errors
///
/// Propagates [`Graph::from_edges`]'s errors.
fn cycle_with_chord(n: usize, from: usize, to: usize) -> Result<Graph> {
    let mut edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    edges.push((from, to));
    Ok(Graph::from_edges(n, &edges)?)
}

/// Rejects `graph` unless every vertex carries `degree` and the edge count is
/// `edges`.
///
/// # Errors
///
/// Reports the first vertex whose degree differs, or the edge count that does.
fn check_shape(label: &str, graph: &Graph, degree: f64, edges: usize) -> Result<()> {
    ensure!(
        graph.edge_count() == edges,
        "{label}: {} edges, expected {edges}",
        graph.edge_count()
    );
    for vertex in 0..graph.order() {
        let observed = graph.degrees()[vertex];
        ensure!(
            (observed - degree).abs() < 1e-12,
            "{label}: vertex {vertex} has degree {observed}, expected {degree}"
        );
    }
    Ok(())
}

/// Rejects a [`circulant`] whose single-offset output differs from the cycle
/// constructor it must reproduce: C_15(1) against `Graph::cycle(15)`, compared
/// entry by entry over the adjacency matrices.
///
/// # Errors
///
/// Propagates the two constructors' errors and reports a mismatch in order or
/// in any adjacency entry.
fn check_builder() -> Result<()> {
    let built = circulant(15, &[1])?;
    let cycle = Graph::cycle(15)?;
    ensure!(
        built.order() == cycle.order(),
        "circulant(15, [1]) has order {}, cycle(15) has {}",
        built.order(),
        cycle.order()
    );
    ensure!(
        built.adjacency() == cycle.adjacency(),
        "circulant(15, [1]) and cycle(15) differ in adjacency"
    );
    tracing::info!("circulant(15, [1]) matches cycle(15) entry for entry");
    Ok(())
}

/// The sweep set: the eight circulants, the Petersen and Frucht graphs, the
/// two near-circulant baselines, and the three D-graphs the committed record
/// already carries, which the retention arm skips.
///
/// # Errors
///
/// Propagates every constructor's errors.
fn subjects() -> Result<Vec<Subject>> {
    let entries: Vec<(&'static str, &'static str, bool, Graph)> = vec![
        ("cycle8", CIRCULANT, true, Graph::cycle(8)?),
        ("cycle12", CIRCULANT, true, Graph::cycle(12)?),
        ("cycle20", CIRCULANT, true, Graph::cycle(20)?),
        ("c12_1_2", CIRCULANT, true, circulant(12, &[1, 2])?),
        ("c12_1_3", CIRCULANT, true, circulant(12, &[1, 3])?),
        ("c15_1_4", CIRCULANT, true, circulant(15, &[1, 4])?),
        ("c20_1_5", CIRCULANT, true, circulant(20, &[1, 5])?),
        ("complete7", CIRCULANT, true, Graph::complete(7)?),
        ("petersen", VERTEX_TRANSITIVE, true, petersen()?),
        ("frucht", ASYMMETRIC, true, frucht()?),
        (
            "cycle15_chord03",
            NEITHER,
            true,
            cycle_with_chord(15, 0, 3)?,
        ),
        ("path15", NEITHER, true, path(15)?),
        ("cycle15", CIRCULANT, false, Graph::cycle(15)?),
        ("grid", NEITHER, false, Graph::grid(4, 4)?),
        ("irregular", NEITHER, false, Graph::irregular()?),
    ];

    Ok(entries
        .into_iter()
        .map(|(stem, class, retention, graph)| Subject {
            stem,
            class,
            retention,
            graph,
        })
        .collect())
}

/// The CSV word for a Node2Vec outcome.
fn outcome_label(outcome: node2vec::Outcome) -> &'static str {
    match outcome {
        node2vec::Outcome::Converged => "converged",
        node2vec::Outcome::StepLimit => "step_limit",
        node2vec::Outcome::Stopped => "stopped",
    }
}

/// The CSV word for a TinyNN stop reason.
fn stop_label(stop: tinynn::StopReason) -> &'static str {
    match stop {
        tinynn::StopReason::Converged => "converged",
        tinynn::StopReason::StepLimit => "step_limit",
        tinynn::StopReason::Aligned => "aligned",
        tinynn::StopReason::Stopped => "stopped",
    }
}

/// A step that may not have occurred, as a CSV field: its number, or empty.
fn step_field(step: Option<usize>) -> String {
    step.map_or_else(String::new, |step| step.to_string())
}

/// The largest of `values`, or NaN over an empty set.
fn maximum(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(f64::NAN, f64::max)
}

impl TiedSummary {
    /// This run's row of the Node2Vec summary CSV.
    fn row(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.stem,
            self.class,
            self.order,
            self.edges,
            self.seed,
            outcome_label(self.outcome),
            self.steps,
            self.residual,
            self.fiedler.start,
            self.fiedler.end,
            maximum(self.members.iter().map(|member| member.norm)),
            maximum(self.members.iter().map(|member| member.share)),
            self.elapsed.as_secs_f64()
        )
    }

    /// This run's rows of the Fiedler-set companion CSV, one per member.
    fn member_rows(&self) -> Vec<String> {
        self.members
            .iter()
            .map(|member| {
                format!(
                    "{},{},{},{},{},{},{},{}",
                    self.stem,
                    self.class,
                    self.seed,
                    member.index,
                    member.norm,
                    member.rayleigh,
                    member.rotation,
                    member.share
                )
            })
            .collect()
    }
}

impl RetentionSummary {
    /// This run's row of the retention summary CSV, its measurement fields
    /// empty when the TinyNN metrics did not accept the graph.
    fn row(&self) -> String {
        let head = format!("{},{},{},{},", self.stem, self.class, self.order, self.seed);
        // Seven empty fields: the measurement columns between `status` and
        // `elapsed_seconds`.
        let tail = self.measured.as_ref().map_or_else(
            || ",".repeat(6),
            |measured| {
                format!(
                    "{},{},{},{},{},{},{}",
                    stop_label(measured.stop),
                    measured.steps,
                    step_field(measured.crossing),
                    measured.peak,
                    step_field(measured.peak_step),
                    measured.final_alignment,
                    measured.final_alignment >= FIEDLER_ALIGNMENT
                )
            },
        );
        let status = if self.measured.is_some() {
            "ok"
        } else {
            "unsupported"
        };
        format!("{head}{status},{tail},{}", self.elapsed.as_secs_f64())
    }
}

/// Writes `header` and then `rows`, one line each.
fn write_csv(path: &Path, header: &str, rows: &[String]) -> Result<(), Error> {
    let mut sink = BufWriter::new(File::create(path)?);
    writeln!(sink, "{header}")?;
    for row in rows {
        writeln!(sink, "{row}")?;
    }
    sink.flush()?;
    Ok(())
}

/// Records each subject's spectral structure once: the degenerate groups of −L
/// at [`DEGENERACY_TOLERANCE`] and the Fiedler-like range the Node2Vec arm
/// measures over, to the log and to `tier1_circulance_graphs.csv`.
///
/// # Errors
///
/// Propagates [`Spectrum::of_negative_laplacian`]'s errors and [`Error::Io`]
/// from writing the CSV.
fn describe(directory: &Path, subjects: &[Subject]) -> Result<(), Error> {
    let mut rows = Vec::with_capacity(subjects.len());
    for subject in subjects {
        let spectrum = Spectrum::of_negative_laplacian(&subject.graph)?;
        let fiedler = fiedler_like_range(&spectrum, fiedler_spread(&spectrum));
        let groups: Vec<String> = spectrum
            .degenerate_groups(DEGENERACY_TOLERANCE)
            .iter()
            .map(|group| format!("{}..{}", group.start, group.end))
            .collect();
        let groups = groups.join("|");
        tracing::info!(
            graph = subject.stem,
            class = subject.class,
            order = subject.graph.order(),
            edges = subject.graph.edge_count(),
            fiedler_start = fiedler.start,
            fiedler_end = fiedler.end,
            degenerate_groups = groups,
            "measured a subject's degenerate-group structure"
        );
        rows.push(format!(
            "{},{},{},{},{},{},{groups}",
            subject.stem,
            subject.class,
            subject.graph.order(),
            subject.graph.edge_count(),
            fiedler.start,
            fiedler.end
        ));
    }

    write_csv(
        &directory.join("tier1_circulance_graphs.csv"),
        "graph,class,order,edges,fiedler_start,fiedler_end,degenerate_groups",
        &rows,
    )
}

/// Runs weight-tied ascent on `subject` at `seed`, writing the per-step history
/// under `directory` and returning the run's summary row.
///
/// # Errors
///
/// Propagates [`node2vec::run_tied`]'s errors.
fn measure_tied(
    directory: &Path,
    subject: &Subject,
    seed: u64,
    params: &node2vec::Params,
    token: &CancellationToken,
) -> Result<TiedSummary, Error> {
    let history = directory.join(format!("tier1_circulance_{}_seed{seed}.csv", subject.stem));

    let started = Instant::now();
    let run = node2vec::run_tied(&subject.graph, params, seed, &history, || {
        token.is_cancelled()
    })?;
    let elapsed = started.elapsed();
    let last = run
        .last()
        .expect("invariant: `run_tied` records its initial state before any stop");

    let spectrum = run.spectrum();
    let fiedler = fiedler_like_range(spectrum, fiedler_spread(spectrum));
    let members: Vec<FiedlerMember> = fiedler
        .clone()
        .map(|index| {
            let norm = last.coefficient_norms()[index];
            let rotation = last.rotations()[index];
            FiedlerMember {
                index,
                norm,
                rayleigh: last.rayleigh()[index],
                rotation,
                share: if norm > 0.0 {
                    rotation / norm
                } else {
                    f64::NAN
                },
            }
        })
        .collect();

    let summary = TiedSummary {
        stem: subject.stem,
        class: subject.class,
        order: subject.graph.order(),
        edges: subject.graph.edge_count(),
        seed,
        outcome: run.outcome(),
        steps: run.steps(),
        residual: last.observation8_residual(),
        fiedler,
        members,
        elapsed,
    };
    tracing::info!(
        graph = summary.stem,
        class = summary.class,
        seed,
        steps = summary.steps,
        outcome = ?summary.outcome,
        observation8_residual = summary.residual,
        fiedler_start = summary.fiedler.start,
        fiedler_end = summary.fiedler.end,
        max_coefficient_norm = maximum(summary.members.iter().map(|member| member.norm)),
        max_rotation_share = maximum(summary.members.iter().map(|member| member.share)),
        elapsed_seconds = elapsed.as_secs_f64(),
        "completed a Node2Vec run"
    );
    Ok(summary)
}

/// Runs the identity-initialized learnable TinyNN regime on `subject` at
/// `seed`, writing the per-step history and the final cosine matrix under
/// `directory` and returning the run's summary row. A graph whose vertex pairs
/// span fewer distances than the TinyNN metrics need is recorded without
/// measurements.
///
/// # Errors
///
/// Propagates [`tinynn::run`]'s errors other than
/// [`Error::InsufficientDistanceShells`].
fn measure_retention(
    directory: &Path,
    subject: &Subject,
    seed: u64,
    params: &tinynn::Params,
    token: &CancellationToken,
) -> Result<RetentionSummary, Error> {
    let stem = format!("tier1_circulance_retention_{}_seed{seed}", subject.stem);
    let history = directory.join(format!("{stem}.csv"));
    let cosines = directory.join(format!("{stem}_cosines.csv"));
    let outputs = Outputs {
        history: &history,
        cosines: &cosines,
    };

    let started = Instant::now();
    let outcome = tinynn::run(&subject.graph, params, seed, &outputs, || {
        token.is_cancelled()
    });
    let elapsed = started.elapsed();

    let run = match outcome {
        Ok(run) => run,
        Err(Error::InsufficientDistanceShells { available }) => {
            tracing::warn!(
                graph = subject.stem,
                seed,
                available,
                "the TinyNN metrics need more distinct distances than this graph has; recording \
                 the configuration without measurements"
            );
            return Ok(RetentionSummary {
                stem: subject.stem,
                class: subject.class,
                order: subject.graph.order(),
                seed,
                measured: None,
                elapsed,
            });
        }
        Err(error) => return Err(error),
    };

    let last = run
        .last()
        .expect("invariant: `tinynn::run` records its initial state before any stop");
    let measured = Measured {
        stop: run.stop_reason(),
        steps: run.steps(),
        crossing: run.alignment_step(FIEDLER_ALIGNMENT),
        peak: run.peak_alignment(),
        peak_step: run
            .records()
            .iter()
            .max_by(|left, right| {
                left.fiedler_alignment()
                    .total_cmp(&right.fiedler_alignment())
            })
            .map(StepRecord::step),
        final_alignment: last.fiedler_alignment(),
    };
    tracing::info!(
        graph = subject.stem,
        class = subject.class,
        seed,
        steps = measured.steps,
        stop = ?measured.stop,
        crossing_step = ?measured.crossing,
        peak_alignment = measured.peak,
        peak_step = ?measured.peak_step,
        final_alignment = measured.final_alignment,
        retained = measured.final_alignment >= FIEDLER_ALIGNMENT,
        elapsed_seconds = elapsed.as_secs_f64(),
        "completed a retention run"
    );

    Ok(RetentionSummary {
        stem: subject.stem,
        class: subject.class,
        order: subject.graph.order(),
        seed,
        measured: Some(measured),
        elapsed,
    })
}

/// Splits a pool's outcomes into the summaries that completed, in
/// configuration order, and the first failure, logging each configuration that
/// did not complete.
fn harvest<T>(
    experiment: &str,
    outcomes: Vec<std::thread::Result<Result<T, Error>>>,
) -> (Vec<T>, Option<Failure>) {
    let mut rows = Vec::new();
    let mut failure = None;
    for outcome in outcomes {
        match outcome {
            Ok(Ok(summary)) => rows.push(summary),
            Ok(Err(error)) => {
                tracing::error!(
                    experiment,
                    error = error.to_string(),
                    "a configuration failed"
                );
                if failure.is_none() {
                    failure = Some(Failure::Errored(error));
                }
            }
            Err(panic) => {
                tracing::error!(experiment, "a configuration panicked");
                if failure.is_none() {
                    failure = Some(Failure::Panicked(panic));
                }
            }
        }
    }
    (rows, failure)
}

/// Re-raises a sweep's first panic, or returns its first error, once the
/// surviving rows have been written.
fn settle(failure: Option<Failure>) -> Result<(), Error> {
    match failure {
        Some(Failure::Panicked(panic)) => std::panic::resume_unwind(panic),
        Some(Failure::Errored(error)) => Err(error),
        None => Ok(()),
    }
}

/// The configurations an arm runs: every subject `include` accepts, crossed
/// with both seeds.
fn configurations(subjects: &[Subject], include: fn(&Subject) -> bool) -> Vec<Config<'_>> {
    subjects
        .iter()
        .filter(|subject| include(subject))
        .flat_map(|subject| SEEDS.map(move |seed| Config { subject, seed }))
        .collect()
}

/// The Node2Vec arm: every subject at both seeds, weight-tied at the Fig-9
/// knobs of `node2vec::Params::default`.
///
/// # Errors
///
/// Propagates [`measure_tied`]'s errors and [`Error::Io`] from writing either
/// summary.
fn tied_sweep(
    directory: &Path,
    pool: &rayon::ThreadPool,
    subjects: &[Subject],
    token: &CancellationToken,
) -> Result<(), Error> {
    let params = node2vec::Params::default();
    let configurations = configurations(subjects, |_| true);
    tracing::info!(
        configurations = configurations.len(),
        dimension = params.dimension,
        sigma = params.sigma,
        learning_rate = params.learning_rate,
        max_steps = params.max_steps,
        tolerance = params.tolerance,
        threads = SWEEP_THREADS,
        "starting the Node2Vec arm"
    );

    let outcomes: Vec<std::thread::Result<Result<TiedSummary, Error>>> = pool.install(|| {
        configurations
            .par_iter()
            .map(|config| {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    measure_tied(directory, config.subject, config.seed, &params, token)
                }))
            })
            .collect()
    });

    let (rows, failure) = harvest("node2vec", outcomes);
    let members: Vec<String> = rows.iter().flat_map(TiedSummary::member_rows).collect();
    write_csv(
        &directory.join("tier1_circulance_summary.csv"),
        "graph,class,order,edges,seed,outcome,steps,observation8_residual,fiedler_start,\
         fiedler_end,max_coefficient_norm,max_rotation_share,elapsed_seconds",
        &rows.iter().map(TiedSummary::row).collect::<Vec<String>>(),
    )?;
    write_csv(
        &directory.join("tier1_circulance_fiedler.csv"),
        "graph,class,seed,index,coefficient_norm,rayleigh,rotation,rotation_share",
        &members,
    )?;
    tracing::info!(
        rows = rows.len(),
        members = members.len(),
        "wrote the Node2Vec summary"
    );
    settle(failure)
}

/// The retention arm: every subject the committed record does not already
/// carry, at both seeds, under an identity-initialized learnable regime at
/// ρ = 1 and η = [`RETENTION_RATE`] over [`RETENTION_STEPS`] applied updates
/// with no geometry stop.
///
/// # Errors
///
/// Propagates [`measure_retention`]'s errors and [`Error::Io`] from writing the
/// summary.
fn retention_sweep(
    directory: &Path,
    pool: &rayon::ThreadPool,
    subjects: &[Subject],
    token: &CancellationToken,
) -> Result<(), Error> {
    let params = tinynn::Params {
        weight_init: WeightInit::Identity,
        weight_rate_ratio: 1.0,
        learning_rate: RETENTION_RATE,
        regime: Regime::LearnableEmbedding,
        alignment_stop: None,
        max_steps: RETENTION_STEPS,
        tolerance: RETENTION_TOLERANCE,
        ..tinynn::Params::default()
    };
    let configurations = configurations(subjects, |subject| subject.retention);
    tracing::info!(
        configurations = configurations.len(),
        width = params.width,
        learning_rate = params.learning_rate,
        max_steps = params.max_steps,
        tolerance = params.tolerance,
        threads = SWEEP_THREADS,
        "starting the retention arm"
    );

    let outcomes: Vec<std::thread::Result<Result<RetentionSummary, Error>>> = pool.install(|| {
        configurations
            .par_iter()
            .map(|config| {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    measure_retention(directory, config.subject, config.seed, &params, token)
                }))
            })
            .collect()
    });

    let (rows, failure) = harvest("retention", outcomes);
    write_csv(
        &directory.join("tier1_circulance_retention_summary.csv"),
        "graph,class,order,seed,status,stop_reason,steps,crossing_step,peak_alignment,peak_step,\
         final_alignment,final_at_or_above,elapsed_seconds",
        &rows
            .iter()
            .map(RetentionSummary::row)
            .collect::<Vec<String>>(),
    )?;
    tracing::info!(rows = rows.len(), "wrote the retention summary");
    settle(failure)
}

#[tokio::main]
async fn main() -> Result<()> {
    logger::setup();
    SETTINGS.ensure_output_dir()?;

    let directory = SETTINGS.output.dir.clone();
    check_builder()?;
    let subjects = subjects()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(SWEEP_THREADS)
        .build()?;

    let runner = Runner::new();
    let handle = runner.spawn(move |token| async move {
        // The runs are CPU-bound and synchronous; spawn_blocking keeps them
        // off the runtime's worker threads.
        tokio::task::spawn_blocking(move || {
            describe(&directory, &subjects)?;
            tied_sweep(&directory, &pool, &subjects, &token)?;
            if token.is_cancelled() {
                tracing::warn!("cancelled before the retention arm");
                return Ok(());
            }
            retention_sweep(&directory, &pool, &subjects, &token)?;
            Ok::<(), Error>(())
        })
        .await
    });

    tokio::select! {
        result = handle => result???,
        result = tokio::signal::ctrl_c() => {
            result?;
            tracing::info!("ctrl-c received, cancelling the run.");
        }
    }

    runner.shutdown().await;
    Ok(())
}
