//! Issue #5: where the spectral geometry sits between the Node2Vec-equivalent
//! corner of the TinyNN family and the committed learnable regime.
//!
//! Experiment 1, the W-sweep, crosses two axes on the learnable regime — W's
//! initializer in {identity, the committed Gaussian draw} and the relative
//! weight rate ρ = η_W/η_E in {0, 1/8, 1/2, 1} — over the four D-graphs and the
//! two seeds at η = 0.01. Each run is capped at [`SWEEP_STEPS`] applied updates
//! and carries two further stopping rules: the relative-update tolerance and
//! the first step whose Fiedler alignment reaches `FIEDLER_ALIGNMENT`. Its
//! (identity, ρ = 0) corner is dynamically Tier 1 — `tinynn`'s
//! `the_identity_corner_keeps_the_tier1_ascent_direction_along_the_run` pins
//! the E update direction there against Lemma 6's CV — and its (Gaussian,
//! ρ = 1) corner is the committed regime.
//!
//! Experiment 2 runs that committed regime under §B.3's optimizer instead:
//! decoupled AdamW at weight decay 0.01 under a linear warm-up into a cosine
//! decay, over the three (peak rate, budget) pairs of the committed
//! constant-rate sweep, the same four graphs and the same two seeds.
//!
//! Each run writes one history CSV and one node-node cosine matrix under the
//! `tier2_transition_` stem, and each experiment writes one summary CSV, into
//! `SETTINGS.output.dir`. The configurations of an experiment go across a rayon
//! pool of `SWEEP_THREADS` workers, each run internally sequential and drawing
//! from its own seeded stream; a configuration that panics is re-raised after
//! the surviving rows are written. The runs are synchronous numerics, so the
//! `Runner` job hands them to `spawn_blocking` and each polls its child token
//! between steps; ctrl-c drains through `Runner::shutdown` and leaves complete
//! CSV rows. The output directory comes from `SETTINGS` here and reaches the
//! library as an explicit argument (decision D8).

#![allow(
    clippy::doc_markdown,
    reason = "the docs carry the model's name and its rate notation — TinyNN, AdamW, η_W/η_E — that the lint reads as unbackticked identifiers"
)]

use std::any::Any;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;
use rayon::prelude::*;
use tokio_util::sync::CancellationToken;

use rediscovery::error::Error;
use rediscovery::graph::Graph;
use rediscovery::logger;
use rediscovery::settings::SETTINGS;
use rediscovery::subsystems::runner::Runner;
use rediscovery::tinynn::{
    self, AdamW, FIEDLER_ALIGNMENT, Optimizer, Outputs, Params, Regime, StopReason, WeightInit,
};

/// Seeds both experiments run, the second being `config/default.toml`'s
/// `rng.seed`.
const SEEDS: [u64; 2] = [20_260_829, 42];

/// Worker threads the sweep pool runs, bounding how many runs of one
/// experiment are in flight at once.
const SWEEP_THREADS: usize = 6;

/// Learning rate the W-sweep runs at, the middle of the committed sweep's
/// three.
const SWEEP_RATE: f64 = 0.01;

/// Applied updates the W-sweep allows before the step limit binds, the budget
/// `node2vec`'s `cycle_reproduces_the_full_fig9_signature` gives Tier 1 at the
/// same η = 0.01, where that run converges at step 15 855.
const SWEEP_STEPS: usize = 20_000;

/// W initializers the sweep crosses, with the file-name stem each writes.
const WEIGHT_INITS: [(&str, WeightInit); 2] = [
    ("identity", WeightInit::Identity),
    ("gaussian", WeightInit::Gaussian),
];

/// Relative weight rates ρ = η_W/η_E the sweep crosses.
const WEIGHT_RATES: [f64; 4] = [0.0, 0.125, 0.5, 1.0];

/// Peak rates and budgets the §B.3 experiment runs: the pairs of the committed
/// constant-rate sweep, which are the timescales Figs. 8 and 22 plot.
const B3_SWEEP: [(f64, usize); 3] = [(0.001, 1_200), (0.01, 200), (0.1, 50)];

/// The measurement columns every summary row carries after its
/// experiment-specific configuration ones.
const SUMMARY_COLUMNS: &str = "steps,stop_reason,alignment_step,peak_alignment,final_alignment,\
     associative_step,peak_associative_score,final_loss,elapsed_seconds";

/// The four graphs of decisions D1–D4, with the file-name stem each writes.
fn d_graphs() -> Result<Vec<(&'static str, Graph)>, Error> {
    Ok(vec![
        ("path_star", Graph::path_star(4, 4)?),
        ("grid", Graph::grid(4, 4)?),
        ("cycle", Graph::cycle(15)?),
        ("irregular", Graph::irregular()?),
    ])
}

/// One run's summary row: its experiment's configuration fields, already
/// comma-separated, followed by the measurements [`SUMMARY_COLUMNS`] names.
struct Summary {
    configuration: String,
    steps: usize,
    stop: StopReason,
    alignment_step: Option<usize>,
    peak_alignment: f64,
    final_alignment: f64,
    associative_step: Option<usize>,
    peak_associative_score: f64,
    final_loss: f64,
    elapsed: Duration,
}

/// One configuration of the W-sweep.
struct SweepConfig<'a> {
    init: &'static str,
    weight_init: WeightInit,
    ratio: f64,
    stem: &'static str,
    graph: &'a Graph,
    seed: u64,
}

/// One configuration of the §B.3 experiment.
struct AdamConfig<'a> {
    peak: f64,
    budget: usize,
    stem: &'static str,
    graph: &'a Graph,
    seed: u64,
}

/// What ended a sweep short of every configuration completing.
enum Failure {
    Panicked(Box<dyn Any + Send>),
    Errored(Error),
}

/// Runs `params` on `graph` at `seed` into the pair of paths `stem` names under
/// `directory`, polling `token` between steps, and returns the row the
/// experiment's summary CSV carries under `configuration`.
fn measure(
    directory: &Path,
    stem: &str,
    configuration: String,
    graph: &Graph,
    params: &Params,
    seed: u64,
    token: &CancellationToken,
) -> Result<Summary, Error> {
    let history = directory.join(format!("{stem}.csv"));
    let cosines = directory.join(format!("{stem}_cosines.csv"));
    let outputs = Outputs {
        history: &history,
        cosines: &cosines,
    };

    let started = Instant::now();
    let run = tinynn::run(graph, params, seed, &outputs, || token.is_cancelled())?;
    let elapsed = started.elapsed();
    let last = run
        .last()
        .expect("invariant: `tinynn::run` records its initial state before any stop");

    let summary = Summary {
        configuration,
        steps: run.steps(),
        stop: run.stop_reason(),
        alignment_step: run.alignment_step(FIEDLER_ALIGNMENT),
        peak_alignment: run.peak_alignment(),
        final_alignment: last.fiedler_alignment(),
        associative_step: run.associative_step(),
        peak_associative_score: run.peak_associative_score(),
        final_loss: last.loss(),
        elapsed,
    };
    tracing::info!(
        stem,
        steps = summary.steps,
        stop = ?summary.stop,
        alignment_step = ?summary.alignment_step,
        peak_alignment = summary.peak_alignment,
        final_alignment = summary.final_alignment,
        associative_step = ?summary.associative_step,
        peak_associative_score = summary.peak_associative_score,
        final_loss = summary.final_loss,
        elapsed_seconds = elapsed.as_secs_f64(),
        "completed a transition run"
    );
    Ok(summary)
}

/// The CSV word for a stop reason.
fn stop_label(stop: StopReason) -> &'static str {
    match stop {
        StopReason::Converged => "converged",
        StopReason::StepLimit => "step_limit",
        StopReason::Aligned => "aligned",
        StopReason::Stopped => "stopped",
    }
}

/// A step that may not have occurred, as a CSV field: its number, or empty.
fn step_field(step: Option<usize>) -> String {
    step.map_or_else(String::new, |step| step.to_string())
}

/// Writes `rows` to `path` under `columns` — the experiment's configuration
/// column names — followed by [`SUMMARY_COLUMNS`]. Floats are written in Rust's
/// shortest round-tripping form.
fn write_summary(path: &Path, columns: &str, rows: &[Summary]) -> Result<(), Error> {
    let mut sink = BufWriter::new(File::create(path)?);
    writeln!(sink, "{columns},{SUMMARY_COLUMNS}")?;
    for row in rows {
        writeln!(
            sink,
            "{},{},{},{},{},{},{},{},{},{}",
            row.configuration,
            row.steps,
            stop_label(row.stop),
            step_field(row.alignment_step),
            row.peak_alignment,
            row.final_alignment,
            step_field(row.associative_step),
            row.peak_associative_score,
            row.final_loss,
            row.elapsed.as_secs_f64()
        )?;
    }
    sink.flush()?;
    Ok(())
}

/// Splits a pool's outcomes into the summaries that completed, in
/// configuration order, and the first failure, logging each configuration that
/// did not complete.
fn harvest(
    experiment: &str,
    outcomes: Vec<std::thread::Result<Result<Summary, Error>>>,
) -> (Vec<Summary>, Option<Failure>) {
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

/// Experiment 1: the 8 (initializer, ρ) configurations across the four graphs
/// and both seeds, at η = [`SWEEP_RATE`] over [`SWEEP_STEPS`] applied updates
/// with the geometry stop armed at `FIEDLER_ALIGNMENT`.
fn weight_sweep(
    directory: &Path,
    pool: &rayon::ThreadPool,
    graphs: &[(&'static str, Graph)],
    token: &CancellationToken,
) -> Result<(), Error> {
    let configurations: Vec<SweepConfig<'_>> = WEIGHT_INITS
        .into_iter()
        .flat_map(|(init, weight_init)| {
            WEIGHT_RATES.into_iter().flat_map(move |ratio| {
                graphs.iter().flat_map(move |(stem, graph)| {
                    SEEDS.map(move |seed| SweepConfig {
                        init,
                        weight_init,
                        ratio,
                        stem,
                        graph,
                        seed,
                    })
                })
            })
        })
        .collect();
    tracing::info!(
        configurations = configurations.len(),
        max_steps = SWEEP_STEPS,
        learning_rate = SWEEP_RATE,
        threads = SWEEP_THREADS,
        "starting the W-sweep"
    );

    let outcomes: Vec<std::thread::Result<Result<Summary, Error>>> = pool.install(|| {
        configurations
            .par_iter()
            .map(|config| {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let params = Params {
                        learning_rate: SWEEP_RATE,
                        max_steps: SWEEP_STEPS,
                        regime: Regime::LearnableEmbedding,
                        weight_init: config.weight_init,
                        weight_rate_ratio: config.ratio,
                        alignment_stop: Some(FIEDLER_ALIGNMENT),
                        ..Params::default()
                    };
                    let stem = format!(
                        "tier2_transition_w_{}_rho{}_{}_seed{}",
                        config.init, config.ratio, config.stem, config.seed
                    );
                    let configuration = format!(
                        "{},{},{},{}",
                        config.init, config.ratio, config.stem, config.seed
                    );
                    measure(
                        directory,
                        &stem,
                        configuration,
                        config.graph,
                        &params,
                        config.seed,
                        token,
                    )
                }))
            })
            .collect()
    });

    let (rows, failure) = harvest("w_sweep", outcomes);
    write_summary(
        &directory.join("tier2_transition_wsweep_summary.csv"),
        "weight_init,weight_rate_ratio,graph,seed",
        &rows,
    )?;
    tracing::info!(rows = rows.len(), "wrote the W-sweep summary");
    settle(failure)
}

/// Experiment 2: the committed learnable regime under §B.3's decoupled AdamW
/// and its warm-up-into-cosine schedule, over the three (peak rate, budget)
/// pairs of [`B3_SWEEP`], the four graphs and both seeds.
fn adamw_sweep(
    directory: &Path,
    pool: &rayon::ThreadPool,
    graphs: &[(&'static str, Graph)],
    token: &CancellationToken,
) -> Result<(), Error> {
    let configurations: Vec<AdamConfig<'_>> = B3_SWEEP
        .into_iter()
        .flat_map(|(peak, budget)| {
            graphs.iter().flat_map(move |(stem, graph)| {
                SEEDS.map(move |seed| AdamConfig {
                    peak,
                    budget,
                    stem,
                    graph,
                    seed,
                })
            })
        })
        .collect();
    let settings = AdamW::default();
    tracing::info!(
        configurations = configurations.len(),
        beta1 = settings.beta1,
        beta2 = settings.beta2,
        epsilon = settings.epsilon,
        weight_decay = settings.weight_decay,
        warmup_fraction = settings.warmup_fraction,
        threads = SWEEP_THREADS,
        "starting the section B.3 optimizer experiment"
    );

    let outcomes: Vec<std::thread::Result<Result<Summary, Error>>> = pool.install(|| {
        configurations
            .par_iter()
            .map(|config| {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let params = Params {
                        learning_rate: config.peak,
                        max_steps: config.budget,
                        regime: Regime::LearnableEmbedding,
                        optimizer: Optimizer::AdamW(settings),
                        alignment_stop: Some(FIEDLER_ALIGNMENT),
                        ..Params::default()
                    };
                    let stem = format!(
                        "tier2_transition_b3_lr{}_{}_seed{}",
                        config.peak, config.stem, config.seed
                    );
                    let configuration = format!(
                        "{},{},{},{}",
                        config.peak, config.budget, config.stem, config.seed
                    );
                    measure(
                        directory,
                        &stem,
                        configuration,
                        config.graph,
                        &params,
                        config.seed,
                        token,
                    )
                }))
            })
            .collect()
    });

    let (rows, failure) = harvest("adamw", outcomes);
    write_summary(
        &directory.join("tier2_transition_adamw_summary.csv"),
        "peak_learning_rate,budget,graph,seed",
        &rows,
    )?;
    tracing::info!(rows = rows.len(), "wrote the section B.3 summary");
    settle(failure)
}

#[tokio::main]
async fn main() -> Result<()> {
    logger::setup();
    SETTINGS.ensure_output_dir()?;

    let directory = SETTINGS.output.dir.clone();
    let graphs = d_graphs()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(SWEEP_THREADS)
        .build()?;

    let runner = Runner::new();
    let handle = runner.spawn(move |token| async move {
        // The runs are CPU-bound and synchronous; spawn_blocking keeps them
        // off the runtime's worker threads.
        tokio::task::spawn_blocking(move || {
            weight_sweep(&directory, &pool, &graphs, &token)?;
            if token.is_cancelled() {
                tracing::warn!("cancelled before the section B.3 experiment");
                return Ok(());
            }
            adamw_sweep(&directory, &pool, &graphs, &token)?;
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
