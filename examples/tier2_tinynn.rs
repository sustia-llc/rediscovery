//! Writes the Tier-2 instrumentation for each of the four D-graphs.
//!
//! Per graph: one frozen-embedding run at η = 0.1 (Fig. 7's associative
//! setting) and one learnable-embedding run per learning rate in
//! {0.001, 0.01, 0.1} (Figs. 8 and 22), each streaming its per-step record —
//! loss, the associative top-d score, the Fiedler alignment, the
//! distance-shell cosine means over the whole graph diameter, and their
//! deepest-shell separation — and its final node-node cosine matrix. The
//! graph's adjacency matrix is written alongside, giving the Fig.-23
//! cosine-versus-adjacency pair under edge-only supervision. The adjacency
//! matrices and the frozen runs are written graph by graph; the twelve
//! learnable runs then go across a rayon pool of `SWEEP_THREADS` workers. The
//! runs are synchronous numerics, so the `Runner` job hands them to
//! `spawn_blocking` and each polls its child token between steps; ctrl-c
//! drains through `Runner::shutdown` and leaves complete CSV rows. The output
//! directory and seed come from `SETTINGS` here and reach the library as
//! explicit arguments (decision D8).

use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;
use tokio_util::sync::CancellationToken;

use rediscovery::error::Error;
use rediscovery::graph::Graph;
use rediscovery::logger;
use rediscovery::output::write_matrix_csv;
use rediscovery::settings::SETTINGS;
use rediscovery::subsystems::runner::Runner;
use rediscovery::tinynn::{self, FIEDLER_ALIGNMENT, Outputs, Params, Regime};

/// Learning rates the geometric sweep covers (decision D10).
const LEARNING_RATES: [f64; 3] = [0.001, 0.01, 0.1];

/// Applied updates the frozen-embedding runs are allowed. Figure 7's claim is
/// about the first two, so a short budget still shows the plateau after them.
const ASSOCIATIVE_STEPS: usize = 20;

/// Applied updates the learnable-embedding runs are allowed, an order of
/// magnitude above the 10² steps Figs. 8 and 22 report.
const GEOMETRIC_STEPS: usize = 2_000;

/// Worker threads the learnable sweep's rayon pool runs, bounding how many of
/// the twelve learnable runs are in flight at once.
const SWEEP_THREADS: usize = 6;

/// The four graphs of decisions D1–D4, with the file-name stem each writes.
fn d_graphs() -> Result<Vec<(&'static str, Graph)>> {
    Ok(vec![
        ("path_star", Graph::path_star(4, 4)?),
        ("grid", Graph::grid(4, 4)?),
        ("cycle", Graph::cycle(15)?),
        ("irregular", Graph::irregular()?),
    ])
}

/// Writes `graph`'s adjacency matrix and its frozen-embedding run under
/// `stem`, at [`ASSOCIATIVE_STEPS`] applied updates, polling `token` between
/// steps.
fn frozen_run(
    directory: &Path,
    seed: u64,
    stem: &str,
    graph: &Graph,
    token: &CancellationToken,
) -> Result<(), Error> {
    let adjacency = directory.join(format!("tier2_adjacency_{stem}.csv"));
    write_matrix_csv(&adjacency, graph.adjacency())?;

    let frozen = Params {
        max_steps: ASSOCIATIVE_STEPS,
        ..Params::default()
    };
    let history = directory.join(format!("tier2_frozen_{stem}.csv"));
    let cosines = directory.join(format!("tier2_frozen_{stem}_cosines.csv"));
    let outputs = Outputs {
        history: &history,
        cosines: &cosines,
    };
    let run = tinynn::run(graph, &frozen, seed, &outputs, || token.is_cancelled())?;
    tracing::info!(
        graph = stem,
        learning_rate = frozen.learning_rate,
        steps = run.steps(),
        outcome = ?run.outcome(),
        associative_step = ?run.associative_step(),
        peak_associative_score = run.peak_associative_score(),
        initial_associative_score = run.records()[0].associative_score(),
        "wrote the frozen-embedding run"
    );
    Ok(())
}

/// Writes `graph`'s learnable-embedding run at `learning_rate` under `stem`, at
/// [`GEOMETRIC_STEPS`] applied updates, polling `token` between steps.
fn learnable_run(
    directory: &Path,
    seed: u64,
    stem: &str,
    graph: &Graph,
    learning_rate: f64,
    token: &CancellationToken,
) -> Result<(), Error> {
    let learnable = Params {
        learning_rate,
        max_steps: GEOMETRIC_STEPS,
        regime: Regime::LearnableEmbedding,
        ..Params::default()
    };
    let history = directory.join(format!("tier2_learnable_{stem}_lr{learning_rate}.csv"));
    let cosines = directory.join(format!(
        "tier2_learnable_{stem}_lr{learning_rate}_cosines.csv"
    ));
    let outputs = Outputs {
        history: &history,
        cosines: &cosines,
    };
    let run = tinynn::run(graph, &learnable, seed, &outputs, || token.is_cancelled())?;
    tracing::info!(
        graph = stem,
        learning_rate,
        steps = run.steps(),
        outcome = ?run.outcome(),
        alignment_step = ?run.alignment_step(FIEDLER_ALIGNMENT),
        peak_alignment = run.peak_alignment(),
        peak_shell_separation = run.peak_deepest_shell_separation(),
        final_shell_means = ?run.last().map(|record| record.shell_means().to_vec()),
        "wrote the learnable-embedding run"
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    logger::setup();
    SETTINGS.ensure_output_dir()?;

    let directory = SETTINGS.output.dir.clone();
    let seed = SETTINGS.rng.seed;
    let graphs = d_graphs()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(SWEEP_THREADS)
        .build()?;

    let runner = Runner::new();
    let handle = runner.spawn(move |token| async move {
        // The runs are CPU-bound and synchronous; spawn_blocking keeps them
        // off the runtime's worker threads.
        tokio::task::spawn_blocking(move || {
            for (stem, graph) in &graphs {
                if token.is_cancelled() {
                    tracing::warn!(graph = *stem, "cancelled before starting this graph");
                    break;
                }
                frozen_run(&directory, seed, stem, graph, &token)?;
            }

            // Each learnable run is internally sequential, draws from its own
            // seeded stream and writes its own pair of paths, so the twelve of
            // them go across the pool. Each polls the child token as before.
            let jobs: Vec<(&str, &Graph, f64)> = graphs
                .iter()
                .flat_map(|(stem, graph)| {
                    LEARNING_RATES.map(|learning_rate| (*stem, graph, learning_rate))
                })
                .collect();
            pool.install(|| {
                jobs.par_iter()
                    .try_for_each(|&(stem, graph, learning_rate)| {
                        if token.is_cancelled() {
                            tracing::warn!(
                                graph = stem,
                                learning_rate,
                                "cancelled before this learning rate"
                            );
                            return Ok::<(), Error>(());
                        }
                        learnable_run(&directory, seed, stem, graph, learning_rate, &token)
                    })
            })?;
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
