//! Writes the Fig. 9 instrumentation for each of the four D-graphs.
//!
//! One weight-tied run per graph at the decision-D7 defaults, each streaming
//! its per-step record to `<SETTINGS.output.dir>/tier1_fig9_<graph>.csv`: the
//! eigenvector projections ‖Vᵀe_i‖₂ (panel b left/middle), the coefficient
//! norms ‖Ce_i‖₂ (panel b right) with their orthogonal split into the signed
//! Rayleigh component r_i = e_iᵀCe_i and the rotation ‖Ce_i − r_i e_i‖₂, the
//! objective, the Observation-8 residual, and the Remark-5 degenerate
//! projection in column `projection_0`. The runs are synchronous numerics,
//! so the `Runner` job hands them to
//! `spawn_blocking` and polls its child token between steps; ctrl-c drains
//! through `Runner::shutdown` and leaves complete CSV rows. The output
//! directory and seed come from `SETTINGS` here and reach the library as
//! explicit arguments (decision D8).

#![allow(
    clippy::doc_markdown,
    reason = "the docs carry matrix notation with subscripts — ‖Vᵀe_i‖₂, ‖Ce_i‖₂ — that the lint reads as unbackticked identifiers"
)]

use anyhow::Result;

use rediscovery::error::Error;
use rediscovery::graph::Graph;
use rediscovery::logger;
use rediscovery::node2vec::{self, Params};
use rediscovery::settings::SETTINGS;
use rediscovery::subsystems::runner::Runner;

/// The four graphs of decisions D1–D4, with the file-name stem each writes.
fn d_graphs() -> Result<Vec<(&'static str, Graph)>> {
    Ok(vec![
        ("path_star", Graph::path_star(4, 4)?),
        ("grid", Graph::grid(4, 4)?),
        ("cycle", Graph::cycle(15)?),
        ("irregular", Graph::irregular()?),
    ])
}

#[tokio::main]
async fn main() -> Result<()> {
    logger::setup();
    SETTINGS.ensure_output_dir()?;

    let directory = SETTINGS.output.dir.clone();
    let seed = SETTINGS.rng.seed;
    let params = Params::default();
    let graphs = d_graphs()?;

    let runner = Runner::new();
    let handle = runner.spawn(move |token| async move {
        // The runs are CPU-bound and synchronous; spawn_blocking keeps them
        // off the runtime's worker threads.
        tokio::task::spawn_blocking(move || {
            for (stem, graph) in graphs {
                if token.is_cancelled() {
                    tracing::warn!(graph = stem, "cancelled before starting this graph");
                    break;
                }

                let path = directory.join(format!("tier1_fig9_{stem}.csv"));
                let run =
                    node2vec::run_tied(&graph, &params, seed, &path, || token.is_cancelled())?;
                tracing::info!(
                    graph = stem,
                    steps = run.steps(),
                    outcome = ?run.outcome(),
                    rows = run.records().len(),
                    path = %path.display(),
                    "wrote Fig. 9 instrumentation"
                );
            }
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
