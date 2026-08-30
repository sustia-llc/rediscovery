//! The Tier-2 spectral geometry turns on and off with W's initialization
//! alone.
//!
//! Two runs of the learnable-embedding regime on the 15-cycle, at seed
//! 20260829, η = 0.01 and a relative weight rate ρ = η_W/η_E of 1 — W training
//! as freely as the embedding — differ in one field of `Params`:
//! `weight_init`. From the identity W the embedding reaches the Fiedler
//! alignment `FIEDLER_ALIGNMENT` that `tinynn` reads as geometric, and the run
//! ends there; from the committed Gaussian draw the alignment stays below that
//! threshold for the whole 2 000-step budget. Each run prints its recorded
//! alignment — every step for the identity run, every hundredth and the last
//! for the Gaussian one — and closes with its step count, stop reason and peak.
//!
//! Run it with `cargo run --release --example w_init_flip`.
//!
//! Nothing reaches the `output` directory: `tinynn::run` streams a per-step CSV
//! and a final cosine matrix to paths the caller names, so this demo points
//! them at a scratch directory it removes on exit and carries its result on
//! stdout instead.
//!
//! `examples/tier2_transition.rs` runs the 64-run W-sweep this pair is drawn
//! from — both initializers crossed with ρ in {0, 1/8, 1/2, 1} over the four
//! D-graphs and both seeds, at a 20 000-step budget — and writes its CSVs.

#![allow(
    clippy::doc_markdown,
    reason = "the docs carry the model's rate notation — η_W/η_E — that the lint reads as an unbackticked identifier"
)]

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;

use rediscovery::graph::Graph;
use rediscovery::tinynn::{
    self, FIEDLER_ALIGNMENT, Outputs, Params, Regime, Run, StopReason, WeightInit,
};

/// Seed both runs draw from, the first of the transition sweep's two.
const SEED: u64 = 20_260_829;

/// Learning rate η both runs descend at, the middle of the committed sweep's
/// three.
const RATE: f64 = 0.01;

/// Applied updates either run is allowed.
const STEPS: usize = 2_000;

/// Vertices of the cycle both runs train on, decision D3's graph.
const CYCLE_ORDER: usize = 15;

/// Steps between the Gaussian run's printed lines, keeping its trace to a
/// screenful.
const GAUSSIAN_STRIDE: usize = 100;

/// A directory under the system temp directory, removed when this value drops.
struct Scratch(PathBuf);

impl Scratch {
    /// Creates a directory named for this process and the current instant.
    fn new() -> Result<Self> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let path = std::env::temp_dir().join(format!(
            "rediscovery-w-init-flip-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path)?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The word this demo prints for a stop reason.
fn stop_label(stop: StopReason) -> &'static str {
    match stop {
        StopReason::Converged => "convergence",
        StopReason::StepLimit => "the step limit",
        StopReason::Aligned => "the geometry criterion",
        StopReason::Stopped => "the caller's stop signal",
    }
}

/// Trains `graph` under `weight_init` at ρ = 1, with the geometry stop armed at
/// `alignment_stop`, writing the library's two CSVs under `scratch` as `stem`.
fn train(
    scratch: &Path,
    stem: &str,
    graph: &Graph,
    weight_init: WeightInit,
    alignment_stop: Option<f64>,
) -> Result<Run> {
    let params = Params {
        learning_rate: RATE,
        max_steps: STEPS,
        regime: Regime::LearnableEmbedding,
        weight_init,
        weight_rate_ratio: 1.0,
        alignment_stop,
        ..Params::default()
    };
    let history = scratch.join(format!("{stem}.csv"));
    let cosines = scratch.join(format!("{stem}_cosines.csv"));
    let outputs = Outputs {
        history: &history,
        cosines: &cosines,
    };
    Ok(tinynn::run(graph, &params, SEED, &outputs, || false)?)
}

/// Prints `header`, then `run`'s Fiedler alignment at every `stride`-th
/// recorded step and at the last one, then the run's step count, stop reason
/// and peak alignment.
fn report(header: &str, run: &Run, stride: usize) {
    println!("{header}");
    let last = run.records().len().saturating_sub(1);
    for (index, record) in run.records().iter().enumerate() {
        if index % stride == 0 || index == last {
            println!(
                "  step {:>4}   fiedler_alignment {}",
                record.step(),
                record.fiedler_alignment()
            );
        }
    }
    println!(
        "  {} steps, ended on {}, peak alignment {}",
        run.steps(),
        stop_label(run.stop_reason()),
        run.peak_alignment()
    );
}

fn main() -> Result<()> {
    let scratch = Scratch::new()?;
    let graph = Graph::cycle(CYCLE_ORDER)?;

    println!(
        "{CYCLE_ORDER}-cycle, learnable embedding, seed {SEED}, eta {RATE}, rho 1, \
         gradient descent, budget {STEPS} steps."
    );
    println!("The two runs differ only in W's initialization.");
    println!("Geometric at fiedler_alignment >= {FIEDLER_ALIGNMENT}.");
    println!();

    let identity = train(
        scratch.path(),
        "identity",
        &graph,
        WeightInit::Identity,
        Some(FIEDLER_ALIGNMENT),
    )?;
    report("W = I, geometry stop armed:", &identity, 1);
    println!();

    let gaussian = train(
        scratch.path(),
        "gaussian",
        &graph,
        WeightInit::Gaussian,
        None,
    )?;
    report(
        "W ~ N(0, weight_sigma^2), no geometry stop:",
        &gaussian,
        GAUSSIAN_STRIDE,
    );
    println!();

    println!(
        "Identity: {} at step {}. Gaussian: peak {} over {} steps.",
        stop_label(identity.stop_reason()),
        identity.steps(),
        gaussian.peak_alignment(),
        gaussian.steps()
    );
    Ok(())
}
