//! Scope pins for the W-initialization flip beyond the D-graphs.
//!
//! The committed Tier-2 finding — no Gaussian-initialized learnable run on
//! the four D-graphs crosses the 0.75 criterion in 20 000 steps at any ρ —
//! does not fix the behavior of other graphs. On `grid(6, 8)` at the
//! committed knobs the outcome differs by seed: at seed 20260829 the
//! Gaussian run crosses the criterion at step 77, while at seed 42 it stays
//! below the criterion for a whole 2 000-step budget (measured peak
//! 0.004342691178681286). Both behaviors are pinned here, so the boundary
//! claim ranges over exactly the graph and the two seeds the assertions
//! touch.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;

use rediscovery::graph::Graph;
use rediscovery::tinynn::{
    self, FIEDLER_ALIGNMENT, Outputs, Params, Regime, StopReason, WeightInit,
};

/// Rows of the grid this scope pin measures.
const GRID_ROWS: usize = 6;

/// Columns of the grid this scope pin measures.
const GRID_COLS: usize = 8;

/// The step at which the seed-20260829 Gaussian run first meets the
/// criterion under the committed sweep knobs.
const CROSSING_STEP: usize = 77;

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
            "rediscovery-flip-scope-{label}-{}-{nanos}-{counter}.csv",
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

/// The committed transition-sweep Gaussian-arm parameters, on the grid this
/// file measures: width 512, η = 0.01, ρ = 1, geometry stop at the
/// criterion, 20 000-step budget.
fn sweep_params() -> Params {
    Params {
        learning_rate: 0.01,
        max_steps: 20_000,
        regime: Regime::LearnableEmbedding,
        weight_init: WeightInit::Gaussian,
        weight_rate_ratio: 1.0,
        alignment_stop: Some(FIEDLER_ALIGNMENT),
        ..Params::default()
    }
}

/// Runs `params` on the pinned grid at `seed` into temp paths.
fn run_on_grid(params: &Params, seed: u64) -> Result<tinynn::Run> {
    let graph = Graph::grid(GRID_ROWS, GRID_COLS)?;
    let history = TempPath::new("history");
    let cosines = TempPath::new("cosines");
    let outputs = Outputs {
        history: history.path(),
        cosines: cosines.path(),
    };
    Ok(tinynn::run(&graph, params, seed, &outputs, || false)?)
}

/// At seed 20260829 the Gaussian-initialized run on the 6×8 grid meets the
/// criterion at step 77 under the committed sweep knobs — the behavior no
/// D-graph Gaussian run shows at either committed seed.
#[test]
fn the_gaussian_run_crosses_on_the_grid_at_the_first_committed_seed() -> Result<()> {
    let run = run_on_grid(&sweep_params(), 20_260_829)?;

    assert_eq!(
        run.stop_reason(),
        StopReason::Aligned,
        "the grid({GRID_ROWS}, {GRID_COLS}) Gaussian run at seed 20260829 ended on {:?} after \
         {} steps, expected the geometry stop",
        run.stop_reason(),
        run.steps()
    );
    assert_eq!(
        run.alignment_step(FIEDLER_ALIGNMENT),
        Some(CROSSING_STEP),
        "the grid({GRID_ROWS}, {GRID_COLS}) Gaussian run at seed 20260829 met the criterion at \
         step {:?}, expected {CROSSING_STEP}",
        run.alignment_step(FIEDLER_ALIGNMENT)
    );
    Ok(())
}

/// At seed 42 the same configuration stays below the criterion for a whole
/// 2 000-step budget, so the grid's crossing is a property of the seed as
/// well as the graph.
#[test]
fn the_gaussian_run_stays_below_the_criterion_on_the_grid_at_the_second_seed() -> Result<()> {
    let params = Params {
        alignment_stop: None,
        max_steps: 2_000,
        ..sweep_params()
    };
    let run = run_on_grid(&params, 42)?;

    assert_eq!(
        run.alignment_step(FIEDLER_ALIGNMENT),
        None,
        "the grid({GRID_ROWS}, {GRID_COLS}) Gaussian run at seed 42 met the criterion at step \
         {:?} inside the {} recorded steps",
        run.alignment_step(FIEDLER_ALIGNMENT),
        run.steps()
    );
    assert!(
        run.peak_alignment() < FIEDLER_ALIGNMENT,
        "the grid({GRID_ROWS}, {GRID_COLS}) Gaussian run at seed 42 peaked at {} over {} steps, \
         at or above the {FIEDLER_ALIGNMENT} criterion",
        run.peak_alignment(),
        run.steps()
    );
    Ok(())
}
