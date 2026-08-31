//! Retention pin for the identity-initialized learnable regime: crossing the
//! criterion is not holding it.
//!
//! On `path_star(4, 4)` at seed 20260829, an identity-initialized learnable
//! run at η = 0.01 and ρ = 1 meets the 0.75 Fiedler criterion at step 11,
//! peaks at 0.9943946281468211 on step 44, and has decayed to
//! 0.6505758720692354 by the end of its 1000-step budget. The claim ranges
//! over exactly that one graph, that one seed and that one budget — the
//! single configuration the assertions below touch.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;

use rediscovery::graph::Graph;
use rediscovery::tinynn::{
    self, FIEDLER_ALIGNMENT, Outputs, Params, Regime, StepRecord, WeightInit,
};

/// Arms of the path-star this pin measures.
const ARMS: usize = 4;

/// Vertices each arm carries beyond the root.
const ARM_LEN: usize = 4;

/// The run seed this pin measures.
const SEED: u64 = 20_260_829;

/// The step budget this pin measures.
const MAX_STEPS: usize = 1_000;

/// The step at which the pinned run first meets the criterion.
const CROSSING_STEP: usize = 11;

/// A floor the pinned run's transient peak clears, placing the peak far above
/// the 0.75 criterion.
const PEAK_FLOOR: f64 = 0.99;

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
            "rediscovery-retention-{label}-{}-{nanos}-{counter}.csv",
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

/// The pinned retention parameters: width 512, an identity W, η = 0.01,
/// ρ = 1, a learnable embedding, no geometry stop, and a 1000-step budget.
fn retention_params() -> Params {
    Params {
        weight_init: WeightInit::Identity,
        weight_rate_ratio: 1.0,
        learning_rate: 0.01,
        regime: Regime::LearnableEmbedding,
        alignment_stop: None,
        max_steps: MAX_STEPS,
        tolerance: 1e-10,
        ..Params::default()
    }
}

/// Runs `params` on the pinned path-star at `seed` into temp paths.
fn run_on_path_star(params: &Params, seed: u64) -> Result<tinynn::Run> {
    let graph = Graph::path_star(ARMS, ARM_LEN)?;
    let history = TempPath::new("history");
    let cosines = TempPath::new("cosines");
    let outputs = Outputs {
        history: history.path(),
        cosines: cosines.path(),
    };
    Ok(tinynn::run(&graph, params, seed, &outputs, || false)?)
}

/// The identity-initialized learnable run on `path_star(4, 4)` at seed
/// 20260829 meets the criterion at step 11 and peaks above 0.99, then ends its
/// 1000-step budget below the criterion.
#[test]
fn the_identity_run_crosses_the_criterion_and_ends_below_it() -> Result<()> {
    let run = run_on_path_star(&retention_params(), SEED)?;

    let peak = run.peak_alignment();
    let peak_step = run
        .records()
        .iter()
        .max_by(|left, right| {
            left.fiedler_alignment()
                .total_cmp(&right.fiedler_alignment())
        })
        .map(StepRecord::step);
    let last = run
        .last()
        .expect("invariant: a run records the state before every stop check, so it records one");

    println!(
        "path_star({ARMS}, {ARM_LEN}) seed {SEED}: crossed at step {:?}, peaked at {peak} on step \
         {peak_step:?}, ended at {} on step {}",
        run.alignment_step(FIEDLER_ALIGNMENT),
        last.fiedler_alignment(),
        last.step()
    );

    assert_eq!(
        run.alignment_step(FIEDLER_ALIGNMENT),
        Some(CROSSING_STEP),
        "the path_star({ARMS}, {ARM_LEN}) identity run at seed {SEED} met the \
         {FIEDLER_ALIGNMENT} criterion at step {:?} over {} steps, expected {CROSSING_STEP}",
        run.alignment_step(FIEDLER_ALIGNMENT),
        run.steps()
    );
    assert!(
        peak > PEAK_FLOOR,
        "the path_star({ARMS}, {ARM_LEN}) identity run at seed {SEED} peaked at {peak} on step \
         {peak_step:?}, expected a peak above {PEAK_FLOOR}"
    );
    assert!(
        last.fiedler_alignment() < FIEDLER_ALIGNMENT,
        "the path_star({ARMS}, {ARM_LEN}) identity run at seed {SEED} ended at {} on step {}, at \
         or above the {FIEDLER_ALIGNMENT} criterion it had peaked past at {peak} on step \
         {peak_step:?}",
        last.fiedler_alignment(),
        last.step()
    );
    Ok(())
}
