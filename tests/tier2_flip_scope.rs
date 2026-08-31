//! Scope pin for the W-initialization flip: the dichotomy is
//! graph-dependent.
//!
//! The committed Tier-2 finding — no Gaussian-initialized learnable run
//! crosses the 0.75 criterion in 20 000 steps at any ρ — was measured on
//! the four D-graphs, `grid(4, 4)` among them. It does not extend
//! unmodified to larger members of the same family: on `grid(6, 8)` at
//! the committed width, η = 0.01 and ρ = 1, the Gaussian-initialized run
//! crosses the criterion (measured peak alignment 0.8160584913387996 at
//! seed 20260829 over a 2 000-step budget). First observed downstream in
//! `spatial-priors` (finding F15, 2026-08-31); reproduced here with this
//! crate's own run loop.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;

use rediscovery::graph::Graph;
use rediscovery::tinynn::{self, FIEDLER_ALIGNMENT, Outputs, Params, Regime, WeightInit};

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

/// On the 6×8 grid the Gaussian-initialized learnable run crosses the
/// geometry criterion — the initialization dichotomy of the committed
/// D-graph finding does not extend to every graph, or even to every grid.
#[test]
fn the_gaussian_run_crosses_the_criterion_on_the_6x8_grid() -> Result<()> {
    let graph = Graph::grid(6, 8)?;
    let params = Params {
        learning_rate: 0.01,
        max_steps: 2_000,
        regime: Regime::LearnableEmbedding,
        weight_init: WeightInit::Gaussian,
        weight_rate_ratio: 1.0,
        alignment_stop: None,
        ..Params::default()
    };
    let history = TempPath::new("history");
    let cosines = TempPath::new("cosines");
    let outputs = Outputs {
        history: history.path(),
        cosines: cosines.path(),
    };

    let run = tinynn::run(&graph, &params, 20_260_829, &outputs, || false)?;

    assert!(
        run.peak_alignment() >= FIEDLER_ALIGNMENT,
        "the Gaussian run crosses on grid(6, 8): measured peak 0.8160584913387996, got {}",
        run.peak_alignment()
    );
    Ok(())
}
