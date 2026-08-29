//! Public-surface seam for the scaffold: `Runner` task lifecycle and
//! `Settings` output-directory handling.
//!
//! Uses exported items only, and never writes inside the repository tree —
//! every filesystem assertion runs against a uniquely named path under the
//! system temp directory, removed when its guard drops. `SETTINGS` is a
//! process-wide `LazyLock`, so these tests construct `Settings` values
//! directly instead of dereferencing it or touching `RUN_MODE`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rediscovery::error::Error;
use rediscovery::settings::{Logger, Output, Rng, Settings};
use rediscovery::subsystems::runner::Runner;

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
            "rediscovery-{label}-{}-{nanos}-{counter}",
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
        if self.0.is_dir() {
            let _ = std::fs::remove_dir_all(&self.0);
        } else {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}

/// A `Settings` value whose `output.dir` is `dir`, built field by field so no
/// config file or environment variable is read.
fn settings_with_output_dir(dir: PathBuf) -> Settings {
    Settings {
        environment: "test".to_string(),
        logger: Logger {
            level: "info".to_string(),
        },
        output: Output { dir },
        rng: Rng { seed: 42 },
    }
}

/// `shutdown()` does not return until a spawned job has finished. The job
/// here is released before `shutdown()` is called, and `#[tokio::test]`
/// builds a current-thread runtime, so the drain inside `shutdown()` is the
/// only point at which the job can make progress.
#[tokio::test]
async fn runner_shutdown_drains_a_spawned_job() {
    let started = Instant::now();

    let runner = Runner::new();
    let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
    let (finished_tx, mut finished_rx) = tokio::sync::oneshot::channel::<()>();

    runner.spawn(move |_token| async move {
        let _ = release_rx.await;
        let _ = finished_tx.send(());
    });

    release_tx.send(()).expect("release receiver is alive");
    runner.shutdown().await;

    assert!(
        finished_rx.try_recv().is_ok(),
        "shutdown() returned before the spawned job finished"
    );

    println!(
        "runner_shutdown_drains_a_spawned_job: {:?}",
        started.elapsed()
    );
}

/// Cancelling the root token reaches a job spawned with a child token. The
/// success path resolves without waiting; the bound only turns the failure
/// mode — an event that never arrives — from a hung binary into a message.
#[tokio::test]
async fn cancellation_token_reaches_a_spawned_job() {
    let runner = Runner::new();
    let (observed_tx, observed_rx) = tokio::sync::oneshot::channel::<()>();

    runner.spawn(|token| async move {
        token.cancelled().await;
        let _ = observed_tx.send(());
    });

    runner.cancellation_token().cancel();

    let observed = tokio::time::timeout(Duration::from_secs(5), observed_rx).await;
    assert!(
        matches!(observed, Ok(Ok(()))),
        "spawned job did not observe cancellation of the root token within 5s: {observed:?}"
    );

    runner.shutdown().await;
}

/// `shutdown()` leaves the root token cancelled and completes with no jobs
/// to drain.
#[tokio::test]
async fn runner_shutdown_with_no_jobs_cancels_the_root_token() {
    let runner = Runner::new();

    runner.shutdown().await;

    assert!(
        runner.cancellation_token().is_cancelled(),
        "shutdown() returned with the root token uncancelled"
    );
}

/// `ensure_output_dir()` creates the configured directory, parents included.
#[test]
fn ensure_output_dir_creates_a_missing_directory() {
    let temp = TempPath::new("output");
    let settings = settings_with_output_dir(temp.path().join("nested").join("run"));

    assert!(
        !settings.output.dir.exists(),
        "fixture path {} already exists",
        settings.output.dir.display()
    );

    settings.ensure_output_dir().expect("ensure_output_dir");

    assert!(
        settings.output.dir.is_dir(),
        "ensure_output_dir() returned Ok but {} is not a directory",
        settings.output.dir.display()
    );
}

/// A directory that cannot be created surfaces as `Error::Io` rather than a
/// silent success.
#[test]
fn ensure_output_dir_surfaces_an_io_error_under_a_file() {
    let temp = TempPath::new("blocker");
    std::fs::write(temp.path(), b"not a directory").expect("write blocker file");

    let settings = settings_with_output_dir(temp.path().join("under-a-file"));

    match settings.ensure_output_dir() {
        Err(Error::Io(source)) => {
            println!(
                "ensure_output_dir_surfaces_an_io_error_under_a_file: kind={:?}",
                source.kind()
            );
        }
        Err(other) => panic!("expected Error::Io, got {other:?}"),
        Ok(()) => panic!(
            "expected an I/O error, but ensure_output_dir() reported success for {}",
            settings.output.dir.display()
        ),
    }

    assert!(
        !settings.output.dir.is_dir(),
        "{} was created under a regular file",
        settings.output.dir.display()
    );
}
