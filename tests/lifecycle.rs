//! Public-surface seam for the scaffold: `Runner` task lifecycle and
//! `Settings` output-directory handling.
//!
//! Filesystem assertions run against uniquely named paths under the system
//! temp directory, removed when their guards drop. `SETTINGS` is a
//! process-wide `LazyLock`, so these tests construct `Settings` values
//! directly instead of dereferencing it or touching `RUN_MODE`. Every await
//! on a `Runner` is bounded by [`SHUTDOWN_BOUND`], turning a lifecycle
//! regression into a named failure instead of a hung binary.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rediscovery::error::Error;
use rediscovery::settings::{Logger, Output, Rng, Settings};
use rediscovery::subsystems::runner::Runner;

/// Upper bound on lifecycle awaits; the passing paths resolve without
/// waiting on it.
const SHUTDOWN_BOUND: Duration = Duration::from_secs(5);

/// Scheduler passes the drain-test job needs before it can finish, chosen
/// well above the await points any single `shutdown()` call contains.
const DRAIN_YIELDS: usize = 64;

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

/// `shutdown().await` bounded by [`SHUTDOWN_BOUND`].
async fn bounded_shutdown(runner: &Runner) {
    tokio::time::timeout(SHUTDOWN_BOUND, runner.shutdown())
        .await
        .expect("shutdown() did not complete within SHUTDOWN_BOUND");
}

/// `shutdown()` does not return until a spawned job has finished. The job
/// needs [`DRAIN_YIELDS`] scheduler passes on this current-thread runtime,
/// so an early return from the drain leaves it unfinished.
#[tokio::test]
async fn runner_shutdown_drains_a_spawned_job() {
    let started = Instant::now();

    let runner = Runner::new();
    let (finished_tx, mut finished_rx) = tokio::sync::oneshot::channel::<()>();

    runner.spawn(move |_token| async move {
        for _ in 0..DRAIN_YIELDS {
            tokio::task::yield_now().await;
        }
        let _ = finished_tx.send(());
    });

    bounded_shutdown(&runner).await;

    assert!(
        finished_rx.try_recv().is_ok(),
        "shutdown() returned before the spawned job finished its {DRAIN_YIELDS} passes"
    );

    println!(
        "runner_shutdown_drains_a_spawned_job: {:?}",
        started.elapsed()
    );
}

/// Each job receives its own child token: cancelling the token one job was
/// handed cancels neither the root nor a sibling job's token.
#[tokio::test]
async fn child_tokens_are_isolated_per_job() {
    let runner = Runner::new();
    let (token_tx, token_rx) = tokio::sync::oneshot::channel();
    let (sibling_tx, mut sibling_rx) = tokio::sync::oneshot::channel::<()>();

    runner.spawn(move |token| async move {
        let _ = token_tx.send(token);
    });
    runner.spawn(move |token| async move {
        token.cancelled().await;
        let _ = sibling_tx.send(());
    });

    let job_token = tokio::time::timeout(SHUTDOWN_BOUND, token_rx)
        .await
        .expect("first job did not hand its token out in time")
        .expect("token sender dropped");

    job_token.cancel();
    tokio::task::yield_now().await;

    assert!(
        !runner.cancellation_token().is_cancelled(),
        "cancelling a job's token cancelled the root token"
    );
    assert!(
        sibling_rx.try_recv().is_err(),
        "cancelling one job's token reached a sibling job's token"
    );

    runner.cancellation_token().cancel();
    let observed = tokio::time::timeout(SHUTDOWN_BOUND, sibling_rx).await;
    assert!(
        matches!(observed, Ok(Ok(()))),
        "sibling job did not observe root cancellation: {observed:?}"
    );

    bounded_shutdown(&runner).await;
}

/// `shutdown()` leaves the root token cancelled and completes with no jobs
/// to drain.
#[tokio::test]
async fn runner_shutdown_with_no_jobs_cancels_the_root_token() {
    let runner = Runner::new();

    bounded_shutdown(&runner).await;

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
