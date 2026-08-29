//! Library-first async task lifecycle core.
//!
//! [`Runner`] owns an internal `TaskTracker` + `CancellationToken` and spawns
//! cancellation-aware jobs, each receiving its own `child_token()`. Callers
//! that need to react to cancellation from outside use
//! [`Runner::cancellation_token`]; [`Runner::shutdown`] implements the
//! standard cancel → close → drain sequence. No `SubsystemHandle`, no signal
//! handling — those are downstream/binary concerns (see the
//! `rust-practical:async-lifecycle` skill).

use std::future::Future;

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

/// Owns the task lifecycle for a set of spawned async jobs.
///
/// Exposes [`Runner::cancellation_token`] so callers can observe or extend
/// cancellation, and [`Runner::shutdown`] for the cancel → close → drain
/// sequence. Holds no domain state — downstream binaries and library
/// consumers layer their own state around it.
#[derive(Debug)]
pub struct Runner {
    task_tracker: TaskTracker,
    cancellation_token: CancellationToken,
}

impl Runner {
    /// Creates a `Runner` with a fresh `TaskTracker` and root `CancellationToken`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            task_tracker: TaskTracker::new(),
            cancellation_token: CancellationToken::new(),
        }
    }

    /// Spawns `job` under the internal `TaskTracker`, handing it a
    /// `child_token()` derived from the root cancellation token.
    ///
    /// `job` runs synchronously up to the point it returns a future; the
    /// future itself is what runs on the runtime. Cancelling the root token
    /// (directly, or via [`Runner::shutdown`]) cancels every spawned job's
    /// child token.
    pub fn spawn<F, Fut, T>(&self, job: F) -> JoinHandle<T>
    where
        F: FnOnce(CancellationToken) -> Fut,
        Fut: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let child_token = self.cancellation_token.child_token();
        self.task_tracker.spawn(job(child_token))
    }

    /// The root cancellation token. Cancelling it directly (bypassing
    /// [`Runner::shutdown`]) signals every spawned job's child token without
    /// closing or draining the tracker.
    #[must_use]
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation_token
    }

    /// Three-step shutdown: cancel the root token, close the tracker to new
    /// spawns, then wait for every spawned job to finish.
    pub async fn shutdown(&self) {
        self.cancellation_token.cancel();
        self.task_tracker.close();
        self.task_tracker.wait().await;
    }
}

impl Default for Runner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// `shutdown()` waits for every spawned job to actually finish (the
    /// "drain" step) before returning, independent of whether the job reacts
    /// to cancellation at all.
    #[tokio::test]
    async fn shutdown_drains_spawned_jobs() {
        let runner = Runner::new();
        let completed = Arc::new(AtomicBool::new(false));
        let completed_clone = Arc::clone(&completed);
        runner.spawn(move |_token| async move {
            completed_clone.store(true, Ordering::SeqCst);
        });

        runner.shutdown().await;

        assert!(
            completed.load(Ordering::SeqCst),
            "shutdown() must not return before spawned jobs complete"
        );
    }

    /// Cancelling the root token (without calling `shutdown()`) reaches a
    /// spawned job's `child_token()`.
    #[tokio::test]
    async fn cancellation_reaches_child_token() {
        let runner = Runner::new();
        let (tx, rx) = tokio::sync::oneshot::channel();
        runner.spawn(|token| async move {
            token.cancelled().await;
            tx.send(()).expect("send cancellation signal");
        });

        runner.cancellation_token().cancel();

        rx.await
            .expect("child token should observe cancellation of the root token");
    }

    /// `shutdown()` with no spawned jobs completes (nothing to drain) and
    /// leaves the root token cancelled.
    #[tokio::test]
    async fn shutdown_with_no_jobs_completes() {
        let runner = Runner::new();

        runner.shutdown().await;

        assert!(runner.cancellation_token().is_cancelled());
    }
}
