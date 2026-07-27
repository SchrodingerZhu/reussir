//! The process pool: bounded fan-out for the build's child processes.
//!
//! Built on compio's dispatcher: `jobs` worker threads, each running its own
//! completion-based runtime, each **awaiting one task to completion before
//! taking the next** (`concurrent(false)`) — so at most `jobs` compiles run
//! at once, with the overflow queued in dispatch order. The main runtime
//! keeps orchestrating (freshness, records, the listing) while the workers
//! carry the processes.
//!
//! Tasks are dispatched as `Send` closures returning a (worker-local)
//! future; results come back over a oneshot the caller awaits. The pool is
//! shared by reference — target compiles today, the per-dependency compiles
//! of the cross-package build next.

use std::future::Future;
use std::num::NonZeroUsize;

use compio::dispatcher::Dispatcher;

/// A fixed-width pool of process-running workers.
pub struct Pool {
    dispatcher: Dispatcher,
    jobs: NonZeroUsize,
}

impl Pool {
    /// Start a pool wide enough for `work` items, at most `jobs` (default:
    /// the machine's available parallelism). Each worker carries its own
    /// runtime — a kernel resource — so the pool never opens more than the
    /// work can occupy.
    pub fn new(jobs: Option<NonZeroUsize>, work: usize) -> Result<Self, String> {
        let jobs = jobs
            .or_else(|| std::thread::available_parallelism().ok())
            .unwrap_or(NonZeroUsize::MIN)
            .min(NonZeroUsize::new(work.max(1)).expect("max(1) is nonzero"));
        let dispatcher = Dispatcher::builder()
            .worker_threads(jobs)
            // One task per worker at a time — this is the pool's bound.
            .concurrent(false)
            .thread_names(|index| format!("rene-worker-{index}"))
            .build()
            .map_err(|e| format!("cannot start the process pool: {e}"))?;
        Ok(Pool { dispatcher, jobs })
    }

    /// The pool's width.
    pub fn jobs(&self) -> NonZeroUsize {
        self.jobs
    }

    /// Run one task on a pool worker; the returned future resolves on the
    /// calling runtime once the worker finishes it.
    pub async fn run<F, Fut, R>(&self, task: F) -> Result<R, String>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = R> + 'static,
        R: Send + 'static,
    {
        let receiver = self
            .dispatcher
            .dispatch(task)
            .map_err(|_| "the process pool is gone (its workers panicked)".to_owned())?;
        receiver
            .await
            .map_err(|_| "a pool worker dropped the task (panicked while running it)".to_owned())
    }

    /// Stop accepting work and wait for the workers to finish what they
    /// hold. Dropping the pool without this also stops it (the queue closes;
    /// threads wind down unjoined), which is fine for an exiting process.
    pub async fn shutdown(self) -> Result<(), String> {
        self.dispatcher
            .join()
            .await
            .map_err(|e| format!("the process pool did not shut down cleanly: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn block_on<T>(fut: impl Future<Output = T>) -> T {
        compio::runtime::Runtime::new()
            .expect("compio runtime")
            .block_on(fut)
    }

    /// Every dispatched task completes and returns its value.
    #[test]
    fn tasks_come_back() {
        block_on(async {
            let pool = Pool::new(Some(NonZeroUsize::new(3).unwrap()), 16).unwrap();
            let results = futures_util::future::try_join_all(
                (0u64..16).map(|i| pool.run(move || async move { i * 2 })),
            )
            .await
            .unwrap();
            assert_eq!(results, (0..16).map(|i| i * 2).collect::<Vec<_>>());
            pool.shutdown().await.unwrap();
        });
    }

    /// `concurrent(false)` with one worker means strict serialization in
    /// dispatch order: no task starts before the previous one ended.
    #[test]
    fn a_single_worker_serializes() {
        block_on(async {
            let pool = Pool::new(Some(NonZeroUsize::MIN), 4).unwrap();
            let log: Arc<Mutex<Vec<String>>> = Arc::default();
            futures_util::future::try_join_all((0..4).map(|i| {
                let log = log.clone();
                pool.run(move || async move {
                    log.lock().unwrap().push(format!("start {i}"));
                    // A real await point: an interleaving executor would slip
                    // the next task's start in here.
                    compio::runtime::time::sleep(std::time::Duration::from_millis(2)).await;
                    log.lock().unwrap().push(format!("end {i}"));
                })
            }))
            .await
            .unwrap();
            let log = log.lock().unwrap().clone();
            let expect: Vec<String> = (0..4)
                .flat_map(|i| [format!("start {i}"), format!("end {i}")])
                .collect();
            assert_eq!(log, expect, "tasks interleaved on a width-1 pool");
        });
    }

    /// Child processes spawn and complete on the workers' own runtimes.
    #[cfg(unix)]
    #[test]
    fn workers_run_processes() {
        block_on(async {
            let pool = Pool::new(Some(NonZeroUsize::new(2).unwrap()), 4).unwrap();
            let outputs = futures_util::future::try_join_all((0..4).map(|i| {
                pool.run(move || async move {
                    let out = compio::process::Command::new("sh")
                        .args(["-c", &format!("echo pooled-{i}")])
                        .stdout(std::process::Stdio::piped())
                        .expect("pipe stdout")
                        .output()
                        .await
                        .expect("spawn");
                    String::from_utf8_lossy(&out.stdout).trim().to_owned()
                })
            }))
            .await
            .unwrap();
            assert_eq!(outputs, ["pooled-0", "pooled-1", "pooled-2", "pooled-3"]);
        });
    }
}
