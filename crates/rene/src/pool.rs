//! The process pool: `-j` admission for the build's child processes, all
//! monitored by the single main event loop.
//!
//! There are no worker threads. Child processes are external work — one
//! completion-based runtime drives any number of them concurrently — so the
//! pool reduces to the admission bound: a semaphore whose permit is held
//! from spawn to completion, keeping at most `jobs` processes in flight at
//! once. Concurrency comes from the callers awaiting several [`Pool::run`]s
//! at a time (`join_all`, the ready-queue pipeline); the permit only bounds
//! how many are past admission.
//!
//! Everything therefore stays on the thread that created the runtime:
//! futures never cross threads, and no cross-thread wakeup is needed for a
//! build to make progress.

use std::future::Future;
use std::num::NonZeroUsize;

use async_lock::Semaphore;

/// A `-j` admission bound over the main event loop.
pub struct Pool {
    permits: Semaphore,
    jobs: NonZeroUsize,
}

impl Pool {
    /// Create the pool with an admission bound of `jobs` (default: the
    /// machine's available parallelism).
    pub fn new(jobs: Option<NonZeroUsize>) -> Self {
        let jobs = jobs
            .or_else(|| std::thread::available_parallelism().ok())
            .unwrap_or(NonZeroUsize::MIN);
        Pool {
            permits: Semaphore::new(jobs.get()),
            jobs,
        }
    }

    /// The admission bound (`-j`).
    pub fn jobs(&self) -> NonZeroUsize {
        self.jobs
    }

    /// Run one task under a `-j` permit: acquired before the task starts,
    /// held until its future completes. The future runs on the caller's own
    /// runtime.
    pub async fn run<F, Fut, R>(&self, task: F) -> R
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = R>,
    {
        let _permit = self.permits.acquire().await;
        task().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn block_on<T>(fut: impl Future<Output = T>) -> T {
        compio::runtime::Runtime::new()
            .expect("compio runtime")
            .block_on(fut)
    }

    /// Every task completes and returns its value.
    #[test]
    fn tasks_come_back() {
        block_on(async {
            let pool = Pool::new(Some(NonZeroUsize::new(3).unwrap()));
            let results = futures_util::future::join_all(
                (0u64..16).map(|i| pool.run(move || async move { i * 2 })),
            )
            .await;
            assert_eq!(results, (0..16).map(|i| i * 2).collect::<Vec<_>>());
        });
    }

    /// `-j 1` strictly serializes in admission order — no task starts before
    /// the previous one ended, even across a real await point.
    #[test]
    fn a_single_permit_serializes() {
        block_on(async {
            let pool = Pool::new(Some(NonZeroUsize::MIN));
            let log: Arc<Mutex<Vec<String>>> = Arc::default();
            futures_util::future::join_all((0..4).map(|i| {
                let log = log.clone();
                pool.run(move || async move {
                    log.lock().unwrap().push(format!("start {i}"));
                    compio::runtime::time::sleep(std::time::Duration::from_millis(2)).await;
                    log.lock().unwrap().push(format!("end {i}"));
                })
            }))
            .await;
            let log = log.lock().unwrap().clone();
            let expect: Vec<String> = (0..4)
                .flat_map(|i| [format!("start {i}"), format!("end {i}")])
                .collect();
            assert_eq!(log, expect, "tasks interleaved under -j 1");
        });
    }

    /// The `-j` bound holds under eager admission: with 2 permits and 8
    /// tasks, at most 2 are ever in flight.
    #[test]
    fn the_permit_bound_holds() {
        block_on(async {
            let pool = Pool::new(Some(NonZeroUsize::new(2).unwrap()));
            let in_flight = Arc::new(AtomicUsize::new(0));
            let high_water = Arc::new(AtomicUsize::new(0));
            futures_util::future::join_all((0..8).map(|_| {
                let in_flight = in_flight.clone();
                let high_water = high_water.clone();
                pool.run(move || async move {
                    let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    high_water.fetch_max(now, Ordering::SeqCst);
                    compio::runtime::time::sleep(std::time::Duration::from_millis(2)).await;
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                })
            }))
            .await;
            assert!(high_water.load(Ordering::SeqCst) <= 2);
        });
    }

    /// One event loop interleaves up to the admission bound: with 8 permits
    /// and 8 tasks, every task starts before the first one ends.
    #[test]
    fn one_thread_interleaves_to_the_bound() {
        block_on(async {
            let pool = Pool::new(Some(NonZeroUsize::new(8).unwrap()));
            let in_flight = Arc::new(AtomicUsize::new(0));
            let high_water = Arc::new(AtomicUsize::new(0));
            futures_util::future::join_all((0..8).map(|_| {
                let in_flight = in_flight.clone();
                let high_water = high_water.clone();
                pool.run(move || async move {
                    let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    high_water.fetch_max(now, Ordering::SeqCst);
                    // Long enough that every task starts before the first
                    // one ends.
                    compio::runtime::time::sleep(std::time::Duration::from_millis(100)).await;
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                })
            }))
            .await;
            let peak = high_water.load(Ordering::SeqCst);
            assert_eq!(peak, 8, "the single loop failed to interleave: peak {peak}");
        });
    }

    /// Child processes spawn and complete on the main event loop.
    #[cfg(unix)]
    #[test]
    fn processes_run_on_the_main_loop() {
        block_on(async {
            let pool = Pool::new(Some(NonZeroUsize::new(4).unwrap()));
            let outputs = futures_util::future::join_all((0..4).map(|i| {
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
            .await;
            assert_eq!(outputs, ["pooled-0", "pooled-1", "pooled-2", "pooled-3"]);
        });
    }
}
