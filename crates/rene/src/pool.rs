//! The process pool: bounded fan-out for the build's child processes, in
//! two independent layers.
//!
//! **Workers** are a handful of threads (at most [`WORKERS`]), each running
//! its own completion-based runtime, dispatched to `concurrent(true)`: a
//! worker spawns every task it receives and interleaves them — child
//! processes are external work, so one event loop drives many of them. The
//! worker count is an I/O plumbing choice, not a parallelism knob.
//!
//! **`-j` (jobs)** is the admission bound: a semaphore whose permit is held
//! from dispatch to completion, so at most `jobs` processes are in flight
//! at once no matter how the workers interleave. It is deliberately not
//! tied to the worker count — the cross-package build gates dispatch by
//! dependency readiness anyway, and the two constraints (what may run, and
//! what drives its I/O) move independently.

use std::future::Future;
use std::num::NonZeroUsize;
use std::sync::Arc;

use async_lock::Semaphore;
use compio::dispatcher::Dispatcher;

/// The worker-thread cap: event loops, not parallelism — each holds a
/// kernel I/O instance (io_uring/IOCP), and a few drive any number of
/// child processes.
const WORKERS: usize = 4;

/// A pool of process-driving workers behind a `-j` admission bound.
pub struct Pool {
    dispatcher: Dispatcher,
    permits: Arc<Semaphore>,
    jobs: NonZeroUsize,
}

impl Pool {
    /// Start the pool with an admission bound of `jobs` (default: the
    /// machine's available parallelism).
    pub fn new(jobs: Option<NonZeroUsize>) -> Result<Self, String> {
        let jobs = jobs
            .or_else(|| std::thread::available_parallelism().ok())
            .unwrap_or(NonZeroUsize::MIN);
        let workers = jobs.min(NonZeroUsize::new(WORKERS).expect("nonzero"));
        let dispatcher = Dispatcher::builder()
            .worker_threads(workers)
            // Workers interleave their tasks; the bound lives in `permits`.
            .concurrent(true)
            .thread_names(|index| format!("rene-worker-{index}"))
            .build()
            .map_err(|e| format!("cannot start the process pool: {e}"))?;
        Ok(Pool {
            dispatcher,
            permits: Arc::new(Semaphore::new(jobs.get())),
            jobs,
        })
    }

    /// The admission bound (`-j`).
    pub fn jobs(&self) -> NonZeroUsize {
        self.jobs
    }

    /// Run one task under a `-j` permit: acquired before dispatch, held
    /// until the worker finishes the task, released when the result comes
    /// back. The returned future resolves on the calling runtime.
    pub async fn run<F, Fut, R>(&self, task: F) -> Result<R, String>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = R> + 'static,
        R: Send + 'static,
    {
        let permit = self.permits.acquire_arc().await;
        let receiver = self
            .dispatcher
            .dispatch(move || {
                let fut = task();
                async move {
                    let result = fut.await;
                    // Explicit: the permit outlives the task, not the dispatch.
                    drop(permit);
                    result
                }
            })
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
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn block_on<T>(fut: impl Future<Output = T>) -> T {
        compio::runtime::Runtime::new()
            .expect("compio runtime")
            .block_on(fut)
    }

    /// Every dispatched task completes and returns its value.
    #[test]
    fn tasks_come_back() {
        block_on(async {
            let pool = Pool::new(Some(NonZeroUsize::new(3).unwrap())).unwrap();
            let results = futures_util::future::try_join_all(
                (0u64..16).map(|i| pool.run(move || async move { i * 2 })),
            )
            .await
            .unwrap();
            assert_eq!(results, (0..16).map(|i| i * 2).collect::<Vec<_>>());
            pool.shutdown().await.unwrap();
        });
    }

    /// `-j 1` strictly serializes in dispatch order — the permit, not the
    /// worker count, is the bound: no task starts before the previous one
    /// ended, even across a real await point.
    #[test]
    fn a_single_permit_serializes() {
        block_on(async {
            let pool = Pool::new(Some(NonZeroUsize::MIN)).unwrap();
            let log: Arc<Mutex<Vec<String>>> = Arc::default();
            futures_util::future::try_join_all((0..4).map(|i| {
                let log = log.clone();
                pool.run(move || async move {
                    log.lock().unwrap().push(format!("start {i}"));
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
            assert_eq!(log, expect, "tasks interleaved under -j 1");
        });
    }

    /// The `-j` bound holds however the workers interleave: with 2 permits
    /// and 8 eager tasks, at most 2 are ever in flight.
    #[test]
    fn the_permit_bound_holds() {
        block_on(async {
            let pool = Pool::new(Some(NonZeroUsize::new(2).unwrap())).unwrap();
            let in_flight = Arc::new(AtomicUsize::new(0));
            let high_water = Arc::new(AtomicUsize::new(0));
            futures_util::future::try_join_all((0..8).map(|_| {
                let in_flight = in_flight.clone();
                let high_water = high_water.clone();
                pool.run(move || async move {
                    let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    high_water.fetch_max(now, Ordering::SeqCst);
                    compio::runtime::time::sleep(std::time::Duration::from_millis(2)).await;
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                })
            }))
            .await
            .unwrap();
            assert!(high_water.load(Ordering::SeqCst) <= 2);
        });
    }

    /// Workers interleave beyond their own count: more tasks run at once
    /// than there are worker threads, because `concurrent(true)` spawns and
    /// keeps receiving. (The bound is `-j`, which here admits all eight.)
    #[test]
    fn workers_interleave_many_tasks() {
        block_on(async {
            let pool = Pool::new(Some(NonZeroUsize::new(8).unwrap())).unwrap();
            let in_flight = Arc::new(AtomicUsize::new(0));
            let high_water = Arc::new(AtomicUsize::new(0));
            futures_util::future::try_join_all((0..8).map(|_| {
                let in_flight = in_flight.clone();
                let high_water = high_water.clone();
                pool.run(move || async move {
                    let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    high_water.fetch_max(now, Ordering::SeqCst);
                    // Long enough that every task starts before the first
                    // one ends, whatever thread it landed on.
                    compio::runtime::time::sleep(std::time::Duration::from_millis(100)).await;
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                })
            }))
            .await
            .unwrap();
            let peak = high_water.load(Ordering::SeqCst);
            assert!(
                peak > WORKERS,
                "peak {peak} never exceeded the worker count {WORKERS}"
            );
        });
    }

    /// Child processes spawn and complete on the workers' own runtimes.
    #[cfg(unix)]
    #[test]
    fn workers_run_processes() {
        block_on(async {
            let pool = Pool::new(Some(NonZeroUsize::new(4).unwrap())).unwrap();
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
