/* OpenMP driver for the atomic rc lifecycle (atomic_rc_omp_e2e.mlir): one
   atomic box, every thread cloning/reading/dropping it in a hot loop, and the
   main thread releasing the final reference after the region. The refcount
   traffic is the test — a lost decrement leaks the box (lsan), a duplicated
   one frees it under readers (asan), and any non-atomic count access is a
   data race (tsan).

   Each thread validates its own sum locally, so the ONLY cross-thread state
   is the box itself: every ordering claim then flows through the count's
   acquire/release chain — the exact property the lowering must provide. In
   particular the final drop's free is ordered after the workers' plain
   payload reads only by that chain, so a weaker-than-release decrement is a
   reportable race under TSan, not just a latent bug. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void *atomic_pair_create(int64_t, int64_t);
extern void atomic_pair_clone(void *);
extern int64_t atomic_pair_sum(void *);
extern void atomic_pair_drop(void *);

int main(void) {
  enum { THREADS = 8, ITERS = 20000 };
  void *box = atomic_pair_create(17, 25);
#pragma omp parallel num_threads(THREADS)
  {
    int64_t local = 0;
    for (int i = 0; i < ITERS; ++i) {
      atomic_pair_clone(box);
      local += atomic_pair_sum(box);
      atomic_pair_drop(box);
    }
    if (local != (int64_t)ITERS * 42) {
      fprintf(stderr, "thread sum=%lld, want %lld\n", (long long)local,
              (long long)ITERS * 42);
      abort();
    }
  }
  /* Release the creating reference; the box must die exactly here. */
  atomic_pair_drop(box);
  puts("all ok");
  return 0;
}
