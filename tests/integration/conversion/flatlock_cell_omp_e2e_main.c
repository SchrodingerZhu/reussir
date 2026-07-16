/* OpenMP driver for the flatlock cell lowering (flatlock_cell_omp_e2e.mlir):
   one flat-combining-lock-guarded i64 counter, every thread incrementing it
   through the region-form `cell.rmw` in a hot loop. Each increment either
   runs inline under the acquired status flag or is packaged as a node and
   served by the combining thread, so the protocol under test is the full
   flat-combining path: request capture, attach, combination, and wake-up.
   Any dropped or duplicated request surfaces as a wrong final count.

   The fetched old values are only sanity-checked for range: under contention
   their order is nondeterministic, but every one of them must be a count some
   prefix of increments produced. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void *flatlock_counter_create(int64_t);
extern int64_t flatlock_counter_fetch_add(void *, int64_t);
extern int64_t flatlock_counter_get(void *);
extern void flatlock_counter_set(void *, int64_t);
extern void flatlock_counter_drop(void *);

int main(void) {
  enum { THREADS = 8, ITERS = 20000 };
  const int64_t total = (int64_t)THREADS * ITERS;
  void *cell = flatlock_counter_create(0);
#pragma omp parallel num_threads(THREADS)
  {
    for (int i = 0; i < ITERS; ++i) {
      int64_t old = flatlock_counter_fetch_add(cell, 1);
      if (old < 0 || old >= total) {
        fprintf(stderr, "fetched old=%lld outside [0, %lld)\n", (long long)old,
                (long long)total);
        abort();
      }
    }
  }
  /* The parallel region's join orders every increment before this read. */
  int64_t count = flatlock_counter_get(cell);
  if (count != total) {
    fprintf(stderr, "count=%lld, want %lld\n", (long long)count,
            (long long)total);
    abort();
  }
  /* Whole-element set/get through the same lock. */
  flatlock_counter_set(cell, -7);
  if (flatlock_counter_get(cell) != -7) {
    fprintf(stderr, "set/get roundtrip failed\n");
    abort();
  }
  /* Release the creating reference; the box must die exactly here. */
  flatlock_counter_drop(cell);
  puts("all ok");
  return 0;
}
