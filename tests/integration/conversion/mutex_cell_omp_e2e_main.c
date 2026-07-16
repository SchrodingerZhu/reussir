/* OpenMP driver for the mutex cell lowering (mutex_cell_omp_e2e.mlir): one
   mutex-guarded i64 counter, every thread incrementing it through the
   region-form `cell.rmw` in a hot loop. Each increment acquires the cell's
   mutex, moves the element through the rmw body, and stores the replacement
   before the unlock, so mutual exclusion is the property under test: any
   fast-path/slow-path defect in the lowered lock — two threads inside the
   critical section, a lost wake, a torn store — surfaces as a wrong final
   count instead of requiring a sanitizer.

   The fetched old values are only sanity-checked for range: under contention
   their order is nondeterministic, but every one of them must be a count some
   prefix of increments produced. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void *mutex_counter_create(int64_t);
extern int64_t mutex_counter_fetch_add(void *, int64_t);
extern int64_t mutex_counter_get(void *);
extern void mutex_counter_set(void *, int64_t);
extern void mutex_counter_drop(void *);

int main(void) {
  enum { THREADS = 8, ITERS = 20000 };
  const int64_t total = (int64_t)THREADS * ITERS;
  void *cell = mutex_counter_create(0);
#pragma omp parallel num_threads(THREADS)
  {
    for (int i = 0; i < ITERS; ++i) {
      int64_t old = mutex_counter_fetch_add(cell, 1);
      if (old < 0 || old >= total) {
        fprintf(stderr, "fetched old=%lld outside [0, %lld)\n", (long long)old,
                (long long)total);
        abort();
      }
    }
  }
  /* The parallel region's join orders every increment before this read. */
  int64_t count = mutex_counter_get(cell);
  if (count != total) {
    fprintf(stderr, "count=%lld, want %lld\n", (long long)count,
            (long long)total);
    abort();
  }
  /* Whole-element set/get through the same lock. */
  mutex_counter_set(cell, -7);
  if (mutex_counter_get(cell) != -7) {
    fprintf(stderr, "set/get roundtrip failed\n");
    abort();
  }
  /* Release the creating reference; the box must die exactly here. */
  mutex_counter_drop(cell);
  puts("all ok");
  return 0;
}
