/* OpenMP driver for the rwlock cell lowering (rwlock_cell_omp_e2e.mlir): one
   rwlock-guarded i64 counter, writer threads incrementing it through the
   region-form `cell.rmw` (a write critical section) while reader threads
   snapshot it through `cell.rdlock` (a read critical section). Mutual
   exclusion between writers is the first property under test: any lost or
   torn update surfaces as a wrong final count. The read path is the second:
   a snapshot taken under the shared lock must be a count some prefix of
   increments produced — inside [0, total] and, per reader thread,
   non-decreasing, since the counter only grows during the parallel phase.

   The snapshot transaction returns the value doubled (the driver halves it)
   so the multiply demonstrably runs inside the read section rather than
   after it. */
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void *rwlock_counter_create(int64_t);
extern int64_t rwlock_counter_fetch_add(void *, int64_t);
extern int64_t rwlock_counter_snapshot(void *);
extern int64_t rwlock_counter_get(void *);
extern void rwlock_counter_set(void *, int64_t);
extern void rwlock_counter_drop(void *);

int main(void) {
  enum { WRITERS = 4, READERS = 4, ITERS = 20000 };
  const int64_t total = (int64_t)WRITERS * ITERS;
  void *cell = rwlock_counter_create(0);
#pragma omp parallel num_threads(WRITERS + READERS)
  {
    int tid = omp_get_thread_num();
    if (tid < WRITERS) {
      for (int i = 0; i < ITERS; ++i) {
        int64_t old = rwlock_counter_fetch_add(cell, 1);
        if (old < 0 || old >= total) {
          fprintf(stderr, "fetched old=%lld outside [0, %lld)\n",
                  (long long)old, (long long)total);
          abort();
        }
      }
    } else {
      int64_t last = 0;
      for (int i = 0; i < ITERS; ++i) {
        int64_t doubled = rwlock_counter_snapshot(cell);
        if (doubled % 2 != 0) {
          fprintf(stderr, "snapshot returned odd doubled value %lld\n",
                  (long long)doubled);
          abort();
        }
        int64_t seen = doubled / 2;
        if (seen < last || seen > total) {
          fprintf(stderr, "snapshot %lld regressed from %lld or exceeds %lld\n",
                  (long long)seen, (long long)last, (long long)total);
          abort();
        }
        last = seen;
      }
    }
  }
  /* The parallel region's join orders every increment before this read. */
  int64_t count = rwlock_counter_get(cell);
  if (count != total) {
    fprintf(stderr, "count=%lld, want %lld\n", (long long)count,
            (long long)total);
    abort();
  }
  /* Whole-element set/get through the write and read sides of the lock. */
  rwlock_counter_set(cell, -7);
  if (rwlock_counter_get(cell) != -7) {
    fprintf(stderr, "set/get roundtrip failed\n");
    abort();
  }
  /* Release the creating reference; the box must die exactly here. */
  rwlock_counter_drop(cell);
  puts("all ok");
  return 0;
}
