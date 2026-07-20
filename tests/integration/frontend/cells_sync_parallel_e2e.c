#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* The exported C-ABI trampolines follow reussir's own convention, not the
   platform aggregate ABI: a struct result is returned through a leading
   sret pointer, and a multi-field argument list is passed as one pointer to
   a packed argument struct. A single scalar argument with a scalar result
   is passed and returned directly. */
struct worker_args {
  void *stats;
  int64_t tid;
  int64_t stride;
  int64_t upto;
};
struct dup_arg {
  void *s;
};
struct pair {
  void *keep;
  void *extra;
};

extern void *make_stats_ffi(void);
extern void dup_ffi(struct pair *out, struct dup_arg *in);
extern int64_t worker_ffi(struct worker_args *);
extern int64_t grand_total_ffi(void *);

enum { THREADS = 4, KEYS = 4000 };

int main(void) {
  void *stats = make_stats_ffi(); /* one owned reference */
  /* Split it into THREADS worker handles plus one for the final fold, each
     an owned reference consumed exactly once — no leak, no over-release. */
  void *handles[THREADS];
  for (int i = 0; i < THREADS; ++i) {
    struct dup_arg in = {stats};
    struct pair out;
    dup_ffi(&out, &in);
    handles[i] = out.extra;
    stats = out.keep;
  }

  omp_set_num_threads(THREADS);
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    struct worker_args a = {handles[tid], tid, THREADS, KEYS};
    worker_ffi(&a); /* consumes handles[tid] */
  }

  int64_t got = grand_total_ffi(stats); /* consumes the last reference */
  int64_t want = (int64_t)KEYS * (KEYS - 1) / 2;
  if (got != want) {
    fprintf(stderr, "grand total: got %lld, want %lld\n", (long long)got,
            (long long)want);
    return EXIT_FAILURE;
  }
  printf("parallel shared map: %lld\n", (long long)got);
  return EXIT_SUCCESS;
}
