#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Float results cross the C-ABI trampoline via an out-parameter (only
   void/integer/pointer returns are direct). */
extern void test_tabulate_sum_ffi(double *);
extern int64_t test_splat_get_ffi(void);
extern void test_unique_update_ffi(double *);
extern void test_cow_ffi(double *);
extern int64_t test_rank2_ffi(void);
extern void test_stencil_ffi(double *);

static int check_f64(const char *name, void (*fn)(double *), double want) {
  double got = 0.0;
  fn(&got);
  if (got != want) {
    fprintf(stderr, "%s: got %f, want %f\n", name, got, want);
    return 1;
  }
  return 0;
}

static int check_i64(const char *name, int64_t got, int64_t want) {
  if (got != want) {
    fprintf(stderr, "%s: got %lld, want %lld\n", name, (long long)got,
            (long long)want);
    return 1;
  }
  return 0;
}

int main(void) {
  int failures = 0;
  failures += check_f64("tabulate_sum", test_tabulate_sum_ffi, 28.0);
  failures += check_i64("splat_get", test_splat_get_ffi(), 7);
  failures += check_f64("unique_update", test_unique_update_ffi, 38.0);
  failures += check_f64("cow", test_cow_ffi, 100.0);
  /* sum(i*10+j over 4x4) = 10*6*4 + 6*4 = 264; m[2][3] = 23 => 287 */
  failures += check_i64("rank2", test_rank2_ffi(), 287);
  /* y[0] = 3, y[1..7] = 3 => 24 */
  failures += check_f64("stencil", test_stencil_ffi, 24.0);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
