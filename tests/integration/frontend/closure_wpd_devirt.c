#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t wpd_single_ffi(int64_t n);
extern int32_t wpd_multi_ffi(int32_t n);

int main(void) {
  /* Single-impl family: devirtualized (or plainly indirect without WPD). */
  if (wpd_single_ffi(21) != 42) {
    fprintf(stderr, "FAIL: single\n");
    abort();
  }
  /* Two-impl family: exercises the branch-funnel / residual-indirect path. */
  if (wpd_multi_ffi(10) != 50) {
    fprintf(stderr, "FAIL: multi\n");
    abort();
  }
  return 0;
}
