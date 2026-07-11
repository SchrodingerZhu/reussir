#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t wpd_single_ffi(int64_t n);
extern int32_t wpd_multi_ffi(int32_t n);

int main(void) {
  /* Single-impl family, applied twice: n * 2 * 2 (devirtualized under WPD,
   * plainly indirect without). */
  if (wpd_single_ffi(21) != 84) {
    fprintf(stderr, "FAIL: single\n");
    abort();
  }
  /* Two-impl family: 2n + 3n + 2*(3n) — exercises the residual-indirect
   * path. */
  if (wpd_multi_ffi(10) != 110) {
    fprintf(stderr, "FAIL: multi\n");
    abort();
  }
  return 0;
}
