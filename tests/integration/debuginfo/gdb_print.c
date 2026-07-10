#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t gdb_print_ffi(int64_t n);

int main(void) {
  /* Three shape probes (42 each), list depth 3, snapshot 7, packed sum
     (700 + 1), and the four observers 1 + 2 + 3 + 4. */
  int64_t r = gdb_print_ffi(7);
  if (r != 3 * 42 + 3 + 7 + 701 + 10) {
    fprintf(stderr, "FAIL: got %lld\n", (long long)r);
    abort();
  }
  return 0;
}
