#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t variant_print_ffi(int64_t n);

int main(void) {
  /* Keeps the debugged program honest: three shape probes (42 each), the
     list/tree/snapshot probes (3 + 11 + 7). */
  int64_t r = variant_print_ffi(7);
  if (r != 3 * 42 + 3 + 11 + 7) {
    fprintf(stderr, "FAIL: got %lld\n", (long long)r);
    abort();
  }
  return 0;
}
