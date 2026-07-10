#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t packed_print_ffi(int64_t n);

int main(void) {
  /* probe: sum = n + 1, twice = 2 * sum, result twice - sum = sum. */
  int64_t r = packed_print_ffi(500);
  if (r != 501) {
    fprintf(stderr, "FAIL: got %lld\n", (long long)r);
    abort();
  }
  return 0;
}
