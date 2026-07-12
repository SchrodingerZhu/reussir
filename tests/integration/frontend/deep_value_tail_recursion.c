#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t deep_test_ffi(int64_t n);

int main(void) {
  // sum(1..1000000) mod 1000000007 — one million recursion frames unless the
  // self tail call runs in constant stack.
  int64_t result = deep_test_ffi(1000000);
  if (result != 496500) {
    fprintf(stderr, "FAIL: expected 496500, got %lld\n", (long long)result);
    abort();
  }
  return 0;
}
