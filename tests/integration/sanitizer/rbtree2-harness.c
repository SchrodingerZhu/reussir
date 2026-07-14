#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int32_t fold_test_ffi(int32_t size);

int main(void) {
  const int32_t n = 1000;
  int32_t p = fold_test_ffi(n);
  if (p != 100) {
    fprintf(stderr, "FAIL: expected 100, got %d\n", (int)p);
    abort();
  }
  return 0;
}
