#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int32_t fold_test_ffi(int32_t size);

int main(void) {
  const int32_t n = 4200000;
  int32_t p = fold_test_ffi(n);
  if (p != 420000) {
    fprintf(stderr, "FAIL: expected 420000, got %d\n", (int32_t)p);
    abort();
  }
  return 0;
}
