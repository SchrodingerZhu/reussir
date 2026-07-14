#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int32_t vapp_test_ffi(void);

int main(void) {
  int32_t got = vapp_test_ffi();
  if (got != 1) {
    fprintf(stderr, "FAIL: expected 1, got %d\n", got);
    abort();
  }
  return 0;
}
