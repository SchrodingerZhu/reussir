#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t advanced_print_ffi(int64_t n);

int main(void) {
  /* The four observers return 1 + 2 + 3 + 4. */
  int64_t r = advanced_print_ffi(7);
  if (r != 10) {
    fprintf(stderr, "FAIL: got %lld\n", (long long)r);
    abort();
  }
  return 0;
}
