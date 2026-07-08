#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t tail_spin_ffi(int64_t depth);

int main(void) {
  /* Far more frames than the default 8 MiB stack can hold: only a
     preserved tail call survives this depth. */
  int64_t r = tail_spin_ffi(50000000);
  if (r != 0) {
    fprintf(stderr, "FAIL: expected 0, got %lld\n", (long long)r);
    abort();
  }
  return 0;
}
