#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t mixed_arms_test_ffi(int64_t n);

int main(void) {
  /* rotate(One{i}, 3, 0): One(i) -> Two{i,i+1} -> Three{i,i+1,2i+1} ->
     One{3i+1}, acc 3; total adds 3i+1 => 3i+4 per iteration.
     sum_{i=1..200000} (3i+4) = 3*200000*200001/2 + 4*200000. */
  int64_t r = mixed_arms_test_ffi(200000);
  int64_t expected = 3 * (200000LL * 200001LL / 2) + 4 * 200000LL;
  if (r != expected) {
    fprintf(stderr, "FAIL: expected %lld, got %lld\n", (long long)expected,
            (long long)r);
    abort();
  }
  return 0;
}
