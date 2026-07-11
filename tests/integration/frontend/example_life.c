#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t life_test_ffi(int64_t);

/* Population after 100 generations of the hash-seeded soup; the reference
   value comes from an independent simulation of the same rule and seed. */
int main(void) {
  int64_t got = life_test_ffi(100);
  if (got != 520) {
    fprintf(stderr, "life(100): got %lld, want 520\n", (long long)got);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
