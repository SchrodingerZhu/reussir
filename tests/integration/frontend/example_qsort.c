#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t qsort_test_ffi(int64_t);

/* Checksum of two sort rounds (seeds 43 and 44); the reference value comes
   from an independent sort of the same MINSTD sequences. A -1 means the
   result was not sorted. */
int main(void) {
  int64_t got = qsort_test_ffi(2);
  if (got != 738424691) {
    fprintf(stderr, "qsort(2): got %lld, want 738424691\n", (long long)got);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
