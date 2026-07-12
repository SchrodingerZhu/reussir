#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t queue_test_ffi(void);

/* 64 interleaved rounds enqueue 0..127 and everything is dequeued; the sum
   0+1+...+127 = 8128. */
int main(void) {
  int64_t got = queue_test_ffi();
  if (got != 8128) {
    fprintf(stderr, "hm_queue: got %lld, want 8128\n", (long long)got);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
