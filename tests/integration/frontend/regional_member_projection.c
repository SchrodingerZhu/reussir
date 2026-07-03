#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern uint64_t read_both_ffi(uint64_t);
extern uint64_t use_inner_ffi(uint64_t);

#define EXPECT(what, got, want)                                                \
  do {                                                                         \
    uint64_t g = (got), w = (want);                                            \
    if (g != w) {                                                              \
      fprintf(stderr, "FAIL %s: got %llu want %llu\n", what,                   \
              (unsigned long long)g, (unsigned long long)w);                   \
      abort();                                                                 \
    }                                                                          \
  } while (0)

int main(void) {
  EXPECT("read_both", read_both_ffi(41), 41 + 42);
  EXPECT("use_inner", use_inner_ffi(9), 9);
  /* Run a second round so a double-free / use-after-free in the rc
     bookkeeping of the frozen graphs has a chance to surface. */
  EXPECT("read_both#2", read_both_ffi(100), 100 + 101);
  EXPECT("use_inner#2", use_inner_ffi(0), 0);
  return 0;
}
