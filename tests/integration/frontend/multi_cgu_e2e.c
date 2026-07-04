#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t run_cube_ffi(void);
extern int64_t run_sum_squares_ffi(void);
extern int64_t run_mixed_ffi(void);

#define ASSERT_EQ(name, actual, expected)                                      \
  do {                                                                         \
    if ((actual) != (expected)) {                                              \
      fprintf(stderr, "FAIL: %s: expected %lld, got %lld\n", name,             \
              (long long)(expected), (long long)(actual));                     \
      abort();                                                                 \
    }                                                                          \
  } while (0)

int main() {
  ASSERT_EQ("cube", run_cube_ffi(), 729);            // cube(9) = 9^3
  ASSERT_EQ("sum_squares", run_sum_squares_ffi(), 385); // 1^2 + ... + 10^2
  ASSERT_EQ("mixed", run_mixed_ffi(), 110);          // 64 + 16 + (1+4+9+16)
  return 0;
}
