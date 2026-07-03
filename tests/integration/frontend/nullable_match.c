#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int32_t read_wrapped_ffi(int32_t);
extern int32_t read_empty_ffi(void);
extern int32_t drop_unmatched_ffi(int32_t);
extern uint64_t second_ffi(uint64_t);
extern uint64_t end_of_list_ffi(uint64_t);

#define EXPECT(what, got, want)                                                \
  do {                                                                         \
    long long g = (long long)(got), w = (long long)(want);                     \
    if (g != w) {                                                              \
      fprintf(stderr, "FAIL %s: got %lld want %lld\n", what, g, w);            \
      abort();                                                                 \
    }                                                                          \
  } while (0)

int main(void) {
  EXPECT("read_wrapped", read_wrapped_ffi(42), 42);
  EXPECT("read_empty", read_empty_ffi(), -1);
  EXPECT("drop_unmatched", drop_unmatched_ffi(5), 7);
  EXPECT("second", second_ffi(41), 42);           /* tail.v = v + 1 */
  EXPECT("end_of_list", end_of_list_ffi(7), 108); /* tail.v + 100 */
  /* Second round: surface double-free / use-after-free in the rc paths. */
  EXPECT("read_wrapped#2", read_wrapped_ffi(-3), -3);
  EXPECT("second#2", second_ffi(0), 1);
  return 0;
}
