#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t fold_then_traverse_ffi(void);
extern int64_t traverse_twice_ffi(void);
extern int64_t fold_then_peek_ffi(void);

static void expect(const char *what, int64_t got, int64_t want) {
  if (got != want) {
    fprintf(stderr, "FAIL: %s = %lld, want %lld\n", what, (long long)got,
            (long long)want);
    abort();
  }
}

int main(void) {
  /* one digit per observation: a failure names the check that broke */
  expect("fold_then_traverse", fold_then_traverse_ffi(), 1111);
  expect("traverse_twice", traverse_twice_ffi(), 11);
  expect("fold_then_peek", fold_then_peek_ffi(), 11);
  return 0;
}
