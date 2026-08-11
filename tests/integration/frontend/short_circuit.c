#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Each case reports `evaluations of the right operand * 10 + result bit`, so
   a failure says whether the answer or the path to it was wrong. */
extern int64_t and_short_ffi(void);
extern int64_t and_full_ffi(void);
extern int64_t or_short_ffi(void);
extern int64_t or_full_ffi(void);
extern int64_t and_chain_ffi(void);
extern int64_t mixed_chain_ffi(void);
extern int64_t guarded_zero_ffi(void);
extern int64_t guarded_nonzero_ffi(void);
extern int64_t truth_table_ffi(void);

static void expect(const char *what, int64_t got, int64_t want) {
  if (got != want) {
    fprintf(stderr, "FAIL: %s = %lld, want %lld\n", what, (long long)got,
            (long long)want);
    abort();
  }
}

int main(void) {
  /* `false && p` — p never runs, result false */
  expect("and_short", and_short_ffi(), 0);
  /* `true && p` — p runs once and decides it */
  expect("and_full", and_full_ffi(), 11);
  /* `true || p` — p never runs, result true */
  expect("or_short", or_short_ffi(), 1);
  /* `false || p` — p runs once and decides it */
  expect("or_full", or_full_ffi(), 10);
  /* the first false stops the rest of the chain: one evaluation, not three */
  expect("and_chain", and_chain_ffi(), 10);
  /* `true || (p && p)` skips both operands of the right side */
  expect("mixed_chain", mixed_chain_ffi(), 1);
  /* the guard holds: a strict `&&` would divide by zero here instead */
  expect("guarded_zero", guarded_zero_ffi(), 0);
  expect("guarded_nonzero", guarded_nonzero_ffi(), 1);
  /* and/or over the full truth table: 0b1000 and 0b1110 */
  expect("truth_table", truth_table_ffi(), 8 * 100 + 14);
  return 0;
}
