#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t test_scalar_ffi(void);
extern int64_t test_managed_element_ffi(void);
extern int64_t test_nested_cell_ffi(void);
extern int64_t test_record_member_ffi(void);
extern int64_t test_generic_ffi(void);
extern int64_t test_in_use_ffi(void);
extern int64_t test_rc_cycle_ffi(void);

static int check(const char *name, int64_t got, int64_t want) {
  if (got == want)
    return 0;
  fprintf(stderr, "%s: got %lld, want %lld\n", name, (long long)got,
          (long long)want);
  return 1;
}

int main(void) {
  int failures = 0;
  failures += check("scalar", test_scalar_ffi(), 32);
  failures += check("managed_element", test_managed_element_ffi(), 16);
  failures += check("nested_cell", test_nested_cell_ffi(), 24);
  failures += check("record_member", test_record_member_ffi(), 42);
  failures += check("generic", test_generic_ffi(), 41);
  failures += check("in_use", test_in_use_ffi(), 7);
  failures += check("rc_cycle", test_rc_cycle_ffi(), 42);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
