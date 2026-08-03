#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t impl_value_ffi(void);
extern int64_t impl_arc_ffi(void);
extern int64_t impl_regional_ffi(void);
extern int64_t impl_generic_ffi(void);

static int check(const char *name, int64_t got, int64_t want) {
  if (got == want)
    return 0;
  fprintf(stderr, "%s: got %lld, want %lld\n", name, (long long)got,
          (long long)want);
  return 1;
}

int main(void) {
  int failures = 0;
  failures += check("value", impl_value_ffi(), 42);
  failures += check("arc", impl_arc_ffi(), 42);
  /* bump: 3 + 1 = 4, refrozen, read: 4 * 10 = 40 */
  failures += check("regional", impl_regional_ffi(), 40);
  /* 21 * 2 + 7 * 3 = 63 */
  failures += check("generic", impl_generic_ffi(), 63);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
