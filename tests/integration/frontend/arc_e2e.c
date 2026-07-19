#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t test_share_ffi(void);
extern int64_t test_enum_ffi(void);
extern int64_t test_nullable_ffi(void);
extern int64_t test_generic_ffi(void);
extern int64_t test_recursive_ffi(void);
extern int64_t test_tree_ffi(void);

static int check(const char *name, int64_t got, int64_t want) {
  if (got == want)
    return 0;
  fprintf(stderr, "%s: got %lld, want %lld\n", name, (long long)got,
          (long long)want);
  return 1;
}

int main(void) {
  int failures = 0;
  failures += check("share", test_share_ffi(), 42);
  failures += check("enum", test_enum_ffi(), 42);
  failures += check("nullable", test_nullable_ffi(), 11);
  failures += check("generic", test_generic_ffi(), 9);
  failures += check("recursive", test_recursive_ffi(), 42);
  failures += check("tree", test_tree_ffi(), 42);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
