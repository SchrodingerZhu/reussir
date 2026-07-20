#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int64_t test_mutex_ffi(void);
extern int64_t test_atomic_ffi(void);
extern int64_t test_flatlock_ffi(void);
extern int64_t test_rwlock_ffi(void);

static int check(const char *name, int64_t got, int64_t want) {
  if (got == want)
    return 0;
  fprintf(stderr, "%s: got %lld, want %lld\n", name, (long long)got,
          (long long)want);
  return 1;
}

int main(void) {
  int failures = 0;
  failures += check("mutex", test_mutex_ffi(), 42);
  failures += check("atomic", test_atomic_ffi(), 42);
  failures += check("flatlock", test_flatlock_ffi(), 33);
  failures += check("rwlock", test_rwlock_ffi(), 42);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
