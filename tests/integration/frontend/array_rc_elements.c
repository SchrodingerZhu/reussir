#include <stdio.h>
#include <stdlib.h>

extern long test_splat_get_ffi(void);
extern long test_cow_ffi(void);
extern long test_unique_overwrite_ffi(void);
extern long test_rank2_ffi(void);
extern long test_rank2_cow_ffi(void);
extern long test_roundtrip_ffi(void);
extern long test_generic_ffi(void);

static void expect(const char *name, long got, long want) {
  if (got != want) {
    fprintf(stderr, "%s: got %ld, want %ld\n", name, got, want);
    exit(1);
  }
}

int main(void) {
  expect("splat_get", test_splat_get_ffi(), 14);
  expect("cow", test_cow_ffi(), 20);
  expect("unique_overwrite", test_unique_overwrite_ffi(), 111);
  expect("rank2", test_rank2_ffi(), 40);
  expect("rank2_cow", test_rank2_cow_ffi(), 10);
  expect("roundtrip", test_roundtrip_ffi(), 31);
  expect("generic", test_generic_ffi(), 3);
  return 0;
}
