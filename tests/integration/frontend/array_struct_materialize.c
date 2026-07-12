#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Float results cross the C-ABI trampoline via an out-parameter (only
   void/integer/pointer returns are direct). */
extern void test_shared_ffi(double *);
extern int64_t test_value_ffi(void);
extern void test_enum_blob_ffi(double *);
extern void test_enum_empty_ffi(double *);
extern void test_cow_field_ffi(double *);
extern void test_multi_ffi(double *);
extern int64_t test_generic_ffi(void);
extern int64_t test_nested_ffi(void);
extern int64_t test_rank2_field_ffi(void);

static int check_f64(const char *name, void (*fn)(double *), double want) {
  double got = 0.0;
  fn(&got);
  if (got != want) {
    fprintf(stderr, "%s: got %f, want %f\n", name, got, want);
    return 1;
  }
  return 0;
}

static int check_i64(const char *name, int64_t got, int64_t want) {
  if (got != want) {
    fprintf(stderr, "%s: got %lld, want %lld\n", name, (long long)got,
            (long long)want);
    return 1;
  }
  return 0;
}

int main(void) {
  int failed = 0;
  /* shared: get(splat(2.5), 3) * 4.0 = 10.0 */
  failed |= check_f64("shared", test_shared_ffi, 10.0);
  /* value: i*i at 3 and 7 through two copies = 9 + 49 = 58 */
  failed |= check_i64("value", test_value_ffi(), 58);
  /* enum blob: fold(splat(7.0)) = 28.0; empty arm = -1.0 */
  failed |= check_f64("enum_blob", test_enum_blob_ffi, 28.0);
  failed |= check_f64("enum_empty", test_enum_empty_ffi, -1.0);
  /* cow through field: original 1.0 + updated 99.0 = 100.0 */
  failed |= check_f64("cow_field", test_cow_field_ffi, 100.0);
  /* multi: (0+1+2+3) twice + 2.0 = 14.0 */
  failed |= check_f64("multi", test_multi_ffi, 14.0);
  /* generic Pair<[i64;4]>: fst[2] = 3, snd[3] = 6 → 9 */
  failed |= check_i64("generic", test_generic_ffi(), 9);
  /* nested: (i+1) tabulate at 5 = 6 */
  failed |= check_i64("nested", test_nested_ffi(), 6);
  /* rank-2: cells[1][2] = 12, fold = 36 → 48 */
  failed |= check_i64("rank2_field", test_rank2_field_ffi(), 48);
  if (failed)
    abort();
  puts("all ok");
  return 0;
}
