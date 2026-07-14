#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int32_t test_fn_lift_basic_ffi(void);
extern int32_t test_fn_partial_ffi(void);
extern int32_t test_fn_struct_field_ffi(void);
extern int32_t test_fn_lift_multi_arg_ffi(void);
extern int32_t test_fn_partial_multi_ffi(void);
extern int32_t test_generic_lift_ffi(void);
extern int32_t test_generic_partial_ffi(void);

#define ASSERT_EQ(name, actual, expected)                                      \
  do {                                                                         \
    if ((actual) != (expected)) {                                              \
      fprintf(stderr, "FAIL: %s: expected %lld, got %lld\n", name,             \
              (long long)(expected), (long long)(actual));                     \
      abort();                                                                 \
    }                                                                          \
  } while (0)

int main() {
  ASSERT_EQ("fn_lift_basic", test_fn_lift_basic_ffi(), 42);
  ASSERT_EQ("fn_partial", test_fn_partial_ffi(), 42);
  ASSERT_EQ("fn_struct_field", test_fn_struct_field_ffi(), 42);
  ASSERT_EQ("fn_lift_multi_arg", test_fn_lift_multi_arg_ffi(), 42);
  ASSERT_EQ("fn_partial_multi", test_fn_partial_multi_ffi(), 42);
  ASSERT_EQ("generic_lift", test_generic_lift_ffi(), 42);
  ASSERT_EQ("generic_partial", test_generic_partial_ffi(), 42);
  return 0;
}
