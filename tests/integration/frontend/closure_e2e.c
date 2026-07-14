#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int32_t test_basic_call_ffi(void);
extern int32_t test_capture_ffi(void);
extern int32_t test_multi_arg_ffi(void);
extern int32_t test_partial_ffi(void);
extern int32_t test_multi_capture_ffi(void);
extern int32_t test_nested_ffi(void);
extern int32_t test_higher_order_ffi(void);
extern uint64_t test_abstract_ffi(void);
extern int32_t test_partial_false_ffi(void);

#define ASSERT_EQ(name, actual, expected)                                      \
  do {                                                                         \
    if ((actual) != (expected)) {                                              \
      fprintf(stderr, "FAIL: %s: expected %lld, got %lld\n", name,             \
              (long long)(expected), (long long)(actual));                     \
      abort();                                                                 \
    }                                                                          \
  } while (0)

int main() {
  ASSERT_EQ("basic_call", test_basic_call_ffi(), 42);
  ASSERT_EQ("capture", test_capture_ffi(), 42);
  ASSERT_EQ("multi_arg", test_multi_arg_ffi(), 42);
  ASSERT_EQ("partial", test_partial_ffi(), 100);
  ASSERT_EQ("multi_capture", test_multi_capture_ffi(), 121);
  ASSERT_EQ("nested", test_nested_ffi(), 41);
  ASSERT_EQ("higher_order", test_higher_order_ffi(), 42);
  ASSERT_EQ("abstract", test_abstract_ffi(), 99);
  ASSERT_EQ("partial_false", test_partial_false_ffi(), 0);
  return 0;
}
