#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int32_t test_vbox_single_ffi(void);
extern int32_t test_vbox_multi_ffi(void);
extern int32_t test_sbox_single_ffi(void);
extern int32_t test_sbox_multi_ffi(void);
extern int32_t test_vbox_capture_multi_ffi(void);
extern int32_t test_sbox_capture_multi_ffi(void);
extern int32_t test_both_structs_ffi(void);
extern int32_t test_vbox_pass_multi_ffi(void);

#define ASSERT_EQ(name, actual, expected)                                      \
  do {                                                                         \
    if ((actual) != (expected)) {                                              \
      fprintf(stderr, "FAIL: %s: expected %lld, got %lld\n", name,             \
              (long long)(expected), (long long)(actual));                     \
      abort();                                                                 \
    }                                                                          \
  } while (0)

int main() {
  /* vbox_single: inc(41) = 42 */
  ASSERT_EQ("vbox_single", test_vbox_single_ffi(), 42);

  /* vbox_multi: double(5) + double(10) = 10 + 20 = 30 */
  ASSERT_EQ("vbox_multi", test_vbox_multi_ffi(), 30);

  /* sbox_single: add3(39) = 42 */
  ASSERT_EQ("sbox_single", test_sbox_single_ffi(), 42);

  /* sbox_multi: triple(4) + triple(10) = 12 + 30 = 42 */
  ASSERT_EQ("sbox_multi", test_sbox_multi_ffi(), 42);

  /* vbox_capture_multi: (1+100) + (2+100) + (3+100) = 306 */
  ASSERT_EQ("vbox_capture_multi", test_vbox_capture_multi_ffi(), 306);

  /* sbox_capture_multi: (3*10) + (7*10) = 30 + 70 = 100 */
  ASSERT_EQ("sbox_capture_multi", test_sbox_capture_multi_ffi(), 100);

  /* both_structs: inc(20) + inc(20) = 21 + 21 = 42 */
  ASSERT_EQ("both_structs", test_both_structs_ffi(), 42);

  /* vbox_pass_multi: double(5) + double(15) = 10 + 30 = 40 */
  ASSERT_EQ("vbox_pass_multi", test_vbox_pass_multi_ffi(), 40);

  return 0;
}
