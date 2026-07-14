#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int32_t test_closure_reuse_ffi(void);
extern int32_t test_capture_and_use_ffi(void);
extern int32_t test_shared_capture_ffi(void);
extern int32_t test_closure_multi_apply_ffi(void);
extern int32_t test_multi_use_capture_ffi(void);
extern int32_t test_partial_reuse_ffi(void);
extern int32_t test_rc_closure_reuse_ffi(void);
extern int32_t test_rc_capture_and_consume_ffi(void);

#define ASSERT_EQ(name, actual, expected)                                      \
  do {                                                                         \
    if ((actual) != (expected)) {                                              \
      fprintf(stderr, "FAIL: %s: expected %lld, got %lld\n", name,             \
              (long long)(expected), (long long)(actual));                     \
      abort();                                                                 \
    }                                                                          \
  } while (0)

int main() {
  /* same closure called 3 times: (10+1) + (20+1) + (30+1) = 63 */
  ASSERT_EQ("closure_reuse", test_closure_reuse_ffi(), 63);

  /* captured RC val used inside (10+7=17) and outside (7*5=35): 17+35 = 52 */
  ASSERT_EQ("capture_and_use", test_capture_and_use_ffi(), 52);

  /* two closures share RC capture: (10+100) + (210-100) = 110 + 110 = 220 */
  ASSERT_EQ("shared_capture", test_shared_capture_ffi(), 220);

  /* closure applied 3 times: 5*2*2*2 = 40 */
  ASSERT_EQ("closure_multi_apply", test_closure_multi_apply_ffi(), 40);

  /* RC base=10: f1(5)=15, f2(3)=30, direct=12: 15+30+12 = 57 */
  ASSERT_EQ("multi_use_capture", test_multi_use_capture_ffi(), 57);

  /* partial reuse: add5(10) + add5(20) = 15 + 25 = 40 */
  ASSERT_EQ("partial_reuse", test_partial_reuse_ffi(), 40);

  /* RC closure reuse: (10+5) + (20+5) + (30+5) = 75 */
  ASSERT_EQ("rc_closure_reuse", test_rc_closure_reuse_ffi(), 75);

  /* RC capture and consume: (10+8) + 8 = 26 */
  ASSERT_EQ("rc_capture_and_consume", test_rc_capture_and_consume_ffi(), 26);

  return 0;
}
