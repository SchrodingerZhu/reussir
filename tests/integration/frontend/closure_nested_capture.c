#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int32_t test_compose_ffi(void);
extern int32_t test_compose_reuse_ffi(void);
extern int32_t test_compose_chain_ffi(void);
extern int32_t test_capture_higher_order_ffi(void);

#define ASSERT_EQ(name, actual, expected)                                      \
  do {                                                                         \
    if ((actual) != (expected)) {                                              \
      fprintf(stderr, "FAIL: %s: expected %lld, got %lld\n", name,             \
              (long long)(expected), (long long)(actual));                     \
      abort();                                                                 \
    }                                                                          \
  } while (0)

int main() {
  /* compose(double, inc)(5) = double(inc(5)) = double(6) = 12 */
  ASSERT_EQ("compose", test_compose_ffi(), 12);

  /* compose reuse: f(3)=double(inc(3))=8, f(10)=double(inc(10))=22; 8+22=30 */
  ASSERT_EQ("compose_reuse", test_compose_reuse_ffi(), 30);

  /* chain: g = compose(compose(inc,double), inc)
     g(4) = inc(double(inc(4))) = inc(double(5)) = inc(10) = 11 */
  ASSERT_EQ("compose_chain", test_compose_chain_ffi(), 11);

  /* capture_higher_order: compose(add5, add5)(10) = add5(add5(10)) = 20 */
  ASSERT_EQ("capture_higher_order", test_capture_higher_order_ffi(), 20);

  return 0;
}
