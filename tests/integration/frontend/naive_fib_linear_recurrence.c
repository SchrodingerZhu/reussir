#include <stdint.h>
#include <stdlib.h>

extern uint64_t naive_fibonacci_ffi(uint64_t n);

int main() {
  if (naive_fibonacci_ffi(0) != 0)
    abort();
  if (naive_fibonacci_ffi(1) != 1)
    abort();
  if (naive_fibonacci_ffi(2) != 1)
    abort();
  if (naive_fibonacci_ffi(10) != 55)
    abort();
  // Unreachable for the exponential recursion within any reasonable time.
  if (naive_fibonacci_ffi(90) != 2880067194370816120ULL)
    abort();
  // Wraps mod 2^64; still bit-exact after the rewrites.
  if (naive_fibonacci_ffi(200) != 17323038258947941269ULL)
    abort();
  // Far beyond the exponential call tree; instant via matrix exponentiation
  // and still bounded (a few seconds) if only the linearized loop fired, so
  // the test cannot hang CI. The O(log n) guarantee itself is pinned by the
  // llvmpass/linear_recurrence_full_e2e.ll test.
  if (naive_fibonacci_ffi(1000000000ULL) != 3311503426941990459ULL)
    abort();
  return 0;
}
