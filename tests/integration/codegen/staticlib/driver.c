// Drives a Reussir static library through its exported trampolines. Both are
// called, so a partitioned build (`--codegen-units N`, which scatters the two
// functions across members) has to resolve symbols across archive members.
#include <stdint.h>
#include <stdio.h>

extern uint64_t fib_ffi(uint64_t n);
extern uint64_t twice_ffi(uint64_t n);

int main(void) {
  unsigned long long fib = (unsigned long long)fib_ffi(10);
  unsigned long long twice = (unsigned long long)twice_ffi(21);
  printf("fib=%llu twice=%llu\n", fib, twice);
  return (fib == 55 && twice == 42) ? 0 : 1;
}
