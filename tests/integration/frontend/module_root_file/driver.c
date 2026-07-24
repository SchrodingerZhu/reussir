#include <stdint.h>
#include <stdlib.h>

extern uint64_t entry_ffi(uint64_t n);

int main() {
  if (entry_ffi(5) != 15)
    abort();
  if (entry_ffi(0) != 10)
    abort();
  return 0;
}
