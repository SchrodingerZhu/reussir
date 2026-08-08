// A use-after-free planted in memory the *runtime* allocated, so the report
// depends on the sanitizers being able to see the runtime's allocator.
//
// The sanitized runtime variants are built `--no-default-features`
// (crates/reussir-rt/CMakeLists.txt), which routes allocation through the
// platform allocator the sanitizers interpose. The ordinary runtime allocates
// from mimalloc's private heap, which they cannot see — the runtime's own
// Cargo.toml records this as the reason the variants exist. So a silent run
// here means the wrong runtime was linked or loaded, not that the memory is
// fine.
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

extern uint8_t *__reussir_allocate(size_t align, size_t size);
extern void __reussir_deallocate(uint8_t *ptr, size_t align, size_t size);

int main(void) {
  uint8_t *box = __reussir_allocate(8, 32);
  box[8] = 42;
  __reussir_deallocate(box, 8, 32);
  printf("read after free: %d\n", (int)box[8]);
  return 0;
}
