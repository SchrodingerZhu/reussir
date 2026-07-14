#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct rc_list {
  uint32_t refcount; /* fused 8-byte header: i32 count + i32 tag */
  uint32_t tag;
  struct rc_list *tail; /* members are packed by descending alignment */
  int32_t head;
  int32_t _pad;
} rc_list_t;

#define LIST_NIL 0
#define LIST_CONS 1

/*
 * Special pointer tag (aarch64 default): a nullary variant may be encoded as
 * an unboxed immediate — a pointer whose top byte is `tag + 1` (the low bits
 * point at a compiler-internal dummy box, so on-target dereferences still
 * see the right tag under TBI). Portable foreign code should decode the tag
 * from the pointer itself without dereferencing. A zero top byte means an
 * ordinary RC box (also the layout with --disable-special-pointer-tag).
 */
static inline int64_t list_tag(const rc_list_t *p) {
  uint64_t top = (uint64_t)(uintptr_t)p >> 56;
  return top != 0 ? (int64_t)(top - 1) : p->tag;
}

extern rc_list_t *make_tree_to_list_ffi(int32_t size);

int main(void) {
  const int32_t n = 256;
  rc_list_t *p = make_tree_to_list_ffi(n);

  for (int32_t i = 0; i < n; ++i) {
    if (list_tag(p) != LIST_CONS) {
      fprintf(stderr, "FAIL: expected Cons at index %d, got Nil\n", (int)i);
      abort();
    }
    if (p->head != i) {
      fprintf(stderr, "FAIL: expected %d at index %d, got %d\n", (int)i, (int)i,
              (int)p->head);
      abort();
    }
    p = p->tail;
  }

  if (list_tag(p) != LIST_NIL) {
    fprintf(stderr, "FAIL: expected Nil terminator, got tag=%ld\n",
            (long)list_tag(p));
    abort();
  }

  puts("PASS: make_tree_to_list_ffi(256) returned [0..255]");
  return 0;
}
