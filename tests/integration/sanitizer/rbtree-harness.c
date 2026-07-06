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

extern rc_list_t *make_tree_to_list_ffi(int32_t size);
extern int __lsan_do_recoverable_leak_check(void);

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

static NOINLINE void leak_c_allocation(void) {
    volatile char *leaked = (volatile char *)malloc(64);
    if (leaked == NULL) {
        abort();
    }
    leaked[0] = 1;
    leaked = NULL;
}

static NOINLINE void clear_stack_roots(void) {
    volatile char buffer[4096];
    for (size_t i = 0; i < sizeof(buffer); ++i) {
        buffer[i] = 0;
    }
}

int main(void) {
    const int32_t n = 50;
    rc_list_t *p = make_tree_to_list_ffi(n);

    for (int32_t i = 0; i < n; ++i) {
        if (p->tag != LIST_CONS) {
            fprintf(stderr, "FAIL: expected Cons at index %d, got Nil\n", (int)i);
            abort();
        }
        if (p->head != i) {
            fprintf(stderr, "FAIL: expected %d at index %d, got %d\n", (int)i, (int)i, (int)p->head);
            abort();
        }
        p = p->tail;
    }

    if (p->tag != LIST_NIL) {
        fprintf(stderr, "FAIL: expected Nil terminator, got tag=%ld\n", (long)p->tag);
        abort();
    }

    leak_c_allocation();
    clear_stack_roots();
    return __lsan_do_recoverable_leak_check() > 0 ? 1 : 0;
}
