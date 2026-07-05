#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern uint64_t ring3_sum_ffi(uint64_t rounds);
extern uint64_t ring_n_sum_ffi(uint64_t n, uint64_t rounds);
extern uint64_t make_and_drop_ffi(uint64_t n);

static void expect(const char *what, uint64_t got, uint64_t want) {
    if (got != want) {
        fprintf(stderr, "FAIL: %s = %llu, want %llu\n", what,
                (unsigned long long)got, (unsigned long long)want);
        abort();
    }
}

int main(void) {
    expect("ring3_sum(4)", ring3_sum_ffi(4), 4 * (1 + 2 + 3));
    expect("ring_n_sum(100, 3)", ring_n_sum_ffi(100, 3), 3 * (100 * 101 / 2));
    expect("make_and_drop(64)", make_and_drop_ffi(64), 1);
    return 0;
}
