#include <stdio.h>
#include <stdlib.h>

extern long test_set_in_match(long x);

int main(void) {
  long got = test_set_in_match(7);
  if (got != 1) {
    fprintf(stderr, "set_in_match: got %ld, want 1\n", got);
    exit(1);
  }
  return 0;
}
