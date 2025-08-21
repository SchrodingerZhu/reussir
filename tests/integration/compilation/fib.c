#include <stdint.h>
#include <stdio.h>
size_t fibonacci(size_t n) __asm__("fib::fibonacci");
int main() { printf("%zu\n", fibonacci(42)); }
