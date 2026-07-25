// Stands in for the `main` an executable build will generate: hand the
// `#[main]` function's C-ABI export to the runtime's starter, which runs it
// the way Rust runs its own `main` and yields the process status.
extern int __reussir_start(void (*entry)(void));
extern void __reussir_main(void);

int main(void) { return __reussir_start(__reussir_main); }
