---
name: run-repl
description: Run the Reussir REPL to interpret Reussir source code.
license: MPL-2.0
---

You can use `rrepl` (the Rust REPL, built from `crates/reussir-repl`) to run the
Reussir REPL. Build it with `cmake --build build --target rrepl`; the binary is
at `build/target/<profile>/rrepl` (e.g. `build/target/release/rrepl`).

It has several CLI options (`rrepl --help` for the full list):

```text
Usage: rrepl [OPTIONS]

  -O, --opt-level <OPT_LEVEL>   none, default, aggressive, size, tpde
  -l, --log-level <LOG_LEVEL>   error, warning, info, debug, trace  [default: warning]
  -i, --input <INPUT>           Input file for line-by-line execution (script mode)
      --no-tui                  Use the plain line-based prompt (implied by script mode / piped I/O)
```

The REPL can read from a file (`-i`) or interactively. It provides `:dump`
commands to inspect interpreter state.

Assume the input file `tests/integration/repl-rs/polymorphic_math.repl`:

```text
fn cos_squared(x: f64) -> f64 { let y = core::intrinsic::math::cos(x, 0); y * y }
fn sin_squared(x: f64) -> f64 { let y = core::intrinsic::math::sin(x, 0); y * y }
:dump context
fn poly_add<T : Num>(x : T) -> T { (cos_squared(x as f64) + sin_squared(x as f64)) as T }
poly_add(127)
poly_add(127.0)
:dump compiled
```

We can run it with:

```bash
build/target/release/rrepl -i tests/integration/repl-rs/polymorphic_math.repl -lerror
```

And get output like:

```text
Definition added.
Definition added.
fn #cos_squared(v0 (x): f64) -> f64 {
    let v1 (y) = intrinsic#math#cos#0(v0 : f64) : f64;
    (v1 : f64 * v1 : f64) : f64
}

fn #sin_squared(v0 (x): f64) -> f64 {
    let v1 (y) = intrinsic#math#sin#0(v0 : f64) : f64;
    (v1 : f64 * v1 : f64) : f64
}

Definition added.
1 : i64
1.0 : f64
=== Compiled Functions ===
 - _RC11cos_squared
 - _RC11sin_squared
 - _RIC8poly_adddE
 - _RIC8poly_addxE
```

The REPL's own lit suite lives under `tests/integration/repl-rs/` and runs as
part of `cmake --build build --target check` once `rrepl` has been built.
