---
name: elaboration
description: Elaborate Reussir source code.
license: MPL-2.0
---

You can use `rrc` (built with `cmake --build build --target rrc`) to inspect the
type-checking and lowering of Reussir source code by dumping an intermediate
stage with `--emit`:

- `--emit hir`: the **Semi** IR. Local expression types are inferred and
  checked, but generic substitutions are not attempted — generics stay open
  (`$0`, `$1`) for later instantiation.
- `--emit mir`: the **Full** IR. All types are checked and monomorphized, and
  memory-management modality is inserted: Rc-managed objects are wrapped, and
  functions and records are mangled without generic parameters.

Add `--no-source-locations` for a structural dump (no file table / byte spans);
use `-o -` to stream to stdout.

For example, `build/bin/rrc tests/integration/frontend/region.rr --emit hir --no-source-locations -o -`
generates:

```rust
[regional] struct #Cell<$0> { "value": $0 };

[regional] struct #Container<$1> { "cell": field Nullable<[regional] #Cell::<$1>> };

fn #trivial() -> u64 {
    region { 1 : u64 } : u64
}

fn #freeze_cell() -> [rigid] #Cell::<u64> {
    region { #Cell::<u64>{1 : u64} : [flex] #Cell::<u64> } : [rigid] #Cell::<u64>
}

regional fn #regional_function(v0 (x): u64) -> [flex] #Cell::<u64> {
    #Cell::<u64>{v0 : u64} : [flex] #Cell::<u64>
}
```

While `build/bin/rrc tests/integration/frontend/region.rr --emit mir --no-source-locations -o -`
generates the monomorphized, mangled Full IR:

```rust
record @_RIC4CellyE : regional struct Cell::<u64> { "value": u64 };

fn @_RC11freeze_cell() -> [rigid] Cell::<u64> {
    region { @_RIC4CellyE{1 : u64} : [flex] Cell::<u64> } : [rigid] Cell::<u64>
}

regional fn @_RC17regional_function(v0 (x): u64) -> [flex] Cell::<u64> {
    @_RIC4CellyE{v0 : u64} : [flex] Cell::<u64>
}
```

Both the `hir` and `mir` text dumps round-trip: a dump can be fed back into
`rrc` with `--from hir` / `--from mir` to resume the pipeline.
