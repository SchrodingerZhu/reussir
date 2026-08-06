# `rene` end-to-end examples

Each `<name>.rr` file at this level is a lit test driving `rene` against the
package tree in the sibling `<name>/` directory (whose `lit.local.cfg`
excludes the package sources from test discovery). The suite grows along two
axes: what `rene` itself must do (resolution, freshness, the bake, profiles),
and how much of the language and polyffi surface a realistic multi-package
program exercises.

| test           | shape                                                        |
|----------------|--------------------------------------------------------------|
| `new.rr`       | `rene new` scaffolding: kinds, cross target, the first scan  |
| `build.rr`     | one package, targets/profiles, freshness, plan inspection    |
| `deps.rr`      | dependency graph views, pubgrub feasibility, build plan      |
| `source_graph.rr` | the source-graph table and its rebuild conditions         |
| `fib.rr`       | one package + polyffi console, release profile               |
| `calc.rr`      | interpreter app + polyffi I/O dep: lexer/parser/eval modules |
| `inventory.rr` | four deps, two polyffi units, cross-package generics         |
| `stats.rr`     | float pipeline + polyffi vector I/O, multi-unit release      |
| `shapes.rr`    | impl methods over a vendor dep: value/Arc/regional receivers, cross-package generic method, private fields behind methods |
| `std_option_test.rr` | numbered unit-case dispatcher for `std::option::Option` methods |
| `std_collections_pure_list_test.rr` | numbered unit-case dispatcher for persistent `std` list methods |
| `std_collections_pure_rtqueue_test.rr` | generic Hood-Melville queue API, persistence, and incremental rotations |
| `wasi_examples.rr` | the packages above cross-built for `wasm32-wasip1`, run on wasmer |
| `wasi_threads.rr` | `wasm32-wasip1-threads`: real threads over a shared map   |
