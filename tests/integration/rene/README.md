# `rene` end-to-end examples

Each `<name>.rr` file at this level is a lit test driving `rene` against the
package tree in the sibling `<name>/` directory (whose `lit.local.cfg`
excludes the package sources from test discovery). The suite grows along two
axes: what `rene` itself must do (resolution, freshness, the bake, profiles),
and how much of the language and polyffi surface a realistic multi-package
program exercises.

| test           | shape                                                        |
|----------------|--------------------------------------------------------------|
| `build.rr`     | one package, targets/profiles, freshness, plan inspection    |
| `deps.rr`      | dependency graph views, pubgrub feasibility, build plan      |
| `source_graph.rr` | the source-graph table and its rebuild conditions         |
| `fib.rr`       | one package + polyffi console, release profile               |
| `calc.rr`      | interpreter app + polyffi I/O dep: lexer/parser/eval modules |
| `inventory.rr` | four deps, two polyffi units, cross-package generics         |
| `stats.rr`     | float pipeline + polyffi vector I/O, multi-unit release      |
