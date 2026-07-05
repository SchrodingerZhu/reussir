---
name: parser-tree
description: Parse Reussir source code and inspect / validate the surface AST.
license: MPL-2.0
---

You can use `reussir-syntax` (built with `cmake --build build --target
reussir-syntax`; binary at `build/bin/reussir-syntax`) to parse Reussir source
code. It emits the surface AST as JSON, and can also just validate that a file
parses.

To check that a file parses (report diagnostics, no output on success):

```bash
build/bin/reussir-syntax --check tests/integration/frontend/fibonacci.rr
```

On a syntax error it renders each diagnostic with source context on stderr and
exits non-zero; the parser recovers and reports every error, not just the first.

To emit the JSON surface AST (to stdout, or `-o PATH`):

```bash
build/bin/reussir-syntax tests/integration/frontend/fibonacci.rr -o -
```

The output is a single JSON document encoding the spanned AST, e.g.:

```json
[{"contents":{"spanStartOffset":...,"spanValue":{"contents":{"recordDefaultCap":"Value",
"recordFields":{...}}}}}, ...]
```

For a human-readable view of the *elaborated* program instead of the raw parse
tree, prefer `rrc --emit hir` (see the `elaboration` skill). The detailed
structural assertions the parser needs are covered by `reussir-syntax`'s own
Rust unit tests (`cmake --build build --target rrc-test` runs the crate suites).
