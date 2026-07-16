---
name: reussir-mlir
description: Run MLIR conversions and translations.
license: MPL-2.0
---

You can examine individual MLIR backend passes and their effects on the IR using
`reussir-opt`.

It can be built using `ninja reussir-opt` in the build directory. The executable
is under `build/bin`.

To convert MLIR to LLVM IR, use `reussir-translate --reussir-to-llvmir`. The tool
is also under `build/bin` and it can be built in the same way.

To run the full pipeline and compile the final code, you can use somthing
like the following:

```bash
reussir-opt <input-file> \
 --reussir-token-instantiation \
 --reussir-closure-outlining \
 --reussir-lowering-region-patterns \
 --reussir-inc-dec-cancellation \
 --reussir-rc-decrement-expansion \
 --reussir-infer-variant-tag \
 --reussir-acquire-drop-expansion \
 --reussir-convert-to-std \
 --reussir-inc-dec-cancellation \
 --reussir-acquire-drop-expansion='expand-decrement=1 outline-record=1' \
 --reussir-token-reuse \
 --reussir-convert-to-std \
 --reussir-compile-polymorphic-ffi \
 --convert-scf-to-cf \
 | \
reussir-translate \
 --reussir-to-llvmir | \
opt -O3 |\
llc -relocation-model=pic -filetype=obj -o <output-file>
```

For each pass's meaning, examine `include/Reussir/Conversion/Passes.td` for more
details.
