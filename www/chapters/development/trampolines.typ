#import "/book.typ": book-page

#show: book-page.with(title: "Trampolines")

== Trampolines

`reussir.trampoline` models the backend FFI boundary for ordinary native calls.
In both directions `target` names the function with the *native*
(Reussir-internal) signature and `sym_name` names the C-facing boundary
symbol; the direction picks which side the lowering materializes:

- `reussir.trampoline export "C" @ffi_name = @target`
- `reussir.trampoline import "C" @c_boundary = @native_decl`

`export` exposes an internal Reussir function to C-facing callers: the
lowering creates `@ffi_name` with the boundary signature, unpacking the
boundary arguments and calling `@target`.

`import` is the mirror image: `@native_decl` is an existing *body-less*
declaration (Reussir callers `func.call` it natively); the lowering replaces
the declaration with a definition that packs the native arguments per the
boundary convention and calls the external boundary symbol `@c_boundary`,
which it declares. Trivial signatures forward directly in both directions.

=== Trivial Boundary

A function is *trivial* for the backend trampoline when:

- it accepts fewer than four arguments;
- every argument is an integral or pointer LLVM value;
- it returns either no result or exactly one integral or pointer LLVM value.

Trivial trampolines do not need ABI reshaping. The backend therefore lowers
them as direct forwards with best-effort minimal overhead.

=== Nontrivial Boundary

Any function outside the trivial class uses the explicit trampoline boundary
below.

For nontrivial returns, the trampoline always uses an explicit leading return
pointer parameter. The backend does not attach ABI-specific `sret` metadata to
that parameter. This rule is backend-defined and no longer depends on
platform-specific C ABI aggregate classification.

For nontrivial calls, the trampoline packs all logical arguments into a literal
LLVM struct in source order and passes a pointer to that struct. This applies
even when the original arguments are individually integral or pointer values,
for example once the arity reaches four.

The resulting boundary therefore has one of these shapes:

- trivial call: direct values in, direct value or `void` out;
- nontrivial return only: `void` return plus leading return pointer;
- nontrivial arguments only: direct return plus one pointer to the packed
  argument struct;
- both nontrivial: leading return pointer plus one pointer to the packed argument
  struct.

=== Import And Export Responsibilities

For `export`, the backend creates only the boundary symbol; the external C
caller must obey the boundary shape (packed-argument pointer, explicit return
storage).

For `import`, the backend materializes the native-side marshaling itself —
the packed-argument alloca, the return slot, and the call — so Reussir-side
callers use the plain native signature. The external boundary symbol must be
defined elsewhere with the boundary shape; the frontend's polymorphic FFI
generates exactly that shape for its Rust wrapper functions (see the
Polymorphic FFI chapter and `docs/design/polymorphic-ffi.md`).

=== Operation Shape

The operation includes an import/export enum attribute in assembly syntax:

```mlir
reussir.trampoline export "C" @sum_ffi = @sum_internal
reussir.trampoline import "C" @sum_c_boundary = @sum_native
```

Currently only the `"C"` ABI is supported.

=== Frontend Surface

`extern "C" trampoline "sym" = path<T,...>;` emits the export direction. The
import direction is not written by hand: every `#[ffi(import)]` function
instance emits one, binding its generated Rust boundary wrapper to the
native declaration (`docs/design/polymorphic-ffi.md`).
