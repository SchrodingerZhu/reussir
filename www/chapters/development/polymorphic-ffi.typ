#import "/book.typ": book-page

#show: book-page.with(title: "Polymorphic FFI")

== Polymorphic FFI

=== Functional Imperative Wrapper

An important feature of RC-based functional runtime is that it provides a straightforward way to wrap imperative data structures with functional interfaces.

Consider `Vec<f64>`, which is a mutable vector of floating-point numbers. We can wrap it with a functional interface like this:
```rust
#[repr(transparent)]
struct FunctionalVec<f64>(Rc<Vec<f64>>);
impl FunctionalVec<f64> {
  fn new() -> Self {
    Self(Rc::new(Vec::new()))
  }
  fn push(mut self, value: f64) -> Self {
    self.0.make_mut().push(value);
    self
  }
}
```

As long as the underlying data structure has well-defined *clone* and *drop* behavior, we can use a similar approach to wrap it with a functional interface. Moreover, as
long as the data structure is used
linearly, the overhead is minimal.

=== Polymorphic Types

The problem arises, however, when a data structure is polymorphic. Since the underlying data structure is implemented in an imperative language (Rust in this case), we will
need to find a way to monophimize the FFI.

`Lean4` takes an easy approach by inventing the so-called *uniform ABI*, where every type, scalar or composite, are always encoded into a pointer. For small values, this can
done with a simple tagging overhead, while for larger values, additional allocation is required. This ease the burden of passing data across the FFI boundary.

We adopt a diffrent and much more aggressive approach. Again, we abuse the simplicity of RC-based functional runtime and provide a real measure to generate monomorphic types
at compile time.

=== Polymorphic IR Operation

The `reussir.polyffi` operation is the core mechanism for compile-time monomorphization of polymorphic FFI code. It takes a Rust source template and a dictionary of substitutions, then compiles the monomorphized result into LLVM bitcode.

==== Syntax

```mlir
reussir.polyffi
    texture("...rust source template...")
    substitutions({"key1" = value1, "key2" = value2})
```

The operation accepts three optional attributes:
- `texture`: A Rust source code template with `[:key:]` placeholders
- `substitutions`: A dictionary mapping placeholder names to values
- `compiled`: The resulting LLVM bitcode blob (after compilation)

==== Substitution Mechanism

Placeholders in the template follow the `[:key:]` syntax. The substitution values can be:

- *String attributes*: Directly inserted as text
- *Integer attributes*: Inserted as numeric literals
- *Type attributes*: Generates proper Rust type definitions

For type substitutions, the compiler generates appropriate Rust code based on the MLIR type:
- Primitive types (`i32`, `f64`, etc.) are printed directly
- `RcType` generates `::reussir_rt::rc::Rc<T>` wrappers
- `RecordType` generates `#[repr(C)]` structs with `Drop` and `Clone` implementations

==== Example: Vec\<f64\>

```mlir
!Vecf64 = !reussir.rc<!reussir.ffi_object<
    "::reussir_rt::collections::vec::Vec<f64>",
    @__reussir_polyffi_Vec_f64_drop>>

reussir.polyffi
    texture("#![feature(linkage)]
extern crate reussir_rt;
use reussir_rt::collections::vec::Vec;

#[linkage = \"weak_odr\"]
#[unsafe(no_mangle)]
pub unsafe extern \"C\" fn __reussir_polyffi_Vec_[:typename:]_new()
    -> Vec<[:typename:]> { Vec::new() }

#[linkage = \"weak_odr\"]
#[unsafe(no_mangle)]
pub unsafe extern \"C\" fn __reussir_polyffi_Vec_[:typename:]_push(
    vec: Vec<[:typename:]>, ele: [:typename:]
) -> Vec<[:typename:]> { Vec::push(vec, ele) }")
    substitutions({"typename" = "f64"})

func.func private @__reussir_polyffi_Vec_f64_new() -> !Vecf64
func.func private @__reussir_polyffi_Vec_f64_push(!Vecf64, f64) -> !Vecf64
```

After the `CompilePolymorphicFFI` pass runs, all `[:typename:]` occurrences are replaced with `f64`, the Rust code is compiled to LLVM bitcode, and the operation becomes:

```mlir
reussir.polyffi compiled(dense_resource<blob> : tensor<1024xi8>)
```

==== Nested Type Instantiation

For nested polymorphic types like `Vec<Vec<f64>>`, we can use type substitutions:

```mlir
!Vecf64 = !reussir.rc<!reussir.ffi_object<...>>
!VecVec64 = !reussir.rc<!reussir.ffi_object<
    "::reussir_rt::collections::vec::Vec<
        ::reussir_rt::collections::vec::Vec<f64>>",
    @__reussir_polyffi_Vec_NestedVec_drop>>

reussir.polyffi
    texture("... [:NestedVec:] ... Vec<[:typename:]> ...")
    substitutions({"typename" = "NestedVec", "NestedVec" = !Vecf64})
```

When `[:NestedVec:]` is encountered with a type attribute, the compiler generates:
```rust
type NestedVec = ::reussir_rt::collections::vec::Vec<f64>;
```

This type alias is inserted into the Rust source, and then `[:typename:]` is replaced with the string `"NestedVec"`, resulting in functions operating on `Vec<NestedVec>` (i.e., `Vec<Vec<f64>>`).

==== Compilation Pipeline

The polymorphic FFI compilation happens in two distinct phases during the overall translation process.

*Phase 1: Monomorphization (`CompilePolymorphicFFI` pass)*

The `CompilePolymorphicFFI` pass runs early in the pipeline:

+ *Collect* all `reussir.polyffi` operations with uncompiled textures
+ *Monomorphize* each template by substituting `[:key:]` placeholders
+ *Compile* the generated Rust source to LLVM bitcode via `rustc`
+ *Embed* the bitcode as a `DenseElementsAttr` tensor
+ *Remove* the `texture` and `substitutions` attributes

After this pass, each `reussir.polyffi` operation contains only the `compiled` attribute with embedded LLVM bitcode.

*Phase 2: Linking (`translateToModule`)*

The `translateToModule` function in `Bridge.cppm` orchestrates the final assembly:

+ *Parse* the MLIR module and attach target data layout
+ *Run* `compilePolymorphicFFI` to ensure all FFI operations are compiled
+ *Gather* all compiled bitcode blobs via `gatherCompiledModules`:
  + Walk the module to find all `reussir.polyffi` operations with `compiledModule`
  + Parse each bitcode blob back into an LLVM module
  + Link all parsed modules together into a single `finalModule`
+ *Lower* the MLIR module through the standard pipeline (SCF ops → basic ops → LLVM dialect)
+ *Translate* the lowered MLIR to LLVM IR (`translateModuleToLLVMIR`)
+ *Link* the gathered FFI modules into the main LLVM module (`llvm::Linker::linkModules`)

This two-phase approach allows the Rust FFI code to be compiled independently and then merged with the main program during final code generation.

*Code Flow*

```txt
MLIR Module with reussir.polyffi operations
    │
    ▼ compilePolymorphicFFI()
    │   └─ For each polyffi op with texture:
    │       ├─ monomorphize() → substitute [:key:] placeholders
    │       ├─ compileRustSourceToBitcode() → invoke rustc
    │       └─ store bitcode in compiledModule attribute
    │
    ▼ gatherCompiledModules()
    │   └─ For each polyffi op with compiledModule:
    │       ├─ parseBitcodeFile() → LLVM Module
    │       └─ llvm::Linker::linkModules() → merge
    │
    ▼ PassManager.run()
    │   └─ ConvertToSTD → Basic ops lowering → LLVM dialect
    │
    ▼ translateModuleToLLVMIR()
    │   └─ Main program LLVM IR
    │
    ▼ llvm::Linker::linkModules()
        └─ Final linked LLVM module (main + FFI)
```

== Frontend Surface

The frontend generates `reussir.polyffi` textures itself — users never write
`[:key:]` placeholders or MLIR. Three source forms drive it:

```
extern "rust" [{ use reussir_rt::collections::vec::Vec as RVec; }];

#[ffi(rust = "::reussir_rt::collections::vec::Vec")]
pub struct Vec<T>;

#[ffi(import)]
pub fn push<T>(v: Vec<T>, x: T) -> Vec<T> [{ RVec::push(v, x) }];
```

Per monomorphized instance, the compiler renders a self-contained Rust
wrapper texture in the trampoline boundary shape, emits an import
trampoline binding it to the native declaration, and — for opaque record
instances — a drop-hook texture. Shared Reussir records crossing the
boundary become generated `#[repr(transparent)]` pointer wrappers whose
`Clone`/`Drop` call compiler-emitted `rc.inc`/`rc.dec` glue, so clone and
drop flow correctly in both directions. Ownership is uniform: the boundary
consumes, exactly like every other call.

See `docs/design/polymorphic-ffi.md` for the full design (boundary
classification, symbol scheme, the `Rc`-wrapper contract for exposing new
types).
