//===----------------------------------------------------------------------===//
//
// Part of the Reussir Project, dual licensed under the Apache License v2.0 or
// the MIT License.
// See https://github.com/reussir-lang/reussir/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the lowering and conversion passes for the Reussir
/// dialect.
///
//===----------------------------------------------------------------------===//

#ifndef REUSSIR_CONVERSION_PASSES_TD
#define REUSSIR_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

class ReussirLoweringPass<string mnemonic, string target = "">
    : Pass<"reussir-lowering-"#mnemonic, target>;

//===----------------------------------------------------------------------===//
// BasicOpsLowering
//===----------------------------------------------------------------------===//
def ReussirBasicOpsLoweringPass
    : ReussirLoweringPass<"basic-ops", "::mlir::ModuleOp"> {
  let summary = "prepare Reussir basic lowering for convert-to-llvm";
  let description = [{
    `reussir-lowering-basic-ops` prepares modules for LLVM lowering by
    materializing the Reussir runtime declarations and normalizing fused debug
    information. The actual lowering of Reussir basic operations is provided by
    the registered LLVM lowering interface and is driven by `convert-to-llvm`.

    With `closure-wpd` enabled, the pass stamps each outlined closure vtable
    with its whole-program-devirtualization type id (one per closure RETURN
    type — apply/uniqify/clone never change a vtable, so the return type is
    the one thing every value it backs agrees on; see `ClosureWpd.h`) and
    marks the module (`reussir.wpd`). The vtable conversion pattern carries
    the id onto the vtable global, and the dispatch conversion patterns
    (eval/clone/drop) assert it with `reussir.closure.wpd_test` on the very
    vtable-pointer SSA value the slot call hangs off. The dialect's LLVM
    translation interface turns both into the real artifacts — `!type` /
    `!vcall_visibility` metadata and `llvm.type.test` + `llvm.assume` — and
    the backend pipeline's speculative `WholeProgramDevirt` folds
    single-implementation families into guarded direct calls (sound on any
    world: a vtable the module has never seen takes the indirect fallback
    arm). See docs/design/closure-wpd.md.
  }];
  let options =
      [Option<
           "closureWpd", "closure-wpd", "bool", /*default=*/"false",
           "emit whole-program devirtualization artifacts for closure vtables">,
  ];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::LLVM::LLVMDialect",
                           "::mlir::func::FuncDialect", "::mlir::ub::UBDialect",
                           "::reussir::ReussirDialect"];
}

//===----------------------------------------------------------------------===//
// DebugInfoConversion
//===----------------------------------------------------------------------===//
def ReussirDebugInfoConversionPass
    : ReussirLoweringPass<"debug-info", "::mlir::ModuleOp"> {
  let summary = "lower Reussir fused debug-info attributes to LLVM DI";
  let description = [{
    `reussir-lowering-debug-info` translates the Reussir `DBG*` attributes that
    ride on fused locations (subprograms, local variables, function arguments)
    into LLVM debug-info attributes, and materializes `llvm.dbg.value` ops. It
    must run *after* `convert-to-llvm`: the conversion preserves the fused
    locations and the `reussir.dbg_func_args` attribute onto the `llvm.func`,
    and a function's debug info only survives translation once its `llvm.func`
    carries a `DISubprogram`.
  }];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// ConvertToSTD
//===----------------------------------------------------------------------===//
def ReussirConvertToSTDPass
    : Pass<"reussir-convert-to-std", "::mlir::ModuleOp"> {
  let summary = "lower high-level Reussir operations to standard MLIR dialects";
  let description = [{
    `reussir-convert-to-std` lowers high-level Reussir operations to standard
    MLIR dialects such as arith, func, memref, and scf. It also expands
    high-level sync operations into structured control flow and runtime calls,
    while leaving straight-line Reussir and sync bridge operations for the
    final LLVM conversion.

    Cell creation and non-atomic Cell get/set/rmw/in-use operations become RC
    borrows, mutable slot accesses, and MLIR SCF guards.
    Cell get acquires the exact loaded value, set drops the old slot before
    storing the caller-owned replacement, and rmw moves the old and replacement
    values through its region without implicit ownership traffic.
    Loop-shaped atomic read-modify-writes (region bodies and the multiply
    kinds LLVM cannot express as a single atomicrmw) become `scf.while` retry
    loops committed with the straight-line `reussir.ref.cmpxchg` bridge, so
    all Cell control flow is crafted as structured IR here. Straight-line
    atomic accesses (get, set, native atomicrmw kinds, and the bridge itself)
    remain intact for the LLVM lowering, which performs no CFG changes.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::bufferization::BufferizationDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::reussir::ReussirDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::ptr::PtrDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::sync::SyncDialect",
                           "::mlir::tensor::TensorDialect",
                           "::mlir::ub::UBDialect"];
}

//===----------------------------------------------------------------------===//
// AttachRegionVtables
//===----------------------------------------------------------------------===//
def ReussirRegionPatternsPass : ReussirLoweringPass<"region-patterns"> {
  let summary = "attach vtables to regions";
  let description = [{
    `reussir-lowering-region-patterns` will do the following converions:
    1. attaches vtables to regions.
    2. convert region run operations to memref alloca_scope operations.
    3. convert yield operations to freeze operations.
  }];
  let dependentDialects = ["::reussir::ReussirDialect",
                           "::mlir::memref::MemRefDialect"];
}

//===----------------------------------------------------------------------===//
// AcquireDropExpansion
//===----------------------------------------------------------------------===//
def ReussirAcquireDropExpansionPass : Pass<"reussir-acquire-drop-expansion"> {
  let summary = "expand Reussir acquire and drop operations";
  let description = [{
    `reussir-acquire-drop-expansion` greedily expands Reussir acquire and drop
    operations.

    For drop operations:
    1. it eliminates no-op drop operations.
    2. for normal drop, it inserts destruction operations for inner elements.

    For acquire operations:
    1. it eliminates no-op acquire operations (trivially copyable types).
    2. for normal acquire, it emits ownership acquisition (e.g., rc.inc for Rc
       fields, recursive acquisition for compound/variant records).

    Notice that this pass terminates normally for recursive types, as such types
    are backed by RC pointers. The decrement operation will not be inlined at
    this pass. After the expansion, the optimizer has chances to fuse increment
    and decrement operations. Eventually, remaining drop operations expanded from
    decrement operations will be outlined to avoid recursive inlining.

    To also expand rc decrement operations, use `expand-decrement` option. When
    enabled, conversion patterns from `rc-decrement-expansion` will also be applied
    in the greedy matching process.

    If you have recursive types, if `expand-decrement` is enabled, the pass may
    not terminate as dec-drop will be expanded infinitely. To avoid this, you
    should enable `outline-record` option, which converts record destructions to
    outline function calls.
  }];
  let options =
      [Option<"outlineRecord", "outline-record", "bool", /*default=*/"false",
              "whether to outline record destruction and acquisition">,
       Option<"expandDecrement", "expand-decrement", "bool",
              /*default=*/"false",
              "whether to expand rc decrement operations (using the "
              "`rc-decrement-expansion` pass)">,
  ];
  let dependentDialects = ["::reussir::ReussirDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::sync::SyncDialect"];
}

//===----------------------------------------------------------------------===//
// RcDecrementExpansion
//===----------------------------------------------------------------------===//
def ReussirRcDecrementExpansionPass : Pass<"reussir-rc-decrement-expansion"> {
  let summary = "expand Reussir rc decrement operations";
  let description = [{
    `reussir-rc-decrement-expansion` expands Reussir rc decrement operations.
    This is separated from the ConvertToSTD pass because it may be used in
    early stages of the pipeline.
  }];

  let dependentDialects = ["::reussir::ReussirDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::scf::SCFDialect"];
}

//===----------------------------------------------------------------------===//
// TokenInstantiation
//===----------------------------------------------------------------------===//
def ReussirTokenInstantiationPass
    : Pass<"reussir-token-instantiation", "::mlir::func::FuncOp"> {
  let summary = "insert token instantiations for TokenAcceptor operations";
  let description = [{
    `reussir-token-instantiation` inserts token allocation operations for operations
    implementing the TokenAcceptor interface that do not have a token assigned.
    For each such operation, a `reussir.token.alloc` operation is inserted
    immediately before it, and the token is assigned to the operation.
  }];

  let dependentDialects = ["::reussir::ReussirDialect",
                           "::mlir::sync::SyncDialect"];
}

//===----------------------------------------------------------------------===//
// ClosureOutlining
//===----------------------------------------------------------------------===//
def ReussirClosureOutliningPass
    : Pass<"reussir-closure-outlining", "::mlir::ModuleOp"> {
  let summary = "outline closures from Reussir operations";
  let description = [{
    `reussir-closure-outlining` outlines closures from Reussir operations.
    This is required to instantiate closures for further lowering.
  }];

  let dependentDialects = ["::reussir::ReussirDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// Compile Polymorphic FFI
//===----------------------------------------------------------------------===//
def ReussirCompilePolymorphicFFIPass
    : Pass<"reussir-compile-polymorphic-ffi", "::mlir::ModuleOp"> {
  let summary = "compile polymorphic FFI calls";
  let description = [{
    `reussir-compile-polymorphic-ffi` compiles polymorphic FFI template
    into concrete LLVM bitcode and embeds them into the module.
  }];
  let options = [Option<"optimized", "optimized", "bool", /*default=*/"false",
                        "enable optimizations when compiling FFI templates">,
                 Option<"rustPath", "rust-path", "std::string",
                        /*default=*/"\"\"",
                        "explicit rustc executable used to compile FFI "
                        "templates (falls back to REUSSIR_RUSTC and probing)">,
                 ListOption<"libDirs", "libdirs", "std::string",
                        "directories searched for Rust packages when "
                        "compiling FFI templates (falls back to "
                        "REUSSIR_RUSTC_DEPS and probing)">,
  ];
  let dependentDialects = ["::reussir::ReussirDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::scf::SCFDialect"];
}

//===----------------------------------------------------------------------===//
// InstrumentNonlinearFFI
//===----------------------------------------------------------------------===//
def ReussirInstrumentNonlinearFFIPass
    : Pass<"reussir-instrument-nonlinear-ffi", "::mlir::ModuleOp"> {
  let summary = "report non-linearly used ffi/array values consumed by FFI "
                "calls at runtime";
  let description = [{
    `reussir-instrument-nonlinear-ffi` instruments every call to an
    FFI-imported function (the `target` of a `reussir.trampoline import`):
    for each `!reussir.rc` argument whose payload is an `ffi_object` or an
    `array`, it emits a guarded runtime report before the call:

    ```mlir
    %count = reussir.rc.fetch (%arg : ...) : index
    %shared = arith.cmpi ne, %count, %c1
    scf.if %shared {  // reussir.expect'ed cold
      llvm.call @__reussir_report_nonlinear_usage(%file, %line, %col)
    }
    ```

    The owned calling convention consumes every rc argument, so a reference
    count above one at the boundary means the value is used non-linearly and
    the foreign side degrades to copy-on-write instead of updating in place.
    The runtime function reports the call's source location on stderr. The
    check is purely diagnostic and never changes program semantics; schedule
    it after the rc optimization passes so the observed count is the final
    one.
  }];
  let dependentDialects = ["::reussir::ReussirDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// ConvertToLLVM
//===----------------------------------------------------------------------===//
def ReussirConvertToLLVMPass
    : Pass<"reussir-convert-to-llvm", "::mlir::ModuleOp"> {
  let summary =
      "interface-driven convert-to-llvm honoring the module data layout";
  let description = [{
    `reussir-convert-to-llvm` runs the same `ConvertToLLVMPatternInterface`
    driven conversion as the upstream `convert-to-llvm`, but builds its
    `LLVMTypeConverter` from the module's stamped `llvm.data_layout` — index
    width included. The upstream pass freezes a default-options converter (a
    64-bit index regardless of the target) before ever seeing a module, while
    every converter Reussir builds itself derives the index width from the
    data layout (`getReussirToLLVMOptions`); on a 32-bit-pointer target
    (e.g. wasm32) the two disagree, and the runtime declarations
    (`__reussir_allocate` at `i32`) no longer match the call sites the
    conversion emits (`i64`).

    The upstream `dynamic` mode is not sufficient either: it derives the
    *index width* from `DataLayoutAnalysis`, but the converter's embedded
    `llvm::DataLayout` — which `convertRecordType` uses to compute record
    bodies (member padding and the pad-lift into wider integers) — stays at
    the 64-bit default. On wasm32 a `{i32, ptr}` record then lowers as
    `{i64, ptr}` (16 bytes) while the stamped module layout sizes its token
    allocation at 8, corrupting the heap. Building the converter from the
    stamped string fixes both.
  }];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// AttachNativeTarget
//===----------------------------------------------------------------------===//
def ReussirAttachNativeTargetPass
    : Pass<"reussir-attach-native-target", "::mlir::ModuleOp"> {
  let summary = "attach native target information to the module";
  let description = [{
    `reussir-attach-native-target` attaches native target information to the module.
    It adds the following attributes to the module:
    - `llvm.data_layout`: The data layout string of the native target.
    - `llvm.target_triple`: The target triple string of the native target.
    - `dlti.dl_spec`: The data layout spec of the native target.
  }];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect", "::mlir::DLTIDialect"];
}

#endif // REUSSIR_CONVERSION_PASSES_TD
