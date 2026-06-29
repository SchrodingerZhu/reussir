//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Part of the Reussir project, dual licensed under the Apache License v2.0 or
// the MIT License.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//
//
// This file defines the lowering and conversion passes for the Reussir
// dialect.
//
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
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::LLVM::LLVMDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::ub::UBDialect",
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
// SCFOpsLowering
//===----------------------------------------------------------------------===//
def ReussirSCFOpsLoweringPass
    : ReussirLoweringPass<"scf-ops"> {
  let summary = "lower Reussir SCF operations to MLIR SCF operations";
  let description = [{
    `reussir-lowering-scf-ops` lowers Reussir SCF operations to MLIR SCF
    operations.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::bufferization::BufferizationDialect",
                           "::mlir::func::FuncDialect",
                           "::mlir::linalg::LinalgDialect",
                           "::reussir::ReussirDialect",
                           "::mlir::math::MathDialect",
                           "::mlir::memref::MemRefDialect",
                           "::mlir::scf::SCFDialect",
                           "::mlir::tensor::TensorDialect",
                           "::mlir::ub::UBDialect"];
}

//===----------------------------------------------------------------------===//
// AttachRegionVtables
//===----------------------------------------------------------------------===//
def ReussirRegionPatternsPass
    : ReussirLoweringPass<"region-patterns"> {
  let summary = "attach vtables to regions";
  let description = [{
    `reussir-lowering-region-patterns` will do the following converions:
    1. attaches vtables to regions.
    2. convert region run operations to memref alloca_scope operations.
    3. convert yield operations to freeze operations.
  }];
  let dependentDialects = [
    "::reussir::ReussirDialect", 
    "::mlir::memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// AcquireDropExpansion
//===----------------------------------------------------------------------===//
def ReussirAcquireDropExpansionPass
    : Pass<"reussir-acquire-drop-expansion"> {
  let summary = "expand Reussir acquire and drop operations";
  let description = [{
    `reussir-acquire-drop-expansion` expands Reussir acquire and drop operations.
    This is a greedy pattern conversion pass.

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
  let options = [
    Option<"outlineRecord", "outline-record", "bool", /*default=*/"false",
        "whether to outline record destruction and acquisition">,
    Option<"expandDecrement", "expand-decrement", "bool", /*default=*/"false",
        "whether to expand rc decrement operations (using the `rc-decrement-expansion` pass)">,
  ];
  let dependentDialects = ["::reussir::ReussirDialect"];
}

//===----------------------------------------------------------------------===//
// RcDecrementExpansion
//===----------------------------------------------------------------------===//
def ReussirRcDecrementExpansionPass
    : Pass<"reussir-rc-decrement-expansion"> {
  let summary = "expand Reussir rc decrement operations";
  let description = [{
    `reussir-rc-decrement-expansion` expands Reussir rc decrement operations.
    This is separated from scf expansion pass as it may be used in early stages
    of the pipeline.
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

  let dependentDialects = ["::reussir::ReussirDialect"];
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
                           "::mlir::LLVM::LLVMDialect"
                          ];
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
  let options = [
    Option<"optimized", "optimized", "bool", /*default=*/"false",
        "enable optimizations when compiling FFI templates">,
  ];
  let dependentDialects = ["::reussir::ReussirDialect", 
                           "::mlir::func::FuncDialect", 
                           "::mlir::scf::SCFDialect"];
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
  let dependentDialects = ["::mlir::LLVM::LLVMDialect", 
                           "::mlir::DLTIDialect"];
}

#endif // REUSSIR_CONVERSION_PASSES_TD
