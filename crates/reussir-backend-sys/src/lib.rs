//! Raw FFI bindings to the Reussir C API.
//!
//! This crate exposes the C entry points provided by `libReussirCAPI` together
//! with a re-export of [`mlir_sys`], the raw MLIR C API the bindings are built
//! on. Safe wrappers live in the `reussir-backend` crate; everything here is
//! `unsafe` and mirrors the C signatures one-to-one.

#![allow(non_snake_case)]

pub use mlir_sys;

use core::ffi::{c_char, c_int};

use mlir_sys::{
    MlirAttribute, MlirContext, MlirDialectHandle, MlirDialectRegistry, MlirModule, MlirPass,
    MlirType,
};

//==-- Reussir type enums --==//
//
// These mirror the dialect's `I32EnumAttr` definitions one-to-one. They are
// passed by value to the type constructors below. `#[repr(C)]` gives them the C
// `int` representation the C API expects.

/// Reference/record capability. Mirrors `reussir::Capability`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReussirCapability {
    Unspecified = 0,
    Value = 1,
    Shared = 2,
    Flex = 3,
    Rigid = 4,
    Field = 5,
    Regional = 6,
}

/// Reference-count atomicity. Mirrors `reussir::AtomicKind`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReussirAtomicKind {
    Normal = 0,
    Atomic = 1,
}

/// Record flavour. Mirrors `reussir::RecordKind`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReussirRecordKind {
    Compound = 0,
    Variant = 1,
}

/// String lifetime scope. Mirrors `reussir::LifeScope`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReussirLifeScope {
    Global = 0,
    Local = 1,
}

// Opaque LLVM-C handles used by the bitcode-gathering helper. The next stage of
// the port wires in `llvm-sys`; until then these are declared as opaque pointers
// so the C API surface is complete and linkable, matching `llvm-c/Types.h`.
#[repr(C)]
pub struct LlvmOpaqueModule {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LlvmOpaqueContext {
    _private: [u8; 0],
}
#[repr(C)]
pub struct LlvmOpaqueMemoryBuffer {
    _private: [u8; 0],
}
/// Mirrors `LLVMModuleRef`.
pub type LLVMModuleRef = *mut LlvmOpaqueModule;
/// Mirrors `LLVMContextRef`.
pub type LLVMContextRef = *mut LlvmOpaqueContext;
/// Mirrors `LLVMMemoryBufferRef`.
pub type LLVMMemoryBufferRef = *mut LlvmOpaqueMemoryBuffer;

unsafe extern "C" {
    //==-- Dialect registration --==//

    /// Returns the dialect handle for the Reussir dialect. The handle can be
    /// inserted into a dialect registry or registered into a context.
    pub fn mlirGetDialectHandle__reussir__() -> MlirDialectHandle;

    /// Registers the Reussir dialect together with every upstream dialect,
    /// extension and LLVM/builtin translation it relies on into `context`, then
    /// loads all available dialects.
    pub fn reussirRegisterAllDialects(context: MlirContext);

    /// Populates a dialect registry with the Reussir dialect and everything the
    /// backend pipeline depends on. Build a context from the registry so dialect
    /// extensions are applied as dialects load.
    pub fn reussirPopulateRegistry(registry: MlirDialectRegistry);

    //==-- Reussir passes --==//

    pub fn reussirCreateUniqueCarryingRecursionAnalysisPass() -> MlirPass;
    pub fn reussirCreateTokenInstantiationPass() -> MlirPass;
    pub fn reussirCreateClosureOutliningPass() -> MlirPass;
    pub fn reussirCreateRegionPatternsPass() -> MlirPass;
    pub fn reussirCreateIncDecCancellationPass() -> MlirPass;
    pub fn reussirCreateRcDecrementExpansionPass() -> MlirPass;
    pub fn reussirCreateInferVariantTagPass() -> MlirPass;
    pub fn reussirCreateSCFOpsLoweringPass() -> MlirPass;
    pub fn reussirCreateRcCreateSinkPass() -> MlirPass;

    /// Encodes nullary variants of shared rc-boxed enums as tagged pointer
    /// immediates (top byte = tag + 1) and stamps the module so the LLVM
    /// lowering steers refcount stores away from the dummy boxes. Only added
    /// when the scheme is enabled; `arch_independent` selects the encoding:
    /// false = `tbi` (top-byte tag, aarch64), true = `immortal` (plain dummy
    /// address with an immortal refcount, any target).
    pub fn reussirCreateSpecialPointerTagPass(arch_independent: bool) -> MlirPass;
    pub fn reussirCreateRcDispatchFusionPass() -> MlirPass;
    pub fn reussirCreateRcCreateFusionPass() -> MlirPass;
    pub fn reussirCreateTRMCRecursionAnalysisPass() -> MlirPass;
    pub fn reussirCreateCompilePolymorphicFFIPass(optimized: bool) -> MlirPass;
    pub fn reussirCreateInvariantGroupAnalysisPass() -> MlirPass;
    pub fn reussirCreateBasicOpsLoweringPass() -> MlirPass;
    pub fn reussirCreateDebugInfoConversionPass() -> MlirPass;
    pub fn reussirCreateAcquireDropExpansionPass(
        expand_decrement: bool,
        outline_record: bool,
    ) -> MlirPass;
    pub fn reussirCreateTokenReusePass(reuse_across_call: bool) -> MlirPass;

    //==-- Upstream passes used by the pipeline --==//

    pub fn reussirCreateDefaultInlinerPass() -> MlirPass;
    pub fn reussirCreateCanonicalizerPass() -> MlirPass;
    pub fn reussirCreateCSEPass() -> MlirPass;
    pub fn reussirCreateControlFlowSinkPass() -> MlirPass;
    pub fn reussirCreateSCFToControlFlowPass() -> MlirPass;
    pub fn reussirCreateConvertToLLVMPass() -> MlirPass;
    pub fn reussirCreateReconcileUnrealizedCastsPass() -> MlirPass;

    //==-- Standalone helpers --==//

    /// Monomorphizes and compiles polymorphic FFI operations in the module.
    /// Returns true on success.
    pub fn reussirCompilePolymorphicFFI(module: MlirModule, optimized: bool) -> bool;

    /// Gathers the LLVM bitcode modules attached to compiled operations into a
    /// single LLVM module owned by `context`. Returns null on failure; otherwise
    /// the caller owns the returned module.
    pub fn reussirGatherCompiledModules(
        module: MlirModule,
        context: LLVMContextRef,
        data_layout: *const c_char,
    ) -> LLVMModuleRef;

    /// Rewrites the `{ tag, payload-union }` debug type emitted for each enum
    /// into a real DWARF `DW_TAG_variant_part`, so a debugger shows only the
    /// active case. Operates in place; a no-op when `module` has no debug info.
    pub fn reussirFixupVariantDebugInfo(module: LLVMModuleRef);

    /// Reports whether TPDE support was compiled into the backend.
    pub fn reussirHasTPDE() -> c_int;

    /// Stamps `module` with the target's layout facts before the lowering
    /// pipeline runs: the `llvm.data_layout` string, the `llvm.target_triple`,
    /// and the translated `dlti.dl_spec` that MLIR `DataLayout` queries read
    /// (without it MLIR falls back to conservative defaults, understating
    /// sizes and alignments). Returns false if `data_layout` does not parse.
    pub fn reussirModuleAttachTargetSpec(
        module: MlirModule,
        data_layout: *const c_char,
        triple: *const c_char,
    ) -> bool;

    //==-- LLVM-side codegen helpers (Jit.h) --==//

    /// Runs the Reussir LLVM optimization pipeline on `module` in place at the
    /// requested level (a no-op for `None`/`Tpde`). `opt` mirrors the backend's
    /// `ReussirOptOption`/`ReussirJitOptLevel` C enum.
    pub fn reussirRunBackendLLVMPipeline(module: LLVMModuleRef, opt: c_int);

    /// Compiles `module` to an ELF object with TPDE after stamping the given data
    /// layout and triple on it. Returns null if TPDE is unavailable or
    /// compilation fails; otherwise the caller owns the returned memory buffer.
    pub fn reussirTpdeCompileToObject(
        module: LLVMModuleRef,
        data_layout: *const c_char,
        triple: *const c_char,
    ) -> LLVMMemoryBufferRef;

    //==-- Reussir type constructors --==//
    //
    // The dialect types use custom printers/parsers, so the generic MLIR C API
    // cannot build them; each constructor wraps the type's C++ `get` builder.

    pub fn reussirRawPtrTypeGet(element_type: MlirType) -> MlirType;
    pub fn reussirTokenTypeGet(context: MlirContext, align: usize, size: usize) -> MlirType;
    pub fn reussirRegionTypeGet(context: MlirContext) -> MlirType;
    pub fn reussirRcTypeGet(
        element_type: MlirType,
        capability: ReussirCapability,
        atomic_kind: ReussirAtomicKind,
    ) -> MlirType;
    pub fn reussirNullableTypeGet(pointer_type: MlirType) -> MlirType;
    pub fn reussirRefTypeGet(
        element_type: MlirType,
        capability: ReussirCapability,
        atomic_kind: ReussirAtomicKind,
    ) -> MlirType;
    pub fn reussirHoleTypeGet(element_type: MlirType) -> MlirType;
    pub fn reussirRcBoxTypeGet(element_type: MlirType, regional: bool) -> MlirType;
    pub fn reussirClosureTypeGet(
        context: MlirContext,
        n_inputs: isize,
        input_types: *const MlirType,
        output_type: MlirType,
    ) -> MlirType;
    pub fn reussirClosureBoxTypeGet(
        context: MlirContext,
        n_payloads: isize,
        payload_types: *const MlirType,
    ) -> MlirType;
    pub fn reussirArrayTypeGet(
        n_dims: isize,
        shape: *const i64,
        element_type: MlirType,
    ) -> MlirType;
    pub fn reussirViewTypeGet(is_mutable: bool, array_type: MlirType) -> MlirType;
    pub fn reussirFFIObjectTypeGet(
        context: MlirContext,
        ffi_name: MlirAttribute,
        cleanup_hook: MlirAttribute,
    ) -> MlirType;
    pub fn reussirStrTypeGet(context: MlirContext, life_scope: ReussirLifeScope) -> MlirType;

    pub fn reussirRecordTypeGetComplete(
        context: MlirContext,
        n_members: isize,
        members: *const MlirType,
        member_is_field: *const bool,
        name: MlirAttribute,
        kind: ReussirRecordKind,
        default_capability: ReussirCapability,
    ) -> MlirType;
    pub fn reussirRecordTypeGetIncomplete(
        context: MlirContext,
        name: MlirAttribute,
        kind: ReussirRecordKind,
    ) -> MlirType;
    pub fn reussirRecordTypeComplete(
        record: MlirType,
        n_members: isize,
        members: *const MlirType,
        member_is_field: *const bool,
        default_capability: ReussirCapability,
    );
    pub fn reussirRecordTypeIsComplete(record: MlirType) -> bool;

    //==-- Reussir debug-info attribute constructors --==//
    //
    // The debug-info attributes describe a value's debug type/variable for the
    // debug-info conversion pass; their custom storage cannot be built through
    // the generic MLIR C API, so each constructor wraps the attribute's C++
    // `get` builder.

    pub fn reussirDBGIntTypeAttrGet(
        context: MlirContext,
        inner_type: MlirType,
        is_signed: bool,
        dbg_name: MlirAttribute,
    ) -> MlirAttribute;
    pub fn reussirDBGFPTypeAttrGet(
        context: MlirContext,
        inner_type: MlirType,
        dbg_name: MlirAttribute,
    ) -> MlirAttribute;
    pub fn reussirDBGRecordMemberAttrGet(
        context: MlirContext,
        name: MlirAttribute,
        type_attr: MlirAttribute,
    ) -> MlirAttribute;
    pub fn reussirDBGRecordTypeAttrGet(
        context: MlirContext,
        n_members: isize,
        members: *const MlirAttribute,
        is_variant: bool,
        underlying_type: MlirType,
        dbg_name: MlirAttribute,
    ) -> MlirAttribute;
    pub fn reussirDBGSubprogramAttrGet(
        context: MlirContext,
        raw_name: MlirAttribute,
        n_type_params: isize,
        type_params: *const MlirAttribute,
    ) -> MlirAttribute;
    pub fn reussirDBGBoxedTypeAttrGet(
        context: MlirContext,
        dbg_type: MlirAttribute,
        regional: bool,
    ) -> MlirAttribute;
    pub fn reussirDBGLocalVarAttrGet(
        context: MlirContext,
        dbg_type: MlirAttribute,
        var_name: MlirAttribute,
    ) -> MlirAttribute;
    pub fn reussirDBGFuncArgAttrGet(
        context: MlirContext,
        dbg_type: MlirAttribute,
        arg_name: MlirAttribute,
        arg_index: core::ffi::c_uint,
    ) -> MlirAttribute;

    /// Set an operation's location (`mlir-sys` does not bind
    /// `mlirOperationSetLocation`). Used to attach a fused debug location to a
    /// local variable's defining op.
    pub fn reussirOperationSetLocation(
        op: mlir_sys::MlirOperation,
        location: mlir_sys::MlirLocation,
    );
}
