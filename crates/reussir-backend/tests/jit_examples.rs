//! End-to-end examples: build Reussir IR with the high-level wrappers
//! ([`reussir_backend::dialect`] op builders + [`reussir_backend::dialect::ty`]
//! type constructors), lower it with the Reussir pipeline, and execute it
//! through the [`Jit`].
//!
//! Each test constructs a small `func.func` out of Reussir operations, lowers it
//! to the LLVM dialect, JIT-compiles it (with the Reussir runtime library loaded
//! so allocator calls resolve), and asserts on the value it returns.

use reussir_backend::dialect::{self, ty};
use reussir_backend::jit::Jit;
use reussir_backend::melior::Context;
use reussir_backend::melior::dialect::{arith, func};
use reussir_backend::melior::ir::attribute::{
    DenseI32ArrayAttribute, IntegerAttribute, StringAttribute, TypeAttribute,
};
use reussir_backend::melior::ir::block::BlockLike;
use reussir_backend::melior::ir::operation::{OperationBuilder, OperationLike};
use reussir_backend::melior::ir::r#type::FunctionType;
use reussir_backend::melior::ir::{
    Attribute, Block, Identifier, Location, Module, Operation, Region, RegionLike, Type, Value,
};
use reussir_backend::pipeline::{LoweringOptions, run_lowering_pipeline};

// Absolute path to the Reussir runtime shared library, resolved relative to the
// crate at build time so it is found regardless of the test's working directory.
fn runtime_library() -> String {
    let ext = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    format!(
        "{}/../../build/lib/libreussir_rt.{ext}",
        env!("CARGO_MANIFEST_DIR")
    )
}

// Adds a nullary, C-interface `func.func @name() -> result_type` whose body is
// produced by `body` (which must terminate the block, e.g. with `func.return`)
// to `module`.
fn add_function<'c>(
    context: &'c Context,
    module: &Module<'c>,
    name: &str,
    result_type: Type<'c>,
    body: impl FnOnce(&Block<'c>),
) {
    let location = Location::unknown(context);
    let block = Block::new(&[]);
    body(&block);

    let region = Region::new();
    region.append_block(block);

    let function = func::func(
        context,
        StringAttribute::new(context, name),
        TypeAttribute::new(FunctionType::new(context, &[], &[result_type]).into()),
        region,
        // `llvm.emit_c_interface` lets the engine invoke it by the packed ABI
        // used by `Jit::invoke_packed`.
        &[(
            Identifier::new(context, "llvm.emit_c_interface"),
            Attribute::parse(context, "unit").unwrap(),
        )],
        location,
    );
    module.body().append_operation(function);
}

// Appends `operation` and returns its first result as a value. Accepts both
// melior's plain `Operation` builders and the generated typed Reussir op structs
// (which convert via `Into<Operation>`).
fn result_of<'c, 'a>(block: &'a Block<'c>, operation: impl Into<Operation<'c>>) -> Value<'c, 'a> {
    block
        .append_operation(operation.into())
        .result(0)
        .unwrap()
        .into()
}

// Builds a `reussir.rc.create` that takes only a value (no token/region). The
// op carries `AttrSizedOperandSegments`, and melior's generated builder does not
// populate `operandSegmentSizes`, so we construct it directly and set the
// segment sizes ([value, token, region] = [1, 0, 0]). The pipeline's
// token-instantiation pass supplies the missing allocation token.
fn rc_create_value<'c, 'a>(
    context: &'c Context,
    block: &'a Block<'c>,
    rc_type: Type<'c>,
    value: Value<'c, '_>,
    location: Location<'c>,
) -> Value<'c, 'a> {
    let operation = OperationBuilder::new("reussir.rc.create", location)
        .add_operands(&[value])
        .add_attributes(&[(
            Identifier::new(context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(context, &[1, 0, 0]).into(),
        )])
        .add_results(&[rc_type])
        .build()
        .expect("valid reussir.rc.create");
    block.append_operation(operation).result(0).unwrap().into()
}

// Loads the Reussir runtime into the global symbol namespace so the JIT's
// process symbol generator can resolve `__reussir_allocate` and friends. (The
// engine's `shared_library_paths` alone do not expose them on this platform.)
fn preload_runtime() {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int, c_void};
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    }
    const RTLD_NOW: c_int = 0x2;
    const RTLD_GLOBAL: c_int = 0x100;
    let path = CString::new(runtime_library()).unwrap();
    let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
    assert!(
        !handle.is_null(),
        "failed to load the Reussir runtime library"
    );
}

// Lowers `module` to the LLVM dialect and returns a JIT with the Reussir runtime
// loaded.
fn lower_and_jit(context: &Context, module: &mut Module) -> Jit {
    assert!(
        module.as_operation().verify(),
        "constructed module should verify before lowering"
    );
    run_lowering_pipeline(context, module, &LoweringOptions::default())
        .expect("lowering pipeline should succeed");
    assert!(
        module.as_operation().verify(),
        "lowered module should verify"
    );
    // The runtime is loaded into the global namespace (see `preload_runtime`),
    // so no per-engine shared libraries are needed for symbol resolution.
    preload_runtime();
    Jit::new(module, 2, &[])
}

// Invokes a nullary function returning a 64-bit-sized scalar and reads it back.
fn call_returning_i64(jit: &Jit, name: &str) -> i64 {
    let mut result: i64 = 0;
    unsafe {
        jit.invoke_packed(name, &mut [&mut result as *mut i64 as *mut ()])
            .expect("invocation should succeed");
    }
    result
}

/// Example 1: allocate and free a memory token through the runtime allocator.
/// `reussir.token.alloc` / `reussir.token.free` lower to `__reussir_allocate` /
/// `__reussir_deallocate`; the function returns 0 to prove it ran.
#[test]
fn token_alloc_free_round_trip() {
    let context = reussir_backend::context();
    let i64_type = Type::parse(&context, "i64").unwrap();
    let mut module = Module::new(Location::unknown(&context));

    add_function(&context, &module, "tok", i64_type, |block| {
        let location = Location::unknown(&context);
        let token_type = ty::token(&context, 8, 16);

        let token = result_of(block, dialect::token_alloc(&context, token_type, location));
        block.append_operation(dialect::token_free(&context, token, location).into());

        let zero = result_of(
            block,
            arith::constant(
                &context,
                IntegerAttribute::new(i64_type, 0).into(),
                location,
            ),
        );
        block.append_operation(func::r#return(&[zero], location));
    });

    let jit = lower_and_jit(&context, &mut module);
    assert_eq!(call_returning_i64(&jit, "tok"), 0);
}

/// Example 2: a manual heap round-trip with no reference counting. Allocate a
/// token, reinterpret it as a typed reference, store 42 through it, load the
/// value back, then free the token. Exercises `reussir.token.alloc`,
/// `reussir.token.reinterpret`, `reussir.ref.store`, `reussir.ref.load` and
/// `reussir.token.free`.
#[test]
fn token_reinterpret_store_load() {
    let context = reussir_backend::context();
    let i64_type = Type::parse(&context, "i64").unwrap();
    let mut module = Module::new(Location::unknown(&context));

    add_function(&context, &module, "store_load", i64_type, |block| {
        let location = Location::unknown(&context);
        // Token alignment must match the reinterpreted element's alignment.
        let token_type = ty::token(&context, 4, 8);
        // `ref.store` requires a `field`-capability reference.
        let ref_type = ty::r#ref(
            i64_type,
            ty::ReussirCapability::Field,
            ty::ReussirAtomicKind::Normal,
        );

        let token = result_of(block, dialect::token_alloc(&context, token_type, location));
        let reference = result_of(
            block,
            dialect::token_reinterpret(&context, ref_type, token, location),
        );
        let value = result_of(
            block,
            arith::constant(
                &context,
                IntegerAttribute::new(i64_type, 42).into(),
                location,
            ),
        );
        block.append_operation(dialect::ref_store(&context, reference, value, location).into());
        let loaded = result_of(
            block,
            dialect::ref_load(&context, i64_type, reference, location),
        );
        block.append_operation(dialect::token_free(&context, token, location).into());
        block.append_operation(func::r#return(&[loaded], location));
    });

    let jit = lower_and_jit(&context, &mut module);
    assert_eq!(call_returning_i64(&jit, "store_load"), 42);
}

/// Example 3: create an `!reussir.rc<i64>` on the heap and read its reference
/// count with `reussir.rc.fetch`. A freshly created Rc has a count of 1.
#[test]
fn rc_create_fetch_refcount() {
    let context = reussir_backend::context();
    let i64_type = Type::parse(&context, "i64").unwrap();
    let index_type = Type::parse(&context, "index").unwrap();
    let mut module = Module::new(Location::unknown(&context));

    add_function(&context, &module, "rc_refcount", index_type, |block| {
        let location = Location::unknown(&context);
        let rc_type = ty::rc(
            i64_type,
            ty::ReussirCapability::Shared,
            ty::ReussirAtomicKind::Normal,
        );

        let value = result_of(
            block,
            arith::constant(
                &context,
                IntegerAttribute::new(i64_type, 0).into(),
                location,
            ),
        );
        let rc = rc_create_value(&context, block, rc_type, value, location);
        let count = result_of(block, dialect::rc_fetch(&context, index_type, rc, location));
        block.append_operation(func::r#return(&[count], location));
    });

    let jit = lower_and_jit(&context, &mut module);
    // `index` is pointer-sized; on a 64-bit host it reads back as an i64.
    assert_eq!(call_returning_i64(&jit, "rc_refcount"), 1);
}
