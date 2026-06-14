//! End-to-end examples: build Reussir IR with the high-level wrappers
//! ([`reussir_backend::dialect`] op builders + [`reussir_backend::dialect::ty`]
//! type constructors), lower it with the Reussir pipeline, and execute it
//! through the [`OrcJit`].
//!
//! Each test constructs a small `func.func` out of Reussir operations, lowers it
//! to the LLVM dialect, adds it to a persistent ORC session (with the Reussir
//! runtime library loaded so allocator calls resolve), looks the function up and
//! calls it directly through the C ABI.

use reussir_backend::dialect::{self, ty};
use reussir_backend::melior::Context;
use reussir_backend::melior::dialect::{arith, func};
use reussir_backend::melior::ir::attribute::{
    DenseI32ArrayAttribute, IntegerAttribute, StringAttribute, TypeAttribute,
};
use reussir_backend::melior::ir::block::BlockLike;
use reussir_backend::melior::ir::operation::{OperationBuilder, OperationLike};
use reussir_backend::melior::ir::r#type::FunctionType;
use reussir_backend::melior::ir::{
    Block, Identifier, Location, Module, Operation, Region, RegionLike, Type, Value,
};
use reussir_backend::pipeline::{LoweringOptions, run_lowering_pipeline};
use reussir_jit::{OptLevel, OrcJit};

// Adds a nullary `func.func @name() -> result_type` whose body is produced by
// `body` (which must terminate the block, e.g. with `func.return`) to `module`.
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
        &[],
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

// Builds a `reussir.rc.create` that takes only a value (no token/region). The op
// carries `AttrSizedOperandSegments`, and melior's generated builder does not
// populate `operandSegmentSizes`, so we construct it directly and set the segment
// sizes ([value, token, region] = [1, 0, 0]). The pipeline's token-instantiation
// pass supplies the missing allocation token.
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

// Locates the Reussir runtime shared library for these tests.
//
// TODO: `CARGO_MANIFEST_DIR` only happens to sit next to the CMake `build/`
// tree in this checkout — it is a build-time hint, not where the runtime ships
// in a real install. A production lookup should go through an explicit
// mechanism (e.g. `REUSSIR_RT_LIBRARY` / the install layout, as
// `reussir_jit::runtime_library_path` does) rather than the crate manifest dir.
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

// Lowers `module` to the LLVM dialect and adds it to a fresh ORC session with the
// Reussir runtime loaded.
fn jit_module(context: &Context, module: &mut Module) -> OrcJit {
    run_lowering_pipeline(context, module, &LoweringOptions::default())
        .expect("lowering pipeline should succeed");
    assert!(
        module.as_operation().verify(),
        "lowered module should verify"
    );

    let jit = OrcJit::new().expect("create JIT");
    jit.add_library(&runtime_library())
        .expect("load the Reussir runtime");
    jit.add_module(module, OptLevel::Default)
        .expect("add module to JIT");
    jit
}

// Looks up a nullary function returning a 64-bit-sized scalar and calls it.
fn call_returning_i64(jit: &OrcJit, name: &str) -> i64 {
    let address = jit.lookup(name).expect("symbol should be found");
    let function: extern "C" fn() -> i64 = unsafe { std::mem::transmute(address as usize) };
    function()
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

    let jit = jit_module(&context, &mut module);
    assert_eq!(call_returning_i64(&jit, "tok"), 0);
}

/// Example 2: a manual heap round-trip with no reference counting. Allocate a
/// token, reinterpret it as a typed reference, store 42 through it, load the value
/// back, then free the token. Exercises `reussir.token.alloc`,
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

    let jit = jit_module(&context, &mut module);
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

    let jit = jit_module(&context, &mut module);
    // `index` is pointer-sized; on a 64-bit host it reads back as an i64.
    assert_eq!(call_returning_i64(&jit, "rc_refcount"), 1);
}
