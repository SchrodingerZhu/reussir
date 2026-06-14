//! Self-contained example modules used by the command-line tool and the lit
//! tests. These exercise the builder and writer end to end and double as living
//! documentation of how a frontend assembles IR.
//!
//! Each demo returns the finished `builtin.module` operation, ready to hand to
//! `write_module`.

use reussir_bytecode::builder::ModuleBuilder;
use reussir_bytecode::context::{Atomicity, Capability, Context, FloatKind};
use reussir_bytecode::dialects::CmpIPredicate;
use reussir_bytecode::ir::Op;

/// Build the demo selected by `name`, or return `None` for an unknown name.
pub fn build<'a>(ctx: &Context<'a>, name: &str) -> Option<Op<'a>> {
    match name {
        "basic" => Some(basic(ctx)),
        "reussir_types" => Some(reussir_types(ctx)),
        "control_flow" => Some(control_flow(ctx)),
        "calls_and_locs" => Some(calls_and_locs(ctx)),
        "intrinsics" => Some(intrinsics(ctx)),
        "records" => Some(records(ctx)),
        "dispatch" => Some(dispatch(ctx)),
        "misc" => Some(misc(ctx)),
        "ffi" => Some(ffi(ctx)),
        "types" => Some(types(ctx)),
        _ => None,
    }
}

/// The names of all available demos.
pub const NAMES: &[&str] = &[
    "basic",
    "reussir_types",
    "control_flow",
    "calls_and_locs",
    "intrinsics",
    "ffi",
    "records",
    "dispatch",
    "misc",
    "types",
];

/// A module whose function signature exercises the remaining wrapped Reussir
/// types: `raw_ptr`, `hole`, `closure_box`, `array`, and `ffi_object`.
pub fn types<'a>(ctx: &Context<'a>) -> Op<'a> {
    let i32 = ctx.int(32);
    let i64 = ctx.int(64);
    let raw_ptr = ctx.raw_ptr(i32);
    let hole = ctx.hole(i32);
    let closure_box = ctx.closure_box(&[i32, i64]);
    let array = ctx.array(&[2, 3], i32);
    let ffi_object = ctx.ffi_object("Vec", "cleanup");
    let mut module = ModuleBuilder::new(ctx, "types");
    module.function(
        "sig",
        &[raw_ptr, hole, closure_box, array, ffi_object],
        &[],
        |f| {
            f.ret(&[]);
        },
    );
    module.finish()
}

/// A module with a single integer-adding function, exercising builtin types,
/// the `func` and `arith` dialects, operands, and results.
///
/// ```mlir
/// module @demo {
///   func.func @add(%a: i32, %b: i32) -> i32 {
///     %0 = arith.addi %a, %b : i32
///     func.return %0 : i32
///   }
/// }
/// ```
pub fn basic<'a>(ctx: &Context<'a>) -> Op<'a> {
    let i32 = ctx.int(32);
    let mut module = ModuleBuilder::new(ctx, "demo");
    module.function("add", &[i32, i32], &[i32], |f| {
        let (op, sum) = f.ctx().arith_binary("arith.addi", f.arg(0), f.arg(1), i32);
        f.push(op);
        f.ret(&[sum]);
    });
    module.finish()
}

/// A module exercising the `arith` and `math` intrinsic helpers: integer and
/// floating point arithmetic, comparison, conversion, a math call, and a select.
///
/// ```mlir
/// module @intrinsics {
///   func.func @compute(%a: i32, %b: i32, %x: f32) -> f32 {
///     %0 = arith.addi %a, %b : i32
///     %1 = arith.cmpi sgt, %a, %b : i32
///     %2 = arith.sitofp %0 : i32 to f32
///     %3 = math.sqrt %x : f32
///     %4 = arith.addf %2, %3 : f32
///     %5 = arith.select %1, %4, %x : f32
///     func.return %5 : f32
///   }
/// }
/// ```
pub fn intrinsics<'a>(ctx: &Context<'a>) -> Op<'a> {
    let i32 = ctx.int(32);
    let f32 = ctx.float(FloatKind::F32);
    let mut module = ModuleBuilder::new(ctx, "intrinsics");
    module.function("compute", &[i32, i32, f32], &[f32], |f| {
        let c = f.ctx();
        let (a, b, x) = (f.arg(0), f.arg(1), f.arg(2));
        let (sum_op, sum) = c.arith_binary("arith.addi", a, b, i32);
        let (cmp_op, cmp) = c.arith_cmpi(CmpIPredicate::Sgt, a, b);
        let (conv_op, conv) = c.arith_cast("arith.sitofp", sum, f32);
        let (sqrt_op, sqrt) = c.math_unary("math.sqrt", x, f32);
        let (add_op, total) = c.arith_binary("arith.addf", conv, sqrt, f32);
        let (sel_op, result) = c.arith_select(cmp, total, x, f32);
        for op in [sum_op, cmp_op, conv_op, sqrt_op, add_op, sel_op] {
            f.push(op);
        }
        f.ret(&[result]);
    });
    module.finish()
}

/// A module exercising nested, non-isolated regions (`scf.if`), the value
/// numbering that spans region boundaries, and operation properties carried in
/// the attribute dictionary (`arith.constant`'s `value`).
///
/// ```mlir
/// module @cf {
///   func.func @pick(%cond: i1) -> i32 {
///     %c10 = arith.constant 10 : i32
///     %c20 = arith.constant 20 : i32
///     %r = scf.if %cond -> (i32) {
///       scf.yield %c10 : i32
///     } else {
///       scf.yield %c20 : i32
///     }
///     func.return %r : i32
///   }
/// }
/// ```
pub fn control_flow<'a>(ctx: &Context<'a>) -> Op<'a> {
    let i1 = ctx.int(1);
    let i32 = ctx.int(32);
    let mut module = ModuleBuilder::new(ctx, "cf");
    module.function("pick", &[i1], &[i32], |f| {
        let c = f.ctx();
        let cond = f.arg(0);

        let (c10_op, c10) = c.arith_constant_int(10, i32);
        let (c20_op, c20) = c.arith_constant_int(20, i32);
        f.push(c10_op);
        f.push(c20_op);

        // Each branch yields one of the outer constants, so the regions define
        // no values of their own but reference values numbered above them.
        let then_block = c.block_unknown_locs(&[], &[c.scf_yield(&[c10])]);
        let else_block = c.block_unknown_locs(&[], &[c.scf_yield(&[c20])]);
        let (if_op, results) = c.scf_if(
            cond,
            &[i32],
            c.region(&[then_block]),
            c.region(&[else_block]),
        );
        f.push(if_op);
        f.ret(results);
    });
    module.finish()
}

/// A module exercising symbol references (`func.call`) and source locations.
/// The call carries a file/line/column location that survives the round-trip
/// (visible with `--mlir-print-debuginfo`).
///
/// ```mlir
/// module @calls {
///   func.func @callee(%x: i32) -> i32 { func.return %x : i32 }
///   func.func @caller(%y: i32) -> i32 {
///     %r = func.call @callee(%y) : (i32) -> i32 loc("caller.rs":2:9)
///     func.return %r : i32
///   }
/// }
/// ```
pub fn calls_and_locs<'a>(ctx: &Context<'a>) -> Op<'a> {
    let i32 = ctx.int(32);
    let mut module = ModuleBuilder::new(ctx, "calls");
    module.function("callee", &[i32], &[i32], |f| {
        let x = f.arg(0);
        f.ret(&[x]);
    });
    module.function("caller", &[i32], &[i32], |f| {
        let c = f.ctx();
        let y = f.arg(0);
        // func.call carries a source location threaded through from the frontend.
        let callee = c.attr_symbol("callee");
        let (call, results) = c
            .op("func.call")
            .attrs(c.attr_dict(&[("callee", callee)]))
            .operand(y)
            .result(i32)
            .loc(c.loc_file("caller.rs", 2, 9))
            .build();
        f.push(call);
        f.ret(results);
    });
    module.finish()
}

/// A module exercising the remaining catalog: nullable checks/creation, the
/// string operations, `scf.index_switch`, and `reussir.panic`.
pub fn misc<'a>(ctx: &Context<'a>) -> Op<'a> {
    use reussir_bytecode::context::LifeScope;
    let i1 = ctx.int(1);
    let i32 = ctx.int(32);
    let index = ctx.index();
    let rc_i32 = ctx.rc(i32, Capability::Unspecified, Atomicity::NonAtomic);
    let nullable_rc = ctx.nullable(rc_i32);
    let str_global = ctx.str_ty(LifeScope::Global);
    let str_local = ctx.str_ty(LifeScope::Local);

    let mut module = ModuleBuilder::new(ctx, "misc");

    // A global string and operations over it.
    module.push(ctx.str_global("greeting", "hello"));
    module.function("string_len", &[], &[index], |f| {
        let c = f.ctx();
        let (lit, g) = c.str_literal("greeting", str_global);
        let (cast, l) = c.str_cast(g, str_local);
        let (len, n) = c.str_len(l);
        f.push(lit);
        f.push(cast);
        f.push(len);
        f.ret(&[n]);
    });

    // Nullable: test and construct.
    module.function("is_null", &[nullable_rc], &[i1], |f| {
        let (op, b) = f.ctx().nullable_check(f.arg(0));
        f.push(op);
        f.ret(&[b]);
    });
    module.function("wrap", &[rc_i32], &[nullable_rc], |f| {
        let (op, n) = f.ctx().nullable_create(Some(f.arg(0)), nullable_rc);
        f.push(op);
        f.ret(&[n]);
    });

    // scf.index_switch with two cases and a default.
    module.function("classify", &[index], &[i32], |f| {
        let c = f.ctx();
        let case0 = {
            let (k, v) = c.arith_constant_int(10, i32);
            c.region(&[c.block_unknown_locs(&[], &[k, c.scf_yield(&[v])])])
        };
        let case1 = {
            let (k, v) = c.arith_constant_int(20, i32);
            c.region(&[c.block_unknown_locs(&[], &[k, c.scf_yield(&[v])])])
        };
        let default = {
            let (k, v) = c.arith_constant_int(0, i32);
            c.region(&[c.block_unknown_locs(&[], &[k, c.scf_yield(&[v])])])
        };
        let (sw, results) = c.scf_index_switch(f.arg(0), &[0, 1], &[case0, case1], default, &[i32]);
        f.push(sw);
        f.ret(results);
    });

    // A function that always panics.
    module.function("boom", &[], &[i32], |f| {
        let c = f.ctx();
        f.push(c.panic("unreachable"));
        let (k, v) = c.arith_constant_int(0, i32);
        f.push(k);
        f.ret(&[v]);
    });

    module.finish()
}

/// A module exercising the foreign-function interface ops: a `reussir.trampoline`
/// exporting a function under a C ABI, and a `reussir.polyffi` stub.
pub fn ffi<'a>(ctx: &Context<'a>) -> Op<'a> {
    let i32 = ctx.int(32);
    let mut module = ModuleBuilder::new(ctx, "ffi");
    module.function("id_target", &[i32], &[i32], |f| {
        let x = f.arg(0);
        f.ret(&[x]);
    });
    // Export id_target under the C ABI (direction 1 = export, 0 = import).
    module.push(ctx.trampoline("id_target_ffi", "id_target", "C", 1));
    // A polymorphic FFI stub with a substitution dictionary mapping a type
    // parameter and a size constant.
    let subs = ctx.attr_dict(&[
        ("Elem", ctx.attr_type(i32)),
        ("Size", ctx.attr_int(4, ctx.int(64))),
    ]);
    module.push(ctx.polyffi("extern_call(%0)", subs));
    module.finish()
}

/// A module exercising record types — including a recursive one — record
/// construction, and `reussir.rc.create` with its `operandSegmentSizes`.
///
/// ```mlir
/// module @records {
///   func.func @make_pair(%a: i32, %b: i64)
///       -> !reussir.rc<!reussir.record<compound "Pair" [value] {i32, i64}>> {
///     %0 = reussir.record.compound(%a, %b) ...
///     %1 = reussir.rc.create(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>}> ...
///     func.return %1
///   }
///   func.func @id(%l: !reussir.rc<!reussir.record<variant "List" {...}>>) -> ... {
///     reussir.rc.inc(%l) ; func.return %l
///   }
/// }
/// ```
pub fn records<'a>(ctx: &Context<'a>) -> Op<'a> {
    let i32 = ctx.int(32);
    let i64 = ctx.int(64);

    // A simple product type Pair { i32, i64 }.
    ctx.define_compound(
        "Pair",
        Capability::Value,
        &[(ctx.field(i32), false), (ctx.field(i64), false)],
    );
    let pair = ctx.record("Pair");
    let rc_pair = ctx.rc(pair, Capability::Unspecified, Atomicity::NonAtomic);

    // A recursive sum type: List = Cons(i32, List) | Nil.
    ctx.define_variant(
        "List",
        &[
            ctx.field_record("List::Cons"),
            ctx.field_record("List::Nil"),
        ],
    );
    ctx.define_compound(
        "List::Cons",
        Capability::Value,
        &[(ctx.field(i32), false), (ctx.field_record("List"), false)],
    );
    ctx.define_compound("List::Nil", Capability::Value, &[]);
    let list = ctx.record("List");
    let rc_list = ctx.rc(list, Capability::Unspecified, Atomicity::NonAtomic);

    let mut module = ModuleBuilder::new(ctx, "records");
    module.function("make_pair", &[i32, i64], &[rc_pair], |f| {
        let c = f.ctx();
        let (compound, p) = c.record_compound(&[f.arg(0), f.arg(1)], pair);
        let (create, rc) = c.rc_create(p, None, None, rc_pair);
        f.push(compound);
        f.push(create);
        f.ret(&[rc]);
    });
    module.function("id", &[rc_list], &[rc_list], |f| {
        let l = f.arg(0);
        f.push(f.ctx().rc_inc(l));
        f.ret(&[l]);
    });
    module.finish()
}

/// A module exercising `reussir.record.dispatch`, `reussir.ref.project`,
/// `reussir.ref.load`, and `reussir.scf.yield`: unwrap an `Option<i32>` held
/// behind a reference, returning a default for the `None` arm.
pub fn dispatch<'a>(ctx: &Context<'a>) -> Op<'a> {
    let i32 = ctx.int(32);

    // Option = Some(i32) | None.
    ctx.define_variant(
        "Option",
        &[
            ctx.field_record("Option::Some"),
            ctx.field_record("Option::None"),
        ],
    );
    ctx.define_compound(
        "Option::Some",
        Capability::Value,
        &[(ctx.field(i32), false)],
    );
    ctx.define_compound("Option::None", Capability::Value, &[]);

    let some = ctx.record("Option::Some");
    let none = ctx.record("Option::None");
    let option = ctx.record("Option");
    let ref_option = ctx.ref_ty(option, Capability::Unspecified, Atomicity::NonAtomic);
    let ref_some = ctx.ref_ty(some, Capability::Unspecified, Atomicity::NonAtomic);
    let ref_none = ctx.ref_ty(none, Capability::Unspecified, Atomicity::NonAtomic);
    let ref_i32 = ctx.ref_ty(i32, Capability::Unspecified, Atomicity::NonAtomic);

    let mut module = ModuleBuilder::new(ctx, "dispatch");
    module.function("unwrap_or", &[ref_option, i32], &[i32], |f| {
        let c = f.ctx();
        let opt = f.arg(0);
        let default = f.arg(1);

        // Some arm: project field 0, load it, yield it.
        let some_arg = c.value(ref_some);
        let (proj, field_ref) = c.ref_project(some_arg, 0, ref_i32);
        let (load, val) = c.ref_load(field_ref, i32);
        let some_block =
            c.block_unknown_locs(&[some_arg], &[proj, load, c.reussir_scf_yield(&[val])]);

        // None arm: yield the default.
        let none_arg = c.value(ref_none);
        let none_block = c.block_unknown_locs(&[none_arg], &[c.reussir_scf_yield(&[default])]);

        let (disp, results) = c.record_dispatch(
            opt,
            &[&[0], &[1]],
            &[c.region(&[some_block]), c.region(&[none_block])],
            &[i32],
        );
        f.push(disp);
        f.ret(results);
    });
    module.finish()
}

/// A module exercising Reussir dialect types and operations: a function over a
/// reference-counted integer that increments its count and returns it.
///
/// ```mlir
/// module @rc_demo {
///   func.func @bump(%p: !reussir.rc<i32>) -> !reussir.rc<i32> {
///     reussir.rc.inc(%p : !reussir.rc<i32>)
///     func.return %p : !reussir.rc<i32>
///   }
/// }
/// ```
pub fn reussir_types<'a>(ctx: &Context<'a>) -> Op<'a> {
    let i32 = ctx.int(32);
    let rc_i32 = ctx.rc(i32, Capability::Unspecified, Atomicity::NonAtomic);
    let mut module = ModuleBuilder::new(ctx, "rc_demo");
    module.function("bump", &[rc_i32], &[rc_i32], |f| {
        let p = f.arg(0);
        // reussir.rc.inc has no results; it just takes the rc pointer.
        f.push(f.ctx().rc_inc(p));
        f.ret(&[p]);
    });
    module.finish()
}
