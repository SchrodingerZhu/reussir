//! End-to-end check that `rrc -g` emits discriminated enum debug info.
//!
//! An enum lowers (in the debug-info conversion) to a `{ tag, payload-union }`
//! composite because MLIR's `DICompositeTypeAttr` cannot carry a discriminator.
//! The post-translation `fixupVariantDebugInfo` step then rewrites it into a
//! real DWARF `DW_TAG_variant_part`, and the compile unit is tagged Rust so
//! debuggers expose it. This drives the whole AOT pipeline and asserts the
//! emitted LLVM IR carries that discriminated form.

use std::path::PathBuf;
use std::process::Command;

fn emit_llvm_ir(source: &str, stem: &str) -> String {
    let dir = std::env::temp_dir();
    let rr = dir.join(format!("{stem}.rr"));
    let ll = dir.join(format!("{stem}.ll"));
    std::fs::write(&rr, source).expect("write source");
    let status = Command::new(env!("CARGO_BIN_EXE_rrc"))
        .args(["-O", "none", "-g", "--emit", "llvm-ir"])
        .arg("-o")
        .arg(&ll)
        .arg(&rr)
        .status()
        .expect("run rrc");
    assert!(status.success(), "rrc failed for {stem}");
    let ir = std::fs::read_to_string(&ll).expect("read llvm-ir");
    let _ = std::fs::remove_file(&rr);
    let _ = std::fs::remove_file(PathBuf::from(&ll));
    ir
}

#[test]
fn enum_debug_info_is_a_discriminated_variant_part() {
    // A value enum and a recursive shared enum; both must become variant parts.
    let ir = emit_llvm_ir(
        r#"
        enum [value] Shape { Dot, Circle(i64), Rect(i64, i64) }
        enum List { Nil, Cons(i64, List) }
        fn see_shape(s: Shape) -> i64 { let k = 0; k }
        fn see_list(l: List) -> i64 { let z = 0; z }
        pub fn run(n: i64) -> i64 {
            see_shape(Shape::Rect{n, n + 1}) + see_list(List::Cons{n, List::Nil})
        }
        extern "C" trampoline "run" = run;
        "#,
        "variant_dbg_e2e",
    );

    // Rust compile unit — this is what makes debuggers honour the variant part.
    assert!(
        ir.contains("DW_LANG_Rust"),
        "compile unit should be tagged Rust:\n{ir}"
    );
    // The discriminated form: a variant part with per-case discriminant values.
    assert!(
        ir.contains("DW_TAG_variant_part"),
        "enum debug info should be a DW_TAG_variant_part:\n{ir}"
    );
    assert!(
        ir.contains("extraData: i64"),
        "variant members should carry a discriminant value:\n{ir}"
    );
    // One variant part per enum (Shape and List), not the old `{tag, union}`.
    let parts = ir.matches("DW_TAG_variant_part").count();
    assert!(
        parts >= 2,
        "expected a variant part per enum, found {parts}"
    );
    // The old union form must be gone — the fixup rewrites it in place.
    assert!(
        !ir.contains("DW_TAG_union_type"),
        "the `payload` union should have been replaced by the variant part:\n{ir}"
    );
}
