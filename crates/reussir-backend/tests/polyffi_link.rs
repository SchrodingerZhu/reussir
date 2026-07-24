//! Exercises the polymorphic-FFI gather/link step shared by the JIT and the AOT
//! compiler: a `reussir.polyffi` template is compiled to LLVM bitcode, gathered
//! ([`LlvmLowering::prepare`]), and linked into the finalized module
//! ([`LlvmLowering::finish`]) so its function ends up *defined* — not merely
//! declared — in the resulting LLVM IR. Without the gather/link, the symbol would
//! remain an undefined external and fail to link.
//!
//! Ignored by default: it shells out to `rustc` to compile the embedded Rust
//! (needs `REUSSIR_RUSTC`/`rustc` on a known path, and a `REUSSIR_RUSTC_DEPS`
//! directory for its `-L`). Run explicitly with:
//!
//! ```text
//! cargo test -p reussir-backend --test polyffi_link -- --ignored
//! ```

use std::ffi::CString;

use reussir_backend::context;
use reussir_backend::llvm::{LlvmLowering, PolyffiPaths};
use reussir_backend::melior::ir::Module;
use reussir_backend::pipeline::{LoweringOptions, run_lowering_pipeline};

// A self-contained polyffi template (no `extern crate reussir_rt`, so the
// `-L` deps dir only has to exist) defining one `extern "C"` function, plus a
// caller that references it through a `func.func` declaration.
const MODULE: &str = r##"
module {
  reussir.polyffi texture("#[no_mangle] pub unsafe extern \"C\" fn __reussir_polyffi_answer() -> i64 { 42 }")
  func.func private @__reussir_polyffi_answer() -> i64
  func.func @entry() -> i64 {
    %0 = func.call @__reussir_polyffi_answer() : () -> i64
    func.return %0 : i64
  }
}
"##;

#[test]
#[ignore = "requires rustc + a REUSSIR_RUSTC_DEPS directory for the polyffi bitcode compile"]
fn polyffi_function_is_linked_in() {
    // The Rust-compiler helper requires a non-empty package directory even when
    // the template pulls in no external crates; any existing directory will do,
    // passed explicitly through `PolyffiPaths` (rustc still resolves through
    // the environment/probe discovery).
    let paths = PolyffiPaths {
        libdirs: vec![env!("CARGO_MANIFEST_DIR").to_owned()],
        ..PolyffiPaths::default()
    };

    let context = context();
    let mut module = Module::parse(&context, MODULE).expect("polyffi module parses");

    // Gather (compiles the template + collects its bitcode) must run before the
    // lowering pipeline erases the polyffi op; the link happens in `finish`.
    let prepared = LlvmLowering::prepare(&module, "", false, &paths)
        .expect("polyffi compile + gather succeeds");
    run_lowering_pipeline(&context, &mut module, &LoweringOptions::default())
        .expect("lowering pipeline succeeds");
    let finalized = prepared.finish(&module).expect("translate + link succeeds");

    unsafe {
        let name = CString::new("__reussir_polyffi_answer").unwrap();
        let function = llvm_sys::core::LLVMGetNamedFunction(finalized.module, name.as_ptr());
        assert!(
            !function.is_null(),
            "polyffi function missing from the linked module"
        );
        assert_eq!(
            llvm_sys::core::LLVMIsDeclaration(function),
            0,
            "polyffi function is only declared, not defined — gather/link did not run"
        );
        llvm_sys::core::LLVMDisposeModule(finalized.module);
        llvm_sys::core::LLVMContextDispose(finalized.context);
    }
}
