// RUN: %reussir-opt --reussir-compile-polymorphic-ffi=optimized %s \
// RUN:   -reussir-lowering-basic-ops --convert-to-llvm \
// RUN:   --reconcile-unrealized-casts | %reussir-translate --reussir-to-llvmir \
// RUN:   | %opt -O3 | %llc -relocation-model=pic -filetype=obj -o %t.o
// RUN: %cc %t.o -L%library_path -lreussir_rt \
// RUN:   %rpath_flag %extra_sys_libs -o %t.exe
// RUN: %t.exe
!Veci64 = !reussir.rc<!reussir.ffi_object<"::reussir_rt::collections::vec::Vec<i64>", @__reussir_polyffi_Vec_i64_drop>>
!VecVeci64 = !reussir.rc<!reussir.ffi_object<"::reussir_rt::collections::vec::Vec<::reussir_rt::collections::vec::Vec<i64>>", @__reussir_polyffi_Vec_NestedVec_drop>>
module {
    reussir.polyffi 
        texture("#![feature(linkage)] extern crate reussir_rt; use reussir_rt::collections::vec::Vec; #[linkage = \"weak_odr\"]#[unsafe(no_mangle)] pub unsafe extern \"C\" fn __reussir_polyffi_Vec_[:typename:]_new() -> Vec<[:typename:]> {Vec::new()} #[linkage = \"weak_odr\"]#[unsafe(no_mangle)] pub unsafe extern \"C\" fn __reussir_polyffi_Vec_[:typename:]_drop(_ : Vec<[:typename:]>) {} #[linkage = \"weak_odr\"]#[unsafe(no_mangle)] pub unsafe extern \"C\" fn __reussir_polyffi_Vec_[:typename:]_push(vec : Vec<[:typename:]>, ele: [:typename:]) -> Vec<[:typename:]> {Vec::push(vec, ele)}")
        substitutions({"typename" = "i64"})
    reussir.polyffi 
        texture("#![feature(linkage)] extern crate reussir_rt; use reussir_rt::collections::vec::Vec; [:NestedVec:] #[linkage = \"weak_odr\"]#[unsafe(no_mangle)] pub unsafe extern \"C\" fn __reussir_polyffi_Vec_[:typename:]_new() -> Vec<[:typename:]> {Vec::new()} #[linkage = \"weak_odr\"]#[unsafe(no_mangle)] pub unsafe extern \"C\" fn __reussir_polyffi_Vec_[:typename:]_drop(_ : Vec<[:typename:]>) {} #[linkage = \"weak_odr\"]#[unsafe(no_mangle)] pub unsafe extern \"C\" fn __reussir_polyffi_Vec_[:typename:]_push(vec : Vec<[:typename:]>, ele: [:typename:]) -> Vec<[:typename:]> {Vec::push(vec, ele)}")
        substitutions({"typename" = "NestedVec", "NestedVec" = !Veci64})
    func.func private @__reussir_polyffi_Vec_i64_new() -> !Veci64
    func.func private @__reussir_polyffi_Vec_i64_drop(!Veci64)
    func.func private @__reussir_polyffi_Vec_i64_push(!Veci64, i64) -> !Veci64
    func.func private @__reussir_polyffi_Vec_NestedVec_new() -> !VecVeci64
    func.func private @__reussir_polyffi_Vec_NestedVec_drop(!VecVeci64)
    func.func private @__reussir_polyffi_Vec_NestedVec_push(!VecVeci64, !Veci64) -> !VecVeci64
    func.func public @main() -> i32 {
        // single test
        %0 = func.call @__reussir_polyffi_Vec_i64_new() : () -> !Veci64
        reussir.rc.inc ( %0 : !Veci64 )
        reussir.rc.dec ( %0 : !Veci64 )
        reussir.rc.dec ( %0 : !Veci64 )
        // nested test
        %1 = func.call @__reussir_polyffi_Vec_NestedVec_new() : () -> !VecVeci64
        %2 = func.call @__reussir_polyffi_Vec_i64_new() : () -> !Veci64
        reussir.rc.inc ( %2 : !Veci64 )
        %3 = func.call @__reussir_polyffi_Vec_NestedVec_push(%1, %2) : (!VecVeci64, !Veci64) -> (!VecVeci64)
        %4 = func.call @__reussir_polyffi_Vec_NestedVec_push(%3, %2) : (!VecVeci64, !Veci64) -> (!VecVeci64)
        reussir.rc.dec ( %4 : !VecVeci64 )
        // return
        %c0 = arith.constant 0 : i32
        func.return %c0 : i32
    }
}
