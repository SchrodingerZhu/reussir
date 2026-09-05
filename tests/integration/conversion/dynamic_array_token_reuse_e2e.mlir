// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-attach-native-target,func.func(reussir-token-instantiation),reussir-rc-decrement-expansion,reussir-acquire-drop-expansion,reussir-convert-to-std,func.func(reussir-token-reuse),reussir-convert-to-std,canonicalize,cse,convert-scf-to-cf,reussir-lowering-basic-ops,reussir-convert-to-llvm,reconcile-unrealized-casts,canonicalize)' -o %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | %opt -S -O2 -o %t.ll
// RUN: %llc %t.ll -relocation-model=pic -filetype=obj -o %t.o
// RUN: %cc %t.o -o %t.exe -L%library_path -lreussir_rt %rpath_flag %extra_sys_libs
// RUN: %t.exe

// Dropping one dynamic box and constructing a larger one must preserve the
// runtime allocation size on both the unique and shared decrement paths.
!array = !reussir.array<? x i32>
!rc = !reussir.rc<!array>
!view = memref<?xi32, strided<[?], offset: ?>>
module {
  func.func private @fill(%xs: !rc) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %ref = reussir.rc.borrow (%xs : !rc) : !reussir.ref<!array>
    %view = reussir.array.view(%ref : !reussir.ref<!array>) : !view
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = memref.dim %view, %c0 : !view
    scf.for %i = %c0 to %n step %c1 {
      %value = arith.index_cast %i : index to i32
      memref.store %value, %view[%i] : !view
    }
    return
  }

  func.func private @make(%n: index) -> !rc attributes {llvm.linkage = #llvm.linkage<internal>} {
    %poison = ub.poison : !array
    %xs = reussir.rc.create value(%poison : !array) extents(%n) : !rc
    func.call @fill(%xs) : (!rc) -> ()
    return %xs : !rc
  }

  func.func private @replace(%old: !rc, %n: index) -> !rc attributes {llvm.linkage = #llvm.linkage<internal>} {
    reussir.rc.dec (%old : !rc)
    %poison = ub.poison : !array
    %xs = reussir.rc.create value(%poison : !array) extents(%n) : !rc
    func.call @fill(%xs) : (!rc) -> ()
    return %xs : !rc
  }

  func.func private @check(%xs: !rc, %expected: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %ref = reussir.rc.borrow (%xs : !rc) : !reussir.ref<!array>
    %view = reussir.array.view(%ref : !reussir.ref<!array>) : !view
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0i = arith.constant 0 : i32
    %n = memref.dim %view, %c0 : !view
    %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0i) -> i32 {
      %value = memref.load %view[%i] : !view
      %next = arith.addi %acc, %value : i32
      scf.yield %next : i32
    }
    %bad = arith.cmpi ne, %sum, %expected : i32
    scf.if %bad {
      reussir.panic "dynamic array replacement lost its size or contents"
    }
    return
  }

  func.func @main() -> i32 {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c64 = arith.constant 64 : index
    %c0i = arith.constant 0 : i32
    %c3i = arith.constant 3 : i32
    %c2016 = arith.constant 2016 : i32

    %unique = func.call @make(%c3) : (index) -> !rc
    %larger = func.call @replace(%unique, %c64) : (!rc, index) -> !rc
    func.call @check(%larger, %c2016) : (!rc, i32) -> ()
    reussir.rc.dec (%larger : !rc)

    %shared = func.call @make(%c3) : (index) -> !rc
    reussir.rc.inc (%shared : !rc)
    %fresh = func.call @replace(%shared, %c64) : (!rc, index) -> !rc
    func.call @check(%shared, %c3i) : (!rc, i32) -> ()
    func.call @check(%fresh, %c2016) : (!rc, i32) -> ()
    reussir.rc.dec (%shared : !rc)

    %empty = func.call @replace(%fresh, %c0) : (!rc, index) -> !rc
    func.call @check(%empty, %c0i) : (!rc, i32) -> ()
    reussir.rc.dec (%empty : !rc)
    return %c0i : i32
  }
}
