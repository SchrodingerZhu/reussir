// Failure modes of `rrc --transform-script` (issue #349): a bad schedule is
// always a clean compile error naming the problem, never a silent no-op.
//
// An unknown anchor name is a driver usage error.
// RUN: %not %rrc %s --emit mlir-llvm \
// RUN:   --transform-script %S/Inputs/dot_unroll.transform.mlir@bogus -o - \
// RUN:   2>&1 | %FileCheck %s --check-prefix=BADANCHOR
// BADANCHOR: unknown anchor `bogus`
//
// A script that never defines the anchor's entry point fails the pipeline
// with the interpreter's diagnostic naming the missing sequence.
// RUN: %not %rrc %s --emit mlir-llvm \
// RUN:   --transform-script %S/Inputs/no_entry_point.transform.mlir@entry -o - \
// RUN:   2>&1 | %FileCheck %s --check-prefix=NOENTRY
// NOENTRY: could not find a nested named sequence with name: __reussir_anchor_entry
//
// A script file that does not exist fails the preload pass.
// RUN: %not %rrc %s --emit mlir-llvm \
// RUN:   --transform-script %S/Inputs/does_not_exist.transform.mlir -o - \
// RUN:   2>&1 | %FileCheck %s --check-prefix=NOFILE
// NOFILE: does_not_exist.transform.mlir' is neither a file nor a directory

module {
  func.func @answer() -> i32 {
    %0 = arith.constant 42 : i32
    func.return %0 : i32
  }
}
