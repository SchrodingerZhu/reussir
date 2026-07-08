// RUN: %reussir-opt %s --convert-to-llvm | \
// RUN: %reussir-translate --mlir-to-llvmir | %FileCheck %s

// LLVM lowering under the *taggedbox* special-pointer-tag encoding (module
// stamped `reussir.special_ptr_tag = "taggedbox"`). This is the Koka/Lean
// low-bit convention: a nullary variant becomes the pure value
// `(tag << 1) | 1` — there is no dummy box, so the immediate is not
// dereferenceable. Every rc access first tests the pointer's low bit and,
// when set, skips the box behind a predicted-not-taken branch (`!prof`
// weights 1:2000). The variant tag is decoded as `imm >> 1` rather than
// loaded. Recognition is a cheap `and`, not the immortal encoding's
// count-magnitude load.

!list = !reussir.record<variant "List" {!reussir.record<compound "List.Cons" [value] {i64, !reussir.record<variant "List">}>, !reussir.record<compound "List.Nil" [value] {}>}>
!rclist = !reussir.rc<!list>

module @test attributes { reussir.special_ptr_tag = "taggedbox", dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {

  // Immediate materialization: the pure value (tag << 1) | 1 = 3 for tag 1.
  // No dummy box global, no `or`, just an inttoptr of the constant.
  // CHECK-LABEL: define ptr @make_nil()
  // CHECK-NOT: @_RNvC17REUSSIR_TAG_DUMMY
  // CHECK: ret ptr inttoptr (i64 3 to ptr)
  func.func @make_nil() -> !rclist {
    %0 = reussir.rc.tagged tag(1) : !rclist
    return %0 : !rclist
  }

  // Fetch: the immediate has no box, so the count load is guarded by the low
  // bit and yields a sentinel `2` (!= 1 -> routes to the shared branch).
  // CHECK-LABEL: define i64 @fetch(ptr %0)
  // CHECK: %[[RAW:.+]] = ptrtoint ptr %0 to i64
  // CHECK: %[[BIT:.+]] = and i64 %[[RAW]], 1
  // CHECK: %[[TAGGED:.+]] = icmp ne i64 %[[BIT]], 0
  // CHECK: br i1 %[[TAGGED]], label %[[CONT:.+]], label %[[LOAD:.+]], !prof
  // CHECK: [[LOAD]]:
  // CHECK: %[[CNT:.+]] = load i32, ptr %0
  // CHECK: [[CONT]]:
  // CHECK: %[[PHI:.+]] = phi i32 [ %[[CNT]], %[[LOAD]] ], [ 2, %1 ]
  func.func @fetch(%rc: !rclist) -> index {
    %cnt = reussir.rc.fetch (%rc : !rclist) : index
    return %cnt : index
  }

  // Tag read: a low-bit immediate decodes `imm >> 1`; only a real box loads.
  // CHECK-LABEL: define i64 @tag(ptr %0)
  // CHECK: %[[R:.+]] = ptrtoint ptr %0 to i64
  // CHECK: %[[SH:.+]] = lshr i64 %[[R]], 1
  // CHECK: %[[DEC:.+]] = trunc i64 %[[SH]] to i32
  // CHECK: br i1 %{{.+}}, label %[[TCONT:.+]], label %[[TLOAD:.+]], !prof
  // CHECK: [[TLOAD]]:
  // CHECK: %[[LT:.+]] = load i32, ptr
  // CHECK: [[TCONT]]:
  // CHECK: phi i32 [ %[[LT]], %[[TLOAD]] ], [ %[[DEC]], %1 ]
  func.func @tag(%ref: !reussir.ref<!list shared>) -> index {
    %tag = reussir.record.tag(%ref : !reussir.ref<!list shared>) : index
    return %tag : index
  }

  // The decrementing store: recognize an immediate by the low bit and skip
  // the store behind an unlikely branch (the immediate is not writable).
  // CHECK-LABEL: define void @set(ptr %0, i64 %1)
  // CHECK: %[[SR:.+]] = ptrtoint ptr %0 to i64
  // CHECK: %[[SB:.+]] = and i64 %[[SR]], 1
  // CHECK: %[[ST:.+]] = icmp ne i64 %[[SB]], 0
  // CHECK: br i1 %[[ST]], label %[[SCONT:.+]], label %[[SSTORE:.+]], !prof
  // CHECK: [[SSTORE]]:
  // CHECK: store i32 %{{.+}}, ptr %0
  // CHECK: br label %[[SCONT]]
  func.func @set(%rc: !rclist, %v: index) {
    reussir.rc.set(%rc : !rclist, %v : index)
    return
  }

  // Increment: the whole load+add+store is skipped for an immediate (no box
  // to grow), behind the same predicted-not-taken low-bit branch.
  // CHECK-LABEL: define void @inc(ptr %0)
  // CHECK: %[[IR:.+]] = ptrtoint ptr %0 to i64
  // CHECK: %[[IB:.+]] = and i64 %[[IR]], 1
  // CHECK: %[[IT:.+]] = icmp ne i64 %[[IB]], 0
  // CHECK: br i1 %[[IT]], label %[[ICONT:.+]], label %[[IINC:.+]], !prof
  // CHECK: [[IINC]]:
  // CHECK: %[[OLD:.+]] = load i32
  // CHECK: %[[NEW:.+]] = add i32 %[[OLD]], 1
  // CHECK: store i32 %[[NEW]]
  // CHECK: br label %[[ICONT]]
  func.func @inc(%rc: !rclist) {
    reussir.rc.inc(%rc : !rclist)
    return
  }
}
