// RUN: %reussir-opt %s --pass-pipeline='builtin.module(reussir-rc-decrement-expansion,func.func(reussir-token-reuse))' | %FileCheck %s

// A dropped-but-never-reused token used to surface as a top-level
// `token.free` of the expanded decrement's *nullable* result — a runtime
// call even on the common shared (rc > 1) branch, where the token is null.
// Token reuse now sinks a statically non-null free into the unique branch
// instead: the shared branch no longer touches the allocator, and the
// nullable result is dead.

!rc64 = !reussir.rc<i64>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
// CHECK: func.func @unused_dec(%[[rc:[a-z0-9]+]]: !reussir.rc<i64>)
// CHECK: scf.if
// CHECK:   reussir.ref.drop
// CHECK:   %[[tok:[a-z0-9]+]] = reussir.rc.reinterpret(%[[rc]] : !reussir.rc<i64>) : !reussir.token<align : 8, size : 16>
// CHECK:   reussir.token.free(%[[tok]] : !reussir.token<align : 8, size : 16>)
// CHECK: } else {
// CHECK:   reussir.rc.set
// CHECK-NOT: reussir.token.free
// CHECK: return
    func.func @unused_dec(%0: !rc64) {
        %1 = reussir.rc.dec (%0 : !rc64) : !reussir.nullable<!reussir.token<align: 8, size: 16>>
        return
    }
}
