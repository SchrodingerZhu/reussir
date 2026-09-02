// RUN: %reussir-opt %s -reussir-token-reuse | %FileCheck %s
// RUN: %reussir-opt %s --reussir-token-reuse=emit-remarks=1 --remarks-filter=TokenReuse --remark-format=emitRemark -o %t.remarks.mlir 2>&1 | %FileCheck %s --check-prefix=REMARK

// REMARK: remark: [Passed] TokenReused | Category:TokenReuse:OneShot | Function=reuse | AvailableTokens=1, CompatibleTokens=1, RemarkId={{[0-9]+}}, Score=2, Source=loc("{{.*}}straight_line_reuse.mlir":{{[0-9]+}}:14), Strategy=ensure
// REMARK: remark: [Missed] TokenNotReused | Category:TokenReuse:OneShot | Function=no_cross_bin_realloc | AvailableTokens=1, CompatibleTokens=0, Reason=no-compatible-token, RemarkId={{[0-9]+}}
// REMARK: remark: [Passed] TokenReused | Category:TokenReuse:OneShot | Function=same_bin_realloc | AvailableTokens=1, CompatibleTokens=1, RemarkId={{[0-9]+}}, Score=1, Source=loc("{{.*}}straight_line_reuse.mlir":{{[0-9]+}}:14), Strategy=realloc
// REMARK: remark: [Missed] TokenNotReused | Category:TokenReuse:OneShot | Function=no_available | AvailableTokens=0, CompatibleTokens=0, Reason=no-available-token, RemarkId={{[0-9]+}}

!rc64 = !reussir.rc<i64>
!rc64x2 = !reussir.rc<!reussir.record<compound "test" {i64, i64}>>
!rc64x3 = !reussir.rc<!reussir.record<compound "test3" {i64, i64, i64}>>
// 8 i64 fields + 8-byte header = 72 B; 9 fields = 80 B. Both round to the same
// 80-byte allocator bin (bins step by 16 above 64), so a dead 72-byte token
// feeds an 80-byte create in place.
!rc64x8 = !reussir.rc<!reussir.record<compound "test8" {i64, i64, i64, i64, i64, i64, i64, i64}>>
!rc64x9 = !reussir.rc<!reussir.record<compound "test9" {i64, i64, i64, i64, i64, i64, i64, i64, i64}>>

module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    func.func @reuse(%0: !rc64) -> !rc64 {
        %1 = reussir.rc.borrow (%0 : !rc64) : !reussir.ref<i64>
        %2 = reussir.ref.load (%1 : !reussir.ref<i64>) : i64
        %3 = reussir.rc.dec (%0 : !rc64) : !reussir.nullable<!reussir.token<align: 8, size: 16>>
        %4 = arith.addi %2, %2 : i64
        // CHECK:      %[[a:[a-z0-9]+]] = reussir.token.ensure(%{{[a-z0-9]+}} : <!reussir.token<align : 8, size : 16>>) : <align : 8, size : 16>
        // CHECK-NEXT: %{{[a-z0-9]+}} = reussir.rc.create value(%{{[a-z0-9]+}} : i64) token(%[[a]] : !reussir.token<align : 8, size : 16>) : !reussir.rc<i64>
        %tk = reussir.token.alloc : !reussir.token<align: 8, size: 16>
        %5 = reussir.rc.create value(%4 : i64) token(%tk : !reussir.token<align: 8, size: 16>) : !rc64
        return %5 : !rc64
    }

    // A 16-byte token cannot feed a 24-byte create: with 8-granular bins
    // (MI_MAX_ALIGN_SIZE=8) 16 and 24 are distinct bins, so such a realloc
    // would move. The matcher must leave the fresh alloc alone and free the
    // dead token.
    // CHECK-LABEL: @no_cross_bin_realloc
    func.func @no_cross_bin_realloc(%0: !rc64) -> !rc64x2 {
        %1 = reussir.rc.borrow (%0 : !rc64) : !reussir.ref<i64>
        %2 = reussir.ref.load (%1 : !reussir.ref<i64>) : i64
        %3 = reussir.rc.dec (%0 : !rc64) : !reussir.nullable<!reussir.token<align: 8, size: 16>>
        %4 = arith.addi %2, %2 : i64
        %5 = reussir.record.compound (%4, %4 : i64, i64) : !reussir.record<compound "test" {i64, i64}>
        // CHECK-NOT:  reussir.token.realloc
        // CHECK:      %[[tk:[a-z0-9]+]] = reussir.token.alloc : <align : 8, size : 24>
        // CHECK-NEXT: %{{[a-z0-9]+}} = reussir.rc.create value(%{{[a-z0-9]+}} : !reussir.record<compound "test" {i64, i64}>) token(%[[tk]] : !reussir.token<align : 8, size : 24>) : !reussir.rc<!reussir.record<compound "test" {i64, i64}>>
        // CHECK:      reussir.token.free
        %tk = reussir.token.alloc : !reussir.token<align: 8, size: 24>
        %6 = reussir.rc.create value(%5 : !reussir.record<compound "test" {i64, i64}>) token(%tk : !reussir.token<align: 8, size: 24>) : !rc64x2
        return %6 : !rc64x2
    }

    // 72 and 80 share a mimalloc bin (both served from the 80-byte bin), so
    // the dead 72-byte token feeds the 80-byte create via token.realloc.
    // CHECK-LABEL: @same_bin_realloc
    func.func @same_bin_realloc(%0: !rc64x8) -> !rc64x9 {
        %1 = reussir.rc.borrow (%0 : !rc64x8) : !reussir.ref<!reussir.record<compound "test8" {i64, i64, i64, i64, i64, i64, i64, i64}>>
        %3 = reussir.rc.dec (%0 : !rc64x8) : !reussir.nullable<!reussir.token<align: 8, size: 72>>
        %4 = arith.constant 0 : i64
        %5 = reussir.record.compound (%4, %4, %4, %4, %4, %4, %4, %4, %4 : i64, i64, i64, i64, i64, i64, i64, i64, i64) : !reussir.record<compound "test9" {i64, i64, i64, i64, i64, i64, i64, i64, i64}>
        // CHECK:      %[[a:[a-z0-9]+]] = reussir.token.realloc(%{{[a-z0-9]+}} : !reussir.nullable<!reussir.token<align : 8, size : 72>>) : <align : 8, size : 80>
        // CHECK-NEXT: %{{[a-z0-9]+}} = reussir.rc.create value(%{{[a-z0-9]+}} : !reussir.record<compound "test9" {i64, i64, i64, i64, i64, i64, i64, i64, i64}>) token(%[[a]] : !reussir.token<align : 8, size : 80>) : !reussir.rc<!reussir.record<compound "test9" {i64, i64, i64, i64, i64, i64, i64, i64, i64}>>
        %tk = reussir.token.alloc : !reussir.token<align: 8, size: 80>
        %6 = reussir.rc.create value(%5 : !reussir.record<compound "test9" {i64, i64, i64, i64, i64, i64, i64, i64, i64}>) token(%tk : !reussir.token<align: 8, size: 80>) : !rc64x9
        return %6 : !rc64x9
    }

    func.func @no_available() -> !rc64 {
        %value = arith.constant 0 : i64
        %tk = reussir.token.alloc : !reussir.token<align: 8, size: 16>
        %result = reussir.rc.create value(%value : i64) token(%tk : !reussir.token<align: 8, size: 16>) : !rc64
        return %result : !rc64
    }
}
