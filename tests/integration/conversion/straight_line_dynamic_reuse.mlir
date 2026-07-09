// RUN: %reussir-opt %s -reussir-token-reuse | %FileCheck %s

// Straight-line reuse of a *dynamic* token (`token<align, ?>`, #361). An
// unpinned decrement of a non-uniform variant yields a dynamic donor whose
// size is only known at runtime. Unlike a statically-sized donor — which
// only pairs within its mimalloc bin (see straight_line_reuse.mlir) — a
// dynamic token is a *universal fallback* donor: TokenReuse pairs it with any
// acceptor via `token.realloc`, which resizes to the acceptor's exact size at
// runtime. Binning is only a heuristic for *which* donor to prefer; it never
// gates this pairing.

!list = !reussir.record<variant "List" {!reussir.record<compound "List.Cons" [value] {i64, !reussir.record<variant "List">}>, !reussir.record<compound "List.Var" [value] {i64}>, !reussir.record<compound "List.Nil" [value] {}>}>
!rclist = !reussir.rc<!list>

module @test attributes { reussir.per_constructor_box_sizing, dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {
    // The dead dynamic donor feeds a 16-byte `Var` construction. The fresh
    // 16-byte alloc is dropped; the donor is realloc'd to size 16 in place of it.
    // CHECK-LABEL: @dyn_reuse_leaf
    func.func @dyn_reuse_leaf(%0: !rclist, %x: i64) -> !rclist {
        %tok = reussir.rc.dec (%0 : !rclist) : !reussir.nullable<!reussir.token<align: 8, size: ?>>
        // CHECK-NOT:  reussir.token.alloc
        // CHECK:      %[[t:[a-z0-9]+]] = reussir.token.realloc(%{{[a-z0-9]+}} : !reussir.nullable<!reussir.token<align : 8, size : ?>>) : <align : 8, size : 16>
        // CHECK-NEXT: reussir.rc.create_variant"(%{{[a-z0-9]+}}, %[[t]]){{.*}}tag = 1
        %tk = reussir.token.alloc : !reussir.token<align: 8, size: 16>
        %r = "reussir.rc.create_variant"(%x, %tk) <{operandSegmentSizes = array<i32: 0, 1, 1, 0>, tag = 1 : index}> : (i64, !reussir.token<align: 8, size: 16>) -> !rclist
        return %r : !rclist
    }

    // Same dynamic donor, but the acceptor is a 24-byte `Cons`. A statically
    // 16-byte donor could never reach this (16 and 24 are different bins), but
    // the dynamic donor reallocs to 24 all the same — the universal-fallback
    // property that `token<?>` exists to provide.
    // CHECK-LABEL: @dyn_reuse_any_size
    func.func @dyn_reuse_any_size(%0: !rclist, %x: i64, %tail: !rclist) -> !rclist {
        %tok = reussir.rc.dec (%0 : !rclist) : !reussir.nullable<!reussir.token<align: 8, size: ?>>
        // CHECK-NOT:  reussir.token.alloc
        // CHECK:      %[[t:[a-z0-9]+]] = reussir.token.realloc(%{{[a-z0-9]+}} : !reussir.nullable<!reussir.token<align : 8, size : ?>>) : <align : 8, size : 24>
        // CHECK-NEXT: reussir.rc.create_variant"(%{{[a-z0-9]+}}, %{{[a-z0-9]+}}, %[[t]]){{.*}}tag = 0
        %tk = reussir.token.alloc : !reussir.token<align: 8, size: 24>
        %r = "reussir.rc.create_variant"(%x, %tail, %tk) <{operandSegmentSizes = array<i32: 0, 2, 1, 0>, tag = 0 : index}> : (i64, !rclist, !reussir.token<align: 8, size: 24>) -> !rclist
        return %r : !rclist
    }
}
