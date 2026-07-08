// RUN: %reussir-opt %s -reussir-lowering-scf-ops | %FileCheck %s

// A same-bin token.realloc never moves — the non-null block already
// satisfies the new layout in place — so it expands to a null-dispatch
// (launder | alloc) with no runtime realloc call. Cross-bin reallocs keep
// the op and lower to __reussir_reallocate later.

module @test {
    // CHECK-LABEL: @same_bin
    // CHECK-NOT: reussir.token.realloc
    // CHECK: scf.if
    // CHECK: reussir.token.launder
    // CHECK: reussir.token.alloc
    func.func @same_bin(%t: !reussir.nullable<!reussir.token<align: 8, size: 24>>) -> !reussir.token<align: 8, size: 32> {
        %0 = reussir.token.realloc(%t : !reussir.nullable<!reussir.token<align: 8, size: 24>>) : !reussir.token<align: 8, size: 32>
        return %0 : !reussir.token<align: 8, size: 32>
    }

    // A plain (non-nullable) same-bin token needs no dispatch at all.
    // CHECK-LABEL: @same_bin_plain
    // CHECK-NOT: reussir.token.realloc
    // CHECK: reussir.token.launder
    func.func @same_bin_plain(%t: !reussir.token<align: 8, size: 24>) -> !reussir.token<align: 8, size: 32> {
        %0 = reussir.token.realloc(%t : !reussir.token<align: 8, size: 24>) : !reussir.token<align: 8, size: 32>
        return %0 : !reussir.token<align: 8, size: 32>
    }

    // CHECK-LABEL: @cross_bin
    // CHECK: reussir.token.realloc
    func.func @cross_bin(%t: !reussir.nullable<!reussir.token<align: 8, size: 16>>) -> !reussir.token<align: 8, size: 32> {
        %0 = reussir.token.realloc(%t : !reussir.nullable<!reussir.token<align: 8, size: 16>>) : !reussir.token<align: 8, size: 32>
        return %0 : !reussir.token<align: 8, size: 32>
    }
}
