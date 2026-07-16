// RUN: %reussir-opt %s -reussir-convert-to-std | %FileCheck %s

// A `token.realloc` is a real resize between two distinct layouts. Whether
// the two layouts share an allocator bin is only a *pairing heuristic* in
// TokenReuse — it says nothing about whether the block can be relabeled in
// place (that holds only on a bin allocator that ignores the free size). So
// ConvertToSTD never elides a realloc: every one is left for the direct
// `__reussir_reallocate` lowering, which resizes for real and is sound on
// any allocator (mimalloc or a size-strict one alike).

module @test {
    // Same-bin (24 and 32 both in the 32-byte mimalloc bin): still a real
    // realloc, not a launder.
    // CHECK-LABEL: @same_bin
    // CHECK: reussir.token.realloc
    // CHECK-NOT: reussir.token.launder
    func.func @same_bin(%t: !reussir.nullable<!reussir.token<align: 8, size: 24>>) -> !reussir.token<align: 8, size: 32> {
        %0 = reussir.token.realloc(%t : !reussir.nullable<!reussir.token<align: 8, size: 24>>) : !reussir.token<align: 8, size: 32>
        return %0 : !reussir.token<align: 8, size: 32>
    }

    // CHECK-LABEL: @same_bin_plain
    // CHECK: reussir.token.realloc
    // CHECK-NOT: reussir.token.launder
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
