// RUN: %reussir-opt %s | %reussir-opt | %FileCheck %s

// The dynamic-size token (`token<align, size: ?>`, #325) round-trips through
// the custom assembly, and a static token is unchanged. A dynamic token is
// produced by an unpinned decrement of a *non-uniform* variant under
// per-constructor box sizing; it carries its size at runtime (lowering to a
// fat `{ptr, size}` pair) so its free/realloc pass the exact size on any
// allocator.

// CHECK-LABEL: @dynamic
// CHECK-SAME: !reussir.token<align : 16, size : ?>
func.func private @dynamic(!reussir.token<align: 16, size: ?>)

// CHECK-LABEL: @static
// CHECK-SAME: !reussir.token<align : 8, size : 24>
func.func private @static(!reussir.token<align: 8, size: 24>)

// CHECK-LABEL: @also_dynamic
// CHECK-SAME: !reussir.token<align : 8, size : ?>
func.func private @also_dynamic(!reussir.token<align: 8, size: ?>)
