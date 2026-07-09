// RUN: %reussir-opt %s | %reussir-opt | %FileCheck %s

// The dynamic-size token (`token<align, size: ?>`, #325) round-trips through
// the custom assembly, and a static token is unchanged. A dynamic token is
// produced by an unpinned decrement of a *non-uniform* variant under the
// default per-constructor box sizing (a `fixed` variant would instead yield a
// static uniform token). It lowers to a bare pointer (#366); its free/realloc
// recover the exact size from the size-recovering allocator, so no size need
// be carried.

// CHECK-LABEL: @dynamic
// CHECK-SAME: !reussir.token<align : 16, size : ?>
func.func private @dynamic(!reussir.token<align: 16, size: ?>)

// CHECK-LABEL: @static
// CHECK-SAME: !reussir.token<align : 8, size : 24>
func.func private @static(!reussir.token<align: 8, size: 24>)

// CHECK-LABEL: @also_dynamic
// CHECK-SAME: !reussir.token<align : 8, size : ?>
func.func private @also_dynamic(!reussir.token<align: 8, size: ?>)
