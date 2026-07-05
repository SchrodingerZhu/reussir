// RUN: %reussir-opt %s --reussir-lowering-scf-ops | %FileCheck %s

// A free of a *nullable* token must be guarded before it reaches the runtime
// call: the null case (the shared branch of an expanded rc decrement) must
// not pay a `__reussir_deallocate` call just to return. A plain-token free
// is statically non-null and stays a direct free.

module @test {
// CHECK: func.func @free_nullable(%[[t:[a-z0-9]+]]: !reussir.nullable<!reussir.token<align : 8, size : 16>>)
// CHECK:   %[[flag:[a-z0-9]+]] = reussir.nullable.check(%[[t]] : <!reussir.token<align : 8, size : 16>>) : i1
// CHECK:   scf.if %[[flag]] {
// CHECK:     %[[tok:[a-z0-9]+]] = reussir.nullable.coerce(%[[t]] : <!reussir.token<align : 8, size : 16>>) : !reussir.token<align : 8, size : 16>
// CHECK:     reussir.token.free(%[[tok]] : !reussir.token<align : 8, size : 16>)
// CHECK:   }
// CHECK:   return
    func.func @free_nullable(%t: !reussir.nullable<!reussir.token<align: 8, size: 16>>) {
        reussir.token.free (%t : !reussir.nullable<!reussir.token<align: 8, size: 16>>)
        return
    }

// CHECK: func.func @free_plain(%[[p:[a-z0-9]+]]: !reussir.token<align : 8, size : 16>)
// CHECK-NOT: reussir.nullable.check
// CHECK:   reussir.token.free(%[[p]] : !reussir.token<align : 8, size : 16>)
// CHECK:   return
    func.func @free_plain(%t: !reussir.token<align: 8, size: 16>) {
        reussir.token.free (%t : !reussir.token<align: 8, size: 16>)
        return
    }
}
