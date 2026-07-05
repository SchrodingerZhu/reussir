// RUN: %reussir-opt %s --reussir-rc-decrement-expansion --reussir-infer-variant-tag | %FileCheck %s

!list_ = !reussir.record<variant "List" incomplete>
!list_nil = !reussir.record<compound "List::Nil" [value] { }>
!list_cons = !reussir.record<compound "List::Cons" [value] { i64, !list_ }>
!list = !reussir.record<variant "List" { !list_nil, !list_cons }>

module {
    // CHECK-LABEL: reussir.ref.drop
    // CHECK-SAME: variant[1]
    func.func @case_1(
        %rc: !reussir.rc<!list>
    ) {
        %ref = reussir.rc.borrow(%rc : !reussir.rc<!list>) : !reussir.ref<!list>
        %ref_ = reussir.record.coerce [1] (%ref : !reussir.ref<!list>) : !reussir.ref<!list_cons>
        %token = reussir.rc.dec (%rc : !reussir.rc<!list>) : !reussir.nullable<!reussir.token<align: 8, size: 24>>
        reussir.token.free (%token : !reussir.nullable<!reussir.token<align: 8, size: 24>>)
        return
    }

    func.func @case_2(
        %rc: !reussir.rc<!list>
    ) {
        %ref = reussir.rc.borrow(%rc : !reussir.rc<!list>) : !reussir.ref<!list>
        reussir.record.dispatch(%ref : !reussir.ref<!list>) {
            // CHECK-LABEL: reussir.ref.drop
            // CHECK-SAME: variant[0]
            [0] -> {
                ^bb0(%nil_ref : !reussir.ref<!list_nil>):
                    %token = reussir.rc.dec (%rc : !reussir.rc<!list>) : !reussir.nullable<!reussir.token<align: 8, size: 24>>
                    reussir.token.free (%token : !reussir.nullable<!reussir.token<align: 8, size: 24>>)
                    reussir.scf.yield
            }
            // CHECK-LABEL: reussir.ref.drop
            // CHECK-SAME: variant[1]
            [1] -> {
                ^bb0(%cons_ref : !reussir.ref<!list_cons>):
                    %token = reussir.rc.dec (%rc : !reussir.rc<!list>) : !reussir.nullable<!reussir.token<align: 8, size: 24>>
                    reussir.token.free (%token : !reussir.nullable<!reussir.token<align: 8, size: 24>>)
                    reussir.scf.yield
            }
        }
        return
    }
}
