// RUN: %reussir-opt %s -reussir-token-reuse | %FileCheck %s --check-prefix=NONE
// RUN: %reussir-opt %s -reussir-token-reuse="domain-bias=two-phase" | \
// RUN:   %FileCheck %s --check-prefix=DOMAIN

// Two records with identical layout ("A" and "B") produce identical token
// types, so a dead "A" box is a *perfect-size* candidate for a "B" create.
// Without a domain bias, the greedy in-order matcher hands the token to the
// first consumer — the cross-domain "B" — starving the same-domain "A"
// consumer right behind it. The two-phase bias forms same-domain pairs first,
// so the "A" box is recycled into the new "A" and the "B" allocates fresh.

!recA = !reussir.record<compound "A" {i64, i64}>
!recB = !reussir.record<compound "B" {i64, i64}>
!rcA = !reussir.rc<!recA>
!rcB = !reussir.rc<!recB>
!tk = !reussir.token<align: 8, size: 24>

module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-i64:64-n8:16:32:64-S128"} {
    // NONE-LABEL: func.func @bias
    // NONE: %[[T:.+]] = reussir.token.ensure
    // NONE-NEXT: reussir.rc.create value(%{{.+}} : !reussir.record<compound "B"
    // NONE-SAME: token(%[[T]]
    // NONE: reussir.token.alloc
    // NONE-NEXT: reussir.rc.create value(%{{.+}} : !reussir.record<compound "A"

    // DOMAIN-LABEL: func.func @bias
    // DOMAIN: reussir.token.alloc
    // DOMAIN-NEXT: reussir.rc.create value(%{{.+}} : !reussir.record<compound "B"
    // DOMAIN: %[[T:.+]] = reussir.token.ensure
    // DOMAIN-NEXT: reussir.rc.create value(%{{.+}} : !reussir.record<compound "A"
    // DOMAIN-SAME: token(%[[T]]
    func.func @bias(%dead: !rcA, %va: !recA, %vb: !recB) -> (!rcB, !rcA) {
        %t = reussir.rc.dec (%dead : !rcA) : !reussir.nullable<!tk>
        %tkb = reussir.token.alloc : !tk
        %b = reussir.rc.create value(%vb : !recB) token(%tkb : !tk) : !rcB
        %tka = reussir.token.alloc : !tk
        %a = reussir.rc.create value(%va : !recA) token(%tka : !tk) : !rcA
        return %b, %a : !rcB, !rcA
    }
}
