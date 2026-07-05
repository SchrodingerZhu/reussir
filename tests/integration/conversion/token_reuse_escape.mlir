// RUN: %reussir-opt %s -reussir-token-reuse | %FileCheck %s

// A member decrement expands *inside* its owner's expanded decrement (the
// owner's unique branch releases its fields), so the member's reuse token is
// born one region deep and — by SSA dominance — dies at the owner's join,
// while same-shaped creates right below allocate fresh memory. TokenReuse's
// escape pre-phase widens the owner: the trapped token rides out through an
// extra result (null on the shared path) and both dead boxes feed the two
// creates. No token.alloc survives.

!inner = !reussir.record<compound "pair" {i64, i64}>
!rcInner = !reussir.rc<!inner>
!outer = !reussir.record<compound "node" {!inner, i64}>
!rcOuter = !reussir.rc<!outer>
!tk = !reussir.token<align: 8, size: 24>

module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-i64:64-n8:16:32:64-S128"} {
    // CHECK-LABEL: func.func @escape
    // The owner's expanded decrement is widened with the escaped member
    // token; the shared path yields null for it.
    // CHECK: %[[TOKS:.+]]:2 = scf.if %{{.+}} -> (!reussir.nullable<!reussir.token<align : 8, size : 24>>, !reussir.nullable<!reussir.token<align : 8, size : 24>>)
    // CHECK: scf.yield %{{.+}}, %{{.+}} :
    // CHECK: } else {
    // CHECK: scf.yield %{{.+}}, %{{.+}} :
    // CHECK: } {reussir.expanded_decrement}
    // Both creates reuse: one token each, no fresh allocation.
    // CHECK: reussir.token.ensure
    // CHECK-NEXT: reussir.rc.create
    // CHECK: reussir.token.ensure
    // CHECK-NEXT: reussir.rc.create
    // CHECK-NOT: reussir.token.alloc
    func.func @escape(%o: !rcOuter, %va: !inner, %vb: !inner) -> (!rcInner, !rcInner) {
        %prev = reussir.rc.fetch (%o : !rcOuter) : index
        %c1 = arith.constant 1 : index
        %isOne = arith.cmpi eq, %prev, %c1 : index
        %unique = reussir.expect (%isOne : i1, true) : i1
        %t = scf.if %unique -> (!reussir.nullable<!tk>) {
            // unique: release the member (its own expanded decrement — the
            // trapped producer), then take the owner's box.
            %ref = reussir.rc.borrow (%o : !rcOuter) : !reussir.ref<!outer>
            %slot = reussir.ref.project (%ref : !reussir.ref<!outer>) [0] : !reussir.ref<!rcInner>
            %m = reussir.ref.load (%slot : !reussir.ref<!rcInner>) : !rcInner
            %mprev = reussir.rc.fetch (%m : !rcInner) : index
            %misOne = arith.cmpi eq, %mprev, %c1 : index
            %munique = reussir.expect (%misOne : i1, true) : i1
            %mt = scf.if %munique -> (!reussir.nullable<!tk>) {
                %mtok = reussir.rc.reinterpret (%m : !rcInner) : !tk
                %mnn = reussir.nullable.create (%mtok : !tk) : !reussir.nullable<!tk>
                scf.yield %mnn : !reussir.nullable<!tk>
            } else {
                %mdec = arith.subi %mprev, %c1 : index
                reussir.rc.set (%m : !rcInner, %mdec : index)
                %mnull = reussir.nullable.create : !reussir.nullable<!tk>
                scf.yield %mnull : !reussir.nullable<!tk>
            } {reussir.expanded_decrement}
            %tok = reussir.rc.reinterpret (%o : !rcOuter) : !tk
            %nn = reussir.nullable.create (%tok : !tk) : !reussir.nullable<!tk>
            scf.yield %nn : !reussir.nullable<!tk>
        } else {
            %dec = arith.subi %prev, %c1 : index
            reussir.rc.set (%o : !rcOuter, %dec : index)
            %null = reussir.nullable.create : !reussir.nullable<!tk>
            scf.yield %null : !reussir.nullable<!tk>
        } {reussir.expanded_decrement}
        %tka = reussir.token.alloc : !tk
        %a = reussir.rc.create value(%va : !inner) token(%tka : !tk) : !rcInner
        %tkb = reussir.token.alloc : !tk
        %b = reussir.rc.create value(%vb : !inner) token(%tkb : !tk) : !rcInner
        return %a, %b : !rcInner, !rcInner
    }
}
