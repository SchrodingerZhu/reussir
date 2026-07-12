// RUN: %reussir-opt %s -reussir-token-reuse --reussir-lowering-scf-ops | %FileCheck %s

// The ensure lowering may replace its null check with the producing expanded
// decrement's own condition — but only for the decrement's *own* token
// (result 0: then yields a nonnull `nullable.create`, else yields null),
// where "condition holds" and "token is nonnull" coincide. Token reuse also
// widens these ifs with *escaped* results carrying inner member-decrement
// tokens out; such a result's nullness follows the inner decrement's count,
// not the outer condition, so its ensure must keep a genuine
// `nullable.check`. Substituting the outer condition there hands a null
// token to the reuse path of `rc.create` (issue #398: a unique owner whose
// dead member is a tagged-immediate nullary — the immediate never takes the
// unique path, its escaped token is always null, and the in-place
// initialization writes through null).

!inner = !reussir.record<compound "pair" {i64, i64}>
!rcInner = !reussir.rc<!inner>
!outer = !reussir.record<compound "node" {!inner, i64}>
!rcOuter = !reussir.rc<!outer>
!tk = !reussir.token<align: 8, size: 24>

module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-i64:64-n8:16:32:64-S128"} {
    // CHECK-LABEL: func.func @escape
    // The widened decrement: condition %[[UNIQ]], own token + escaped token.
    // CHECK: %[[UNIQ:.+]] = reussir.expect
    // CHECK: %[[TOKS:.+]]:2 = scf.if %[[UNIQ]]
    // The escaped token (%[[TOKS]]#1): its ensure keeps a real null check.
    // CHECK: %[[CHK:.+]] = reussir.nullable.check(%[[TOKS]]#1
    // CHECK: %[[CHKEXP:.+]] = reussir.expect(%[[CHK]]
    // CHECK: scf.if %[[CHKEXP]]
    // The own token (%[[TOKS]]#0): its ensure reuses the decrement's
    // condition directly.
    // CHECK: reussir.nullable.check(%[[TOKS]]#0
    // CHECK: scf.if %[[UNIQ]]
    func.func @escape(%o: !rcOuter, %va: !inner, %vb: !inner) -> (!rcInner, !rcInner) {
        %prev = reussir.rc.fetch (%o : !rcOuter) : index
        %c1 = arith.constant 1 : index
        %isOne = arith.cmpi eq, %prev, %c1 : index
        %unique = reussir.expect (%isOne : i1, true) : i1
        %t = scf.if %unique -> (!reussir.nullable<!tk>) {
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
