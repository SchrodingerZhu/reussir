// RUN: %reussir-opt %s | %reussir-opt 
!list_incomplete = !reussir.record<variant "List" incomplete>
!cons = !reussir.record<compound "List::Cons" { i32, [shared] !list_incomplete }>
!nil = !reussir.record<compound "List::Nil" {}>
!list = !reussir.record<variant "List" {!cons, !nil}>

module {
  func.func @cons(%fst : i32, %tail : !reussir.rc<!list>) -> !reussir.rc<!list> {
    %0 = reussir.record.compound(%fst, %tail : i32, !reussir.rc<!list>) : !cons
    %1 = reussir.record.variant [0] (%0 : !cons) : !list
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 24>
    %rc = reussir.rc.create 
        value(%1 : !list) 
        token(%token : !reussir.token<align: 8, size: 24>) : !reussir.rc<!list>
    return %rc : !reussir.rc<!list>
  }
}
