// RUN: %reussir-opt %s --split-input-file --reussir-token-instantiation | %FileCheck %s

// Only *token-producing* arms count toward a dec's token type. Under the
// special-pointer-tag scheme (module stamped `reussir.special_ptr_tag`, as
// the pass that runs before token instantiation does), a nullary arm of a
// taggable variant is an unboxed immediate: it never allocates and its
// decrement never yields a token. So `Tree { Branch(32B), Leaf(nullary) }`
// has exactly one real arm and an unpinned dec keeps a STATIC 32-byte token
// — preserving in-place reuse (`token.ensure`) instead of a runtime realloc
// per node (the 2x rbtree regression this pins against). Without the module
// attribute (boxed nullary encoding) the Leaf boxes are real 8-byte cells,
// the arms are genuinely non-uniform, and the dec is dynamic `token<?>`.

!tree = !reussir.record<variant "Tree" {!reussir.record<compound "Tree.Branch" [value] {i64, !reussir.record<variant "Tree">, !reussir.record<variant "Tree">}>, !reussir.record<compound "Tree.Leaf" [value] {}>}>
!rctree = !reussir.rc<!tree>

// Tagged module: nullary Leaf is an immediate -> static token (8 header + 24
// payload = 32; both value/borrow paths must agree).
// CHECK-LABEL: @tagged
module @tagged attributes { reussir.special_ptr_tag = "immortal", dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {
  // CHECK: reussir.rc.dec{{.*}}: !reussir.nullable<!reussir.token<align : 8, size : 32>>
  func.func @dec(%0: !rctree) {
    reussir.rc.dec (%0 : !rctree)
    return
  }
}

// -----

!tree = !reussir.record<variant "Tree" {!reussir.record<compound "Tree.Branch" [value] {i64, !reussir.record<variant "Tree">, !reussir.record<variant "Tree">}>, !reussir.record<compound "Tree.Leaf" [value] {}>}>
!rctree = !reussir.rc<!tree>

// Boxed module (no attribute): Leaf boxes exist (8B) -> arms non-uniform ->
// dynamic token.
// CHECK-LABEL: @boxed
module @boxed attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>> } {
  // CHECK: reussir.rc.dec{{.*}}: !reussir.nullable<!reussir.token<align : 8, size : ?>>
  func.func @dec(%0: !rctree) {
    reussir.rc.dec (%0 : !rctree)
    return
  }
}
