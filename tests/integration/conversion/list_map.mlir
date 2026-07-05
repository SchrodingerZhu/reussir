// RUN: %reussir-opt %s \
// RUN: --reussir-token-instantiation \
// RUN: --reussir-closure-outlining \
// RUN: --reussir-lowering-region-patterns \
// RUN: --reussir-inc-dec-cancellation \
// RUN: --reussir-rc-decrement-expansion \
// RUN: --reussir-infer-variant-tag \
// RUN: --reussir-acquire-drop-expansion \
// RUN: --reussir-lowering-scf-ops \
// RUN: --reussir-inc-dec-cancellation \
// RUN: --reussir-acquire-drop-expansion='expand-decrement=1 outline-record=1' \
// RUN: --reussir-token-reuse \
// RUN: --reussir-lowering-scf-ops \
// RUN: --reussir-compile-polymorphic-ffi \
// RUN: --convert-scf-to-cf \
// RUN: --convert-to-llvm | \
// RUN: %reussir-translate \
// RUN: --reussir-to-llvmir | \
// RUN: %opt -O3 |\
// RUN: %llc -relocation-model=pic -filetype=obj -o %t.o
// RUN: %cc %t.o %S/print_i64.c -o %t.exe -L%library_path -lreussir_rt \
// RUN:    %rpath_flag %extra_sys_libs
// RUN: %t.exe | %FileCheck %s
// CHECK: 01235432
!cell = !reussir.record<compound "Cell" {i64}>
!rc_i64 = !reussir.rc<!cell>
!list_cons = !reussir.record<compound "list.cons" [value] {!cell, !reussir.record<variant "list" incomplete>}>
!list_nil = !reussir.record<compound "list.nil" [value] {}>
!list = !reussir.record<variant "list" {!list_cons, !list_nil}>
!rc_list = !reussir.rc<!list>
!token_rc_i64 = !reussir.token<align: 8, size: 16>
!token_rc_list = !reussir.token<align: 8, size: 24>
!token_closure = !reussir.token<align: 8, size: 40>
!nullable_token_rc_list = !reussir.nullable<!reussir.token<align: 8, size: 24>>
!closure = !reussir.rc<!reussir.closure<(!rc_i64) -> !rc_i64>>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} 
{
    func.func private @get_closure(%delta: i64) -> !closure {
        %init = reussir.closure.create -> !reussir.rc<!reussir.closure<(i64, !rc_i64) -> !rc_i64>> {
            body {
                ^bb0(%arg0 : i64, %arg2 : !rc_i64):
                %borrow_ = reussir.rc.borrow (%arg2 : !rc_i64) : !reussir.ref<!cell>
                %borrow = reussir.ref.project (%borrow_ : !reussir.ref<!cell>) [0] : !reussir.ref<i64>
                %loaded = reussir.ref.load (%borrow : !reussir.ref<i64>) : i64
                %added_ = arith.addi %loaded, %arg0 : i64
                %added = reussir.record.compound(%added_ : i64) : !cell
                %new_rc = reussir.rc.create value(%added : !cell) : !rc_i64
                reussir.closure.yield %new_rc : !rc_i64
            }
        }
        %res = reussir.closure.apply (%delta : i64) to 
            (%init : !reussir.rc<!reussir.closure<(i64, !rc_i64) -> !rc_i64>>) 
            : !closure
        func.return %res : !closure
    }
    func.func private @cons(%data: i64, %tail: !rc_list) -> !rc_list attributes {
        llvm.linkage = #llvm.linkage<internal>
    } {
        %cell = reussir.record.compound(%data : i64) : !cell
        %rc = reussir.rc.create value(%cell : !cell) : !rc_i64
        %cons = reussir.record.compound(%rc, %tail : !rc_i64, !rc_list) : !list_cons
        %list = reussir.record.variant[0] (%cons : !list_cons) : !list
        %res = reussir.rc.create value(%list : !list) : !rc_list
        func.return %res : !rc_list
    }
    func.func @list_0123() -> !rc_list {
        %nil = reussir.record.compound : !list_nil
        %nil_variant = reussir.record.variant[1] (%nil : !list_nil) : !list
        %nil_rc = reussir.rc.create value(%nil_variant : !list) : !rc_list

        %0 = arith.constant 0 : i64
        %1 = arith.constant 1 : i64
        %2 = arith.constant 2 : i64
        %3 = arith.constant 3 : i64

        %list3 = func.call @cons(%3, %nil_rc) : (i64, !rc_list) -> !rc_list
        %list2 = func.call @cons(%2, %list3) : (i64, !rc_list) -> !rc_list
        %list1 = func.call @cons(%1, %list2) : (i64, !rc_list) -> !rc_list
        %list0 = func.call @cons(%0, %list1) : (i64, !rc_list) -> !rc_list
        func.return %list0 : !rc_list
    }
    func.func private @print_i64(i64)
    // print list with consuming
    func.func private @print_list(%list: !rc_list) {
        %borrow = reussir.rc.borrow (%list : !rc_list) : !reussir.ref<!list>
        reussir.record.dispatch (%borrow : !reussir.ref<!list>) {
            [0] -> {
                ^bb0(%cons_ref: !reussir.ref<!list_cons>):
                %head_ref = reussir.ref.project (%cons_ref : !reussir.ref<!list_cons>) [0] : !reussir.ref<!rc_i64>
                %head_rc = reussir.ref.load (%head_ref : !reussir.ref<!rc_i64>) : !rc_i64
                %head_val_ref_ = reussir.rc.borrow (%head_rc : !rc_i64) : !reussir.ref<!cell>
                %head_val_ref = reussir.ref.project (%head_val_ref_ : !reussir.ref<!cell>) [0] : !reussir.ref<i64>
                %head = reussir.ref.load (%head_val_ref : !reussir.ref<i64>) : i64
                %tail_ref = reussir.ref.project (%cons_ref : !reussir.ref<!list_cons>) [1] : !reussir.ref<!rc_list>
                %tail_rc = reussir.ref.load (%tail_ref : !reussir.ref<!rc_list>) : !rc_list
                reussir.rc.inc (%tail_rc : !rc_list)
                %to_free = reussir.rc.dec (%list : !rc_list) : !nullable_token_rc_list
                func.call @print_i64(%head) : (i64) -> ()
                func.call @print_list(%tail_rc) : (!rc_list) -> ()
                reussir.scf.yield
            }
            [1] -> {
                ^bb1(%nil_ref: !reussir.ref<!list_nil>):
                %to_free = reussir.rc.dec (%list : !rc_list) : !nullable_token_rc_list
                reussir.scf.yield
            }
        }
        func.return
    }
    func.func @apply(%closure: !closure, %list: !rc_list) -> !rc_list {
        %list_ref = reussir.rc.borrow (%list : !rc_list) : !reussir.ref<!list>
        %res = reussir.record.dispatch (%list_ref : !reussir.ref<!list>) -> !rc_list {
            [0] -> {
                ^bb0(%cons_ref: !reussir.ref<!list_cons>):
                %head_ref = reussir.ref.project (%cons_ref : !reussir.ref<!list_cons>) [0] : !reussir.ref<!rc_i64>
                %head_rc = reussir.ref.load (%head_ref : !reussir.ref<!rc_i64>) : !rc_i64
                reussir.rc.inc (%head_rc : !rc_i64)
                %tail_ref = reussir.ref.project (%cons_ref : !reussir.ref<!list_cons>) [1] : !reussir.ref<!rc_list>
                %tail_rc = reussir.ref.load (%tail_ref : !reussir.ref<!rc_list>) : !rc_list
                reussir.rc.inc (%tail_rc : !rc_list)
                %to_free = reussir.rc.dec (%list : !rc_list) : !nullable_token_rc_list
                reussir.rc.inc (%closure : !closure)
                %closure_ = reussir.closure.uniqify (%closure : !closure) : !closure
                %head_applied_ = reussir.closure.apply (%head_rc : !rc_i64) to (%closure_ : !closure) : !reussir.rc<!reussir.closure<() -> !rc_i64>>
                %head_applied = reussir.closure.eval (%head_applied_ : !reussir.rc<!reussir.closure<() -> !rc_i64>>) : !rc_i64
                %tail_applied = func.call @apply(%closure, %tail_rc) : (!closure, !rc_list) -> !rc_list
                %new_cons = reussir.record.compound(%head_applied, %tail_applied : !rc_i64, !rc_list) : !list_cons
                %new_list = reussir.record.variant[0] (%new_cons : !list_cons) : !list
                %res_list = reussir.rc.create value(%new_list : !list) : !rc_list
                reussir.scf.yield %res_list : !rc_list
            }
            [1] -> {
                ^bb1(%nil_ref: !reussir.ref<!list_nil>):
                reussir.rc.dec (%closure : !closure)
                reussir.scf.yield %list : !rc_list
            }
        }
        func.return %res : !rc_list
    }
    func.func @reverse_impl(%list: !rc_list, %acc: !rc_list) -> !rc_list 
        attributes { llvm.linkage = #llvm.linkage<internal> }
    {
        %list_ref = reussir.rc.borrow (%list : !rc_list) : !reussir.ref<!list>
        %res = reussir.record.dispatch (%list_ref : !reussir.ref<!list>) -> !rc_list {
            [0] -> {
                ^bb0(%cons_ref: !reussir.ref<!list_cons>):
                %head_ref = reussir.ref.project (%cons_ref : !reussir.ref<!list_cons>) [0] : !reussir.ref<!rc_i64>
                %head_rc = reussir.ref.load (%head_ref : !reussir.ref<!rc_i64>) : !rc_i64
                reussir.rc.inc (%head_rc : !rc_i64)
                %tail_ref = reussir.ref.project (%cons_ref : !reussir.ref<!list_cons>) [1] : !reussir.ref<!rc_list>
                %tail_rc = reussir.ref.load (%tail_ref : !reussir.ref<!rc_list>) : !rc_list
                reussir.rc.inc (%tail_rc : !rc_list)
                %to_free = reussir.rc.dec (%list : !rc_list) : !nullable_token_rc_list
                %new_cons = reussir.record.compound(%head_rc, %acc : !rc_i64, !rc_list) : !list_cons
                %new_list = reussir.record.variant[0] (%new_cons : !list_cons) : !list
                %new_acc = reussir.rc.create value(%new_list : !list) : !rc_list
                %res_list = func.call @reverse_impl(%tail_rc, %new_acc) : (!rc_list, !rc_list) -> !rc_list
                reussir.scf.yield %res_list : !rc_list
            }
            [1] -> {
                ^bb1(%nil_ref: !reussir.ref<!list_nil>):
                reussir.rc.dec (%list : !rc_list)
                reussir.scf.yield %acc : !rc_list
            }
        }
        func.return %res : !rc_list
    }
    func.func @reverse(%list: !rc_list) -> !rc_list {
        %nil = reussir.record.compound : !list_nil
        %nil_variant = reussir.record.variant[1] (%nil : !list_nil) : !list
        %nil_rc = reussir.rc.create value(%nil_variant : !list) : !rc_list
        %res = func.call @reverse_impl(%list, %nil_rc) : (!rc_list, !rc_list) -> !rc_list
        func.return %res : !rc_list
    }
    func.func @main() -> i32 {
        %list = func.call @list_0123() : () -> !rc_list
        reussir.rc.inc (%list : !rc_list)
        func.call @print_list(%list) : (!rc_list) -> ()
        %one = arith.constant 1 : i64
        %closure = func.call @get_closure(%one) : (i64) -> !closure
        reussir.rc.inc (%closure : !closure)
        %applied_once = func.call @apply(%closure, %list) : (!closure, !rc_list) -> !rc_list
        %applied_twice = func.call @apply(%closure, %applied_once) : (!closure, !rc_list) -> !rc_list
        %reversed = func.call @reverse(%applied_twice) : (!rc_list) -> !rc_list
        func.call @print_list(%reversed) : (!rc_list) -> ()
        %zero = arith.constant 0 : i32
        func.return %zero : i32
    }
}
