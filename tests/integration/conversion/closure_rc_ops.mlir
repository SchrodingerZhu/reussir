// RUN: %reussir-opt \
// RUN:   -reussir-closure-outlining \
// RUN:   -reussir-convert-to-std \
// RUN:   -convert-scf-to-cf \
// RUN:   -reussir-lowering-basic-ops \
// RUN:   -convert-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN:   -o %t.mlir %s
// RUN: %FileCheck %s --check-prefix=CHECK-MLIR < %t.mlir
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | \
// RUN:   %FileCheck %s --check-prefix=CHECK-LLVM
// RUN: %reussir-translate --mlir-to-llvmir %t.mlir | \
// RUN:   %opt -S -O3 | \
// RUN:   %llc -relocation-model=pic -filetype=obj -o %t.o
// RUN: %cc %t.o -o %t.exe -L%library_path -lreussir_rt \
// RUN:    %rpath_flag %extra_sys_libs
// RUN: %t.exe

// Test closure operations: uniqify, rc.inc, rc.dec, and closures with multiple arguments
module @test attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>>} {
  // Helper: create a closure that adds 1 to its argument
  func.func private @create_add_one_closure() -> !reussir.rc<!reussir.closure<(i32) -> i32>> attributes { llvm.linkage = #llvm.linkage<internal> } {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %closure = reussir.closure.create -> !reussir.rc<!reussir.closure<(i32) -> i32>> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%v0 : i32):
          %one = arith.constant 1 : i32
          %add = arith.addi %v0, %one : i32 
          reussir.closure.yield %add : i32
      }
    }
    return %closure : !reussir.rc<!reussir.closure<(i32) -> i32>>
  }

  // Helper: create a closure that adds two arguments together
  func.func private @create_two_arg_closure() -> !reussir.rc<!reussir.closure<(i32, i32) -> i32>> attributes { llvm.linkage = #llvm.linkage<internal> } {
    %token = reussir.token.alloc : !reussir.token<align: 8, size: 32>
    %closure = reussir.closure.create -> !reussir.rc<!reussir.closure<(i32, i32) -> i32>> {
      token(%token : !reussir.token<align: 8, size: 32>)
      body {
        ^bb0(%v0 : i32, %v1 : i32):
          %add = arith.addi %v0, %v1 : i32 
          reussir.closure.yield %add : i32
      }
    }
    return %closure : !reussir.rc<!reussir.closure<(i32, i32) -> i32>>
  }

  // Test 1: Test rc.inc and rc.dec on closure
  // Create closure, increment refcount, decrement refcount, then use it
  func.func private @test_rc_inc_dec() -> i32 attributes { llvm.linkage = #llvm.linkage<internal> } {
    %c14 = arith.constant 14 : i32
    %c15 = arith.constant 15 : i32
    
    // Create a closure that adds 1
    %closure = func.call @create_add_one_closure() : () -> !reussir.rc<!reussir.closure<(i32) -> i32>>
    
    // Increment reference count
    reussir.rc.inc (%closure : !reussir.rc<!reussir.closure<(i32) -> i32>>)
    
    // Decrement reference count (closure should still be valid, refcount back to 1)
    reussir.rc.dec (%closure : !reussir.rc<!reussir.closure<(i32) -> i32>>)
    
    // Apply and evaluate the closure: 14 + 1 = 15
    %applied = reussir.closure.apply (%c14 : i32) to (%closure : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<() -> i32>>
    %result = reussir.closure.eval (%applied : !reussir.rc<!reussir.closure<() -> i32>>) : i32
    
    // Return 15 - 15 = 0 on success
    %res = arith.subi %result, %c15 : i32
    return %res : i32
  }

  // Test 2: Test closure.uniqify with unique closure (should return same closure)
  func.func private @test_uniqify_unique() -> i32 attributes { llvm.linkage = #llvm.linkage<internal> } {
    %c26 = arith.constant 26 : i32
    %c27 = arith.constant 27 : i32
    
    // Create a closure that adds 1
    %closure = func.call @create_add_one_closure() : () -> !reussir.rc<!reussir.closure<(i32) -> i32>>
    
    // Uniqify - closure is already unique (refcount = 1), should return same pointer
    %uniqified = reussir.closure.uniqify (%closure : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<(i32) -> i32>>
    
    // Apply and evaluate: 26 + 1 = 27
    %applied = reussir.closure.apply (%c26 : i32) to (%uniqified : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<() -> i32>>
    %result = reussir.closure.eval (%applied : !reussir.rc<!reussir.closure<() -> i32>>) : i32
    
    // Return 27 - 27 = 0 on success
    %res = arith.subi %result, %c27 : i32
    return %res : i32
  }

  // Test 3: Test closure.uniqify with shared closure (should clone)
  func.func private @test_uniqify_shared() -> i32 attributes { llvm.linkage = #llvm.linkage<internal> } {
    %c32 = arith.constant 32 : i32
    %c33 = arith.constant 33 : i32
    
    // Create a closure that adds 1
    %closure = func.call @create_add_one_closure() : () -> !reussir.rc<!reussir.closure<(i32) -> i32>>
    
    // Increment refcount to make it shared (refcount = 2)
    reussir.rc.inc (%closure : !reussir.rc<!reussir.closure<(i32) -> i32>>)
    
    // Uniqify - closure is shared, should clone
    %uniqified = reussir.closure.uniqify (%closure : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<(i32) -> i32>>
    
    // Decrement the original closure's refcount (we're done with the extra reference)
    reussir.rc.dec (%closure : !reussir.rc<!reussir.closure<(i32) -> i32>>)

    // Apply and evaluate the uniqified (cloned) closure: 32 + 1 = 33
    %applied = reussir.closure.apply (%c32 : i32) to (%uniqified : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<() -> i32>>
    %result = reussir.closure.eval (%applied : !reussir.rc<!reussir.closure<() -> i32>>) : i32
    
    // Return 33 - 33 = 0 on success
    %res = arith.subi %result, %c33 : i32
    return %res : i32
  }

  // Test 4: Closure with multiple partial applications
  func.func private @test_multi_arg_closure() -> i32 attributes { llvm.linkage = #llvm.linkage<internal> } {
    %c100 = arith.constant 100 : i32
    %c50 = arith.constant 50 : i32
    %c150 = arith.constant 150 : i32
    
    // Create a two-argument closure
    %closure = func.call @create_two_arg_closure() : () -> !reussir.rc<!reussir.closure<(i32, i32) -> i32>>
    
    // First application: apply 100
    %applied1 = reussir.closure.apply (%c100 : i32) to (%closure : !reussir.rc<!reussir.closure<(i32, i32) -> i32>>) : !reussir.rc<!reussir.closure<(i32) -> i32>>
    
    // Second application: apply 50
    %applied2 = reussir.closure.apply (%c50 : i32) to (%applied1 : !reussir.rc<!reussir.closure<(i32) -> i32>>) : !reussir.rc<!reussir.closure<() -> i32>>
    
    // Evaluate: 100 + 50 = 150
    %result = reussir.closure.eval (%applied2 : !reussir.rc<!reussir.closure<() -> i32>>) : i32
    
    // Return 150 - 150 = 0 on success
    %res = arith.subi %result, %c150 : i32
    return %res : i32
  }

  // Main function: run all tests with panic on failure
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : i32
    
    // Test 1: rc.inc/rc.dec
    %t1 = func.call @test_rc_inc_dec() : () -> i32
    %fail1 = arith.cmpi ne, %t1, %c0 : i32
    scf.if %fail1 {
      reussir.panic "Test 1 failed: test_rc_inc_dec"
    }
    
    // Test 2: uniqify unique closure
    %t2 = func.call @test_uniqify_unique() : () -> i32
    %fail2 = arith.cmpi ne, %t2, %c0 : i32
    scf.if %fail2 {
      reussir.panic "Test 2 failed: test_uniqify_unique"
    }
    
    // Test 3: uniqify shared closure
    %t3 = func.call @test_uniqify_shared() : () -> i32
    %fail3 = arith.cmpi ne, %t3, %c0 : i32
    scf.if %fail3 {
      reussir.panic "Test 3 failed: test_uniqify_shared"
    }
    
    // Test 4: multi-arg closure
    %t4 = func.call @test_multi_arg_closure() : () -> i32
    %fail4 = arith.cmpi ne, %t4, %c0 : i32
    scf.if %fail4 {
      reussir.panic "Test 4 failed: test_multi_arg_closure"
    }
    
    // All tests passed
    func.return %c0 : i32
  }
}

// CHECK-MLIR-DAG: llvm.func @__reussir_allocate
// CHECK-MLIR-DAG: llvm.call @__reussir_allocate

// CHECK-LLVM-DAG: declare {{.*}} ptr @__reussir_allocate
// CHECK-LLVM-DAG: call ptr @__reussir_allocate
