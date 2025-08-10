// RUN: %reussir-opt %s | %reussir-opt 
// Test basic reference projection operations

// Define a simple record type with multiple fields
!point = !reussir.record<compound "Point" {i32, i32}>

// Define a record with different capability fields
!complex = !reussir.record<compound "Complex" {i64, [field] i32, [shared] i64}>

module {
  // Test basic projection of a simple record
  func.func @project_simple(%ref : !reussir.ref<!point>) -> !reussir.ref<i32> {
    %x = reussir.ref.project (%ref : !reussir.ref<!point>) [0] : !reussir.ref<i32>
    %y = reussir.ref.project (%ref : !reussir.ref<!point>) [1] : !reussir.ref<i32>
    return %x : !reussir.ref<i32>
  }

  // Test projection with rigid capability
  func.func @project_rigid(%ref : !reussir.ref<!point rigid>) -> !reussir.ref<i32 rigid> {
    %x = reussir.ref.project (%ref : !reussir.ref<!point rigid>) [0] : !reussir.ref<i32 rigid>
    return %x : !reussir.ref<i32 rigid>
  }

  // Test projection with shared capability
  func.func @project_shared(%ref : !reussir.ref<!point shared>) -> !reussir.ref<i32 shared> {
    %x = reussir.ref.project (%ref : !reussir.ref<!point shared>) [1] : !reussir.ref<i32 shared>
    return %x : !reussir.ref<i32 shared>
  }

  // Test projection of complex record with field capability
  // When projecting a field capability member under a rigid reference, 
  // the projected type becomes a nullable rigid rc pointer
  func.func @project_complex(%ref : !reussir.ref<!complex rigid>) -> 
    !reussir.ref<!reussir.nullable<!reussir.rc<i32 rigid>> rigid> {
    %field0 = reussir.ref.project (%ref : !reussir.ref<!complex rigid>) [0] 
        : !reussir.ref<i64 rigid>
    %field1 = reussir.ref.project (%ref : !reussir.ref<!complex rigid>) [1] 
        : !reussir.ref<!reussir.nullable<!reussir.rc<i32 rigid>> rigid>
    %field2 = reussir.ref.project (%ref : !reussir.ref<!complex rigid>) [2] 
        : !reussir.ref<!reussir.rc<i64 shared> rigid>
    return %field1 : !reussir.ref<!reussir.nullable<!reussir.rc<i32 rigid>> rigid>
  }
}
