// RUN: %reussir-opt %s --reussir-convert-to-std | %FileCheck %s

module {
  // CHECK: func.func private @[[DISPATCHER1:.*REUSSIR_STRING_DISPATCHER.*]](%{{.*}}: !reussir.str<local>) -> (index, i1)
  // CHECK:      %[[START:.*]] = reussir.str.unsafe_startswith(%{{.*}} : <local>) "extremely_long_unique_pattern" : i1
  // CHECK:      %[[LEN:.*]] = reussir.str.len(%{{.*}} : <local>) : index
  // CHECK:      %[[C29:.*]] = arith.constant 29 : index
  // CHECK:      %[[LEN_OK:.*]] = arith.cmpi eq, %[[LEN]], %[[C29]] : index
  // CHECK:      %[[COND:.*]] = arith.andi %[[START]], %[[LEN_OK]] : i1
  // CHECK:      scf.if %[[COND]] -> (index, i1) {
  // CHECK:        %[[IDX:.*]] = arith.constant 0 : index
  // CHECK:        %[[TRUE:.*]] = arith.constant true
  // CHECK:        scf.yield %[[IDX]], %[[TRUE]] : index, i1
  // CHECK:      } else {
  // CHECK:        %[[POISON:.*]] = ub.poison : index
  // CHECK:        %[[FALSE:.*]] = arith.constant false
  // CHECK:        scf.yield %[[POISON]], %[[FALSE]] : index, i1
  // CHECK:      }
  
  // CHECK: func.func private @[[DISPATCHER2:.*REUSSIR_STRING_DISPATCHER.*]](%{{.*}}: !reussir.str<local>) -> (index, i1)
  // CHECK:      %[[LEN1:.*]] = reussir.str.len(%{{.*}} : <local>) : index
  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[IS_ZERO:.*]] = arith.cmpi eq, %[[LEN1]], %[[C0]] : index
  // CHECK:      scf.if %[[IS_ZERO]] -> (index, i1) {
  // CHECK:        scf.yield
  // CHECK:      } else {
  // CHECK:        %[[BYTE:.*]] = reussir.str.unsafe_byte_at
  // CHECK:        %[[CAST:.*]] = arith.index_cast %[[BYTE]]
  // CHECK:        scf.index_switch %[[CAST]]
  // CHECK:        case 102 {
  // CHECK:          %[[START:.*]] = reussir.str.unsafe_startswith(%{{.*}} : <local>) "oo" : i1
  // CHECK:          %[[LEN_CASE:.*]] = reussir.str.len(%{{.*}} : <local>) : index
  // CHECK:          %[[C2:.*]] = arith.constant 2 : index
  // CHECK:          %[[LEN_OK:.*]] = arith.cmpi eq, %[[LEN_CASE]], %[[C2]] : index
  // CHECK:          %[[MATCH:.*]] = arith.andi %[[START]], %[[LEN_OK]] : i1
  // CHECK:          scf.if %[[MATCH]] -> (index, i1) {
  // CHECK:            %[[IDX:.*]] = arith.constant 0 : index
  // CHECK:            scf.yield
  // CHECK:          }
  // CHECK:          scf.yield
  // CHECK:        }
  // CHECK:        case 98 {
  // CHECK:          %[[START:.*]] = reussir.str.unsafe_startswith(%{{.*}} : <local>) "ar" : i1
  // CHECK:          %[[LEN_CASE:.*]] = reussir.str.len(%{{.*}} : <local>) : index
  // CHECK:          %[[C2:.*]] = arith.constant 2 : index
  // CHECK:          %[[LEN_OK:.*]] = arith.cmpi eq, %[[LEN_CASE]], %[[C2]] : index
  // CHECK:          %[[MATCH:.*]] = arith.andi %[[START]], %[[LEN_OK]] : i1
  // CHECK:          scf.if %[[MATCH]] -> (index, i1) {
  // CHECK:            %[[IDX:.*]] = arith.constant 1 : index
  // CHECK:            scf.yield
  // CHECK:          }
  // CHECK:          scf.yield
  // CHECK:        }
  // CHECK:      }

  // CHECK-LABEL: @test_select_compressed_short
  func.func @test_select_compressed_short(%str: !reussir.str<local>) -> (index, i1) {
    // CHECK: call @[[DISPATCHER2]]
    %idx, %found = reussir.str.select (%str) ["foo", "bar"] : (!reussir.str<local>) -> (index, i1)
    return %idx, %found : index, i1
  }

  // CHECK-LABEL: @test_select_compressed_long
  func.func @test_select_compressed_long(%str: !reussir.str<local>) -> (index, i1) {
     // CHECK: call @[[DISPATCHER1]]
     %idx, %found = reussir.str.select (%str) ["extremely_long_unique_pattern"] : (!reussir.str<local>) -> (index, i1)
     return %idx, %found : index, i1
  }
}
