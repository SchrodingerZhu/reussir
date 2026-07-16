// RUN: %reussir-opt %s \
// RUN:   --reussir-convert-to-std \
// RUN:   --convert-scf-to-cf \
// RUN:   --reussir-attach-native-target \
// RUN:   --convert-to-llvm | \
// RUN:   %reussir-translate --reussir-to-llvmir | \
// RUN:   %opt -S -O3 -o %t.ll
// RUN: %llc %t.ll -relocation-model=pic -filetype=obj -o %t.o
// RUN: %cc %S/str_select_e2e.c %t.o -o %t.exe -L%library_path \
// RUN:    %rpath_flag %extra_sys_libs
// RUN: %t.exe

module {
  // Function to test basic pattern matching: matches "foo" (index 0), "bar" (index 1), or none
  func.func @test_basic_(%str: !reussir.str<local>) -> (index, i1) {
    %idx, %found = reussir.str.select (%str) ["foo", "bar"] : (!reussir.str<local>) -> (index, i1)
    return %idx, %found : index, i1
  }

  reussir.trampoline export "C" @test_basic = @test_basic_

  // Function to test pattern with common prefix: "abc" (0), "abd" (1), "xyz" (2)
  func.func @test_prefix_(%str: !reussir.str<local>) -> (index, i1) {
    %idx, %found = reussir.str.select (%str) ["abc", "abd", "xyz"] : (!reussir.str<local>) -> (index, i1)
    return %idx, %found : index, i1
  }

  reussir.trampoline export "C" @test_prefix = @test_prefix_

  // Function to test long pattern
  func.func @test_long_(%str: !reussir.str<local>) -> (index, i1) {
    %idx, %found = reussir.str.select (%str) ["extremely_long_unique_pattern_that_is_over_32_chars"] : (!reussir.str<local>) -> (index, i1)
    return %idx, %found : index, i1
  }

  reussir.trampoline export "C" @test_long = @test_long_

  // Function to test many patterns with various first bytes
  func.func @test_many_(%str: !reussir.str<local>) -> (index, i1) {
    %idx, %found = reussir.str.select (%str) ["apple", "banana", "cherry", "date", "elderberry"] : (!reussir.str<local>) -> (index, i1)
    return %idx, %found : index, i1
  }

  reussir.trampoline export "C" @test_many = @test_many_
}
