#include "Reussir/IR/ReussirTypes.h"
#include <gtest/gtest.h>

import reussir.test;

namespace reussir {
TEST_F(ReussirTest, SimpleRecordScanner) {
  withType<reussir::RecordType>(
      SIMPLE_LAYOUT,
      R"(!reussir.record<compound "Test" [regional] {i32, i64, [field] f128}>)",
      [](mlir::ModuleOp module, reussir::RecordType type) {
        llvm::SmallVector<int32_t> buffer;
        mlir::DataLayout dataLayout = mlir::DataLayout(module);
        type.emitScannerInstructions(buffer, dataLayout, {});
        llvm::SmallVector<int32_t> expected = {
            scanner::advance(8), scanner::field(), scanner::end()};
        EXPECT_EQ(buffer, expected);
      });
}

TEST_F(ReussirTest, NestedRecordScanner) {
  withType<reussir::RecordType>(
      SIMPLE_LAYOUT,
      R"(!reussir.record<compound "Test" [regional] {
        i32, 
        i64, 
        !reussir.record<compound "Nested" [value] {
          i8, 
          [field] i16
        }>,
        f128,
        [field] i32
      }>)",
      [](mlir::ModuleOp module, reussir::RecordType type) {
        llvm::SmallVector<int32_t> buffer;
        mlir::DataLayout dataLayout = mlir::DataLayout(module);
        type.emitScannerInstructions(buffer, dataLayout, {});
        llvm::SmallVector<int32_t> expected = {
            scanner::advance(24), scanner::field(), scanner::advance(16),
            scanner::field(), scanner::end()};
        EXPECT_EQ(buffer, expected);
      });
}

TEST_F(ReussirTest, VariantRecordScanner) {
  withType<reussir::RecordType>(
      SIMPLE_LAYOUT,
      R"(!reussir.record<variant "Test" [regional] {
        [field] i32, 
        i64, 
        !reussir.record<compound "Nested" [value]  {
          i8, 
          [field] i16
        }>,
        f128,
        !reussir.record<compound "Nested2" [value] {
          [field] i16,
          f128, 
          [field] i16
        }>
      }>)",
      [](mlir::ModuleOp module, reussir::RecordType type) {
        llvm::SmallVector<int32_t> buffer;
        mlir::DataLayout dataLayout = mlir::DataLayout(module);
        type.emitScannerInstructions(buffer, dataLayout, {});
        llvm::SmallVector<int32_t> expected = {
            // Fused header: advance past the i32 count slot to the i32 tag,
            // then read the 4-byte tag.
            scanner::advance(4), scanner::variant(4),
            // skip table
            scanner::skip(5), scanner::skip(8), scanner::skip(9),
            scanner::skip(12), scanner::skip(13),
            // first variant
            scanner::advance(12), scanner::field(), scanner::advance(32),
            scanner::skip(14),
            // second variant
            scanner::advance(44), scanner::skip(12),
            // third variant
            scanner::advance(12), scanner::field(), scanner::advance(32),
            scanner::skip(8),
            // fourth variant
            scanner::advance(44), scanner::skip(6),
            // fifth variant
            scanner::advance(28), scanner::field(), scanner::advance(8),
            scanner::field(), scanner::advance(8),
            // final end
            scanner::end()};
        EXPECT_EQ(buffer, expected);
      });
}
} // namespace reussir
