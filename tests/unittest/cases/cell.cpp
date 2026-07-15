#include "Reussir/IR/ReussirOps.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <optional>

import reussir.test;

namespace reussir {
TEST_F(ReussirTest, ParseCellTypeTest) {
  withType<CellType>(SIMPLE_LAYOUT, R"(!reussir.cell<!reussir.rc<i64>>)",
                     [](mlir::ModuleOp module, CellType type) {
                       EXPECT_TRUE(llvm::isa<RcType>(type.getElementType()));

                       mlir::DataLayout layout(module);
                       EXPECT_EQ(layout.getTypeSize(type),
                                 layout.getTypeSize(type.getElementType()));
                       EXPECT_EQ(
                           layout.getTypeABIAlignment(type),
                           layout.getTypeABIAlignment(type.getElementType()));
                     });
}

TEST_F(ReussirTest, ParseExclusiveCellTypeTest) {
  withType<CellType>(
      SIMPLE_LAYOUT, R"(!reussir.cell<!reussir.rc<i64> exclusive>)",
      [](mlir::ModuleOp module, CellType type) {
        EXPECT_TRUE(type.getExclusive());
        EXPECT_TRUE(llvm::isa<RcType>(type.getElementType()));

        // The trailing i1 in-use flag adds one byte, padded back to the
        // element's alignment: a pointer element yields a 16-byte cell.
        mlir::DataLayout layout(module);
        EXPECT_EQ(layout.getTypeSize(type),
                  2 * layout.getTypeSize(type.getElementType()));
        EXPECT_EQ(layout.getTypeABIAlignment(type),
                  layout.getTypeABIAlignment(type.getElementType()));
      });
}

TEST_F(ReussirTest, ParseAtomicCellTypeTest) {
  withType<CellType>(SIMPLE_LAYOUT, R"(!reussir.cell<i64 atomic>)",
                     [](mlir::ModuleOp module, CellType type) {
                       EXPECT_TRUE(type.getAtomic());
                       EXPECT_FALSE(type.getExclusive());
                       EXPECT_EQ(type.getKind(), CellKind::atomic);

                       // The atomic scalar stays inline in the RC payload and
                       // has the same size and alignment as its element. Unlike
                       // an exclusive cell, it has no trailing in-use flag.
                       mlir::DataLayout layout(module);
                       EXPECT_EQ(layout.getTypeSize(type),
                                 layout.getTypeSize(type.getElementType()));
                       EXPECT_EQ(
                           layout.getTypeABIAlignment(type),
                           layout.getTypeABIAlignment(type.getElementType()));
                     });
}

TEST_F(ReussirTest, ParseAtomicCellOrderingTest) {
  withModule(
      R"mlir(
        module {
          func.func @test(
              %cell: !reussir.rc<!reussir.cell<i64 atomic>>,
              %value: i64) -> i64 {
            %loaded = reussir.cell.get(
                %cell : !reussir.rc<!reussir.cell<i64 atomic>>
              ) ordering(monotonic) : i64
            reussir.cell.set(
                %value : i64,
                %cell : !reussir.rc<!reussir.cell<i64 atomic>>
              ) ordering(seq_cst)
            %old = reussir.cell.rmw addi(
                %loaded : i64,
                %cell : !reussir.rc<!reussir.cell<i64 atomic>>
              ) ordering(acquire) -> i64
            return %old : i64
          }
        }
      )mlir",
      [](mlir::ModuleOp module) {
        std::optional<mlir::LLVM::AtomicOrdering> getOrdering;
        std::optional<mlir::LLVM::AtomicOrdering> setOrdering;
        std::optional<mlir::LLVM::AtomicOrdering> rmwOrdering;
        module.walk(
            [&](ReussirCellGetOp op) { getOrdering = op.getOrdering(); });
        module.walk(
            [&](ReussirCellSetOp op) { setOrdering = op.getOrdering(); });
        module.walk(
            [&](ReussirCellRmwOp op) { rmwOrdering = op.getOrdering(); });

        ASSERT_TRUE(getOrdering);
        EXPECT_EQ(*getOrdering, mlir::LLVM::AtomicOrdering::monotonic);
        ASSERT_TRUE(setOrdering);
        EXPECT_EQ(*setOrdering, mlir::LLVM::AtomicOrdering::seq_cst);
        ASSERT_TRUE(rmwOrdering);
        EXPECT_EQ(*rmwOrdering, mlir::LLVM::AtomicOrdering::acquire);
      });
}

TEST_F(ReussirTest, CellMemberProjectsToSharedRc) {
  auto i64Type = mlir::IntegerType::get(context.get(), 64);
  auto cellType = CellType::get(context.get(), i64Type);

  auto projected = llvm::dyn_cast<RcType>(
      getProjectedType(cellType, /*fieldCap=*/false, Capability::value));
  ASSERT_TRUE(projected);
  EXPECT_EQ(projected.getElementType(), cellType);
  EXPECT_EQ(projected.getCapability(), Capability::shared);

  mlir::Type storage = memberStorageType(context.get(), cellType,
                                         /*isField=*/false);
  EXPECT_TRUE(llvm::isa<mlir::LLVM::LLVMPointerType>(storage));
  EXPECT_EQ(memberStorageType(context.get(), cellType, /*isField=*/false,
                              /*memBoxInternal=*/true),
            cellType);
}
} // namespace reussir
