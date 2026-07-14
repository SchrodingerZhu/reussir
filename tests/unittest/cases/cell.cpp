#include "Reussir/IR/ReussirTypes.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

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
