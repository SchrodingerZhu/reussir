#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirTypes.h"
#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Verifier.h>

import reussir.test;
import reussir.test.value;

namespace reussir {

// Ownership acquisition over an array has two shapes, chosen by
// `kArrayOwnershipUnrollThreshold` in lib/IR/ReussirOps.cpp: below the
// threshold the traversal is unrolled into straight-line code, at or above it
// the traversal becomes an `scf.for` nest with one loop per dimension.
//
// Both shapes have to be counted over the whole function rather than its
// entry block. The loop form puts every project and inc inside the loop body,
// so an entry-block scan reports zero of each and reads as "nothing was
// emitted" — which is how these counts silently stopped describing the
// compiler when the loop form landed.
struct AcquisitionShape {
  size_t views = 0;
  size_t projects = 0;
  size_t incs = 0;
  llvm::SmallVector<int64_t> tripCounts;
};

static AcquisitionShape inspectAcquisition(mlir::func::FuncOp funcOp) {
  AcquisitionShape shape;
  funcOp.walk([&](mlir::Operation *op) {
    if (llvm::isa<ReussirArrayViewOp>(op))
      ++shape.views;
    if (llvm::isa<ReussirArrayProjectOp>(op))
      ++shape.projects;
    if (llvm::isa<ReussirRcIncOp>(op))
      ++shape.incs;
    // Trip counts, not just loop count: the point of the loop form is that it
    // visits every element, which only the bounds can attest to.
    if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
      if (auto bound = forOp.getUpperBound()
                           .getDefiningOp<mlir::arith::ConstantIndexOp>())
        shape.tripCounts.push_back(bound.value());
    }
  });
  return shape;
}

TEST_F(ReussirTest, ParseArrayTypeTest) {
  withType<reussir::ArrayType>(
      SIMPLE_LAYOUT, R"(!reussir.array<4 x 8 x !reussir.rc<i64>>)",
      [](mlir::ModuleOp module, reussir::ArrayType type) {
        EXPECT_EQ(type.getShape().size(), 2u);
        EXPECT_EQ(type.getShape()[0], 4);
        EXPECT_EQ(type.getShape()[1], 8);
        EXPECT_TRUE(llvm::isa<RcType>(type.getElementType()));
      });
}

TEST_F(ReussirTest, ViewTypeElementTypeTest) {
  auto i8Type = mlir::IntegerType::get(context.get(), 8);
  auto arrayType = reussir::ArrayType::get(context.get(), {4, 8}, i8Type);
  auto viewType =
      reussir::ViewType::get(context.get(), /*isMutable=*/true, arrayType);

  EXPECT_EQ(viewType.getArrayType(), arrayType);
  EXPECT_EQ(viewType.getElementType(), i8Type);
}

TEST_F(ReussirTest, RcTypeIsValidMemRefElementType) {
  auto i64Type = mlir::IntegerType::get(context.get(), 64);
  auto rcType = reussir::RcType::get(context.get(), i64Type);

  EXPECT_TRUE(llvm::isa<mlir::MemRefElementTypeInterface>(rcType));
  EXPECT_TRUE(mlir::BaseMemRefType::isValidElementType(rcType));

  withType<mlir::MemRefType>(
      SIMPLE_LAYOUT, R"(memref<2x!reussir.rc<i64>>)",
      [](mlir::ModuleOp module, mlir::MemRefType type) {
        EXPECT_TRUE(type.hasStaticShape());
        EXPECT_EQ(type.getShape().size(), 1u);
        EXPECT_EQ(type.getShape()[0], 2);
        EXPECT_TRUE(llvm::isa<reussir::RcType>(type.getElementType()));
      });
}

// Two elements, under the unroll threshold: straight-line code, one project
// and one inc per element, no loop.
TEST_F(ReussirValueTransformTest, RefToArrayOfRcAcquisition) {
  testValueAcquisition("!reussir.ref<!reussir.array<2 x !reussir.rc<i32>>>",
                       [](mlir::func::FuncOp funcOp) {
                         auto shape = inspectAcquisition(funcOp);
                         EXPECT_EQ(shape.views, 1u);
                         EXPECT_TRUE(shape.tripCounts.empty());
                         EXPECT_EQ(shape.projects, 2u);
                         EXPECT_EQ(shape.incs, 2u);
                       });
}

// Still under the threshold, but nested: the unrolled form projects the outer
// dimension once and then each element of the inner one. This is the nested
// unrolling the 2x2 case used to cover before it crossed the threshold.
TEST_F(ReussirValueTransformTest, RefToSmallNestedArrayOfRcAcquisition) {
  testValueAcquisition(
      "!reussir.ref<!reussir.array<1 x 2 x !reussir.rc<i32>>>",
      [](mlir::func::FuncOp funcOp) {
        auto shape = inspectAcquisition(funcOp);
        EXPECT_EQ(shape.views, 1u);
        EXPECT_TRUE(shape.tripCounts.empty());
        // one projection to the single inner row, then one per element in it
        EXPECT_EQ(shape.projects, 3u);
        EXPECT_EQ(shape.incs, 2u);
      });
}

// Four elements is exactly the threshold, so this is the smallest shape that
// takes the loop form: a loop per dimension over that dimension's extent, and
// a single project-per-dimension and inc in the innermost body, executed once
// per element rather than emitted once per element.
TEST_F(ReussirValueTransformTest, RefToNestedArrayOfRcAcquisition) {
  testValueAcquisition("!reussir.ref<!reussir.array<2 x 2 x !reussir.rc<i32>>>",
                       [](mlir::func::FuncOp funcOp) {
                         auto shape = inspectAcquisition(funcOp);
                         EXPECT_EQ(shape.views, 1u);
                         EXPECT_EQ(shape.tripCounts.size(), 2u);
                         for (int64_t trip : shape.tripCounts)
                           EXPECT_EQ(trip, 2);
                         EXPECT_EQ(shape.projects, 2u);
                         EXPECT_EQ(shape.incs, 1u);
                       });
}

TEST_F(ReussirTest, ArrayViewAllowsTensorResult) {
  withModule(R"(
!arr4 = !reussir.array<4 x i8>

module {
  func.func @borrow_tensor(%xs : !reussir.ref<!arr4>) -> tensor<4xi8> {
    %view = reussir.array.view(%xs : !reussir.ref<!arr4>) : tensor<4xi8>
    return %view : tensor<4xi8>
  }
}
)",
             [](mlir::ModuleOp module) {
               EXPECT_TRUE(mlir::succeeded(mlir::verify(module)));
             });
}

TEST_F(ReussirTest, ArrayWithUniqueViewAllowsTensorBodyArgument) {
  withModule(R"(
!arr4 = !reussir.array<4 x i8>
!rc_arr4 = !reussir.rc<!arr4>

module {
  func.func @borrow_tensor(%xs : !rc_arr4) -> !rc_arr4 {
    %updated = reussir.array.with_unique_view (%xs : !rc_arr4) -> !rc_arr4 {
      ^bb0(%view: tensor<4xi8>):
        reussir.scf.yield
    }
    return %updated : !rc_arr4
  }
}
)",
             [](mlir::ModuleOp module) {
               EXPECT_TRUE(mlir::succeeded(mlir::verify(module)));
             });
}
} // namespace reussir
