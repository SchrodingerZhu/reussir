//===----------------------------------------------------------------------===//
//
// Part of the Reussir Project, dual licensed under the Apache License v2.0 or
// the MIT License.
// See https://github.com/reussir-lang/reussir/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//===----------------------------------------------------------------------===//

#include "Reussir-c/Passes.h"

#include <gtest/gtest.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FileUtilities.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Remarks.h>

namespace {

TEST(TokenReuseRemarks, DeduplicatesLocationsAndGenericInstances) {
  llvm::SmallString<128> path;
  ASSERT_FALSE(
      llvm::sys::fs::createTemporaryFile("reussir-reuse", "json", path));
  llvm::FileRemover remove(path);

  {
    mlir::MLIRContext context;
    MlirStringRef outputPath{path.data(), path.size()};
    ASSERT_TRUE(
        reussirContextEnableTokenReuseRemarks(wrap(&context), outputPath));

    mlir::Location source =
        mlir::FileLineColRange::get(&context, "generic.rr", 2, 3, 2, 9);
    mlir::Location reuseSink =
        mlir::FileLineColRange::get(&context, "generic.rr", 8, 5, 8, 19);
    mlir::Location allocateSink =
        mlir::FileLineColRange::get(&context, "generic.rr", 12, 7, 12, 15);

    // Monomorphized functions commonly have distinct debug metadata wrapped
    // around the same source location. The JSON streamer deliberately strips
    // one-child fused wrappers so both instances share one decision/location.
    for (llvm::StringRef instance : {"generic<i32>", "generic<i64>"}) {
      mlir::Attribute metadata = mlir::StringAttr::get(&context, instance);
      mlir::Location wrappedSource =
          mlir::FusedLoc::get(&context, {source}, metadata);
      mlir::Location wrappedSink =
          mlir::FusedLoc::get(&context, {reuseSink}, metadata);
      auto remark = mlir::remark::passed(
          wrappedSink, mlir::remark::RemarkOpts::name("TokenReused")
                           .category("TokenReuse")
                           .subCategory("OneShot")
                           .function(instance));
      ASSERT_TRUE(remark);
      remark << mlir::remark::detail::Remark::Arg(
                    "Source", static_cast<mlir::LocationAttr>(wrappedSource))
             << mlir::remark::metric("Strategy", "ensure")
             << mlir::remark::metric("Score", 2)
             << mlir::remark::metric("AvailableTokens", 1)
             << mlir::remark::metric("CompatibleTokens", 1);
    }

    mlir::remark::missed(allocateSink,
                         mlir::remark::RemarkOpts::name("TokenNotReused")
                             .category("TokenReuse")
                             .subCategory("OneShot")
                             .function("allocate"))
        << mlir::remark::metric("Reason", "no-available-token")
        << mlir::remark::metric("AvailableTokens", 0)
        << mlir::remark::metric("CompatibleTokens", 0);
  }

  auto buffer = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(static_cast<bool>(buffer)) << buffer.getError().message();
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  ASSERT_TRUE(static_cast<bool>(parsed)) << llvm::toString(parsed.takeError());
  const llvm::json::Object *report = parsed->getAsObject();
  ASSERT_NE(report, nullptr);
  EXPECT_EQ(report->getString("schema"), "reussir.token-reuse.v1");

  const llvm::json::Array *files = report->getArray("files");
  ASSERT_NE(files, nullptr);
  ASSERT_EQ(files->size(), 1u);
  EXPECT_EQ((*files)[0].getAsString(), "generic.rr");

  const llvm::json::Array *locations = report->getArray("locations");
  ASSERT_NE(locations, nullptr);
  EXPECT_EQ(locations->size(), 3u);

  const llvm::json::Array *decisions = report->getArray("decisions");
  ASSERT_NE(decisions, nullptr);
  ASSERT_EQ(decisions->size(), 2u);
  const llvm::json::Object *reuse = (*decisions)[0].getAsObject();
  ASSERT_NE(reuse, nullptr);
  EXPECT_EQ(reuse->getString("kind"), "reuse");
  EXPECT_NE(reuse->getInteger("source"), reuse->getInteger("sink"));
  EXPECT_EQ(reuse->getInteger("occurrences"), 2);
  const llvm::json::Array *instances = reuse->getArray("instances");
  ASSERT_NE(instances, nullptr);
  ASSERT_EQ(instances->size(), 2u);
  EXPECT_EQ((*instances)[0].getAsString(), "generic<i32>");
  EXPECT_EQ((*instances)[1].getAsString(), "generic<i64>");

  const llvm::json::Object *allocate = (*decisions)[1].getAsObject();
  ASSERT_NE(allocate, nullptr);
  EXPECT_EQ(allocate->getString("kind"), "allocate");
  EXPECT_EQ(allocate->getString("reason"), "no-available-token");
  EXPECT_FALSE(allocate->get("source"));
}

} // namespace
