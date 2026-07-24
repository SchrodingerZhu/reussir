#include "Reussir/RustCompiler.h"
#include <gtest/gtest.h>
#include <llvm/IR/LLVMContext.h>

namespace reussir {
constexpr llvm::StringRef EXAMPLE_SOURCE = R"(
extern crate reussir_rt as rt;
use rt::collections::vec::Vec;
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_extern_Vec_f64_new() -> Vec<f64> {
    Vec::new()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_extern_Vec_f64_push(vec : Vec<f64>, value: f64) -> Vec<f64> {
    vec.push(value)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __reussir_extern_Vec_f64_drop(_ : Vec<f64>) {
}
)";
TEST(RustCompilerTest, CompileSimpleSource) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> m = compileRustSource(context, EXAMPLE_SOURCE);
  ASSERT_NE(m, nullptr);
  EXPECT_NE(m->getFunction("__reussir_extern_Vec_f64_new"), nullptr);
  EXPECT_NE(m->getFunction("__reussir_extern_Vec_f64_push"), nullptr);
  EXPECT_NE(m->getFunction("__reussir_extern_Vec_f64_drop"), nullptr);
}

TEST(RustCompilerTest, ExplicitPathsOverrideDiscovery) {
  // Explicit paths short-circuit both the environment and the probe list, and
  // several package directories survive side by side.
  EXPECT_EQ(findRustCompiler("/explicit/bin/rustc"), "/explicit/bin/rustc");
  llvm::SmallVector<std::string> explicitDeps =
      findRustCompilerDeps({"/explicit/lib", "/explicit/other-lib"});
  ASSERT_EQ(explicitDeps.size(), 2u);
  EXPECT_EQ(explicitDeps[0], "/explicit/lib");
  EXPECT_EQ(explicitDeps[1], "/explicit/other-lib");

  // Feeding the discovered locations back as explicit arguments — alongside a
  // second, package-free directory — exercises the override path end to end.
  llvm::StringRef rustc = findRustCompiler();
  llvm::SmallVector<std::string> deps = findRustCompilerDeps();
  ASSERT_FALSE(rustc.empty());
  ASSERT_FALSE(deps.empty());
  llvm::SmallVector<llvm::StringRef> depDirs(deps.begin(), deps.end());
  depDirs.push_back(".");
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> m =
      compileRustSource(context, EXAMPLE_SOURCE, {}, rustc, depDirs);
  ASSERT_NE(m, nullptr);
  EXPECT_NE(m->getFunction("__reussir_extern_Vec_f64_new"), nullptr);
}
} // namespace reussir
