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
  // An explicit path short-circuits both the environment and the probe list.
  EXPECT_EQ(findRustCompiler("/explicit/bin/rustc"), "/explicit/bin/rustc");
  EXPECT_EQ(findRustCompilerDeps("/explicit/lib"), "/explicit/lib");

  // Feeding the discovered locations back as explicit arguments exercises the
  // override path end to end.
  llvm::StringRef rustc = findRustCompiler();
  llvm::StringRef deps = findRustCompilerDeps();
  ASSERT_FALSE(rustc.empty());
  ASSERT_FALSE(deps.empty());
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> m =
      compileRustSource(context, EXAMPLE_SOURCE, {}, rustc, deps);
  ASSERT_NE(m, nullptr);
  EXPECT_NE(m->getFunction("__reussir_extern_Vec_f64_new"), nullptr);
}
} // namespace reussir
