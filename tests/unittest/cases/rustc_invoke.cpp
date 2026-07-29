#include "Reussir/RustCompiler.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <llvm/IR/LLVMContext.h>

namespace reussir {
namespace {
// The build tree stages neither a toolchain nor runtime artifacts the
// probe list could find, so default the discovery environment to the
// configure-time paths (tests/unittest/CMakeLists.txt). An environment the
// caller already set — the lit suite's, or a developer pinning their own
// toolchain — is left alone.
void setDefaultEnv(const char *name, const char *value) {
  if (std::getenv(name))
    return;
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, /*overwrite=*/0);
#endif
}

void primeRustDiscoveryEnv() {
  setDefaultEnv("REUSSIR_RUSTC", REUSSIR_UT_RUSTC);
  setDefaultEnv("REUSSIR_RUSTC_DEPS", REUSSIR_UT_RUSTC_DEPS);
}
} // namespace

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
  primeRustDiscoveryEnv();
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> m = compileRustSource(context, EXAMPLE_SOURCE);
  ASSERT_NE(m, nullptr);
  EXPECT_NE(m->getFunction("__reussir_extern_Vec_f64_new"), nullptr);
  EXPECT_NE(m->getFunction("__reussir_extern_Vec_f64_push"), nullptr);
  EXPECT_NE(m->getFunction("__reussir_extern_Vec_f64_drop"), nullptr);
}

TEST(RustCompilerTest, ExplicitPathsOverrideDiscovery) {
  primeRustDiscoveryEnv();
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

TEST(RustCompilerTest, BareCompilerNameResolvesThroughPath) {
  primeRustDiscoveryEnv();
  llvm::SmallVector<std::string> deps = findRustCompilerDeps();
  ASSERT_FALSE(deps.empty());
  llvm::SmallVector<llvm::StringRef> depDirs(deps.begin(), deps.end());
#ifdef _WIN32
  constexpr llvm::StringRef bareRustc = "rustc.exe";
#else
  constexpr llvm::StringRef bareRustc = "rustc";
#endif
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> m =
      compileRustSource(context, EXAMPLE_SOURCE, {}, bareRustc, depDirs);
  ASSERT_NE(m, nullptr);
  EXPECT_NE(m->getFunction("__reussir_extern_Vec_f64_new"), nullptr);
}
} // namespace reussir
