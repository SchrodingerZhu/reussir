#!/usr/bin/env bash
# Cross-builds the Windows toolchain (rrc, rene, rrepl, reussir-lsp) from a
# Linux host. Run inside `nix develop .#windows-cross`, after
# `reussir-bake-msvc-llvm` has staged the conda-forge MSVC LLVM/MLIR.
#
# Pipeline:
#   1. cargo xwin populates the MSVC CRT/SDK splat and cross-builds the
#      LLVM-free crates.
#   2. CMake (cmake/toolchains/linux-to-windows-msvc.cmake: GNU-driver clang
#      -> COFF) cross-compiles the C++ MLIR dialect + C API static archives;
#      cmake/FindLLVM.cmake stages the wine-backed llvm-config wrapper.
#   3. cargo xwin links rrc/rrepl against those archives and the conda MSVC
#      LLVM; proc-macros (tblgen/melior-macro) use the HOST LLVM prefixes
#      the windows-cross shell exports.
#
# Outputs land in ${BUILD_DIR}/target/x86_64-pc-windows-msvc/release/.
set -euo pipefail

BUILD_DIR="${REUSSIR_XWIN_BUILD_DIR:-build-xwin}"
CONDA="${XDG_CACHE_HOME:-$HOME/.cache}/reussir-msvc-conda/Library"
TARGET=x86_64-pc-windows-msvc

if [ ! -e "$CONDA/lib/cmake/mlir/MLIRConfig.cmake" ]; then
  echo "error: conda MSVC LLVM/MLIR not baked; run reussir-bake-msvc-llvm" >&2
  exit 1
fi

# 1. Rust-only crates first: this also populates the xwin CRT/SDK splat the
# CMake toolchain file needs.
cargo xwin build --locked --release --target "$TARGET" \
  -p rene -p reussir-lsp

# 2. C++ backend static archives.
launchers=()
if command -v sccache >/dev/null; then
  launchers=(-DCMAKE_C_COMPILER_LAUNCHER=sccache
             -DCMAKE_CXX_COMPILER_LAUNCHER=sccache)
fi
# NIX_STORE is unset for the configure: the CMakeLists nix branch injects
# HOST include paths for the scan-deps workaround, which must not leak into
# a Windows-target compile.
env -u NIX_STORE cmake -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux-to-windows-msvc.cmake \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DREUSSIR_ENABLE_TESTS=OFF \
  -DCMAKE_PREFIX_PATH="$CONDA" \
  -DLLVM_DIR="$CONDA/lib/cmake/llvm" \
  -DMLIR_DIR="$CONDA/lib/cmake/mlir" \
  "${launchers[@]}"

# The staged archives rrc links: exactly what reussir-backend-sys asks for
# (kept self-maintaining by reading its build script), plus the umbrella
# dialect/CAPI targets.
mapfile -t libs < <(grep -oE 'MLIRReussir[A-Za-z]+|ReussirCAPI' \
  crates/reussir-backend-sys/build.rs | sort -u)
cmake --build "$BUILD_DIR" --parallel --target MLIRReussir "${libs[@]}"

# 3. The LLVM-linking binaries. Target prefixes go through the wine
# llvm-config wrapper FindLLVM.cmake staged; host prefixes
# (TABLEGEN_220_PREFIX etc.) come from the windows-cross shell.
export MLIR_SYS_220_PREFIX="$PWD/$BUILD_DIR/reussir-wine-llvm-config"
export MLIR_SYS_210_PREFIX="$PWD/$BUILD_DIR/reussir-wine-llvm-config"
export LLVM_SYS_221_PREFIX="$PWD/$BUILD_DIR/reussir-wine-llvm-config"
export REUSSIR_CAPI_LIB_DIR="$PWD/$BUILD_DIR/lib"
export REUSSIR_INCLUDE_DIR="$PWD/include"
export RUSTFLAGS="${RUSTFLAGS:--Awarnings}"
cargo xwin build --locked --release --target "$TARGET" \
  -p reussir-compiler -p reussir-repl

echo "Cross-built Windows binaries:"
ls -la target/"$TARGET"/release/{rrc,rrepl,rene,reussir-lsp}.exe
