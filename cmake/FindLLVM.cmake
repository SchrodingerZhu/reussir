find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include(${LLVM_DIR}/AddLLVM.cmake)
include(${LLVM_DIR}/TableGen.cmake)
include(${LLVM_DIR}/HandleLLVMOptions.cmake)

# Reussir is locked to the LLVM 22 release line. The Rust backend (mlir-sys /
# melior / tblgen / llvm-sys) is pinned to the 22 ABI, so any other major
# version is rejected outright rather than silently mis-linking.
if(LLVM_PACKAGE_VERSION VERSION_LESS "22.0.0" OR
   NOT LLVM_PACKAGE_VERSION VERSION_LESS "23.0.0")
  message(FATAL_ERROR
    "Reussir requires LLVM 22.x (found ${LLVM_PACKAGE_VERSION})")
endif()

# Normalized install prefix exported for the Rust crates' cargo invocations
# (MLIR_SYS_220_PREFIX / TABLEGEN_220_PREFIX / LLVM_SYS_221_PREFIX). LLVMConfig
# defines LLVM_INSTALL_PREFIX; fall back to the parent of the binary dir.
if(DEFINED LLVM_INSTALL_PREFIX)
  set(REUSSIR_LLVM_PREFIX "${LLVM_INSTALL_PREFIX}")
else()
  get_filename_component(REUSSIR_LLVM_PREFIX "${LLVM_TOOLS_BINARY_DIR}" DIRECTORY)
endif()
set(REUSSIR_LLVM_MAJOR 22)
message(STATUS "Reussir LLVM prefix for Rust crates: ${REUSSIR_LLVM_PREFIX}")

set(REUSSIR_TABLEGEN_PREFIX "${REUSSIR_LLVM_PREFIX}")
if(WIN32)
  set(REUSSIR_TABLEGEN_PREFIX
    "${CMAKE_BINARY_DIR}/reussir-tablegen-llvm-config")
  set(REUSSIR_REAL_LLVM_CONFIG
    "${REUSSIR_LLVM_PREFIX}/bin/llvm-config${CMAKE_EXECUTABLE_SUFFIX}")
  file(MAKE_DIRECTORY "${REUSSIR_TABLEGEN_PREFIX}/bin")
  file(WRITE "${REUSSIR_TABLEGEN_PREFIX}/bin/llvm-config.py"
"import subprocess\n"
"import sys\n"
"from pathlib import Path\n"
"REAL = r\"${REUSSIR_REAL_LLVM_CONFIG}\"\n"
"args = sys.argv[1:]\n"
"proc = subprocess.run([REAL, *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n"
"out = proc.stdout\n"
"def base_name(token):\n"
"    return token.replace('\\\\', '/').rsplit('/', 1)[-1]\n"
"def msvc_lib_name(token):\n"
"    base = base_name(token)\n"
"    low = base.lower()\n"
"    if low.endswith('.dll.lib'):\n"
"        return base[:-8]\n"
"    if low.endswith('.lib'):\n"
"        return base[:-4]\n"
"    return base\n"
"def msvc_system_lib_name(token):\n"
"    name = msvc_lib_name(token)\n"
"    if name == base_name(token):\n"
"        return token\n"
"    libdir = Path(REAL).resolve().parent.parent / 'lib'\n"
"    if (libdir / f'{name}.lib').exists():\n"
"        return name\n"
"    if not name.lower().startswith('lib') and (libdir / f'lib{name}.lib').exists():\n"
"        return 'lib' + name\n"
"    return name\n"
"if proc.returncode == 0 and '--libnames' in args:\n"
"    tokens = []\n"
"    for token in out.split():\n"
"        name = msvc_lib_name(token)\n"
"        if name != base_name(token):\n"
"            tokens.append('lib' + name + '.a')\n"
"        else:\n"
"            tokens.append(token)\n"
"    out = ' '.join(tokens) + ('\\n' if out.endswith(('\\n', '\\r\\n')) else '')\n"
"elif proc.returncode == 0 and '--system-libs' in args:\n"
"    tokens = []\n"
"    for token in out.split():\n"
"        name = msvc_system_lib_name(token)\n"
"        if name != base_name(token):\n"
"            tokens.append(name)\n"
"        else:\n"
"            tokens.append(token)\n"
"    out = ' '.join(tokens) + ('\\n' if out.endswith(('\\n', '\\r\\n')) else '')\n"
"sys.stdout.write(out)\n"
"sys.stderr.write(proc.stderr)\n"
"raise SystemExit(proc.returncode)\n")
  file(WRITE "${REUSSIR_TABLEGEN_PREFIX}/bin/llvm-config.cmd"
    "@echo off\r\npython \"%~dp0llvm-config.py\" %*\r\n")
  file(WRITE "${REUSSIR_TABLEGEN_PREFIX}/bin/llvm-config.bat"
    "@echo off\r\npython \"%~dp0llvm-config.py\" %*\r\n")
  message(STATUS
    "Reussir TableGen llvm-config wrapper: ${REUSSIR_TABLEGEN_PREFIX}")
endif()

set(REUSSIR_CARGO_CXX_ENV)
if(WIN32)
  set(REUSSIR_CARGO_CXX
    "${REUSSIR_LLVM_PREFIX}/bin/clang-cl${CMAKE_EXECUTABLE_SUFFIX}")
  if(NOT EXISTS "${REUSSIR_CARGO_CXX}")
    message(FATAL_ERROR
      "Windows Cargo C++ build scripts require clang-cl at "
      "${REUSSIR_CARGO_CXX}")
  endif()
  list(APPEND REUSSIR_CARGO_CXX_ENV
    CXX=${REUSSIR_CARGO_CXX}
    # tblgen adds /WX; clang-cl reports warnings in LLVM headers under /W4.
    "CXXFLAGS_x86_64_pc_windows_msvc=/wd4864 /WX- /clang:-Wno-unused-parameter")
  message(STATUS
    "Reussir Cargo C++ compiler for build scripts: ${REUSSIR_CARGO_CXX}")
endif()

