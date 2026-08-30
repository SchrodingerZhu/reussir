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
if(WIN32 AND CMAKE_CROSSCOMPILING)
  # xwin cross build (Linux host → windows-msvc target). Two prefixes are in
  # play: the HOST LLVM for everything that executes at build time (the
  # tblgen crate links LLVM TableGen C++ into a host proc-macro dylib), and
  # the MSVC target LLVM for what rrc links. The caller provides the host
  # prefix via TABLEGEN_220_PREFIX (the nix shells export it).
  if(DEFINED ENV{TABLEGEN_220_PREFIX})
    set(REUSSIR_TABLEGEN_PREFIX "$ENV{TABLEGEN_220_PREFIX}")
  else()
    message(FATAL_ERROR
      "Cross builds need TABLEGEN_220_PREFIX in the environment: the host "
      "LLVM prefix whose llvm-config/TableGen archives the tblgen "
      "proc-macro can execute and link.")
  endif()
  message(STATUS "Reussir host TableGen prefix (cross): ${REUSSIR_TABLEGEN_PREFIX}")

  # llvm-sys / mlir-sys / melior-macro run `<prefix>/bin/llvm-config` on the
  # build host, but the target prefix only ships llvm-config.exe. Stage a
  # wrapper prefix whose llvm-config runs the real .exe under Wine and maps
  # the drive-letter paths in its output back to host paths; include/ and
  # lib/ link back to the real target prefix.
  set(REUSSIR_WINE_LLVM_PREFIX "${CMAKE_BINARY_DIR}/reussir-wine-llvm-config")
  set(REUSSIR_REAL_LLVM_CONFIG
    "${REUSSIR_LLVM_PREFIX}/bin/llvm-config${CMAKE_EXECUTABLE_SUFFIX}")
  file(MAKE_DIRECTORY "${REUSSIR_WINE_LLVM_PREFIX}/bin")
  file(CREATE_LINK "${REUSSIR_LLVM_PREFIX}/include"
    "${REUSSIR_WINE_LLVM_PREFIX}/include" SYMBOLIC)
  file(CREATE_LINK "${REUSSIR_LLVM_PREFIX}/lib"
    "${REUSSIR_WINE_LLVM_PREFIX}/lib" SYMBOLIC)
  file(WRITE "${REUSSIR_WINE_LLVM_PREFIX}/bin/llvm-config.py"
"import re\n"
"import subprocess\n"
"import sys\n"
"REAL = r\"${REUSSIR_REAL_LLVM_CONFIG}\"\n"
"proc = subprocess.run(['wine', REAL, *sys.argv[1:]], text=True,\n"
"                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n"
"def to_host(match):\n"
"    return match.group(0)[2:].replace('\\\\', '/')\n"
"out = re.sub(r'[A-Za-z]:[\\\\/][^\\s;]*', to_host, proc.stdout)\n"
"sys.stdout.write(out)\n"
"sys.stderr.write(proc.stderr)\n"
"raise SystemExit(proc.returncode)\n")
  file(WRITE "${REUSSIR_WINE_LLVM_PREFIX}/bin/llvm-config"
"#!/bin/sh\n"
"exec python3 \"$(dirname \"$0\")/llvm-config.py\" \"$@\"\n")
  file(CHMOD "${REUSSIR_WINE_LLVM_PREFIX}/bin/llvm-config"
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  set(REUSSIR_LLVM_PREFIX "${REUSSIR_WINE_LLVM_PREFIX}")
  message(STATUS
    "Reussir Wine llvm-config wrapper (cross): ${REUSSIR_WINE_LLVM_PREFIX}")

  # conda-forge's LLVM was built with the MSVC DIA SDK, so its exported
  # LLVMDebugInfoPDB target references diaguids.lib at an absolute Visual
  # Studio path that exists on no cross host (and that xwin does not ship).
  # Drop the reference: linkers only pull DIA-using archive members when
  # something calls the DIA-backed PDB readers, which nothing here does.
  if(TARGET LLVMDebugInfoPDB)
    get_target_property(_reussir_pdb_libs LLVMDebugInfoPDB INTERFACE_LINK_LIBRARIES)
    if(_reussir_pdb_libs)
      list(FILTER _reussir_pdb_libs EXCLUDE REGEX "diaguids")
      set_target_properties(LLVMDebugInfoPDB PROPERTIES
        INTERFACE_LINK_LIBRARIES "${_reussir_pdb_libs}")
      message(STATUS "Stripped MSVC DIA SDK reference from LLVMDebugInfoPDB (cross)")
    endif()
  endif()
elseif(WIN32)
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
  if(CMAKE_CROSSCOMPILING)
    # The host clang-cl (from the toolchain file) compiles the build
    # scripts' target C++ with --target=x86_64-pc-windows-msvc.
    set(REUSSIR_CARGO_CXX "${CMAKE_CXX_COMPILER}")
  else()
    set(REUSSIR_CARGO_CXX
      "${REUSSIR_LLVM_PREFIX}/bin/clang-cl${CMAKE_EXECUTABLE_SUFFIX}")
  endif()
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

