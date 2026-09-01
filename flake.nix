{
  description = "Reussir — MLIR-based compiler framework for RC-managed functional programs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, fenix, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # LLVM/MLIR 23 — must match FindLLVM.cmake's version check (23.x only).
        llvmPkgs = pkgs.llvmPackages_23;

        # Pinned nightly Rust toolchain from rust-toolchain.toml.
        # The sha256 covers the channel manifest downloaded by fenix.
        rustToolchain = fenix.packages.${system}.fromToolchainFile {
          file = ./rust-toolchain.toml;
          sha256 = "sha256-ko0a8G9o/p60mphrxmH0dNQsUdWkKMBaGexsqEqtCF4=";
        };

        # Merge LLVM+MLIR outputs into one prefix so that llvm-sys/mlir-sys
        # Rust build scripts (which expect a single "install prefix" with
        # bin/llvm-config, bin/mlir-tblgen, include/, lib/) work correctly.
        #
        # nixpkgs splits each LLVM component into multiple outputs:
        #   llvm.dev  → bin/llvm-config, include/llvm*, lib/cmake/llvm/
        #   llvm.lib  → lib/libLLVM*.so / lib/libLLVM*.a
        #   mlir.dev  → include/mlir*, lib/cmake/mlir/
        #   mlir      → lib/libMLIR*.a
        #   tblgen    → bin/llvm-tblgen, bin/mlir-tblgen, bin/clang-tblgen
        #
        # symlinkJoin (buildEnv) deep-merges these into one virtual prefix.
        #
        # postBuild patches LLVMConfig.cmake to set LLVM_INSTALL_PREFIX to this
        # join path rather than the original llvm.out store path.  cmake/FindLLVM.cmake
        # (project code) derives REUSSIR_LLVM_PREFIX from LLVM_INSTALL_PREFIX and
        # passes it as LLVM_SYS_231_PREFIX to cargo.  llvm-sys needs to find
        # bin/llvm-config at that prefix — but llvm.out doesn't have it; only
        # llvm.dev does.  The patched join path has llvm-config (from llvm.dev).
        llvmMlirJoin = pkgs.symlinkJoin {
          name = "reussir-llvm-mlir-${llvmPkgs.release_version}";
          paths = [
            llvmPkgs.llvm.dev   # llvm-config, headers, cmake configs
            llvmPkgs.llvm.lib   # LLVM shared/static libraries
            llvmPkgs.mlir.dev   # MLIR headers, cmake configs
            llvmPkgs.mlir       # MLIR static libraries
            llvmPkgs.tblgen     # mlir-tblgen, llvm-tblgen
          ];
          postBuild = ''
            sed -i \
              "s|set(LLVM_INSTALL_PREFIX \"[^\"]*\")|set(LLVM_INSTALL_PREFIX \"$out\")|" \
              "$out/lib/cmake/llvm/LLVMConfig.cmake"

            # llvm-config has the llvm.dev/llvm.lib store paths baked in, so
            # --includedir/--libdir point at outputs that lack the MLIR pieces
            # (mlir-c/ headers, libMLIR*.a).  mlir-sys walks those two paths in
            # its build script and fails.  Replace the symlink with a shim that
            # rewrites the llvm output paths to this join, which has everything.
            rm "$out/bin/llvm-config"
            cat > "$out/bin/llvm-config" <<EOF
            #!${pkgs.runtimeShell}
            output=\$("${llvmPkgs.llvm.dev}/bin/llvm-config" "\$@") || exit \$?
            printf '%s\n' "\$output" | sed \
              -e "s|${llvmPkgs.llvm.dev}|$out|g" \
              -e "s|${llvmPkgs.llvm.lib}|$out|g" \
              -e "s|${llvmPkgs.llvm}|$out|g"
            EOF
            chmod +x "$out/bin/llvm-config"
          '';
        };

        # Rust standard library for the MSVC target, at the same pinned
        # nightly as rust-toolchain.toml (same channel manifest, same hash).
        # Kept out of rust-toolchain.toml so rustup users and the default
        # shell don't download a Windows std they never use; only the
        # windows-cross shell below pays for it.
        windowsRustToolchain = fenix.packages.${system}.combine [
          rustToolchain
          (fenix.packages.${system}.targets."x86_64-pc-windows-msvc".toolchainOf {
            channel = "nightly";
            date = "2026-08-31";
            sha256 = "sha256-ko0a8G9o/p60mphrxmH0dNQsUdWkKMBaGexsqEqtCF4=";
          }).rust-std
        ];

        # Bakes the conda-forge MSVC LLVM/MLIR toolchain the Windows CI uses
        # (.github/conda/windows-msvc.yml) into a local win-64 prefix, so the
        # backend crates have the same import libraries and headers to build
        # against as the GitHub runners. Only the library subset of the CI
        # env file is fetched: the host-side tools (cmake, ninja, python,
        # lit) come from nix, and the pip section cannot execute inside a
        # foreign-platform prefix. Versions must track windows-msvc.yml.
        bakeMsvcLlvm = pkgs.writeShellScriptBin "reussir-bake-msvc-llvm" ''
          set -euo pipefail
          prefix="''${XDG_CACHE_HOME:-$HOME/.cache}/reussir-msvc-conda"
          # Full bake and later additions are separately idempotent: a prefix
          # baked (or cache-restored) before clangxx/ml64 joined the recipe
          # still gets them.
          if [ ! -e "$prefix/Library/lib/cmake/mlir/MLIRConfig.cmake" ] || [ "''${1:-}" = "--force" ]; then
            # --platform win-64 extracts the Windows packages without running
            # their activation scripts, which is all a cross link needs.
            # clangxx supplies the windows clang-cl.exe that cc-rs build
            # scripts need when the windows-native cargo runs under Wine.
            ${pkgs.micromamba}/bin/micromamba create --yes \
              --root-prefix "''${XDG_CACHE_HOME:-$HOME/.cache}/reussir-micromamba" \
              --prefix "$prefix" \
              --platform win-64 \
              --channel conda-forge \
              'llvmdev=23.1.0' 'llvm-tools=23.1.0' 'mlir=23.1.0' \
              'clangxx=23.1.0' \
              'compiler-rt=23.1.0' gtest spdlog zlib zstd libxml2
          elif [ ! -x "$prefix/Library/bin/clang-cl.exe" ]; then
            echo "existing bake lacks clangxx; installing into $prefix"
            ${pkgs.micromamba}/bin/micromamba install --yes \
              --root-prefix "''${XDG_CACHE_HOME:-$HOME/.cache}/reussir-micromamba" \
              --prefix "$prefix" \
              --platform win-64 \
              --channel conda-forge \
              'clangxx=23.1.0'
          else
            echo "MSVC LLVM/MLIR already baked at $prefix"
          fi

          # cc-rs assembles MASM (the psm crate) through ml64.exe; llvm-ml64
          # is a drop-in.
          [ -e "$prefix/Library/bin/ml64.exe" ] || \
            cp "$prefix/Library/bin/llvm-ml64.exe" "$prefix/Library/bin/ml64.exe"

          # Import-library aliases the LLVM link line expects but conda does
          # not ship under those names (the native Windows CI generates the
          # same two with dumpbin/lib.exe): zstd.dll.lib aliases zstd.lib,
          # and xml2.lib is synthesized from libxml2.dll's export table —
          # newer conda libxml2 ships no import library at all.
          lib="$prefix/Library/lib"
          [ -e "$lib/zstd.dll.lib" ] || cp "$lib/zstd.lib" "$lib/zstd.dll.lib"
          if [ ! -e "$lib/xml2.lib" ]; then
            def=$(mktemp -d)/xml2.def
            {
              echo "LIBRARY libxml2.dll"
              echo "EXPORTS"
              ${llvmPkgs.llvm}/bin/llvm-readobj --coff-exports \
                "$prefix/Library/bin/libxml2.dll" \
                | sed -n 's/^ *Name: //p' | grep -v '^libxml2\.dll$' | sed 's/^/  /'
            } > "$def"
            ${llvmPkgs.llvm}/bin/llvm-dlltool -m i386:x86-64 \
              -d "$def" -D libxml2.dll -l "$lib/xml2.lib"
          fi
          echo "Baked MSVC LLVM/MLIR into $prefix/Library"
        '';

        # Bakes the official windows-msvc Rust dist (same pinned nightly as
        # rust-toolchain.toml) so the reussir compilers can spawn a
        # windows-native rustc/cargo under Wine: PE parents awaiting PE
        # children work (Wine returns no waitable handle for unix children,
        # which is what broke rrc's polyffi rustc and rene's cargo bake).
        # rustc.exe resolves its sysroot from its own location; rust-lld is
        # additionally staged as lld-link.exe so linker-flavor detection by
        # argv[0] picks the link.exe style.
        bakeMsvcRustc = pkgs.writeShellScriptBin "reussir-bake-msvc-rustc" ''
          set -euo pipefail
          prefix="''${XDG_CACHE_HOME:-$HOME/.cache}/reussir-msvc-rustc"
          if [ -x "$prefix/bin/rustc.exe" ] && [ "''${1:-}" != "--force" ]; then
            echo "windows rustc already baked at $prefix (use --force to redo)"
            exit 0
          fi
          tmp=$(mktemp -d)
          trap 'rm -rf "$tmp"' EXIT
          cd "$tmp"
          for c in rustc cargo rust-std; do
            ${pkgs.curl}/bin/curl -sSfLO \
              "https://static.rust-lang.org/dist/2026-08-31/$c-nightly-x86_64-pc-windows-msvc.tar.xz"
            tar xf "$c-nightly-x86_64-pc-windows-msvc.tar.xz"
          done
          mkdir -p "$prefix"
          cp -r rustc-nightly-x86_64-pc-windows-msvc/rustc/* "$prefix/"
          cp -r cargo-nightly-x86_64-pc-windows-msvc/cargo/* "$prefix/"
          cp -r rust-std-nightly-x86_64-pc-windows-msvc/rust-std-x86_64-pc-windows-msvc/* "$prefix/"
          bindir="$prefix/lib/rustlib/x86_64-pc-windows-msvc/bin"
          cp -f "$bindir/rust-lld.exe" "$bindir/lld-link.exe"
          echo "Baked windows rustc/cargo into $prefix"
        '';

        # System header flags for the clang-scan-deps workaround.
        #
        # cmake 3.28+ with C++23 uses clang-scan-deps (the raw binary, not the
        # nixpkgs clang wrapper) for dependency scanning.  clang-scan-deps uses
        # the clang library API directly — it does NOT invoke the wrapper script —
        # so it never receives the system header paths the wrapper injects from
        # its nix-support/ files.  Without those paths, scanning fails with
        # "fatal error: 'type_traits' file not found" (GCC C++ stdlib) then
        # "'features.h' file not found" (glibc).
        #
        # The fix: embed the missing include paths in CMAKE_CXX_FLAGS (stored in
        # CMakeCache.txt and propagated into compile_commands.json).  clang-scan-
        # deps reads the compilation database, so it sees these flags and the
        # internal clang driver finds the headers.
        #
        # Headers required:
        #   GCC C++ stdlib  →  from libcxx-cxxflags: -cxx-isystem <path> pairs
        #   glibc           →  from libc-cflags:     -idirafter <path>
        #
        # We parse both files at Nix eval time (single space-separated lines) by
        # zipping words with their successor to find each token's argument.
        nixSysFlags =
          let
            # Generic helper: split a nix-support flag file into words, then
            # collect every occurrence of <token> followed by its argument path.
            extractPaths = file: token:
              let
                words  = builtins.filter (s: s != "")
                           (pkgs.lib.splitString " "
                             (builtins.readFile file));
                indexed = pkgs.lib.imap0 (i: v: { inherit i v; }) words;
              in
              map (e: builtins.elemAt words (e.i + 1))
                (builtins.filter (e: e.v == token) indexed);

            gccCxxPaths = extractPaths
              "${llvmPkgs.stdenv.cc}/nix-support/libcxx-cxxflags"
              "-cxx-isystem";

            glibcPaths = extractPaths
              "${llvmPkgs.stdenv.cc}/nix-support/libc-cflags"
              "-idirafter";
          in
          pkgs.lib.concatStringsSep " " (
            (map (p: "-isystem ${p}") gccCxxPaths) ++
            (map (p: "-idirafter ${p}") glibcPaths)
          );

      in {
        # ---------------------------------------------------------------------------
        # Development shell: `nix develop` or `direnv allow` (via .envrc).
        #
        # Provides everything needed to:
        #   cmake --preset nix-dev -B build
        #   cmake --build build
        #
        # LSP support:
        #   C/C++  → clangd (from clang-tools), reads build/compile_commands.json
        #   Rust   → rust-analyzer (from pkgs), reads rust-project.json / Cargo.toml
        # ---------------------------------------------------------------------------
        devShells.default = pkgs.mkShell.override {
          # Build C++ code with clang 23 (same toolchain as the project uses).
          stdenv = llvmPkgs.stdenv;
        } {
          name = "reussir-dev";

          # Disable fortify hardening: cargo compiles build scripts with
          # opt-level=0 (cargo's rule: build scripts always use the dev
          # profile, even for --profile release).  The nixpkgs clang wrapper
          # injects -O2 (before) and -D_FORTIFY_SOURCE=3 (after) via hardening,
          # but cc-rs's -O0 overrides the -O2, leaving FORTIFY_SOURCE defined
          # without optimization.  tblgen's build.rs then adds -Werror which
          # turns the resulting #warning into a fatal error.
          # Disabling fortify in the dev shell is the standard fix; production
          # builds go through their own hardened derivation.
          hardeningDisable = [ "fortify" "fortify3" ];

          packages = [
            # Build orchestration
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config

            # Python (needed by lit and MLIR's Python bindings)
            pkgs.python3
            pkgs.python3Packages.lit

            # C++ test framework (CMakeLists.txt: find_package(GTest REQUIRED))
            pkgs.gtest

            # LLVM/Clang/MLIR toolchain
            llvmPkgs.clang          # clang, clang++
            llvmPkgs.lld            # lld linker
            llvmPkgs.clang-tools    # clangd, clang-tidy, clang-format, clang-scan-deps
            llvmPkgs.llvm           # llvm tools (llvm-ar, opt, …)
            llvmPkgs.mlir           # mlir-opt, mlir-translate, etc.
            llvmPkgs.tblgen         # mlir-tblgen, llvm-tblgen (needed by cmake & mlir-sys)
            # Deliberately the default nixpkgs lldb, not llvmPkgs.lldb: the
            # debuginfo lit suite only needs a DWARF-reading debugger —
            # version lockstep with the compiler is not required.
            pkgs.lldb

            # Rust toolchain (nightly-2026-08-31 as pinned in rust-toolchain.toml)
            rustToolchain

            # Rust LSP — works alongside the pinned toolchain
            pkgs.rust-analyzer

            # VS Code extension: TypeScript bundling and VSIX packaging. The
            # Rust client component's wasm32-unknown-unknown standard library
            # is supplied by rust-toolchain.toml through rustToolchain above.
            pkgs.nodejs_24

            # Generates the JS/TS bindings for the extension's wasm codec.
            # Its version must equal the workspace's exact wasm-bindgen pin
            # (Cargo.toml); scripts/build-wasm.mjs enforces the match.
            pkgs.wasm-bindgen-cli

            # cargo-nextest is convenient for `cargo nextest run` workflows
            pkgs.cargo-nextest

            # spdlog is used by lib/Bridge (find_package(spdlog REQUIRED))
            pkgs.spdlog

            # zlib / libxml2 are LLVM link-time deps on some targets
            pkgs.zlib
            pkgs.libxml2

            # The engine the WebAssembly lit suites run their modules on
            # (tests/integration/rene/wasi_*.rr). Those suites also want the
            # targets' Rust standard libraries, which this pinned toolchain
            # does not carry — `rustup target add wasm32-wasip1
            # wasm32-wasip1-threads` in a rustup shell, or add the targets
            # to the fenix toolchain above. Missing either piece, a suite
            # reports UNSUPPORTED rather than failing.
            pkgs.wasmer
          ]
          # Linux only: gdb cannot properly debug arm64 Mach-O binaries, and
          # shipping it on darwin flips the lit `gdb` feature on — which made
          # tests/integration/debuginfo/gdb_print.rr run (and fail) on macOS
          # where it had always been UNSUPPORTED. lldb covers darwin.
          ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.gdb
          ];

          # libomp: the OpenMP e2e lit tests probe `clang -fopenmp` and are
          # skipped when it fails. As a buildInput the cc wrapper injects
          # omp.h (-isystem) and libomp.so (-L) via NIX_CFLAGS_COMPILE /
          # NIX_LDFLAGS, which `packages` (nativeBuildInputs) would not.
          # Default nixpkgs openmp, not llvmPkgs.openmp: libomp's
          # interface/ABI is stable across majors — the probe and the e2e
          # runs only need a working libomp.
          buildInputs = [
            pkgs.llvmPackages.openmp
          ];

          # --- CMake discovery: tell cmake exactly where the LLVM/MLIR configs are
          LLVM_DIR = "${llvmMlirJoin}/lib/cmake/llvm";
          MLIR_DIR = "${llvmMlirJoin}/lib/cmake/mlir";

          # --- Rust crate build-script discovery variables.
          #     The cmake build exports these via REUSSIR_BACKEND_CARGO_ENV (see
          #     crates/reussir-backend/CMakeLists.txt).  We mirror them here so
          #     that plain `cargo build -p reussir-backend` also works.
          #     Two numbering schemes collide here: the melior family uses
          #     "<llvm-major>0" (230 = LLVM 23), while llvm-sys uses its crate
          #     major, which mashes LLVM major.minor together (231 = LLVM 23.1).
          # read by mlir-sys 220 (the git rev patched in Cargo.toml, retargeted
          # to LLVM 23 — its build script derives 230 from the LLVM major)
          MLIR_SYS_230_PREFIX     = "${llvmMlirJoin}";
          # melior-macro 0.20 still uses the 210 variable name despite targeting
          # MLIR 22+; point it at the same combined prefix.
          MLIR_SYS_210_PREFIX     = "${llvmMlirJoin}";
          # read by the tblgen 0.9.1 fork (llvm22-0 feature retargeted to 23)
          TABLEGEN_230_PREFIX     = "${llvmMlirJoin}";
          # read by llvm-sys 231 — the only LLVM_SYS_* var in Cargo.lock
          LLVM_SYS_231_PREFIX     = "${llvmMlirJoin}";

          # bindgen (mlir-sys / llvm-sys wrapper.h) loads libclang.so at build
          # time; the nixpkgs clang wrapper doesn't put it on any search path.
          LIBCLANG_PATH = "${llvmPkgs.libclang.lib}/lib";
          # bindgen invokes libclang directly (not the nix cc wrapper), so it
          # misses the wrapper-injected system include paths; reuse the same
          # flag set computed for clang-scan-deps.
          BINDGEN_EXTRA_CLANG_ARGS = nixSysFlags;

          # Enable compile_commands.json generation for clangd.
          CMAKE_EXPORT_COMPILE_COMMANDS = "1";

          # cmake/FindMLIR.cmake checks this env var and uses it as MLIR_TABLEGEN_EXE.
          # Without an absolute path, cmake passes "mlir-tblgen" as a bare string to
          # add_custom_command(), which ninja then misinterprets as a file build target
          # (relative to the tablegen output directory) instead of a PATH-found binary.
          MLIR_TABLEGEN_EXE_OVERRIDE = "${llvmPkgs.tblgen}/bin/mlir-tblgen";

          shellHook = ''
            # stdenv.cc.cc.lib supplies libstdc++.so.6: proc-macro dylibs that
            # statically link LLVM TableGen C++ (melior-macro via the tblgen
            # crate) need it at dlopen time when rustc expands the macro, and
            # NixOS has no global libstdc++ — without this, rustc reports the
            # misleading "error[E0463]: can't find crate for `melior_macro`".
            # zlib: rustc links rrc without an rpath entry for it (LLVM's
            # system-libs pull in -lz), so the binary needs it findable at
            # run time — lit tests execute build/bin/rrc directly.
            # openmp: the lit OpenMP probe links via the wrapper's NIX_LDFLAGS
            # (no rpath — `-print-file-name=libomp.so` does not see wrapper
            # -L paths), so the probe binary finds libomp.so only through
            # LD_LIBRARY_PATH.
            export LD_LIBRARY_PATH="${llvmPkgs.llvm.lib}/lib:${llvmPkgs.mlir}/lib:${pkgs.llvmPackages.openmp}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            # Write CMakeUserPresets.json (user-local, gitignored) so that
            # `cmake --preset nix-dev` just works with no extra flags.
            #
            # Key settings explained:
            #
            # LLVM_ENABLE_LIBCXX=ON — skips LLVM's libstdc++ >= 7.4 try_compile
            #   probe (cmake 3.28 uses clang-scan-deps for this probe; without the
            #   GCC C++ headers clang-scan-deps fails on <iosfwd>).  The flag does
            #   NOT force -stdlib=libc++: LLVM_USES_LIBSTDCXX fires afterwards and
            #   detects we are on libstdc++, suppressing that flag.
            #
            # CMAKE_CXX_FLAGS — injects the GCC C++ stdlib include paths that the
            #   nixpkgs clang wrapper normally injects via -cxx-isystem, but which
            #   clang-scan-deps (the raw binary) does not receive.  These paths end
            #   up in compile_commands.json and are therefore visible to clang-scan-
            #   deps when it scans C++23 module dependencies during the build.
            #   Value is computed at Nix eval time from nix-support/libcxx-cxxflags.
            #
            # Nix string interpolations expand to store paths at eval time;
            # cmake's ''${sourceDir} macro is kept literal (single-quoted HEREDOC).
            cat > CMakeUserPresets.json << 'PRESETS_EOF'
{
  "version": 6,
  "configurePresets": [
    {
      "name": "nix-dev",
      "displayName": "Nix devShell (reussir flake)",
      "description": "Auto-generated by the Nix devShell — do not commit",
      "generator": "Ninja",
      "binaryDir": "''${sourceDir}/build",
      "cacheVariables": {
        "CMAKE_C_COMPILER":              "clang",
        "CMAKE_CXX_COMPILER":            "clang++",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "LLVM_ENABLE_LIBCXX":            "ON",
        "CMAKE_CXX_FLAGS":               "${nixSysFlags}",
        "LLVM_DIR":                      "${llvmMlirJoin}/lib/cmake/llvm",
        "MLIR_DIR":                      "${llvmMlirJoin}/lib/cmake/mlir"
      }
    }
  ]
}
PRESETS_EOF

            echo ""
            echo "  Reussir dev shell — LLVM/MLIR ${llvmPkgs.release_version}"
            echo ""
            echo "  Quick start (CMakeUserPresets.json 'nix-dev' preset written):"
            echo "    cmake --preset nix-dev -B build    # configure into ./build"
            echo "    cmake --build build                # compile"
            echo "    cmake --build build --target reussir-vscode-package # build VSIX"
            echo "    ln -sf build/compile_commands.json .   # for clangd"
            echo ""
          '';
        };
        # ---------------------------------------------------------------------------
        # Windows cross shell (optional): `nix develop .#windows-cross`.
        #
        # Cross-builds the LLVM-free Rust crates (the workspace default
        # members) for x86_64-pc-windows-msvc with cargo-xwin and runs their
        # test binaries under Wine:
        #
        #   cargo xwin build --target x86_64-pc-windows-msvc
        #   cargo xwin test  --target x86_64-pc-windows-msvc
        #
        # cargo-xwin downloads the MSVC CRT and Windows SDK once (into
        # ~/.cache/cargo-xwin) via the xwin tool; entering this shell
        # accepts the Microsoft license for those downloads
        # (XWIN_ACCEPT_LICENSE below). The MLIR-linking crates are out of
        # scope here — they would additionally need the conda-forge MSVC
        # LLVM/MLIR toolchain the Windows CI uses.
        # ---------------------------------------------------------------------------
        devShells.windows-cross = pkgs.mkShell {
          name = "reussir-windows-cross";

          # Same rationale as the default shell: cargo builds build scripts
          # at -O0, and the nix cc wrapper's fortify defines turn the glibc
          # "#warning _FORTIFY_SOURCE requires optimization" into a -Werror
          # failure in the tblgen crate's host C++.
          hardeningDisable = [ "fortify" "fortify3" ];

          packages = [
            windowsRustToolchain
            pkgs.cargo-xwin

            # lld-link links the MSVC-target binaries; llvm-lib/llvm-rc back
            # the cc crate's archiver/resource steps. clang-unwrapped supplies
            # clang-cl for the C sources the tree-sitter grammar crates build —
            # and the GNU-driver clang/clang++ that
            # cmake/toolchains/linux-to-windows-msvc.cmake compiles the C++
            # backend with — deliberately unwrapped: the nix cc wrapper would
            # inject host (glibc) include and link paths into a Windows cross
            # compile.
            llvmPkgs.lld
            llvmPkgs.llvm
            llvmPkgs.clang-unwrapped

            # Drive the C++ backend cross build (see the toolchain file
            # above); python backs the wine llvm-config wrapper that
            # cmake/FindLLVM.cmake stages for llvm-sys/mlir-sys, and lit
            # (with psutil for --timeout) runs the integration suite under
            # Wine via the `check` target. util-linux supplies the flock
            # that cmake/ReussirCargo.cmake serializes cargo-xwin with.
            pkgs.cmake
            pkgs.ninja
            pkgs.util-linux
            (pkgs.python3.withPackages (ps: [ ps.lit ps.psutil ]))
            # Host tblgen for the dialect's .td generation under cross.
            llvmPkgs.tblgen

            # 64-bit-only Wine is enough: the msvc target produces x64
            # binaries, and the wow64 build runs them without multilib.
            pkgs.wineWow64Packages.minimal

            # One-shot fetch of the conda-forge MSVC LLVM/MLIR toolchain
            # (same versions as the Windows CI) for backend-crate linking.
            bakeMsvcLlvm
            # One-shot fetch of the windows-native rustc/cargo the reussir
            # compilers spawn under Wine (polyffi, rene's bake).
            bakeMsvcRustc
          ];

          # Host-side link deps for the backend cross build: the tblgen and
          # melior-macro proc-macros statically link the HOST LLVM TableGen
          # C++, whose system-libs pull in -lz and -lxml2 at .so link time.
          buildInputs = [
            pkgs.zlib
            pkgs.libxml2
          ];

          # The user of this shell accepts the Microsoft Software License for
          # the MSVC CRT/SDK components xwin fetches.
          XWIN_ACCEPT_LICENSE = "1";

          # `cargo xwin test` (and `cargo run --target …`) execute the
          # produced .exe files through this runner.
          CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER = "wine";

          # Host prefixes for the backend cross build: proc-macros (tblgen,
          # melior-macro's TableGen link) execute on the HOST, so they get the
          # host LLVM join — the MSVC target prefixes (MLIR_SYS_*/LLVM_SYS_*)
          # are staged per-build by cmake/FindLLVM.cmake's wine wrapper.
          TABLEGEN_230_PREFIX = "${llvmMlirJoin}";
          MLIR_TABLEGEN_EXE_OVERRIDE = "${llvmPkgs.tblgen}/bin/mlir-tblgen";
          LIBCLANG_PATH = "${llvmPkgs.libclang.lib}/lib";

          shellHook = ''
            # Host dylibs for proc-macro dlopen (libstdc++, LLVM/MLIR, zlib):
            # same rationale as the default shell.
            export LD_LIBRARY_PATH="${llvmPkgs.llvm.lib}/lib:${llvmPkgs.mlir}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            # Keep the Wine prefix project-local so first-run initialization
            # never touches ~/.wine.
            export WINEPREFIX="''${XDG_CACHE_HOME:-$HOME/.cache}/reussir-wineprefix"
            # Wine is chatty on stderr (fixme:...) — silence it so cargo test
            # output stays readable.
            export WINEDEBUG="-all"

            # MSVC LLVM/MLIR (conda-forge, as in the Windows CI): advertise
            # the baked prefix when present so backend-crate cross builds can
            # point their *_PREFIX discovery at it.
            msvc_llvm="''${XDG_CACHE_HOME:-$HOME/.cache}/reussir-msvc-conda/Library"
            if [ -e "$msvc_llvm/lib/cmake/mlir/MLIRConfig.cmake" ]; then
              export REUSSIR_MSVC_LLVM_PREFIX="$msvc_llvm"
              # Wine executions of cross-built binaries resolve the conda
              # runtime DLLs (zlib/zstd/libxml2 behind the static LLVM)
              # through the Windows-side PATH extension.
              export WINEPATH="z:''${msvc_llvm//\//\\}\\bin"
            fi

            # The windows-native Rust toolchain (reussir-bake-msvc-rustc):
            # the full spawn-PE-children environment for polyffi and rene
            # under Wine. cc-rs build scripts get clang-cl (conda) with the
            # xwin CRT/SDK through INCLUDE/LIB; cargo links through
            # rust-lld; the shared unix CARGO_HOME keeps the host-side
            # `cargo fetch` cache visible to wine-cargo.
            msvc_rustc="''${XDG_CACHE_HOME:-$HOME/.cache}/reussir-msvc-rustc"
            xwin_splat="''${XDG_CACHE_HOME:-$HOME/.cache}/cargo-xwin/xwin"
            if [ -x "$msvc_rustc/bin/rustc.exe" ]; then
              export REUSSIR_MSVC_RUSTC="$msvc_rustc/bin/rustc.exe"
              export REUSSIR_MSVC_RUSTC_PREFIX="$msvc_rustc"
              export WINEPATH="''${WINEPATH:+$WINEPATH;}z:''${msvc_rustc//\//\\}\\bin"
              export CARGO_HOME="''${CARGO_HOME:-$HOME/.cargo}"
              export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=rust-lld
              export CC_x86_64_pc_windows_msvc=clang-cl
              export CXX_x86_64_pc_windows_msvc=clang-cl
              export AR_x86_64_pc_windows_msvc=llvm-lib
              if [ -d "$xwin_splat/crt/include" ]; then
                _w() { echo "z:''${1//\//\\}"; }
                export LIB="$(_w "$xwin_splat/crt/lib/x86_64");$(_w "$xwin_splat/sdk/lib/um/x86_64");$(_w "$xwin_splat/sdk/lib/ucrt/x86_64")"
                export INCLUDE="$(_w "$xwin_splat/crt/include");$(_w "$xwin_splat/sdk/include/ucrt");$(_w "$xwin_splat/sdk/include/um");$(_w "$xwin_splat/sdk/include/shared")"
                unset -f _w
              fi
            fi

            echo ""
            echo "  Reussir Windows cross shell — cargo-xwin + Wine"
            echo ""
            echo "    cargo xwin build --target x86_64-pc-windows-msvc   # default members"
            echo "    cargo xwin test  --target x86_64-pc-windows-msvc  # tests run under wine"
            echo "    reussir-bake-msvc-llvm   # fetch the conda-forge MSVC LLVM/MLIR (once)"
            echo "    reussir-bake-msvc-rustc  # fetch the windows-native rustc/cargo (once)"
            echo "    # then build the Wine runtime tree for the polyffi/rene suites:"
            echo "    #   cargo fetch --locked && CARGO_TARGET_DIR=\$PWD/build-xwin/target-rt-wine \\"
            echo "    #     CARGO_NET_OFFLINE=true wine \$REUSSIR_MSVC_RUSTC_PREFIX/bin/cargo.exe \\"
            echo "    #     build --offline --locked --release -p reussir-rt"
            echo ""
          '';
        };

        # ---------------------------------------------------------------------------
        # Lean CI shells. The pure-Rust pipelines don't need the LLVM/MLIR
        # closure the default shell drags in, and CI startup time is dominated
        # by store downloads — so give them exactly what they use.
        # ---------------------------------------------------------------------------

        # rene tests and the reussir-rt Miri pipeline: the pinned toolchain
        # (rust-toolchain.toml supplies miri and the wasm stds) plus the
        # stdenv cc that build scripts and test links need.
        devShells.rust = pkgs.mkShell {
          name = "reussir-rust";
          packages = [ rustToolchain ];
        };

        # VS Code extension pipeline: the wasm codec build, the TypeScript
        # bundling, and the VSIX packaging. wasm-bindgen-cli from nixpkgs is
        # the same version-lockstep pin the default shell documents;
        # scripts/build-wasm.mjs still verifies it against Cargo.lock.
        devShells.vscode = pkgs.mkShell {
          name = "reussir-vscode";
          packages = [
            rustToolchain
            pkgs.nodejs_24
            pkgs.wasm-bindgen-cli
          ];
        };
      }
    );
}
