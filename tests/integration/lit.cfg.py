import lit.formats
import os
import sys

config.name = 'Reussir'
# lit 23 (LLVM 23) hard-errors on execute_external=True unless the migration
# escape hatch force_execute_external is set; lit <= 22 does not know that
# kwarg at all. Keep the external shell on both until the suites migrate to
# lit's internal shell.
try:
    config.test_format = lit.formats.ShTest(True, force_execute_external=True)
except TypeError:
    config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.mlir', '.rr', '.repl', '.ll']
# `Inputs/` directories hold companion files (transform scripts, C drivers)
# referenced by tests via %S; they are not tests themselves.
config.excludes = ['Inputs']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_output_root, 'test')

# Toolchain wrappers that inject flags through the environment (e.g. Nix's
# clang wrapper reading NIX_CFLAGS_COMPILE/NIX_LDFLAGS, which are only honored
# together with the wrapper's NIX_CC_WRAPPER_* role markers) need these
# variables to survive lit's environment scrubbing. Without them, `%cc
# -fopenmp` links fine in the probe below (a direct subprocess of this
# config, full environment) but fails inside RUN lines with `cannot find
# -lomp`.
# Windows-target tools driven from a linux host (the xwin cross build, run
# through Wine/binfmt): a feature for the few tests that straddle the
# unix/windows process boundary in ways Wine cannot bridge.
# REUSSIR_CROSS_WINE_UNGATE=1 suppresses the gate for experiments with a
# windows-native rustc under Wine; REUSSIR_RUSTC_OVERRIDE and
# REUSSIR_LIBRARY_PATH_OVERRIDE redirect %rustc_path / %library_path (and
# REUSSIR_RUSTC) without reconfiguring.
if config.reussir_rrc_path.endswith('.exe') and sys.platform != 'win32':
    if not os.environ.get('REUSSIR_CROSS_WINE_UNGATE'):
        config.available_features.add('cross-wine')
if os.environ.get('REUSSIR_RUSTC_OVERRIDE'):
    config.environment['REUSSIR_RUSTC'] = os.environ['REUSSIR_RUSTC_OVERRIDE']
if os.environ.get('REUSSIR_LIBRARY_PATH_OVERRIDE'):
    config.library_path = os.environ['REUSSIR_LIBRARY_PATH_OVERRIDE']
    # rrc discovers the polyffi libdir through this env when the RUN line
    # passes no explicit --polyffi-libdir.
    config.environment['REUSSIR_RUSTC_DEPS'] = os.environ['REUSSIR_LIBRARY_PATH_OVERRIDE']

# WINEPATH rides along for the Windows cross runs: binfmt-launched PE
# binaries resolve the conda runtime DLLs through it.
for _var, _value in os.environ.items():
    if _var.startswith('NIX_') or _var in ('LIBRARY_PATH', 'LD_LIBRARY_PATH',
                                           'WINEPATH'):
        config.environment[_var] = _value

# The runtime dylib links libstd dynamically, and nothing stages libstd next
# to it anymore: put the cargo target directory and the toolchain's own
# libdir on the platform's dynamic-loader path, so linked test executables
# (and the JIT loading the runtime dylib) resolve both libreussir_rt and its
# libstd. An executable rpath cannot do this on Linux — DT_RUNPATH does not
# apply to the dylib's own NEEDED entries.
_loader_var = {'win32': 'PATH', 'darwin': 'DYLD_LIBRARY_PATH'}.get(
    sys.platform, 'LD_LIBRARY_PATH')
_loader_dirs = [d for d in (config.library_path, config.rust_libdir) if d]
_loader_prev = config.environment.get(_loader_var, os.environ.get(_loader_var, ''))
if _loader_prev:
    _loader_dirs.append(_loader_prev)
config.environment[_loader_var] = os.pathsep.join(_loader_dirs)

def sh_path(path):
    return path.replace('\\', '/') if isinstance(path, str) else path

def append_flags(command, flags):
    return ' '.join(
        part for part in (sh_path(command), sh_path(flags)) if part
    )

def bin_tool(name):
    suffix = '.exe' if sys.platform == 'win32' else ''
    path = os.path.join(config.binary_path, name + suffix)
    # Cross runs (windows target, linux host): the staged tool is name.exe
    # even though the host shell is unix — probe instead of assuming. The
    # native Windows pipeline never noticed because CreateProcess appends
    # .exe on its own; a unix shell does not.
    if not os.path.exists(path) and os.path.exists(path + '.exe'):
        path += '.exe'
    return sh_path(path)

config.substitutions.append((r'%reussir-opt',
                             bin_tool('reussir-opt')))
config.substitutions.append((r'%reussir-translate',
                             bin_tool('reussir-translate')))
config.substitutions.append((r'%reussir-llvm-opt',
                             bin_tool('reussir-llvm-opt')))

# Add C compiler substitution using CMake's C compiler
config.substitutions.append((r'%cc', append_flags(config.cc_path, config.cc_runtime_flags)))

config.substitutions.append((r'%FileCheck', sh_path(config.filecheck_path)))
# The std integration executables are targets of one rene package. All of
# their tests deliberately share one build directory; rene's database lock
# serializes concurrent writers while preserving the compiled target cache.
config.substitutions.append((
    r'%std_rene_build',
    sh_path(os.path.join(config.test_exec_root, 'std-rene-build')),
))
linkage_check_prefixes = (
    '--check-prefixes=CHECK,CHECK-COFF'
    if sys.platform == 'win32'
    else '--check-prefixes=CHECK,CHECK-DEDUP'
)
config.substitutions.append((r'%linkage_check_prefixes', linkage_check_prefixes))
config.substitutions.append((r'%not', sh_path(config.not_path)))
config.substitutions.append((r'%opt', sh_path(config.opt_path)))
config.substitutions.append((r'%library_path', sh_path(config.library_path)))
# The developer toolchain's rustc, as resolved at configure time — the same
# value REUSSIR_RUSTC is set to; exposed so tests can exercise the explicit
# --polyffi-rust-path/--polyffi-libdir flags without the environment.
config.substitutions.append((r'%rustc_path', sh_path(config.environment['REUSSIR_RUSTC'])))
config.substitutions.append((r'%llc', sh_path(config.llc_path)))
config.substitutions.append((r'%extra_sys_libs', sh_path(config.extra_sys_libs)))
config.substitutions.append((r'%lli', sh_path(config.lli_path)))
config.substitutions.append((r'%llvm-ar', sh_path(config.llvm_ar_path)))
config.substitutions.append((r'%llvm-nm', sh_path(config.llvm_nm_path)))
config.substitutions.append((r'%llvm-bcanalyzer', sh_path(config.llvm_bcanalyzer_path)))
config.substitutions.append((r'%llvm-dis', sh_path(config.llvm_dis_path)))
config.substitutions.append((r'%llvm-readobj', sh_path(config.llvm_readobj_path)))
# The debug-info rescan suite (debuginfo/): %lldb is a batch, init-free lldb
# with the Reussir data formatters (tool/lldb/reussir_formatters.py)
# preloaded, so tests can drive a compiled executable and FileCheck the
# debugger's variable printing. Gated on the `lldb` feature: configured-in
# and present, or the tests are unsupported.
if config.lldb_path and os.path.exists(config.lldb_path):
    config.available_features.add('lldb')
    config.substitutions.append((
        r'%lldb',
        '%s --batch --no-lldbinit -O "command script import %s"'
        % (sh_path(config.lldb_path), sh_path(config.lldb_formatters_path))))
# gdb decodes the variant DWARF natively — its tests pin the emitted debug
# info with no formatter script in the loop. `-nx` keeps user init files out.
if config.gdb_path and os.path.exists(config.gdb_path):
    config.available_features.add('gdb')
    config.substitutions.append((r'%gdb',
                                 '%s --batch -nx' % sh_path(config.gdb_path)))

# WebAssembly (the rene/wasi_*.rr suites): packages cross-built for a WASI
# target and run on a real engine. Two independent things have to be
# present, so they are separate features.
#
# One per target — the rustc this build compiles against ships that
# target's standard library. `rene` bakes `reussir-rt` for it with cargo,
# `rrc` compiles every polyffi texture for it, and the launcher link is
# another rustc invocation: all three need `rustup target add <target>`.
# `--print target-libdir` renders the path whether or not it is populated,
# so the rlibs are what decide.
def _has_rust_std(target):
    import subprocess
    rustc = config.environment.get('REUSSIR_RUSTC')
    if not rustc:
        return False
    try:
        printed = subprocess.run(
            [rustc, '--print', 'target-libdir', '--target', target],
            capture_output=True, text=True, timeout=60)
    except Exception:
        return False
    libdir = printed.stdout.strip()
    if printed.returncode != 0 or not libdir or not os.path.isdir(libdir):
        return False
    return any(name.startswith('libstd-') for name in os.listdir(libdir))

# The feature is the target's short name: `wasip1` for the baseline WASI
# target, `wasip1-threads` for the one whose programs get real threads.
for _target in ('wasm32-wasip1', 'wasm32-wasip1-threads'):
    if _has_rust_std(_target):
        config.available_features.add(_target.removeprefix('wasm32-'))

# `wasmer` — the engine that runs the module (tests/integration/CMakeLists.txt
# locates it; `-DREUSSIR_WASMER=` pins one). The threads proposal is on by
# default there, so `run` needs no further flags.
if config.wasmer_path and os.path.exists(config.wasmer_path):
    config.available_features.add('wasmer')
    config.substitutions.append((r'%wasmer',
                                 '%s run' % sh_path(config.wasmer_path)))

# OpenMP: multithreaded e2e drivers (the atomic-rc suite) need `-fopenmp` and
# a runtime rpath to wherever the toolchain keeps libomp. Probe by linking a
# trivial parallel program; on failure the tests are unsupported rather than
# broken.
def _probe_openmp():
    import subprocess
    import tempfile
    if not config.cc_path:
        return None
    libomp_dir = ''
    try:
        libomp = subprocess.run(
            [config.cc_path, '-print-file-name=libomp.so'],
            capture_output=True, text=True, timeout=30).stdout.strip()
        if os.path.isabs(libomp):
            libomp_dir = os.path.dirname(libomp)
    except Exception:
        pass
    flags = '-fopenmp'
    if libomp_dir:
        flags += ' -Wl,-rpath,%s' % libomp_dir
    # Windows can hold the probe executable's lock a beat past process
    # exit (observed under x64 emulation); a failed cleanup is not a
    # failed probe.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, 'omp.c')
        exe = os.path.join(tmp, 'omp')
        with open(src, 'w') as f:
            f.write('#include <omp.h>\n'
                    'int main(void) { int n = 0;\n'
                    '#pragma omp parallel num_threads(2) reduction(+ : n)\n'
                    '  n += 1;\n'
                    '  return n == 2 ? 0 : 1; }\n')
        try:
            if subprocess.run([config.cc_path, *flags.split(), src, '-o', exe],
                              capture_output=True, timeout=60).returncode != 0:
                return None
            if subprocess.run([exe], capture_output=True,
                              timeout=60).returncode != 0:
                return None
        except Exception:
            return None
    return flags

_openmp_flags = _probe_openmp()
if _openmp_flags:
    config.available_features.add('openmp')
    config.substitutions.append((r'%openmp_flags', _openmp_flags))

# LTO linking: `rrc --emit staticlib --lto ...` archives bitcode, which only a
# linker carrying the LTO plugin can consume (lld and ld64 do; a plain ld.bfd
# without LLVMgold does not). Probe by linking a trivial program through each
# candidate flag set; the first that works becomes %lto_link_flags, and
# failing them all leaves the `lto-link` feature off so the LTO end-to-end
# test is unsupported rather than broken. The archive-packing checks in
# staticlib_lto.rr do not link, and run either way.
def _probe_lto_link():
    import subprocess
    import tempfile
    if not config.cc_path:
        return None
    for flags in ('-flto=thin -fuse-ld=lld', '-flto=thin', '-flto'):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'lto.c')
            exe = os.path.join(tmp, 'lto')
            with open(src, 'w') as f:
                f.write('int main(void) { return 0; }\n')
            try:
                if subprocess.run([config.cc_path, *flags.split(), src, '-o', exe],
                                  capture_output=True, timeout=120).returncode != 0:
                    continue
                if subprocess.run([exe], capture_output=True,
                                  timeout=60).returncode != 0:
                    continue
            except Exception:
                continue
            return flags
    return None

_lto_link_flags = _probe_lto_link()
if _lto_link_flags:
    config.available_features.add('lto-link')
    config.substitutions.append((r'%lto_link_flags', _lto_link_flags))
config.substitutions.append((r'%rpath_flag', sh_path(config.rpath_flag)))
# The MSVC linker `rrc`'s link step should drive, as a driver flag, resolved
# at configure time (see tests/integration/CMakeLists.txt): left to its own
# discovery under a vcvars environment, rustc degrades to a bare PATH scan
# for `link.exe` and finds Git-for-Windows' coreutils /usr/bin/link instead.
# The path contains spaces (Program Files), so the substitution carries its
# own quoting. Empty — no vcvars at configure, or not Windows — means rustc's
# registry-based discovery, which is right on a plain dev box.
#
# Appended before `%rrc`, which is its prefix: lit applies substitutions in
# list order, and the later `%rrc` would otherwise rewrite this one's name.
_rrc_linker = ''
if sys.platform == 'win32' and config.msvc_linker_path:
    _rrc_linker = '--linker "%s"' % sh_path(config.msvc_linker_path)
config.substitutions.append((r'%rrc_linker', _rrc_linker))
config.substitutions.append((r'%rrc', sh_path(config.reussir_rrc_path)))
# The package manager. It shells out to `rrc` for the source-graph scan, so
# point it at the staged driver rather than whatever `rrc` a PATH lookup finds.
config.substitutions.append((r'%rene', sh_path(config.rene_path)))
config.environment['REUSSIR_RRC'] = sh_path(config.reussir_rrc_path)
# The Rust REPL; its suite lives in repl-rs/ and only runs when the `rrepl`
# CMake target has been built.
config.substitutions.append((r'%rrepl', sh_path(config.rrepl_path)))
if config.rrepl_path and os.path.exists(config.rrepl_path):
    config.available_features.add('rrepl')
config.substitutions.append((r'%reussir-syntax', sh_path(config.reussir_syntax_path)))
config.substitutions.append((r'%reussir-lsp', sh_path(config.reussir_lsp_path)))
config.substitutions.append((r'%python', sh_path(sys.executable)))
config.substitutions.append((r'%asan_flags', config.asan_flags))
config.substitutions.append((r'%lsan_flags', config.lsan_flags))
config.substitutions.append((r'%msan_flags', config.msan_flags))
config.substitutions.append((r'%tsan_flags', config.tsan_flags))
# Windows resolves `reussir_rt.dll` by search order, and the sanitized runtime
# carries the same name as the ordinary one (renaming it would invalidate the
# import library the tests link against). Putting the sanitized directory on
# the loader path globally would hand it to *every* test; putting it in the
# per-case environment prefix hands it only to the sanitizer RUN lines, which
# is what the distinction requires.
_asan_env = config.asan_env
if sys.platform == 'win32' and getattr(config, 'asan_runtime_dir', ''):
    _asan_env = 'PATH="%s;$PATH" %s' % (
        config.asan_runtime_dir.replace('/', '\\'), config.asan_env)
config.substitutions.append((r'%asan_env', _asan_env))
config.substitutions.append((r'%lsan_env', config.lsan_env))
config.substitutions.append((r'%msan_env', config.msan_env))
config.substitutions.append((r'%tsan_env', config.tsan_env))
config.substitutions.append((r'%rpath_san_flag', sh_path(config.rpath_san_flag)))
config.substitutions.append((r'%reussir_rt_asan', sh_path(config.reussir_rt_asan_path)))
config.substitutions.append((r'%reussir_rt_lsan', sh_path(config.reussir_rt_lsan_path)))
config.substitutions.append((r'%reussir_rt_msan', sh_path(config.reussir_rt_msan_path)))
config.substitutions.append((r'%reussir_rt_tsan', sh_path(config.reussir_rt_tsan_path)))

# link.exe itself resolves the CRT and system import libraries through the
# vcvars LIB variable; carry it (and its companions) across lit's
# environment scrubbing.
if sys.platform == 'win32':
    for _var in ('VCINSTALLDIR', 'VSINSTALLDIR', 'VCToolsInstallDir',
                 'VSCMD_ARG_HOST_ARCH', 'VSCMD_ARG_TGT_ARCH',
                 'WindowsSdkDir', 'WindowsSDKVersion', 'WindowsSdkBinPath',
                 'UniversalCRTSdkDir', 'UCRTVersion',
                 'LIB', 'INCLUDE', 'LIBPATH'):
        if _var in os.environ:
            config.environment[_var] = os.environ[_var]

# `rrc --emit executable --lto …` needs the linker *rustc* drives to consume
# this LLVM's bitcode: rust-lld does on the targets where it is rustc's
# default, but a system linker without a matching-version LTO plugin (an old
# ld64, MSVC's link.exe, aarch64-linux's bfd) fails on the first member.
# Probe with the real pipeline; failing leaves `rustc-lto-link` off and the
# LTO link tests unsupported rather than broken.
def _probe_rustc_lto_link():
    import subprocess
    import tempfile
    rrc = sh_path(config.reussir_rrc_path)
    if not rrc or not os.path.exists(rrc):
        return False
    env = dict(os.environ)
    for var in ('REUSSIR_RUSTC', 'REUSSIR_RUSTC_DEPS'):
        if var in config.environment:
            env[var] = config.environment[var]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, 'probe.rr')
        exe = os.path.join(tmp, 'probe.exe')
        with open(src, 'w') as f:
            f.write('fn twice(n: u64) -> u64 { n + n }\n'
                    '#[main]\n'
                    'pub fn entry() {\n'
                    '    let doubled = twice(21);\n'
                    '}\n')
        cmd = [rrc, src, '-o', exe, '--emit', 'executable',
               '--relocation-mode', 'pic', '--lto', 'thin']
        if sys.platform == 'win32' and config.msvc_linker_path:
            cmd += ['--linker', config.msvc_linker_path]
        try:
            if subprocess.run(cmd, env=env, capture_output=True,
                              timeout=300).returncode != 0:
                return False
            if subprocess.run([exe], capture_output=True,
                              timeout=60).returncode != 0:
                return False
        except Exception:
            return False
    return True

if _probe_rustc_lto_link():
    config.available_features.add('rustc-lto-link')

# Executables' platform suffix, for running artifacts whose name a tool
# derived (`rene build`'s `<profile>/<target>`), not a RUN line's `-o`.
config.substitutions.append((r'%exe_ext', '.exe' if sys.platform == 'win32' else ''))

# TODO: should we support macos?
if sys.platform == 'win32':
    config.available_features.add('windows')
    config.substitutions.append((r'%reussir_rt', 'reussir_rt.dll'))
elif sys.platform == 'darwin':
    config.available_features.add('darwin')
    config.substitutions.append((r'%reussir_rt', 'libreussir_rt.dylib'))
else:
    if sys.platform.startswith('linux'):
        config.available_features.add('linux')
    config.substitutions.append((r'%reussir_rt', 'libreussir_rt.so'))

if config.reussir_rt_asan_path and os.path.exists(config.reussir_rt_asan_path):
    config.available_features.add('asan')

if config.reussir_rt_lsan_path and os.path.exists(config.reussir_rt_lsan_path):
    config.available_features.add('lsan')

if config.reussir_rt_msan_path and os.path.exists(config.reussir_rt_msan_path):
    config.available_features.add('msan')

if config.reussir_rt_tsan_path and os.path.exists(config.reussir_rt_tsan_path):
    config.available_features.add('tsan')
