import lit.formats
import os
import sys

config.name = 'Reussir'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.mlir', '.rr', '.repl']
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
for _var, _value in os.environ.items():
    if _var.startswith('NIX_') or _var in ('LIBRARY_PATH', 'LD_LIBRARY_PATH'):
        config.environment[_var] = _value

def sh_path(path):
    return path.replace('\\', '/') if isinstance(path, str) else path

def append_flags(command, flags):
    return ' '.join(
        part for part in (sh_path(command), sh_path(flags)) if part
    )

def bin_tool(name):
    suffix = '.exe' if sys.platform == 'win32' else ''
    return sh_path(os.path.join(config.binary_path, name + suffix))

config.substitutions.append((r'%reussir-opt',
                             bin_tool('reussir-opt')))
config.substitutions.append((r'%reussir-translate',
                             bin_tool('reussir-translate')))

# Add C compiler substitution using CMake's C compiler
config.substitutions.append((r'%cc', append_flags(config.cc_path, config.cc_runtime_flags)))

config.substitutions.append((r'%FileCheck', sh_path(config.filecheck_path)))
linkage_check_prefixes = (
    '--check-prefixes=CHECK,CHECK-COFF'
    if sys.platform == 'win32'
    else '--check-prefixes=CHECK,CHECK-DEDUP'
)
config.substitutions.append((r'%linkage_check_prefixes', linkage_check_prefixes))
config.substitutions.append((r'%not', sh_path(config.not_path)))
config.substitutions.append((r'%opt', sh_path(config.opt_path)))
config.substitutions.append((r'%library_path', sh_path(config.library_path)))
# The rustc the build provisioned for polyffi texture compiles — the same
# value REUSSIR_RUSTC is set to; exposed so tests can exercise the explicit
# --polyffi-rust-path/--polyffi-libdir flags without the environment.
config.substitutions.append((r'%rustc_path', sh_path(config.environment['REUSSIR_RUSTC'])))
config.substitutions.append((r'%llc', sh_path(config.llc_path)))
config.substitutions.append((r'%extra_sys_libs', sh_path(config.extra_sys_libs)))
config.substitutions.append((r'%lli', sh_path(config.lli_path)))
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
    with tempfile.TemporaryDirectory() as tmp:
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
config.substitutions.append((r'%rpath_flag', sh_path(config.rpath_flag)))
config.substitutions.append((r'%rrc', sh_path(config.reussir_rrc_path)))
# The Rust REPL; its suite lives in repl-rs/ and only runs when the `rrepl`
# CMake target has been built.
config.substitutions.append((r'%rrepl', sh_path(config.rrepl_path)))
if config.rrepl_path and os.path.exists(config.rrepl_path):
    config.available_features.add('rrepl')
config.substitutions.append((r'%reussir-syntax', sh_path(config.reussir_syntax_path)))
config.substitutions.append((r'%asan_flags', config.asan_flags))
config.substitutions.append((r'%lsan_flags', config.lsan_flags))
config.substitutions.append((r'%msan_flags', config.msan_flags))
config.substitutions.append((r'%tsan_flags', config.tsan_flags))
config.substitutions.append((r'%asan_env', config.asan_env))
config.substitutions.append((r'%lsan_env', config.lsan_env))
config.substitutions.append((r'%msan_env', config.msan_env))
config.substitutions.append((r'%tsan_env', config.tsan_env))
config.substitutions.append((r'%rpath_san_flag', sh_path(config.rpath_san_flag)))
config.substitutions.append((r'%reussir_rt_asan', sh_path(config.reussir_rt_asan_path)))
config.substitutions.append((r'%reussir_rt_lsan', sh_path(config.reussir_rt_lsan_path)))
config.substitutions.append((r'%reussir_rt_msan', sh_path(config.reussir_rt_msan_path)))
config.substitutions.append((r'%reussir_rt_tsan', sh_path(config.reussir_rt_tsan_path)))

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
