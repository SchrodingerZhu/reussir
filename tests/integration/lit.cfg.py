import lit.formats
import os
import sys

config.name = 'Reussir'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.mlir', '.rr', '.repl']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_output_root, 'test')

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
config.substitutions.append((r'%llc', sh_path(config.llc_path)))
config.substitutions.append((r'%extra_sys_libs', sh_path(config.extra_sys_libs)))
config.substitutions.append((r'%lli', sh_path(config.lli_path)))
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
