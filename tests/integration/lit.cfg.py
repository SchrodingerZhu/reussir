import lit.formats
import os
import sys

config.name = 'Reussir'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.mlir', '.rr', '.repl']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_output_root, 'test')

config.substitutions.append((r'%reussir-opt',
                             os.path.join(config.binary_path, 'reussir-opt')))
config.substitutions.append((r'%reussir-translate',
                             os.path.join(config.binary_path, 'reussir-translate')))

# Add C compiler substitution using CMake's C compiler
config.substitutions.append((r'%cc', config.cc_path))

config.substitutions.append((r'%FileCheck', config.filecheck_path))
config.substitutions.append((r'%not', config.not_path))
config.substitutions.append((r'%opt', config.opt_path))
config.substitutions.append((r'%library_path', config.library_path))
config.substitutions.append((r'%llc', config.llc_path))
config.substitutions.append((r'%extra_sys_libs', config.extra_sys_libs))
config.substitutions.append((r'%lli', config.lli_path))
config.substitutions.append((r'%rpath_flag', config.rpath_flag))
config.substitutions.append((r'%rrc', config.reussir_rrc_path))
# The Rust REPL; its suite lives in repl-rs/ and only runs when the `rrepl`
# CMake target has been built.
config.substitutions.append((r'%rrepl', config.rrepl_path))
if config.rrepl_path and os.path.exists(config.rrepl_path):
    config.available_features.add('rrepl')
config.substitutions.append((r'%reussir-syntax', config.reussir_syntax_path))
config.substitutions.append((r'%asan_flags', config.asan_flags))
config.substitutions.append((r'%lsan_flags', config.lsan_flags))
config.substitutions.append((r'%msan_flags', config.msan_flags))
config.substitutions.append((r'%tsan_flags', config.tsan_flags))
config.substitutions.append((r'%asan_env', config.asan_env))
config.substitutions.append((r'%lsan_env', config.lsan_env))
config.substitutions.append((r'%msan_env', config.msan_env))
config.substitutions.append((r'%tsan_env', config.tsan_env))
config.substitutions.append((r'%rpath_san_flag', config.rpath_san_flag))
config.substitutions.append((r'%reussir_rt_asan', config.reussir_rt_asan_path))
config.substitutions.append((r'%reussir_rt_lsan', config.reussir_rt_lsan_path))
config.substitutions.append((r'%reussir_rt_msan', config.reussir_rt_msan_path))
config.substitutions.append((r'%reussir_rt_tsan', config.reussir_rt_tsan_path))

# TODO: should we support macos?
if sys.platform == 'windows':
    config.substitutions.append((r'%reussir_rt', 'reussir_rt.dll'))
elif sys.platform == 'darwin':
    config.substitutions.append((r'%reussir_rt', 'libreussir_rt.dylib'))
else:
    config.substitutions.append((r'%reussir_rt', 'libreussir_rt.so'))

if config.reussir_rt_asan_path and os.path.exists(config.reussir_rt_asan_path):
    config.available_features.add('asan')

if config.reussir_rt_lsan_path and os.path.exists(config.reussir_rt_lsan_path):
    config.available_features.add('lsan')

if config.reussir_rt_msan_path and os.path.exists(config.reussir_rt_msan_path):
    config.available_features.add('msan')

if config.reussir_rt_tsan_path and os.path.exists(config.reussir_rt_tsan_path):
    config.available_features.add('tsan')
