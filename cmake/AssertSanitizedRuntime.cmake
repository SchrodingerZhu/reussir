# Fail the build if a "sanitized" runtime was not actually instrumented.
#
# The behavioural guard (tests/integration/sanitizer/runtime-allocator-asan.rr)
# checks that the *right* runtime is loaded and that its allocator is visible
# to the sanitizers. It cannot see this failure: `--no-default-features` is a
# cargo argument rather than a rustflag, so a build that loses `-Zsanitizer` —
# an ambient `RUSTFLAGS` replaces the target-specific one, say — still routes
# allocation through the platform allocator, the test program's own
# instrumentation still reports a planted use-after-free, and the suite passes
# with an uninstrumented runtime. Verified on Windows: without the unsets in
# crates/reussir-rt/CMakeLists.txt the DSO imports no sanitizer runtime and
# that test passes anyway.
#
# So assert the cause rather than one of its symptoms, at the point of the
# mistake instead of whenever a test happens to notice.
#
# Leak is deliberately absent from the map: LeakSanitizer is interceptor-based
# rather than compile-time instrumentation, so a correct LSan build has no
# `__lsan_` references and asserting one would make this check falsely red.
#
# Expects: LIBRARY, SANITIZER, READOBJ, NM, IS_WIN32.

set(_marker_address "asan")
set(_marker_memory "msan")
set(_marker_thread "tsan")
if(NOT DEFINED _marker_${SANITIZER})
    message(STATUS "no instrumentation marker for '${SANITIZER}'; nothing to assert")
    return()
endif()
set(_marker "${_marker_${SANITIZER}}")

if(IS_WIN32)
    # PE: instrumented code imports the sanitizer's own runtime DLL.
    execute_process(COMMAND "${READOBJ}" --coff-imports "${LIBRARY}"
        OUTPUT_VARIABLE _dump RESULT_VARIABLE _result ERROR_VARIABLE _err)
    set(_expected "an import of clang_rt.${_marker}")
    string(FIND "${_dump}" "clang_rt.${_marker}" _found)
else()
    # ELF/Mach-O: instrumented code carries undefined `__asan_*` references.
    execute_process(COMMAND "${NM}" --undefined-only "${LIBRARY}"
        OUTPUT_VARIABLE _dump RESULT_VARIABLE _result ERROR_VARIABLE _err)
    set(_expected "an undefined __${_marker}_ reference")
    string(FIND "${_dump}" "__${_marker}_" _found)
endif()

if(NOT _result EQUAL 0)
    message(FATAL_ERROR "cannot inspect ${LIBRARY}: ${_err}")
endif()

if(_found EQUAL -1)
    message(FATAL_ERROR
        "${LIBRARY} was built as the ${SANITIZER} runtime variant but carries no "
        "${SANITIZER} instrumentation (expected ${_expected}).\n"
        "The usual cause is rustc never receiving `-Zsanitizer=${SANITIZER}`: cargo "
        "gives a global RUSTFLAGS precedence over the target-specific form, so an "
        "ambient value replaces these flags instead of adding to them. A runtime in "
        "this state still passes the sanitizer suite while sanitizing nothing.")
endif()
