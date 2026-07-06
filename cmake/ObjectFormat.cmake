# ObjectFormat.cmake
# Detects the object file format for the target platform

# Function to detect object format
function(detect_object_format OUTPUT_VAR)
    # Initialize the result
    set(OBJECT_FORMAT "UNKNOWN")
    
    # Check LLVM's target triple if available
    if(DEFINED LLVM_DEFAULT_TARGET_TRIPLE)
        set(TARGET_TRIPLE ${LLVM_DEFAULT_TARGET_TRIPLE})
        message(STATUS "Using LLVM target triple: ${TARGET_TRIPLE}")
        
        # Parse the triple to determine object format
        # Typical format: <arch>-<vendor>-<os>-<env>
        # ELF: Linux, FreeBSD, etc.
        # Mach-O: Darwin/macOS
        # COFF: Windows (MSVC)
        # Wasm: WebAssembly
        
        if(TARGET_TRIPLE MATCHES "linux|freebsd|netbsd|openbsd|dragonfly|solaris|elf")
            set(OBJECT_FORMAT "ELF")
        elseif(TARGET_TRIPLE MATCHES "darwin|macosx|macos|ios|tvos|watchos")
            set(OBJECT_FORMAT "MachO")
        elseif(TARGET_TRIPLE MATCHES "windows|win32|mingw|cygwin|msvc")
            set(OBJECT_FORMAT "COFF")
        elseif(TARGET_TRIPLE MATCHES "wasm")
            set(OBJECT_FORMAT "Wasm")
        endif()
    endif()
    
    # Fallback to CMAKE variables if LLVM triple not available
    if(OBJECT_FORMAT STREQUAL "UNKNOWN")
        if(CMAKE_SYSTEM_NAME MATCHES "Linux|FreeBSD|NetBSD|OpenBSD|DragonFly|SunOS")
            set(OBJECT_FORMAT "ELF")
        elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
            set(OBJECT_FORMAT "MachO")
        elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
            # Windows uses COFF even when CMake does not classify the Clang driver as MSVC-like.
            set(OBJECT_FORMAT "COFF")
        elseif(CMAKE_SYSTEM_NAME MATCHES "Emscripten")
            set(OBJECT_FORMAT "Wasm")
        endif()
    endif()
    
    # Additional check using compiler preprocessor if still unknown
    if(OBJECT_FORMAT STREQUAL "UNKNOWN")
        set(_object_format_test_source "${CMAKE_BINARY_DIR}/object_format_test.cpp")
        file(WRITE "${_object_format_test_source}" [[
#if defined(__ELF__)
__ELF__
#endif
#if defined(__MACH__)
__MACH__
#endif
#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
_WIN32
#endif
int main() { return 0; }
]])
        try_compile(COMPILE_RESULT
            ${CMAKE_BINARY_DIR}/object_format_test
            ${_object_format_test_source}
            COMPILE_DEFINITIONS -E
            OUTPUT_VARIABLE COMPILER_OUTPUT
        )
        unset(_object_format_test_source)
        
        # Check common preprocessor defines
        if(COMPILER_OUTPUT MATCHES "__ELF__")
            set(OBJECT_FORMAT "ELF")
        elseif(COMPILER_OUTPUT MATCHES "__MACH__")
            set(OBJECT_FORMAT "MachO")
        elseif(COMPILER_OUTPUT MATCHES "_WIN32|_WIN64|__CYGWIN__")
            set(OBJECT_FORMAT "COFF")
        endif()
    endif()
    
    # Set the output variable
    set(${OUTPUT_VAR} ${OBJECT_FORMAT} PARENT_SCOPE)
    message(STATUS "Detected object format: ${OBJECT_FORMAT}")
endfunction()

# Detect object format when this module is included
detect_object_format(TARGET_OBJECT_FORMAT)

# Export the result
set(TARGET_OBJECT_FORMAT ${TARGET_OBJECT_FORMAT} CACHE STRING "Target object file format (ELF, MachO, COFF, Wasm, or UNKNOWN)")
mark_as_advanced(TARGET_OBJECT_FORMAT)
