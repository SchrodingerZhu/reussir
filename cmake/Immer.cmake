include(FetchContent)

set(CMAKE_DISABLE_FIND_PACKAGE_Boost ON)
set(CMAKE_DISABLE_FIND_PACKAGE_BoehmGC ON)
set(CMAKE_WARN_DEPRECATED OFF)

FetchContent_Declare(
  immer
  GIT_REPOSITORY https://github.com/arximboldi/immer.git
  GIT_TAG 06d94ce48a02b11d3b225083a820f82c3d3ef462
)

# Set all Immer build options to OFF
set(immer_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(immer_BUILD_PERSIST_TESTS OFF CACHE BOOL "Build experimental persist tests" FORCE)
set(immer_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(immer_BUILD_DOCS OFF CACHE BOOL "Build docs" FORCE)
set(immer_BUILD_EXTRAS OFF CACHE BOOL "Build extras" FORCE)

# LLVM-style builds disable C++ exceptions; make Immer use its no-exceptions path.
add_compile_definitions(IMMER_NO_EXCEPTIONS)

FetchContent_MakeAvailable(immer)
