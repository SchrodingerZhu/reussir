---
name: check-env
description: Check if the environment is set up correctly for Reussir development.
license: MPL-2.0
---

To check if the environment is set up correctly for Reussir development, please
go through the following checklist:

- Check if there is a `build` directory in the root directory of the project. 
  The directory should looks like a general `CMake` build directory.

  It is preferred to build Reussir with LLVM toolchain (>= 21) and Ninja.

- `rustc --version` should return a version number greater than or equal to 1.90.
  The following is a sample output:
  ```bash
  rustc 1.93.0-nightly (843f8ce2e 2025-11-07)
  ```

