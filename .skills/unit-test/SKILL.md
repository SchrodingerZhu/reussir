---
name: unit-test
description: Run unit tests for Reussir.
license: MPL-2.0
---

Reussir has C++ unit tests (googletest) and Rust crate tests.

To run the C++ unit tests, invoke `ninja reussir-ut` in the build directory.
Then run it under `build/bin`.

The output should look like this:
```bash
❯ ./bin/reussir-ut
Running main() from ./googletest/src/gtest_main.cc
[==========] Running 10 tests from 3 test suites.
[----------] Global test environment set-up.
[----------] 4 tests from ReussirValueTransformTest
[ RUN      ] ReussirValueTransformTest.RcAcquisition
[       OK ] ReussirValueTransformTest.RcAcquisition (19 ms)
[ RUN      ] ReussirValueTransformTest.RefToRcAcquisition
[       OK ] ReussirValueTransformTest.RefToRcAcquisition (3 ms)
[ RUN      ] ReussirValueTransformTest.RefToCompoundRecordAcquisition
[       OK ] ReussirValueTransformTest.RefToCompoundRecordAcquisition (2 ms)
[ RUN      ] ReussirValueTransformTest.RefToVariantRecordAcquisition
[       OK ] ReussirValueTransformTest.RefToVariantRecordAcquisition (2 ms)
[----------] 4 tests from ReussirValueTransformTest (28 ms total)

[----------] 5 tests from ReussirTest
[ RUN      ] ReussirTest.BasicContextTest
[       OK ] ReussirTest.BasicContextTest (2 ms)
[ RUN      ] ReussirTest.ParseRecordTypeTest
[       OK ] ReussirTest.ParseRecordTypeTest (2 ms)
[ RUN      ] ReussirTest.SimpleRecordScanner
[       OK ] ReussirTest.SimpleRecordScanner (2 ms)
[ RUN      ] ReussirTest.NestedRecordScanner
[       OK ] ReussirTest.NestedRecordScanner (2 ms)
[ RUN      ] ReussirTest.VariantRecordScanner
[       OK ] ReussirTest.VariantRecordScanner (2 ms)
[----------] 5 tests from ReussirTest (13 ms total)

[----------] 1 test from RustCompilerTest
[ RUN      ] RustCompilerTest.CompileSimpleSource
[       OK ] RustCompilerTest.CompileSimpleSource (60 ms)
[----------] 1 test from RustCompilerTest (60 ms total)

[----------] Global test environment tear-down
[==========] 10 tests from 3 test suites ran. (101 ms total)
[  PASSED  ] 10 tests.
```

To run the Rust crate tests, invoke the per-crate CMake test targets from the
build directory, e.g. `ninja rrc-test`, `ninja reussir-codegen-test`, and
`ninja reussir-backend-test`. Notice that this may take a while and generates a
long output. Each is expected to exit successfully; there can be warnings from
Reussir's backend, or sample error messages from the compiler's diagnostics
tests.
