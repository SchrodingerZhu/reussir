#import "/book.typ": book-page

#show: book-page.with(title: "Reussir Programming Language")

= Reussir Programming Language

Reussir is designed for high-performance and ergonomic programming. Reussir
provides advanced memory reuse analysis to improve its performance. Built on
top of the MLIR framework, the language provides high extensibility.

The project centers on one claim: memory reuse in reference-counted
functional programs should be represented and optimized directly at the IR
level. Reusable storage is made explicit as SSA tokens, carried through MLIR
structured control flow, and reuse analysis becomes an ordinary compiler
problem. Around that core, Reussir provides an ownership-aware functional
frontend, region-based local mutation with freezing, a polymorphic FFI to
Rust, and a toolchain — the `rrc` compiler driver, the `rene` package
manager, and the `rrepl` REPL.

The sources live at
#link("https://github.com/reussir-lang/reussir")[github.com/reussir-lang/reussir];
the README there covers building the toolchain and running the test suites.
