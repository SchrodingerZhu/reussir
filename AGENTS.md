# Overview

Reussir is a programming language built around **RC-based memory reuse analysis**
and **region-based memory management** for locally mutable data. It began as a
research project focused on leveraging static analysis and language–system
co-design to reduce memory allocations, improve memory locality, and ultimately
deliver better performance. Built on top of LLVM/MLIR, Reussir aims to grow into a
more sophisticated language that addresses real needs in both academic research
and industrial applications.

Reussir predates the rapid rise of Large Language Models (LLMs) and agentic AI
systems. As the project evolves, however, it has become increasingly clear that
AI agents will play an important role in its development. Reussir therefore
rethinks its design to become not only a language that is *developed with the
help of AI*, but also one that is *safe and effective for AI agents to work on*,
and that serves as a medium for communicating ideas between agents and human
programmers.

We adopt an open stance toward AI utilization and aim to develop Reussir
**“with AI, and for AI.”** To achieve this, we expect the Reussir implementation
to exhibit the following qualities:

- Clear logic and architecture  
- Continuous testing for newly added features  
- Clear documentation and well-written comments  
- Efficient implementation, enabling the compiler to tackle complex problems  

## Special Requirements for AI Agents

- Test all changes using both Rust/C++ unit tests and LLVM *lit*-based
  integration tests.
- Do not leave scratch code, ad-hoc scripts, or debugging logs in the repository.
  Clean up all temporary artifacts before completing a task.
- Avoid ad-hoc implementations for general compiler features.
- Ask questions or search for established solutions when uncertain. Prefer
  well-founded approaches and cite authoritative or well-tested references when
  possible.
- Ensure a human-in-the-loop development style. Do not blindly execute vague
  requests (e.g., “build me a 2B-revenue company and report back”). Seek clear
  instructions, constraints, and algorithmic details before proceeding.

## Directions of the Project

It remains an open question what an *AI-friendly programming language* should
ultimately look like. Rather than prescribing a fixed definition, we outline a
set of principles and features we aim to maintain in this codebase:

- **Functional language with managed memory.**  
  Reussir provides token-efficient ways to express solutions without burdening
  the programmer with explicit memory management.

- **Efficient implementation.**  
  Reussir is a highly optimized compiled language. It is not intended to compete
  with Python for rapid prototyping. Instead, AI agents may choose Reussir for
  computationally intensive tasks—such as computing `Fibonacci(42)` or generating
  a `1024×1024` Mandelbrot set in ASCII—when performance and predictability
  matter.

- **A strong glue language for the MLIR ecosystem and native code.**  
  While languages such as Koka and Lean do not emphasize this aspect, an RC-based
  runtime naturally facilitates interoperability with native code. Reussir
  therefore includes a polymorphic FFI design. Built on MLIR, it also provides
  avenues to explore MPI, OpenMP, Async, LinAlg, Enzyme, and other MLIR dialects.
  These capabilities allow AI agents to use Reussir to solve complex,
  systems-level problems.

- **Transparent program compilation.**
  Reussir provides means to inspect its type-checking and elaboration process,
  making it easier to understand and debug the compiler, or trace the
  optimization on programs.

Think of yourself as designing and implementing not just for today’s
developers, but for future **human–AI collaboration**.

## Skills

We provide a set of skills to help you get started with Reussir. They are
under the `.skills` directory within the root of the repository.

- **check-env** — Check if the environment is set up correctly for Reussir development.
- **compilation** — Compile Reussir source code.
- **elaboration** — Elaborate Reussir source code.
- **integration-test** — Run integration tests for Reussir.
- **parser-tree** — Parse Reussir source code into a parser tree.
- **rene** — Build Reussir packages with the rene package manager.
- **reussir-mlir** — Run MLIR conversions and translations.
- **run-repl** — Run Reussir REPL to interpret Reussir source code.
- **sanitizer-testing** — Build and test Reussir with ASan, LSan, MSan, or TSan.
- **unit-test** — Run unit tests for Reussir.
