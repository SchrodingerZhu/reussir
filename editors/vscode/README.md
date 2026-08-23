# Reussir for VS Code

This extension provides semantic highlighting for `.rr` files through the
native `reussir-lsp` server. Its bounded LSP client state and wire framing are
implemented in Rust and shipped as WebAssembly; a small TypeScript host connects
that component to VS Code and the native server process.

Set `reussir.server.path` when `reussir-lsp` is not available on `PATH`. During
development the extension also detects `../../build/bin/reussir-lsp` relative
to this directory.

The initial extension intentionally provides only whole-document semantic
tokens. Inline Rust poly-FFI and MLIR transform schedules are highlighted by
the native server's Tree-sitter integrations.
