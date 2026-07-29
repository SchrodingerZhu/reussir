# The WebAssembly showcase: one package that only ever builds for
# `wasm32-wasip1-threads`, so the machine target lives in the profile rather
# than in the `rene build` command line. Nothing else is target-specific —
# same optimization, debug info and `pic` relocation as any other package
# here, and no `target_features`: the driver derives what a threaded wasm
# target implies (`+atomics,+bulk-memory,+mutable-globals`) from the triple
# itself, which is the whole point of running this program with threads.
{
  package = {
    name | String = "wasi_threads",
    version = "0.1.0",
  },
  targets = {
    threads = { kind = 'executable },
  },
  profiles = {
    dev = {
      default_target_triple = "wasm32-wasip1-threads",
    },
  },
}
