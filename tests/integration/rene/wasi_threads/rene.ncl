# The WebAssembly showcase: one package that only ever builds for
# `wasm32-wasip1-threads`, so the machine target lives in the profile rather
# than in the `rene build` command line. Everything else is a plain `dev`
# build — same optimization, debug info and `pic` relocation as any other
# package here.
#
# `target_features` is the one target-specific knob, and it is not optional:
# the wasm backend emits real atomic read-modify-writes for the atomic
# reference counts (and for the lock-guarded cell's fast paths) only when
# `atomics` is on. Compiled without it the very same code comes out as plain
# `i32.load`/`i32.store` — the target is single-threaded by assumption —
# which is a lost update waiting to happen, not a compile error. rustc
# builds this target's standard library (and the runtime bake, and the
# polyffi textures) with these three, and links with `--shared-memory`, so
# the objects `rrc` emits should carry them too.
#
# The texture rustc compiles for `run_workers` happens to bring the feature
# into the module on its own, so this particular package would come out
# atomic either way. That is an accident of it having a texture at all, not
# a property of the target: a package with no polyffi in it does not, and
# the point of naming the features here is not to depend on the difference.
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
      target_features = "+atomics,+bulk-memory,+mutable-globals",
    },
  },
}
