# The sample-statistics pipeline: numbers arrive through `vecio`'s opaque
# runtime vector, everything after that is functional Reussir. The release
# profile splits the app across three codegen units and turns on
# reuse-across-call — the optimized multi-unit build must be the same
# pipeline.
{
  package = {
    name | String = "stats",
    version = "0.2.0",
  },
  dependencies = {
    vecio = { path = "vendor/vecio", version = "^1" },
  },
  targets = {
    stats = { kind = 'executable },
  },
  profiles = {
    release = {
      codegen_units = 3,
      reuse_across_call = true,
    },
  },
}
