# The package `rene build` compiles end to end: one executable target, and a
# `release` profile refining the built-in (its aggressive optimization stays;
# the unit count is the manifest's).
{
  package = {
    name | String = "demo",
    version = "0.1.0",
  },
  targets = {
    demo = { kind = 'executable },
  },
  profiles = {
    release = { codegen_units = 2 },
  },
}
