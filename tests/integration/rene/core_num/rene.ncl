# An app whose generics ride `core::num` — the numeric tower is now real
# `core` source, sealed and `#[lang]`-bound.
{
  package = {
    name | String = "numdemo",
    version = "1.0.0",
  },
  dependencies = {
    console = { path = "vendor/console", version = "^0.1" },
  },
  targets = {
    numdemo = { kind = 'executable },
  },
}
