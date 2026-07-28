# The inventory report app: domain logic in `store`, generic containers in
# `collections`, and two polyffi packages — `console` for the report's
# output and `pricing` for the money math — all resolved and linked by one
# `rene build`. The app constrains every package it names directly, not
# just its immediate layer.
{
  package = {
    name | String = "inventory",
    version = "0.1.0",
  },
  dependencies = {
    store = { path = "vendor/store", version = "^1" },
    collections = { path = "vendor/collections", version = "^1.4" },
    console = { path = "vendor/console", version = "0.3" },
    pricing = { path = "vendor/pricing", version = "^2" },
  },
  targets = {
    inventory = { kind = 'executable },
  },
}
