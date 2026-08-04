# An app whose manifest names only `console` — the bundled `core` arrives
# implicitly, injected by rene into every build's dependency graph.
{
  package = {
    name | String = "mathdemo",
    version = "1.0.0",
  },
  dependencies = {
    console = { path = "vendor/console", version = "^0.1" },
  },
  targets = {
    mathdemo = { kind = 'executable },
  },
}
