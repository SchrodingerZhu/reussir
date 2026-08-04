# The Hash/Hasher pattern with its three roles in three packages: `hashkit`
# owns the traits, `hashables` implements them (a hasher and hashable
# types), and this app only *uses* them — it implements nothing, just
# names the extern traits in bounds and dispatches. The dependency graph
# is a diamond (app -> hashkit, app -> hashables -> hashkit), so the
# app loads the trait items both directly and through the implementer's
# re-shipped interface.
{
  package = {
    name | String = "hashdemo",
    version = "1.0.0",
  },
  dependencies = {
    console = { path = "vendor/console", version = "^0.1" },
    hashables = { path = "vendor/hashables", version = "^0.1" },
    hashkit = { path = "vendor/hashkit", version = "^0.1" },
  },
  targets = {
    hashdemo = { kind = 'executable },
  },
}
