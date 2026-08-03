# A two-package trait build: the `showable` dependency owns the trait, a
# ground and a generic impl, and a bounded generic; the app adds its own
# impl for a local type and dispatches across the boundary — the ground
# impl method links against the dependency's archive while the generic one
# instantiates here (weak_odr).
{
  package = {
    name | String = "traitdemo",
    version = "1.0.0",
  },
  dependencies = {
    console = { path = "vendor/console", version = "^0.1" },
    showable = { path = "vendor/showable", version = "^0.1" },
  },
  targets = {
    traitdemo = { kind = 'executable },
  },
}
