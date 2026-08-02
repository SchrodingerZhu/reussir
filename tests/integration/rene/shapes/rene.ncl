# Methods over a vendor dep: the app calls `geom`'s impl methods (value,
# Arc, and regional receivers; one generic method at two instantiations)
# across the package boundary, printing through the `console` polyffi
# package. `release` re-runs the same session optimized.
{
  package = {
    name | String = "shapes",
    version = "1.0.0",
  },
  dependencies = {
    geom = { path = "vendor/geom", version = "^1.0" },
    console = { path = "vendor/console", version = "^1.0" },
  },
  targets = {
    shapes = { kind = 'executable },
  },
}
