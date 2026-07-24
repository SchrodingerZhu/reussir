# A demo `rene` package manifest. The manifest is a Nickel program: bindings,
# string interpolation, contracts, and imports all run before `rene` reads the
# evaluated record. Try it from anywhere under this directory:
#
#     rene build
#
# NOTE: fields must not reference other record fields by bare name — Nickel
# records are recursive, so `name = name` is infinite recursion; bind helpers
# with `let` instead.
let pkg_name = "demo" in
{
  package = {
    name | String = pkg_name,
    version = "0.1.0",
  },
  # First stage: dependencies are file-system paths, relative to this file.
  # (Not resolved yet — recorded and dumped only.)
  dependencies = {
    utils = { path = "vendor/utils" },
  },
}
