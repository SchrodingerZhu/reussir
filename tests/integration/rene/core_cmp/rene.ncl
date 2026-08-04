# An app exercising the `core::cmp` tower — the manifest names only
# `console`; the implicit `core` dependency supplies the operator traits.
{
  package = {
    name | String = "cmpdemo",
    version = "1.0.0",
  },
  dependencies = {
    console = { path = "vendor/console", version = "^0.1" },
  },
  targets = {
    cmpdemo = { kind = 'executable },
  },
}
