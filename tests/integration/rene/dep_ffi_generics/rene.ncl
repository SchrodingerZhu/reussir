# The root supplies the key type and its trait evidence; the dependency owns
# the generic Rust-backed container, its FFI imports, and the traits its own
# records bound — the shape std's cow collections ship in.
{
  package = {
    name | String = "holder",
    version = "1.0.0",
  },
  dependencies = {
    hashdep = { path = "vendor/hashdep", version = "^0.1" },
  },
  targets = {
    holder = { kind = 'executable },
  },
}
