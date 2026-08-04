# The Hash/Hasher pattern as a running program: the trait pair and a
# byte-splitting bounded generic at the package root, the djb2 hasher, the
# hashable types, and the API each in their own module, with one polyffi
# console package to make the hash values observable.
{
  package = {
    name | String = "hashdemo",
    version = "1.0.0",
  },
  dependencies = {
    console = { path = "vendor/console", version = "^0.1" },
  },
  targets = {
    hashdemo = { kind = 'executable },
  },
}
