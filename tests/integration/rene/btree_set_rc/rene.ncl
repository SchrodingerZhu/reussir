# The root is the only package that knows about the Rust-backed BTreeSet.
# Its ordered dependency supplies a public shared type and comparison impls,
# exercising bridge selection from an imported RRI.
{
  package = {
    name | String = "btree_rc",
    version = "1.0.0",
  },
  dependencies = {
    ordered = { path = "vendor/ordered", version = "^0.1" },
  },
  targets = {
    btree_rc = { kind = 'executable },
  },
}
