# The implementer: a concrete hasher and hashable types, each in its own
# module, all implementing the extern `hashkit` traits for local types.
{
  package = { name = "hashables", version = "0.1.0" },
  dependencies.hashkit = { path = "../hashkit", version = "^0.1" },
}
