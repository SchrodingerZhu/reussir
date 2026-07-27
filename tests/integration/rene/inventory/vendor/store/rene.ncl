# The domain layer: transactions folded into stock through `collections`'
# generics, line values through `pricing`'s imports. Constraints here are
# deliberately looser than the app's — resolution must satisfy both.
{
  package = { name = "store", version = "1.0.3" },
  dependencies = {
    collections = { path = "../collections", version = "^1.2" },
    pricing = { path = "../pricing", version = "2" },
  },
}
