# Generic containers, pure Reussir: a cons list and a BST map with
# closure-parameterized combination and folds. Consumers instantiate these
# generics from the shipped bodies — the same instance emitted by two
# packages collapses to one definition at link.
{
  package = { name = "collections", version = "1.4.2" },
}
