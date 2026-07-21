# Import

## Motivation

Reussir names every item by its module path. That keeps resolution simple and
unambiguous, but common references — above all the built-in intrinsics — get
long fast:

```text
core::intrinsic::array::set(a, i, core::intrinsic::array::get(a, i) + v)
```

`import` introduces **file-scoped path abbreviations** so the same call can
read:

```text
import core::intrinsic::array;
...
array::set(a, i, array::get(a, i) + v)
```

## Surface syntax

One top-level item form, ending in `;`:

```text
import-stmt ::= 'import' path ('as' name)? ';'
```

* `import a::b::c;` binds the *last segment* (`c`) to the path `a::b::c`.
* `import a::b::c as d;` binds `d` to `a::b::c`.

The target path is not restricted to modules: because a binding is a plain
path rewrite, `import` works identically for a function, a record, a
constructor qualifier, or an intrinsic family — `import pkg::List as L;` and
`import core::intrinsic::math::sqrt as rt;` are ordinary imports. There is
deliberately no separate `alias` item; renaming is spelled with `as`.

`import` is a hard keyword in the lexer but — like every Reussir keyword —
remains usable as an ordinary identifier in identifier positions
([`SyntaxKind::is_ident_like`]). The REPL routes `import` followed by an
identifier to the item parser, mirroring the `fn`/`mod` convention.

## Semantics

A binding maps one identifier to a reference path. Bindings are:

* **File-scoped**: they apply to every item in the file that declares them
  (order-independent, like item declarations) and are invisible to other
  files, including other files of the same package.
* **Private**: `pub import` is an error; bindings do not re-export.
* **Namespace-agnostic**: a binding is a path rewrite, not a def reference.
  The same binding abbreviates a function path, a record path, a constructor
  qualifier, or an intrinsic family, depending on where it is used.
* **Checked for clashes**: two bindings of the same name in one file are an
  error. The names `core`, `root`, and `super` cannot be bound (they are the
  reserved path heads).

### Expansion

Everywhere the elaborator resolves a reference path (function calls,
constructor calls and patterns, variable references, type expressions, and
`extern` trampoline targets), the path is first **expanded**:

* A qualified path whose *first segment* is bound has that segment replaced by
  the binding's target path (`arr::get` with `import core::intrinsic::array
  as arr` becomes `core::intrinsic::array::get`).
* A bare name that is bound is replaced by the target path outright
  (`rt` with `import core::intrinsic::math::sqrt as rt` becomes the full
  path).

Expansion iterates, so a binding may target another binding; a
self-referential chain abandons expansion entirely, so the reference fails
resolution with the ordinary "unknown" diagnostic on the path as written.
Expansion happens *before* the built-in path checks, so bindings work
uniformly for user items and for the built-in `core` intrinsics, `Nullable`,
and `Arc` spellings. Diagnostics print the expanded path, keeping error text
identical to the fully-qualified spelling.

### Shadowing

Local bindings win over imports: a `let`-bound or parameter name shadows a
same-named import in expression positions, exactly as it shadows a top-level
function. Generic parameters likewise win in type positions. An import in turn
takes precedence over module-relative resolution of the same bare name, since
expansion runs before lookup — a file that imports a name it also declares is
better served by renaming the import (`as`).

## Implementation notes

* The parser produces `ImportStmt` nodes; the surface layer projects them
  into `StmtKind::Import(ImportDecl { name, path, … })`.
* The elaborator collects bindings per `FileId` during the statement scan of
  `run_files`, before any resolution runs, and consults them in
  `expand_path` (`semi/ctxt.rs`), which is applied in the shared
  `resolve_*_ref` helpers plus the entry points that perform textual
  built-in checks (`infer_func_call`, `infer_ctor`, `check_ctor_pat`,
  `eval_type_expr`).
* Bindings live in an append-only vector so the REPL checkpoint/rollback
  machinery restores them by truncation like every other elaborator table.
