# Import and Alias

## Motivation

Reussir names every item by its module path. That keeps resolution simple and
unambiguous, but common references — above all the built-in intrinsics — get
long fast:

```text
core::intrinsic::array::set(a, i, core::intrinsic::array::get(a, i) + v)
```

`import` and `alias` introduce **file-scoped path abbreviations** so the same
call can read:

```text
import core::intrinsic::array;
...
array::set(a, i, array::get(a, i) + v)
```

## Surface syntax

Two top-level item forms, both ending in `;`:

```text
import-stmt ::= 'import' path ('as' name)? ';'
alias-stmt  ::= 'alias' name '=' path ';'
```

* `import a::b::c;` binds the *last segment* (`c`) to the path `a::b::c`.
* `import a::b::c as d;` and `alias d = a::b::c;` are equivalent: both bind
  `d` to `a::b::c`. `import … as …` reads naturally when pulling a module
  into scope under a new name; `alias n = p;` reads naturally when naming a
  long path.

`import` and `alias` are hard keywords in the lexer but — like every Reussir
keyword — remain usable as ordinary identifiers in identifier positions
([`SyntaxKind::is_ident_like`]). The REPL routes `import`/`alias` followed by
an identifier to the item parser, mirroring the `fn`/`mod` convention.

## Semantics

A binding maps one identifier to a reference path. Bindings are:

* **File-scoped**: they apply to every item in the file that declares them
  (order-independent, like item declarations) and are invisible to other
  files, including other files of the same package.
* **Private**: `pub import` / `pub alias` is an error; bindings do not
  re-export.
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
  the binding's target path (`arr::get` with `arr = core::intrinsic::array`
  becomes `core::intrinsic::array::get`).
* A bare name that is bound is replaced by the target path outright
  (`get` with `get = core::intrinsic::array::get` becomes the full path).

Expansion iterates, so an alias may refer to another alias; a self-referential
chain abandons expansion entirely, so the reference fails resolution with the
ordinary "unknown" diagnostic on the path as written. Expansion happens *before* the built-in path checks, so
bindings work uniformly for user items and for the built-in `core`
intrinsics, `Nullable`, and `Arc` spellings. Diagnostics print the expanded
path, keeping error text identical to the fully-qualified spelling.

### Shadowing

Local bindings win over imports: a `let`-bound or parameter name shadows a
same-named import in expression positions, exactly as it shadows a top-level
function. Generic parameters likewise win in type positions. An import in turn
takes precedence over module-relative resolution of the same bare name, since
expansion runs before lookup — a file that imports a name it also declares is
better served by renaming the import (`as`).

## Implementation notes

* The parser produces `ImportStmt` / `AliasStmt` nodes; the surface layer
  projects both into one `StmtKind::Import(ImportDecl { name, path, … })`.
* The elaborator collects bindings per `FileId` during the statement scan of
  `run_files`, before any resolution runs, and consults them in
  `expand_path` (`semi/ctxt.rs`), which is applied in the shared
  `resolve_*_ref` helpers plus the entry points that perform textual
  built-in checks (`infer_func_call`, `infer_ctor`, `check_ctor_pat`,
  `eval_type_expr`).
* Bindings live in an append-only vector so the REPL checkpoint/rollback
  machinery restores them by truncation like every other elaborator table.
