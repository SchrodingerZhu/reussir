//! Pattern-match elaboration into decision trees.
//!
//! This compiles a `match` into a [`DecisionTree`] using the clause-matrix
//! algorithm of Maranget, *"Compiling Pattern Matching to Good Decision Trees"*
//! (ML'08). We are free to choose the algorithm here: the decision tree is the
//! frontend's own IR (it drives later code generation), so it need not mirror
//! anything in the backend. A real matrix compiler buys us nested patterns,
//! guards, and exhaustiveness/redundancy checking essentially for free.
//!
//! ## The shape of the algorithm
//!
//! We work over a *clause matrix*. Each **column** is an *occurrence* — a path
//! ([`PatVarRef`]) into the scrutinee value (the root is the empty path, a
//! constructor field `i` extends its parent's path with `i`) together with the
//! type of the value found there. Each **row** is one match arm: a vector of
//! patterns (one per column), the bindings accumulated so far (variable →
//! occurrence), an optional guard, the (already type-checked) body, and the
//! original arm index (for redundancy reporting).
//!
//! `Elaborator::compile_match` then repeats:
//!
//! 1. **Empty matrix** ⇒ nothing matches here: emit [`DecisionTree::Uncovered`]
//!    and flag the match as non-exhaustive.
//! 2. **First row has no constructor tests left** ⇒ it matches unconditionally
//!    (modulo its guard). Emit a [`DecisionTree::Leaf`]; if it has a guard, emit
//!    a [`DecisionTree::Guard`] whose failure branch retries the *remaining*
//!    rows on the same value.
//! 3. **Otherwise** pick a column that still has a constructor and emit a
//!    [`DecisionTree::Switch`] on its occurrence. For each head constructor `c`
//!    we recurse on the *specialized* matrix `S(c, P)` (which expands `c`'s
//!    fields into fresh columns); for open families (ints, strings) we also
//!    recurse on the *default* matrix `D(P)` (the rows that ignore the column).
//!
//! Column choice only affects the *size* of the tree, never correctness; we use
//! the simple "leftmost column that has a constructor" heuristic and leave
//! Maranget's necessity-based heuristics as a future refinement.
//!
//! ## Bindings
//!
//! A variable pattern matches anything but records `(var, occurrence)`. When a
//! column is consumed (specialized or dropped), any binder sitting in it
//! contributes its `(var, occurrence)` to the row; when a row finally becomes a
//! leaf, the remaining binders in its columns are added too. The leaf's
//! `bindings` therefore tell code generation where in the scrutinee to read each
//! bound variable.

use crate::literal::Integer;
use crate::semi::infer::Instantiation;
use crate::semi::ty::{Ty, TyKind};
use crate::surface::{self, Const, PatternKind, Span};
use crate::utils::string::StringToken;

use super::ctxt::{Elaborator, RecordFields};
use super::hir::{DecisionTree, Expr, ExprKind, PatVarRef, SwitchCases, VarId};

/// A head constructor: what a (refutable) pattern tests for at an occurrence.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Ctor<'tcx> {
    /// An enum variant, by index into the enum's variant list.
    Variant(usize),
    /// An integer key, arbitrary-precision (the scrutinee type bounds its
    /// range, checked like any literal at monomorphization).
    Int(&'tcx Integer),
    Char(u32),
    Bool(bool),
    Str(StringToken),
    /// The non-null case of a `Nullable<T>` scrutinee, with one field: the
    /// unwrapped `T`.
    NonNull,
    /// The null case of a `Nullable<T>` scrutinee (no fields).
    Null,
}

impl Ctor<'_> {
    fn family(&self) -> Family {
        match self {
            Ctor::Variant(_) => Family::Variant,
            Ctor::Int(_) => Family::Int,
            Ctor::Char(_) => Family::Char,
            Ctor::Bool(_) => Family::Bool,
            Ctor::Str(_) => Family::Str,
            Ctor::NonNull | Ctor::Null => Family::Nullable,
        }
    }
}

/// The "signature" a column's constructors belong to. A column is well-typed, so
/// every constructor in it shares one family; the family decides how the
/// [`SwitchCases`] are shaped and whether a default branch is needed.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    /// A closed family with one case per enum variant — exhaustive on its own.
    Variant,
    /// A closed two-case family (`true` / `false`).
    Bool,
    /// A closed two-case family (`NonNull` / `Null`).
    Nullable,
    /// Open families: there are always more values than literals, so they need a
    /// default branch.
    Int,
    Char,
    Str,
}

/// A type-checked pattern. Variable bindings carry the [`VarId`] allocated for
/// them during checking; their occurrence is filled in while compiling.
#[derive(Clone, Debug)]
enum Pat<'tcx> {
    /// Matches anything, binds nothing (`_`).
    Wild,
    /// Matches anything, binds the value at this occurrence to `VarId`.
    Bind(VarId),
    /// Matches `ctor` and recurses into its fields.
    Ctor {
        ctor: Ctor<'tcx>,
        fields: Vec<Pat<'tcx>>,
    },
}

impl Pat<'_> {
    fn is_ctor(&self) -> bool {
        matches!(self, Pat::Ctor { .. })
    }
}

/// One column of the clause matrix: where the value lives and what its type is.
#[derive(Clone)]
struct Column<'tcx> {
    occ: PatVarRef,
    ty: Ty<'tcx>,
}

/// One row of the clause matrix: an arm in its current, partially-specialized
/// form.
#[derive(Clone)]
struct Row<'tcx> {
    /// One pattern per column, in column order.
    pats: Vec<Pat<'tcx>>,
    /// Bindings discovered in columns already consumed by specialization.
    bindings: Vec<(VarId, PatVarRef)>,
    guard: Option<Expr<'tcx>>,
    body: Expr<'tcx>,
    /// Index of the originating match arm, for redundancy reporting.
    arm: usize,
}

impl<'tcx> Row<'tcx> {
    /// Drop column `j` from the row, recording the binder (if any) that lived
    /// there. Used by scalar specialization and the default matrix, where the
    /// column carries no sub-fields.
    fn without_col(&self, j: usize, bind: Option<(VarId, PatVarRef)>) -> Row<'tcx> {
        let mut pats = self.pats.clone();
        pats.remove(j);
        let mut bindings = self.bindings.clone();
        bindings.extend(bind);
        Row {
            pats,
            bindings,
            guard: self.guard.clone(),
            body: self.body.clone(),
            arm: self.arm,
        }
    }
}

/// The clause matrix.
struct Matrix<'tcx> {
    cols: Vec<Column<'tcx>>,
    rows: Vec<Row<'tcx>>,
}

impl<'a, 'tcx> Elaborator<'a, 'tcx> {
    /// (MATCH) `Γ ⊢ match scrut { arms } ⇒ (Match(hs, tree) : α)`: synthesize `scrut ⇒ S`
    /// and pick a fresh result hole `α`; for each arm bind its pattern against `⟦S⟧` in a
    /// marked scope, check the optional guard `⇐ Bool`, and check the body `⇐ α` (so all
    /// arms unify to one type). The arms are compiled to a [`DecisionTree`] by the
    /// clause-matrix algorithm, with exhaustiveness and reachability checked here.
    pub(super) fn infer_match(
        &mut self,
        scrut: &surface::Expr,
        arms: &[(surface::Pattern, surface::Expr)],
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let scrut = self.infer_expr(scrut);
        let scrut_ty = self.infer.shallow_resolve(scrut.ty);
        let result = self.infer.new_hole_ty();

        // Type-check every arm, turning each surface pattern into a `Pat` (with
        // binders allocated and in scope for the guard and body).
        let mut rows = Vec::with_capacity(arms.len());
        let mut arm_spans = Vec::with_capacity(arms.len());
        for (i, (pat, body)) in arms.iter().enumerate() {
            let mark = self.vars.mark();
            let p = self.check_pat(&pat.kind(), Some(pat.span()), scrut_ty);
            let guard = pat.guard().map(|g| {
                let bool_ty = self.tcx.mk_bool();
                self.check_expr(&g, bool_ty)
            });
            let checked_body = self.check_expr(body, result);
            self.vars.restore(mark);

            arm_spans.push(Some(body.span()));
            rows.push(Row {
                pats: vec![p],
                bindings: Vec::new(),
                guard,
                body: checked_body,
                arm: i,
            });
        }

        // Compile the single-column matrix rooted at the scrutinee itself.
        let matrix = Matrix {
            cols: vec![Column {
                occ: PatVarRef::default(),
                ty: scrut_ty,
            }],
            rows,
        };
        let mut reached = vec![false; arms.len()];
        let mut exhaustive = true;
        let tree = self.compile_match(matrix, &mut reached, &mut exhaustive);

        // Report arms that can never match, and a non-exhaustive match.
        for (i, hit) in reached.iter().enumerate() {
            if !hit {
                self.warn(arm_spans[i], "unreachable match arm");
            }
        }
        if !exhaustive {
            self.error(span, "non-exhaustive patterns in `match`");
        }

        self.mk_match(Box::new(scrut), tree, result, span)
    }

    fn mk_match(
        &mut self,
        scrut: Box<Expr<'tcx>>,
        tree: DecisionTree<'tcx>,
        ty: Ty<'tcx>,
        span: Option<Span>,
    ) -> Expr<'tcx> {
        let id = self.fresh_expr_id();
        Expr {
            kind: ExprKind::Match(scrut, tree),
            ty,
            span,
            id,
        }
    }

    // ----- pattern type-checking -----

    /// Type-check a surface pattern against `ty`, allocating binders into the
    /// current scope, and return the typed [`Pat`]. `span` is the pattern's
    /// source span, used to blame any error; nested sub-patterns (a constructor's
    /// arguments) carry no span of their own, so they reuse the enclosing
    /// pattern's — coarse, but still traced to source.
    fn check_pat(&mut self, kind: &PatternKind, span: Option<Span>, ty: Ty<'tcx>) -> Pat<'tcx> {
        let ty = self.infer.shallow_resolve(ty);
        match kind {
            PatternKind::Wildcard => Pat::Wild,
            PatternKind::Bind(name) => {
                let var = self.vars.fresh(*name, ty, None);
                Pat::Bind(var)
            }
            PatternKind::Const(c) => self.check_const_pat(c, span, ty),
            PatternKind::Ctor(ctor) => self.check_ctor_pat(ctor, span, ty),
        }
    }

    fn check_const_pat(&mut self, c: &Const, span: Option<Span>, ty: Ty<'tcx>) -> Pat<'tcx> {
        let ctor = match c {
            Const::ConstInt(i) => {
                // The literal must be *some* integer, but its width/signedness
                // is whatever the scrutinee column already is — do not pin it to
                // `i32`, or matching an `i64`/`u8`/… scrutinee against an integer
                // arm would force `i64 = i32` and mis-type the match. Register
                // the `Integral` bound on the column type, exactly as an integer
                // literal *expression* does; the eventual int-switch lowering
                // reads the column's real type to pick the signed/unsigned cast.
                self.register_bound(self.builtins.integral, ty, span);
                Ctor::Int(self.tcx.alloc(i.clone()))
            }
            Const::ConstBool(b) => {
                let want = self.tcx.mk_bool();
                let _ = self.infer.unify(ty, want);
                Ctor::Bool(*b)
            }
            Const::ConstChar(c) => {
                let want = self.tcx.mk_char();
                let _ = self.infer.unify(ty, want);
                Ctor::Char(*c)
            }
            Const::ConstString(s) => {
                let want = self.tcx.mk_str();
                let _ = self.infer.unify(ty, want);
                Ctor::Str(self.strings.allocate(s))
            }
            Const::ConstFloat(_) => {
                self.error(span, "floating-point patterns are not supported");
                return Pat::Wild;
            }
        };
        Pat::Ctor {
            ctor,
            fields: Vec::new(),
        }
    }

    fn check_ctor_pat(
        &mut self,
        ctor: &surface::CtorPat,
        span: Option<Span>,
        ty: Ty<'tcx>,
    ) -> Pat<'tcx> {
        // An `Arc<Enum<…>>` scrutinee dispatches on the inner enum's
        // constructors; the pattern spelling is the plain one. The coloring
        // is the atomic context of everything inside, though: bare shared
        // fields bind at their promoted (`Arc`) types (§3.3).
        let arced = matches!(self.infer.shallow_resolve(ty).kind(), TyKind::Arc(_));
        let ty = self.peel_arc(ty);
        // Expand `import` bindings so an imported qualifier (`L::Cons` with
        // `import pkg::List as L`, or an imported `Nullable`) matches the
        // same as its full spelling.
        let expanded_pat;
        let ctor = match self.expand_path(&ctor.path) {
            Some(path) => {
                expanded_pat = surface::CtorPat {
                    path,
                    args: ctor.args.clone(),
                    has_ellipsis: ctor.has_ellipsis,
                    is_named: ctor.is_named,
                };
                &expanded_pat
            }
            None => ctor,
        };
        // The built-in nullable constructors, by the same qualifier convention
        // as [`Elaborator::infer_ctor`] on the expression side: the canonical
        // `core::intrinsic::nullable::Nullable::…`, or the prelude's bare
        // `Nullable::…` when no user record shadows the name.
        if ctor.path.segments.last().map(|k| self.sym(*k)) == Some("Nullable")
            && self.ctor_qualifier_prelude_applies(&ctor.path, "nullable")
        {
            return self.check_nullable_pat(ctor, span, ty);
        }
        let TyKind::Record { def, args, .. } = ty.kind() else {
            let shown = self.infer.resolve(ty);
            self.error(
                span,
                format!(
                    "cannot match constructor against `{}`",
                    self.ty_display(shown)
                ),
            );
            return Pat::Wild;
        };
        // The enum is named by the path qualifier (`Enum::Variant`, possibly
        // module-qualified: `m::Enum::Variant`), resolved to its def; bare
        // `Variant` falls back to the scrutinee's own record def.
        let want = ctor.path.basename;
        let enum_def = match ctor.path.segments.last() {
            Some(&seg) => match self.resolve_ctor_qualifier(&ctor.path) {
                Some(d) => {
                    self.record_use(seg);
                    d
                }
                None => {
                    if let Some(msg) = self.extern_private_ctor_msg(&ctor.path) {
                        self.error(span, msg);
                        return Pat::Wild;
                    }
                    let hint = self.record_suggestion(seg);
                    self.error(span, format!("unknown enum `{}`{hint}", self.sym(seg)));
                    return Pat::Wild;
                }
            },
            None => *def,
        };
        let Some(record) = self.records.get(&enum_def).cloned() else {
            self.error(span, "unknown enum".to_string());
            return Pat::Wild;
        };
        // Struct patterns do not exist (only enum-variant and Nullable
        // constructors match); if they ever land, private-field bindings must
        // gate on `may_access_private_fields` like projection does.
        let Some(RecordFields::Variants(variants)) = &record.fields else {
            self.error(span, "not an enum".to_string());
            return Pat::Wild;
        };
        let Some(vidx) = variants.iter().position(|v| v.name == want) else {
            self.error(span, format!("no variant `{}`", self.sym(want)));
            return Pat::Wild;
        };

        let payload = variants[vidx].fields.clone();
        if ctor.args.len() > payload.len()
            || (!ctor.has_ellipsis && ctor.args.len() != payload.len())
        {
            self.error(
                span,
                format!(
                    "variant `{}` has {} field(s), but the pattern binds {}",
                    self.sym(want),
                    payload.len(),
                    ctor.args.len()
                ),
            );
        }
        let inst = Instantiation::from_pairs(
            record
                .ty_params
                .iter()
                .map(|(_, g)| *g)
                .zip(args.iter().copied()),
        );
        // Recurse into each declared field. Missing trailing fields (e.g. behind
        // a `..` ellipsis or an arity error) are treated as wildcards.
        let group = if arced {
            self.arc_recursive_group(enum_def)
        } else {
            rustc_hash::FxHashSet::default()
        };
        let fields = payload
            .iter()
            .enumerate()
            .map(|(i, decl)| {
                let field_ty = self.infer.instantiate_ty(*decl, &inst);
                let field_ty = if arced {
                    self.arc_promote_member_ty(&group, field_ty)
                } else {
                    field_ty
                };
                match ctor.args.get(i) {
                    Some(arg) => self.check_pat(&arg.kind, span, field_ty),
                    None => Pat::Wild,
                }
            })
            .collect();
        Pat::Ctor {
            ctor: Ctor::Variant(vidx),
            fields,
        }
    }

    /// `Nullable::NonNull(p)` / `Nullable::Null` patterns against a
    /// `Nullable<T>` scrutinee. `NonNull` binds one sub-pattern at the element
    /// type; `Null` binds nothing.
    fn check_nullable_pat(
        &mut self,
        ctor: &surface::CtorPat,
        span: Option<Span>,
        ty: Ty<'tcx>,
    ) -> Pat<'tcx> {
        // The scrutinee must be (or become) a nullable; a still-unsolved hole
        // is solved to `Nullable<α>` so the sub-pattern can constrain `α`.
        let inner = match ty.kind() {
            TyKind::Nullable(inner) => self.infer.shallow_resolve(*inner),
            TyKind::Hole(_) => {
                let elem = self.infer.new_hole_ty();
                let want = self.tcx.mk_nullable(elem);
                let _ = self.infer.unify(ty, want);
                elem
            }
            _ => {
                let shown = self.infer.resolve(ty);
                self.error(
                    span,
                    format!(
                        "cannot match a nullable pattern against `{}`",
                        self.ty_display(shown)
                    ),
                );
                return Pat::Wild;
            }
        };
        match self.sym(ctor.path.basename) {
            "NonNull" => {
                if ctor.args.len() != 1 {
                    self.error(
                        span,
                        format!(
                            "`Nullable::NonNull` binds exactly 1 value, but the \
                             pattern binds {}",
                            ctor.args.len()
                        ),
                    );
                }
                let field = match ctor.args.first() {
                    Some(arg) => self.check_pat(&arg.kind, span, inner),
                    None => Pat::Wild,
                };
                Pat::Ctor {
                    ctor: Ctor::NonNull,
                    fields: vec![field],
                }
            }
            "Null" => {
                if !ctor.args.is_empty() {
                    self.error(span, "`Nullable::Null` binds no values");
                }
                Pat::Ctor {
                    ctor: Ctor::Null,
                    fields: Vec::new(),
                }
            }
            other => {
                self.error(span, format!("unknown nullable constructor `{other}`"));
                Pat::Wild
            }
        }
    }

    // ----- the matrix compiler -----

    fn compile_match(
        &mut self,
        m: Matrix<'tcx>,
        reached: &mut [bool],
        exhaustive: &mut bool,
    ) -> DecisionTree<'tcx> {
        // (1) No rows can match here: a gap in coverage.
        let Some(first) = m.rows.first() else {
            *exhaustive = false;
            return DecisionTree::Uncovered;
        };

        // (2) The first row tests no constructors: it matches (up to its guard).
        if !first.pats.iter().any(Pat::is_ctor) {
            reached[first.arm] = true;
            // Binders accumulated earlier, plus any still sitting in columns.
            let mut bindings = first.bindings.clone();
            for (col, pat) in m.cols.iter().zip(&first.pats) {
                if let Pat::Bind(v) = pat {
                    bindings.push((*v, col.occ.clone()));
                }
            }
            let body = Box::new(first.body.clone());
            return match &first.guard {
                None => DecisionTree::Leaf { body, bindings },
                Some(guard) => {
                    let guard = Box::new(guard.clone());
                    // If the guard fails, retry the remaining arms on the same
                    // value (their patterns are unchanged: we never specialized).
                    let rest = Matrix {
                        cols: m.cols.clone(),
                        rows: m.rows[1..].to_vec(),
                    };
                    let failure = Box::new(self.compile_match(rest, reached, exhaustive));
                    DecisionTree::Guard {
                        bindings,
                        guard,
                        success: Box::new(DecisionTree::Leaf {
                            body,
                            bindings: Vec::new(),
                        }),
                        failure,
                    }
                }
            };
        }

        // (3) Switch on the leftmost column that still has a constructor.
        let j = m
            .cols
            .iter()
            .enumerate()
            .position(|(c, _)| m.rows.iter().any(|r| r.pats[c].is_ctor()))
            .expect("a constructor column exists when some row tests one");
        let occ = m.cols[j].occ.clone();
        let col_ty = self.infer.shallow_resolve(m.cols[j].ty);
        let family = m
            .rows
            .iter()
            .find_map(|r| match &r.pats[j] {
                Pat::Ctor { ctor, .. } => Some(ctor.family()),
                _ => None,
            })
            .expect("the chosen column has a constructor");

        let cases = match family {
            // Closed: one subtree per variant; an empty variant subtree is an
            // uncovered case (caught by the recursive call returning `Uncovered`).
            Family::Variant => {
                let n = self.variant_count(col_ty);
                let mut subtrees = Vec::with_capacity(n);
                for v in 0..n {
                    let child_tys = self.variant_field_tys(col_ty, v);
                    let sub = self.specialize_variant(&m, j, v, &occ, &child_tys);
                    subtrees.push(self.compile_match(sub, reached, exhaustive));
                }
                SwitchCases::Ctor(subtrees)
            }
            // Closed two-case family.
            Family::Bool => {
                let t = self.specialize_scalar(&m, j, &Ctor::Bool(true), &occ);
                let if_true = Box::new(self.compile_match(t, reached, exhaustive));
                let f = self.specialize_scalar(&m, j, &Ctor::Bool(false), &occ);
                let if_false = Box::new(self.compile_match(f, reached, exhaustive));
                SwitchCases::Bool { if_true, if_false }
            }
            // Closed two-case family. The non-null case exposes one field —
            // the unwrapped element — at the *same* occurrence as the switch
            // itself: the textual-IR arms carry no field index and the
            // codegen dispatch rebinds the scrutinee's path to the unwrapped
            // pointer (unwrapping is a pointer no-op), matching the reference.
            Family::Nullable => {
                let inner = match self.infer.shallow_resolve(col_ty).kind() {
                    TyKind::Nullable(inner) => self.infer.shallow_resolve(*inner),
                    // A non-nullable column type means the pattern arm already
                    // errored in `check_nullable_pat`; recover with Bottom.
                    _ => self.tcx.mk(TyKind::Bottom),
                };
                let nn = self.specialize_nonnull(&m, j, &occ, inner);
                let non_null = Box::new(self.compile_match(nn, reached, exhaustive));
                let nl = self.specialize_scalar(&m, j, &Ctor::Null, &occ);
                let null = Box::new(self.compile_match(nl, reached, exhaustive));
                SwitchCases::Nullable { non_null, null }
            }
            // Open: one case per literal, plus a default for everything else.
            Family::Int => {
                let mut cases = Vec::new();
                for k in self.column_ints(&m, j) {
                    let sub = self.specialize_scalar(&m, j, &Ctor::Int(k), &occ);
                    cases.push((k, self.compile_match(sub, reached, exhaustive)));
                }
                let def = self.default_matrix(&m, j, &occ);
                let default = Box::new(self.compile_match(def, reached, exhaustive));
                SwitchCases::Int { cases, default }
            }
            Family::Char => {
                let mut cases = Vec::new();
                for c in self.column_chars(&m, j) {
                    let sub = self.specialize_scalar(&m, j, &Ctor::Char(c), &occ);
                    cases.push((c, self.compile_match(sub, reached, exhaustive)));
                }
                let def = self.default_matrix(&m, j, &occ);
                let default = Box::new(self.compile_match(def, reached, exhaustive));
                SwitchCases::Char { cases, default }
            }
            Family::Str => {
                let mut cases = Vec::new();
                for s in self.column_strs(&m, j) {
                    let sub = self.specialize_scalar(&m, j, &Ctor::Str(s), &occ);
                    cases.push((s, self.compile_match(sub, reached, exhaustive)));
                }
                let def = self.default_matrix(&m, j, &occ);
                let default = Box::new(self.compile_match(def, reached, exhaustive));
                SwitchCases::String { cases, default }
            }
        };
        DecisionTree::Switch {
            scrutinee: occ,
            cases,
        }
    }

    /// `S(Variant(v), P)`: keep the rows that can match variant `v`, expanding
    /// the matched constructor's fields into fresh columns at `occ ++ [i]`.
    fn specialize_variant(
        &self,
        m: &Matrix<'tcx>,
        j: usize,
        v: usize,
        occ: &PatVarRef,
        child_tys: &[Ty<'tcx>],
    ) -> Matrix<'tcx> {
        let k = child_tys.len();
        let mut cols = m.cols[..j].to_vec();
        for (i, &ty) in child_tys.iter().enumerate() {
            let mut path = occ.0.clone();
            path.push(i as u32);
            cols.push(Column {
                occ: PatVarRef(path),
                ty,
            });
        }
        cols.extend_from_slice(&m.cols[j + 1..]);

        let mut rows = Vec::new();
        for row in &m.rows {
            // What the field columns become, and whether this row's column-`j`
            // pattern bound the whole value.
            let (sub, bind): (Vec<Pat<'tcx>>, Option<(VarId, PatVarRef)>) = match &row.pats[j] {
                Pat::Ctor {
                    ctor: Ctor::Variant(w),
                    fields,
                } if *w == v => (fields.clone(), None),
                // A different variant cannot match this case.
                Pat::Ctor { .. } => continue,
                Pat::Wild => (vec![Pat::Wild; k], None),
                Pat::Bind(var) => (vec![Pat::Wild; k], Some((*var, occ.clone()))),
            };
            let mut pats = row.pats[..j].to_vec();
            pats.extend(sub);
            pats.extend_from_slice(&row.pats[j + 1..]);
            let mut bindings = row.bindings.clone();
            bindings.extend(bind);
            rows.push(Row {
                pats,
                bindings,
                guard: row.guard.clone(),
                body: row.body.clone(),
                arm: row.arm,
            });
        }
        Matrix { cols, rows }
    }

    /// `S(NonNull, P)`: keep the rows that can match the non-null case,
    /// expanding the single unwrapped-element field into a fresh column at the
    /// *same* occurrence `occ` (see the `Family::Nullable` arm — unwrapping a
    /// non-null is a pointer no-op, so the payload occupies the scrutinee's
    /// own path, unlike a variant's `occ ++ [i]` fields).
    fn specialize_nonnull(
        &self,
        m: &Matrix<'tcx>,
        j: usize,
        occ: &PatVarRef,
        inner: Ty<'tcx>,
    ) -> Matrix<'tcx> {
        let mut cols = m.cols[..j].to_vec();
        cols.push(Column {
            occ: occ.clone(),
            ty: inner,
        });
        cols.extend_from_slice(&m.cols[j + 1..]);

        let mut rows = Vec::new();
        for row in &m.rows {
            let (sub, bind): (Pat<'tcx>, Option<(VarId, PatVarRef)>) = match &row.pats[j] {
                Pat::Ctor {
                    ctor: Ctor::NonNull,
                    fields,
                } => (fields[0].clone(), None),
                // A `Null` pattern cannot match this case.
                Pat::Ctor { .. } => continue,
                Pat::Wild => (Pat::Wild, None),
                // Binding the whole nullable: within the non-null case the
                // nullable and its unwrapped element are the same pointer, so
                // the binder's occurrence stays `occ` like everything else.
                Pat::Bind(var) => (Pat::Wild, Some((*var, occ.clone()))),
            };
            let mut pats = row.pats[..j].to_vec();
            pats.push(sub);
            pats.extend_from_slice(&row.pats[j + 1..]);
            let mut bindings = row.bindings.clone();
            bindings.extend(bind);
            rows.push(Row {
                pats,
                bindings,
                guard: row.guard.clone(),
                body: row.body.clone(),
                arm: row.arm,
            });
        }
        Matrix { cols, rows }
    }

    /// `S(c, P)` for a field-less constructor (int/bool/string): keep the rows
    /// matching `c` plus the wildcard rows, dropping the tested column.
    fn specialize_scalar(
        &self,
        m: &Matrix<'tcx>,
        j: usize,
        c: &Ctor<'tcx>,
        occ: &PatVarRef,
    ) -> Matrix<'tcx> {
        let cols = cols_without(&m.cols, j);
        let mut rows = Vec::new();
        for row in &m.rows {
            match &row.pats[j] {
                Pat::Ctor { ctor, .. } if ctor == c => rows.push(row.without_col(j, None)),
                Pat::Ctor { .. } => {}
                Pat::Wild => rows.push(row.without_col(j, None)),
                Pat::Bind(var) => rows.push(row.without_col(j, Some((*var, occ.clone())))),
            }
        }
        Matrix { cols, rows }
    }

    /// `D(P)`: the rows that ignore column `j` (wildcards/binders), dropping it.
    /// This is the body of the default branch for an open family.
    fn default_matrix(&self, m: &Matrix<'tcx>, j: usize, occ: &PatVarRef) -> Matrix<'tcx> {
        let cols = cols_without(&m.cols, j);
        let mut rows = Vec::new();
        for row in &m.rows {
            match &row.pats[j] {
                Pat::Ctor { .. } => {}
                Pat::Wild => rows.push(row.without_col(j, None)),
                Pat::Bind(var) => rows.push(row.without_col(j, Some((*var, occ.clone())))),
            }
        }
        Matrix { cols, rows }
    }

    /// The distinct integer literals tested in column `j`, in first-seen order.
    fn column_ints(&self, m: &Matrix<'tcx>, j: usize) -> Vec<&'tcx Integer> {
        let mut seen = Vec::new();
        for row in &m.rows {
            if let Pat::Ctor {
                ctor: Ctor::Int(k), ..
            } = &row.pats[j]
                && !seen.contains(k)
            {
                seen.push(*k);
            }
        }
        seen
    }

    /// The distinct character literals tested in column `j`, in first-seen order.
    fn column_chars(&self, m: &Matrix<'tcx>, j: usize) -> Vec<u32> {
        let mut seen = Vec::new();
        for row in &m.rows {
            if let Pat::Ctor {
                ctor: Ctor::Char(c),
                ..
            } = &row.pats[j]
                && !seen.contains(c)
            {
                seen.push(*c);
            }
        }
        seen
    }

    /// The distinct string literals tested in column `j`, in first-seen order.
    fn column_strs(&self, m: &Matrix<'tcx>, j: usize) -> Vec<StringToken> {
        let mut seen = Vec::new();
        for row in &m.rows {
            if let Pat::Ctor {
                ctor: Ctor::Str(s), ..
            } = &row.pats[j]
                && !seen.contains(s)
            {
                seen.push(*s);
            }
        }
        seen
    }

    /// The field types of variant `v` of the enum at `ty`, instantiated with the
    /// enum's type arguments.
    fn variant_field_tys(&mut self, ty: Ty<'tcx>, v: usize) -> Vec<Ty<'tcx>> {
        let ty = self.peel_arc(ty);
        let TyKind::Record { def, args, .. } = ty.kind() else {
            return Vec::new();
        };
        let Some(record) = self.records.get(def).cloned() else {
            return Vec::new();
        };
        let Some(RecordFields::Variants(variants)) = &record.fields else {
            return Vec::new();
        };
        let Some(var) = variants.get(v) else {
            return Vec::new();
        };
        let payload = var.fields.clone();
        let inst = Instantiation::from_pairs(
            record
                .ty_params
                .iter()
                .map(|(_, g)| *g)
                .zip(args.iter().copied()),
        );
        payload
            .iter()
            .map(|t| self.infer.instantiate_ty(*t, &inst))
            .collect()
    }

    fn variant_count(&mut self, ty: Ty<'tcx>) -> usize {
        let ty = self.peel_arc(ty);
        if let TyKind::Record { def, .. } = ty.kind()
            && let Some(record) = self.records.get(def)
            && let Some(RecordFields::Variants(vs)) = &record.fields
        {
            return vs.len();
        }
        0
    }

    // ----- zonking -----

    pub(super) fn zonk_tree(&mut self, tree: DecisionTree<'tcx>) -> DecisionTree<'tcx> {
        match tree {
            DecisionTree::Uncovered => DecisionTree::Uncovered,
            DecisionTree::Unreachable => DecisionTree::Unreachable,
            DecisionTree::Leaf { body, bindings } => DecisionTree::Leaf {
                body: Box::new(self.zonk_expr(*body)),
                bindings,
            },
            DecisionTree::Guard {
                bindings,
                guard,
                success,
                failure,
            } => DecisionTree::Guard {
                bindings,
                guard: Box::new(self.zonk_expr(*guard)),
                success: Box::new(self.zonk_tree(*success)),
                failure: Box::new(self.zonk_tree(*failure)),
            },
            DecisionTree::Switch { scrutinee, cases } => DecisionTree::Switch {
                scrutinee,
                cases: self.zonk_cases(cases),
            },
        }
    }

    fn zonk_cases(&mut self, cases: SwitchCases<'tcx>) -> SwitchCases<'tcx> {
        match cases {
            SwitchCases::Int { cases, default } => SwitchCases::Int {
                cases: cases
                    .into_iter()
                    .map(|(k, t)| (k, self.zonk_tree(t)))
                    .collect(),
                default: Box::new(self.zonk_tree(*default)),
            },
            SwitchCases::Bool { if_true, if_false } => SwitchCases::Bool {
                if_true: Box::new(self.zonk_tree(*if_true)),
                if_false: Box::new(self.zonk_tree(*if_false)),
            },
            SwitchCases::Char { cases, default } => SwitchCases::Char {
                cases: cases
                    .into_iter()
                    .map(|(k, t)| (k, self.zonk_tree(t)))
                    .collect(),
                default: Box::new(self.zonk_tree(*default)),
            },
            SwitchCases::Ctor(ts) => {
                SwitchCases::Ctor(ts.into_iter().map(|t| self.zonk_tree(t)).collect())
            }
            SwitchCases::String { cases, default } => SwitchCases::String {
                cases: cases
                    .into_iter()
                    .map(|(k, t)| (k, self.zonk_tree(t)))
                    .collect(),
                default: Box::new(self.zonk_tree(*default)),
            },
            SwitchCases::Nullable { non_null, null } => SwitchCases::Nullable {
                non_null: Box::new(self.zonk_tree(*non_null)),
                null: Box::new(self.zonk_tree(*null)),
            },
        }
    }
}

/// `cols` with index `j` removed.
fn cols_without<'tcx>(cols: &[Column<'tcx>], j: usize) -> Vec<Column<'tcx>> {
    let mut out = cols[..j].to_vec();
    out.extend_from_slice(&cols[j + 1..]);
    out
}

#[cfg(test)]
mod tests {
    use crate::semi::elaborate;
    use crate::surface;
    use crate::with_tcx;

    /// Elaborate a source string and return its diagnostics.
    fn reports_of(source: &str) -> Vec<crate::semi::Report> {
        with_tcx(|tcx| {
            let interner = std::sync::Arc::new(reussir_syntax::new_threaded_interner());
            let parse = reussir_syntax::parse_with_interner(source, interner.clone());
            assert!(parse.ok(), "parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let elab = elaborate(tcx, &prog, &interner);
            elab.reports.clone()
        })
    }

    fn has_message(source: &str, needle: &str) -> bool {
        reports_of(source)
            .iter()
            .any(|r| r.message.contains(needle))
    }

    const COLOR: &str = "enum Color { Red, Green }\nenum Opt { None, Some(Color) }\n";

    #[test]
    fn nested_refutable_patterns_compile() {
        // `Opt::Some(Color::Red)` nests a refutable pattern inside a variant —
        // rejected by the old single-column compiler, exhaustive here.
        let src = format!(
            "{COLOR}fn f(x: Opt) -> i32 {{ match x {{\n\
             Opt::None => 0,\n\
             Opt::Some(Color::Red) => 1,\n\
             Opt::Some(Color::Green) => 2\n\
             }} }}"
        );
        assert!(reports_of(&src).is_empty(), "{:#?}", reports_of(&src));
    }

    #[test]
    fn missing_nested_case_is_non_exhaustive() {
        // `Opt::Some(Color::Green)` and `Opt::None` are uncovered.
        let src = format!(
            "{COLOR}fn f(x: Opt) -> i32 {{ match x {{\n\
             Opt::Some(Color::Red) => 1\n\
             }} }}"
        );
        assert!(
            has_message(&src, "non-exhaustive"),
            "{:#?}",
            reports_of(&src)
        );
    }

    #[test]
    fn full_variant_coverage_is_exhaustive() {
        let src = format!(
            "{COLOR}fn f(c: Color) -> i32 {{ match c {{\n\
             Color::Red => 1,\n\
             Color::Green => 2\n\
             }} }}"
        );
        assert!(reports_of(&src).is_empty(), "{:#?}", reports_of(&src));
    }

    #[test]
    fn arm_after_wildcard_is_unreachable() {
        let src = format!(
            "{COLOR}fn f(c: Color) -> i32 {{ match c {{\n\
             x => 1,\n\
             Color::Red => 2\n\
             }} }}"
        );
        assert!(has_message(&src, "unreachable"), "{:#?}", reports_of(&src));
    }

    #[test]
    fn guards_are_supported() {
        // A guarded arm plus a wildcard fallback: exhaustive, no warnings.
        let src = "fn f(b: bool) -> i32 { match b {\n\
                   x if x => 1,\n\
                   _ => 2\n\
                   } }";
        assert!(reports_of(src).is_empty(), "{:#?}", reports_of(src));
    }

    #[test]
    fn a_lone_guarded_arm_is_non_exhaustive() {
        // The guard may fail, so the match is not exhaustive.
        let src = "fn f(b: bool) -> i32 { match b { x if x => 1 } }";
        assert!(has_message(src, "non-exhaustive"), "{:#?}", reports_of(src));
    }

    #[test]
    fn integer_match_needs_a_default() {
        let src = "fn f(n: i32) -> i32 { match n { 0 => 10, 1 => 11 } }";
        assert!(has_message(src, "non-exhaustive"), "{:#?}", reports_of(src));
    }

    #[test]
    fn integer_match_with_wildcard_is_exhaustive() {
        let src = "fn f(n: i32) -> i32 { match n { 0 => 10, _ => 11 } }";
        assert!(reports_of(src).is_empty(), "{:#?}", reports_of(src));
    }

    #[test]
    fn non_i32_integer_match_typechecks() {
        // An `i64` scrutinee matched against integer-literal arms must not be
        // forced to `i32`. Before the fix, `check_const_pat` unified the column
        // type with `i32`, so this reported a spurious type mismatch.
        let src = "fn f(n: i64) -> i64 { match n { 0 => 10, _ => 11 } }";
        assert!(reports_of(src).is_empty(), "{:#?}", reports_of(src));
    }
}
