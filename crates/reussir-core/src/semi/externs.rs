//! Extern packages joining the consumer's elaboration: the declare-only twin
//! of `run_package`'s scan pass, over a parsed dependency interface.
//!
//! A [`Parsed`] interface carries its own [`Names`](crate::semi::hir::build::Names)
//! interner and dense
//! [`DefTable`](crate::semi::resolve::DefTable) — nothing in it is keyed in
//! the consumer's token or def spaces. Declaring an extern package therefore
//! re-interns every name through the consumer's interner, maps every def to a
//! consumer `DefId` under the same qualified path (paths are the stable
//! cross-package identity), re-allocates generics in the elaborator's global
//! table, and rebuilds every type over the mapped defs. Prototypes and
//! records land in the same `functions`/`records` tables the checker reads;
//! bodies, strings, and ffi textures land in the `extern_*` side tables
//! cross-package monomorphization consumes (`MonoInput::externs`) so they
//! stay out of the consumer's printed program and export surface. Resolution reaches the
//! declared items only through the package head, gated on `pub`
//! (`ctxt::resolve_extern`).

use reussir_syntax::Interner;
use reussir_syntax::kind::{InternKey, TokenKey};
use reussir_syntax::source::FileId;
use rustc_hash::FxHashMap;

use crate::semi::ctxt::{Elaborator, FuncProto, Record, RecordFields, Variant};
use crate::semi::hir::build::Parsed;
use crate::semi::hir::{ClosureExpr, DecisionTree, Expr, ExprKind, Function, SwitchCases};
use crate::semi::resolve::DefKind;
use crate::semi::traits::coherence::{head_key, unify_args};
use crate::semi::traits::def::MethodSig;
use crate::semi::traits::{ImplDef, ImplId, Obligation, TraitDef, TraitId, TraitRef};
use crate::semi::ty::{DefId, GenericId, Ty, TyCtxt, TyKind};

/// One loaded dependency interface, ready to join elaboration: its `--extern`
/// name (the head of every declared path) and the parsed tables. Header
/// gating (format, package name, reserved names) is the driver's job.
pub struct ExternPackage<'p, 'tcx> {
    pub name: &'p str,
    pub parsed: &'p Parsed<'tcx>,
    /// Consumer-cache [`FileId`] per entry of the interface's own file table
    /// (dense, indexed by producer file index). The driver appends the
    /// extern's re-anchored file table to the consumer's source cache and
    /// passes the resulting ids here, so every declared item's `file` lands
    /// in the consumer's id space — reports raised inside an imported body
    /// (cross-package monomorphization) and the debug locations of locally
    /// emitted instances then resolve against the dependency's sources.
    /// Empty when extern locations can never render (text-dump targets stop
    /// before monomorphization); file fields then keep producer-relative ids
    /// nothing reads.
    pub files: &'p [FileId],
}

impl<'tcx> Elaborator<'_, 'tcx> {
    /// Declare `ext`'s items into the consumer's tables, under paths rooted
    /// at the extern package name. Must run before the consumer's own items
    /// are declared; `interner` must back the elaborator's own resolver
    /// (cross-space resolution compares interned keys).
    pub fn declare_extern_package(
        &mut self,
        interner: &mut impl Interner<TokenKey>,
        ext: &ExternPackage<'_, 'tcx>,
    ) {
        let head = interner.get_or_intern(ext.name);
        self.extern_heads.insert(head);
        let parsed = ext.parsed;
        // Producer-relative file ids → consumer-cache ids (spans are byte
        // offsets into the same content, so they carry over untouched). An
        // id outside the map (no map given, or a table-less interface) stays
        // as-is: nothing renders it.
        let file_of = |old: FileId| ext.files.get(old.index()).copied().unwrap_or(old);

        // One consumer key per producer key, densely indexed, so the remap
        // below never threads the interner.
        let keys: Vec<TokenKey> = parsed
            .names
            .entries()
            .iter()
            .map(|s| interner.get_or_intern(s))
            .collect();

        // Map every producer def to a consumer def under its re-interned
        // qualified path — lookup-first, so a path an earlier extern already
        // declared unifies instead of clashing.
        let module = self.defs.module().to_vec();
        let defs: Vec<DefId> = (0..parsed.defs.len())
            .map(|i| {
                let info = parsed.defs.info(DefId(i as u32));
                let segs: Vec<TokenKey> = info
                    .path
                    .0
                    .iter()
                    .map(|k| keys[k.into_u32() as usize])
                    .collect();
                let (name, module) = segs.split_last().expect("paths are never empty");
                let existing = match info.kind {
                    DefKind::Record => self.defs.lookup_record(&segs),
                    DefKind::Function => self.defs.lookup_function(&segs),
                    DefKind::Trait => self.defs.lookup_trait(&segs),
                };
                let def = existing.unwrap_or_else(|| {
                    self.defs.set_module(module.to_vec());
                    match info.kind {
                        DefKind::Record => self.defs.declare_record(*name),
                        DefKind::Function => self.defs.declare_function(*name),
                        DefKind::Trait => self.defs.declare_trait(*name),
                    }
                    .expect("a missed lookup cannot clash")
                });
                self.extern_defs.entry(def).or_insert(head);
                def
            })
            .collect();
        self.defs.set_module(module);

        for (pdef, rec) in &parsed.records {
            let def = defs[pdef.0 as usize];
            let generics =
                self.remap_generics(&rec.ty_params, &keys, &parsed.generic_bounds, interner);
            let remap = Remapper {
                tcx: self.tcx,
                defs: &defs,
                keys: &keys,
                generics: &generics.map,
            };
            let record = Record {
                def,
                name: remap.key(rec.name),
                visibility: rec.visibility,
                ty_params: generics.binder,
                kind: rec.kind,
                default_cap: rec.default_cap,
                repr_fixed: rec.repr_fixed,
                ffi: rec.ffi.clone(),
                fields: rec.fields.as_ref().map(|f| remap.fields(f)),
                regional_generics: remap.generic_ids(&rec.regional_generics),
                span: rec.span,
                file: file_of(rec.file),
            };
            self.records.entry(def).or_insert(record);
        }

        // Trait items: rebuild each declaration in consumer space and
        // register it in the session TraitDb — so bounds on imported
        // binders resolve (the funcs loop below reads them), methods
        // dispatch, and coherence sees the dependency's impls. Two passes,
        // because a super-trait may be declared later in the dump than its
        // sub-trait (exactly like the elaborator's stub/populate split).
        // Registration is idempotent: a def-unified trait (the same package
        // loaded twice, or a transitive re-ship) keeps its first
        // registration.
        let mut pending: Vec<(
            TraitId,
            &crate::semi::hir::TraitText<'tcx>,
            FxHashMap<GenericId, GenericId>,
        )> = Vec::new();
        for t in &parsed.traits {
            let def = defs[t.def.0 as usize];
            if self.traits.trait_by_def(def).is_some() {
                continue;
            }
            let mut ids = FxHashMap::default();
            let binder =
                self.remap_text_generics(&t.generics, &parsed.generic_bounds, interner, &mut ids);
            let (&(_, self_param), params) = binder
                .split_first()
                .expect("trait binders carry `Self` first");
            let id = TraitId(self.traits.traits_len() as u32);
            self.traits.add_trait(TraitDef {
                id,
                def,
                visibility: if t.is_pub {
                    crate::surface::Visibility::Public
                } else {
                    crate::surface::Visibility::Private
                },
                sealed: t.sealed,
                self_param,
                params: params.to_vec(),
                supertraits: Vec::new(),
                methods: Vec::new(),
                assoc_tys: Vec::new(),
                span: t.span,
                file: file_of(t.file),
            });
            pending.push((id, t, ids));
        }
        for (id, t, ids) in pending {
            let supertraits: Vec<TraitRef<'tcx>> = t
                .supers
                .iter()
                .filter_map(|(path, args)| {
                    let segs: Vec<TokenKey> = path
                        .split("::")
                        .map(|s| interner.get_or_intern(s))
                        .collect();
                    let sup = self
                        .defs
                        .lookup_trait(&segs)
                        .and_then(|d| self.traits.trait_by_def(d));
                    if sup.is_none() {
                        self.error(
                            None,
                            format!(
                                "package `{}` names an unknown super-trait `{path}`; \
                                 the dependency set is inconsistent",
                                ext.name
                            ),
                        );
                    }
                    let remap = Remapper {
                        tcx: self.tcx,
                        defs: &defs,
                        keys: &keys,
                        generics: &ids,
                    };
                    Some(TraitRef {
                        trait_id: sup?,
                        args: args.iter().map(|&a| remap.ty(a)).collect(),
                    })
                })
                .collect();
            let methods: Vec<MethodSig<'tcx>> = t
                .methods
                .iter()
                .map(|m| {
                    let mut mids = ids.clone();
                    let generics = self.remap_text_generics(
                        &m.generics,
                        &parsed.generic_bounds,
                        interner,
                        &mut mids,
                    );
                    let remap = Remapper {
                        tcx: self.tcx,
                        defs: &defs,
                        keys: &keys,
                        generics: &mids,
                    };
                    MethodSig {
                        name: interner.get_or_intern(&m.name),
                        generics,
                        receiver: m.receiver,
                        params: m.params.iter().map(|&p| remap.ty(p)).collect(),
                        ret: remap.ty(m.ret),
                        is_regional: m.regional,
                        span: m.span,
                    }
                })
                .collect();
            let def = self.traits.trait_def_mut(id);
            def.supertraits = supertraits;
            def.methods = methods;
        }

        // Impl items. An impl identical to a registered one (same trait and
        // the same method defs — paths unify across loads) is a re-ship and
        // is skipped; a genuinely overlapping one is a load-time coherence
        // error.
        for i in &parsed.impls {
            let tdef = defs[i.trait_def.0 as usize];
            let Some(tid) = self.traits.trait_by_def(tdef) else {
                self.error(
                    None,
                    format!(
                        "package `{}` ships an impl of `{}`, which it does not declare; \
                         the dependency set is inconsistent",
                        ext.name, i.trait_path
                    ),
                );
                continue;
            };
            let mut ids = FxHashMap::default();
            let binder =
                self.remap_text_generics(&i.generics, &parsed.generic_bounds, interner, &mut ids);
            let remap = Remapper {
                tcx: self.tcx,
                defs: &defs,
                keys: &keys,
                generics: &ids,
            };
            let args: Vec<Ty<'tcx>> = i.args.iter().map(|&a| remap.ty(a)).collect();
            let self_ty = args[0];
            let methods: Vec<DefId> = i
                .methods
                .iter()
                .filter_map(|p| {
                    let segs: Vec<TokenKey> =
                        p.split("::").map(|s| interner.get_or_intern(s)).collect();
                    self.defs.lookup_function(&segs)
                })
                .collect();
            let head = head_key(self_ty).expect("shipped impls have ground heads");
            let duplicate = self.traits.impls_for(tid, head).iter().any(|&e| {
                let other = self.traits.impl_def(e);
                other.methods == methods && (!methods.is_empty() || other.trait_ref.args == args)
            });
            if duplicate {
                continue;
            }
            let mut conflict = false;
            for &e in self.traits.impls_for(tid, head) {
                let other = self.traits.impl_def(e);
                let mut vars: rustc_hash::FxHashSet<GenericId> =
                    other.generics.iter().copied().collect();
                vars.extend(binder.iter().map(|&(_, g)| g));
                let mut subst = FxHashMap::default();
                if unify_args(&other.trait_ref.args, &args, &vars, &mut subst) {
                    conflict = true;
                    break;
                }
            }
            if conflict {
                let shown = self.trait_display(tid);
                let ty = self.ty_display(self_ty);
                self.error(
                    None,
                    format!(
                        "conflicting implementations of trait `{shown}` for `{ty}`: \
                         an impl loaded from package `{}` overlaps a registered one",
                        ext.name
                    ),
                );
                continue;
            }
            let where_clauses: Vec<Obligation<'tcx>> = binder
                .iter()
                .flat_map(|&(_, g)| {
                    self.generic_bounds(g)
                        .iter()
                        .map(|&b| {
                            Obligation::Trait(TraitRef {
                                trait_id: b,
                                args: vec![self.tcx.mk_generic(g)],
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            let id = ImplId(self.traits.impls_len() as u32);
            self.traits.add_impl(ImplDef {
                id,
                generics: binder.iter().map(|&(_, g)| g).collect(),
                trait_ref: TraitRef {
                    trait_id: tid,
                    args,
                },
                self_ty,
                where_clauses,
                methods,
                span: i.span,
                file: file_of(i.file),
            });
        }

        // Lang markers: bind the loaded declarations into the session
        // registry (kind re-derived from what the def actually is here).
        for (&pdef, &item) in &parsed.lang {
            let def = defs[pdef.0 as usize];
            let actual = if self.traits.trait_by_def(def).is_some() {
                crate::semi::lang::LangItemKind::Trait
            } else if self.records.contains_key(&def) {
                crate::semi::lang::LangItemKind::Record
            } else {
                crate::semi::lang::LangItemKind::Function
            };
            self.declare_lang(item, def, actual, None);
        }

        for func in &parsed.funcs {
            let def = defs[func.def.0 as usize];
            let generics =
                self.remap_generics(&func.generics, &keys, &parsed.generic_bounds, interner);
            let remap = Remapper {
                tcx: self.tcx,
                defs: &defs,
                keys: &keys,
                generics: &generics.map,
            };
            let function = Function {
                def,
                name: remap.key(func.name),
                visibility: func.visibility,
                generics: generics.binder.clone(),
                regional_generics: remap.generic_ids(&func.regional_generics),
                params: func
                    .params
                    .iter()
                    .map(|&(name, var, ty)| (remap.key(name), var, remap.ty(ty)))
                    .collect(),
                return_ty: remap.ty(func.return_ty),
                is_regional: func.is_regional,
                body: func.body.as_ref().map(|b| remap.expr(b)),
                span: func.span,
                file: file_of(func.file),
            };
            self.functions.entry(def).or_insert_with(|| FuncProto {
                def,
                name: function.name,
                visibility: function.visibility,
                generics: generics.binder,
                regional_generics: function.regional_generics.clone(),
                params: function
                    .params
                    .iter()
                    .map(|&(name, _, ty)| (name, ty))
                    .collect(),
                return_ty: function.return_ty,
                impl_module: None,
                self_ty: None,
                is_regional: function.is_regional,
                span: function.span,
                file: function.file,
            });
            if let Some(import) = parsed.ffi_imports.get(&func.def) {
                let mut import = import.clone();
                import.file = file_of(import.file);
                self.extern_ffi_imports.insert(def, import);
            }
            self.extern_functions.push(function);
        }

        self.extern_strings.extend(parsed.strings.iter().cloned());
        // Preludes pair with imports by file, so both sides remap together.
        self.extern_ffi_preludes
            .extend(parsed.ffi_preludes.iter().map(|prelude| {
                let mut prelude = prelude.clone();
                prelude.file = file_of(prelude.file);
                prelude
            }));
        // Transform metadata of shipped bodies is not carried yet —
        // cross-package monomorphization decides its shape (deferred).
    }

    /// Re-allocate an extern item's generic binder in the global table,
    /// resolving the interface's serialized bound paths (by basename, against
    /// the builtin trait table) into real `TraitId`s. `instantiate` then
    /// registers the bounds as obligations at every consumer call of an
    /// extern generic, so a bad instantiation reports at the consumer's call
    /// site instead of inside the imported body at monomorphization.
    ///
    /// A bound naming an unknown trait (a version-skewed dependency) reports
    /// through the shared "unknown trait bound" diagnostic and the binder
    /// degrades to the resolvable subset; shipped bodies are *elaborated*
    /// HIR, so this can never miscompile — the residue still grounds into an
    /// operation monomorphization rejects with a spanned report.
    /// [`Self::remap_generics`] over a text-view binder (`(id, name?)`
    /// pairs): re-allocate in the global table, resolving serialized bound
    /// paths, and extend `out` with the producer→consumer id mapping.
    fn remap_text_generics(
        &mut self,
        binder: &[(GenericId, Option<String>)],
        bounds: &FxHashMap<GenericId, Vec<crate::semi::hir::Bound>>,
        interner: &mut impl Interner<TokenKey>,
        out: &mut FxHashMap<GenericId, GenericId>,
    ) -> Vec<(TokenKey, GenericId)> {
        binder
            .iter()
            .map(|&(old, ref name)| {
                let tids: Vec<_> = bounds
                    .get(&old)
                    .map(|bs| {
                        bs.iter()
                            .filter_map(|bound| {
                                let segs: Vec<TokenKey> = bound
                                    .trait_path()
                                    .segments
                                    .iter()
                                    .map(|s| interner.get_or_intern(s))
                                    .collect();
                                self.resolve_bound_segments(&segs, None)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let key = interner.get_or_intern(name.as_deref().unwrap_or("_"));
                let fresh = self.fresh_generic(key, tids);
                out.insert(old, fresh);
                (key, fresh)
            })
            .collect()
    }

    fn remap_generics(
        &mut self,
        binder: &[(TokenKey, GenericId)],
        keys: &[TokenKey],
        bounds: &FxHashMap<GenericId, Vec<crate::semi::hir::Bound>>,
        interner: &mut impl Interner<TokenKey>,
    ) -> RemappedGenerics {
        let mut map = FxHashMap::default();
        let binder = binder
            .iter()
            .map(|&(name, old)| {
                let tids: Vec<_> = bounds
                    .get(&old)
                    .map(|bs| {
                        bs.iter()
                            .filter_map(|bound| {
                                // Serialized paths resolve absolutely: one
                                // segment is a crate-root name, where the
                                // builtins live.
                                let segs: Vec<TokenKey> = bound
                                    .trait_path()
                                    .segments
                                    .iter()
                                    .map(|s| interner.get_or_intern(s))
                                    .collect();
                                self.resolve_bound_segments(&segs, None)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let name = keys[name.into_u32() as usize];
                let fresh = self.fresh_generic(name, tids);
                map.insert(old, fresh);
                (name, fresh)
            })
            .collect();
        RemappedGenerics { binder, map }
    }
}

/// An extern item's generic binder after re-allocation: the consumer-space
/// binder list and the producer→consumer id map the type remap reads.
struct RemappedGenerics {
    binder: Vec<(TokenKey, GenericId)>,
    map: FxHashMap<GenericId, GenericId>,
}

/// Per-item remapping state: producer→consumer def and key tables plus the
/// item's generic re-numbering. Rebuilds types through the shared `TyCtxt`;
/// everything else copies structurally.
struct Remapper<'m, 'a, 'tcx> {
    tcx: &'a TyCtxt<'tcx>,
    defs: &'m [DefId],
    keys: &'m [TokenKey],
    generics: &'m FxHashMap<GenericId, GenericId>,
}

impl<'tcx> Remapper<'_, '_, 'tcx> {
    fn def(&self, def: DefId) -> DefId {
        self.defs[def.0 as usize]
    }

    fn key(&self, key: TokenKey) -> TokenKey {
        self.keys[key.into_u32() as usize]
    }

    fn generic(&self, id: GenericId) -> GenericId {
        *self
            .generics
            .get(&id)
            .expect("generic bound by the item's binder")
    }

    fn generic_ids(&self, ids: &[GenericId]) -> Vec<GenericId> {
        ids.iter().map(|&g| self.generic(g)).collect()
    }

    fn ty(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match *ty.kind() {
            TyKind::Record { def, args, flex } => {
                let args: Vec<Ty<'tcx>> = args.iter().map(|&a| self.ty(a)).collect();
                self.tcx.mk_record(self.def(def), &args, flex)
            }
            TyKind::Generic(g) => self.tcx.mk_generic(self.generic(g)),
            TyKind::Nullable(inner) => {
                let inner = self.ty(inner);
                self.tcx.mk_nullable(inner)
            }
            TyKind::Arc(inner) => {
                let inner = self.ty(inner);
                self.tcx.mk_arc(inner)
            }
            TyKind::Cell { elem, kind } => {
                let elem = self.ty(elem);
                self.tcx.mk_cell(elem, kind)
            }
            TyKind::Array { elem, dims } => {
                let elem = self.ty(elem);
                self.tcx.mk_array(elem, dims)
            }
            TyKind::Closure { params, ret } => {
                let params: Vec<Ty<'tcx>> = params.iter().map(|&p| self.ty(p)).collect();
                let ret = self.ty(ret);
                self.tcx.mk_closure(&params, ret)
            }
            // Scalars carry no defs or generics; holes cannot occur in a
            // fully elaborated interface.
            _ => ty,
        }
    }

    fn tys(&self, tys: &[Ty<'tcx>]) -> Vec<Ty<'tcx>> {
        tys.iter().map(|&t| self.ty(t)).collect()
    }

    fn fields(&self, fields: &RecordFields<'tcx>) -> RecordFields<'tcx> {
        match fields {
            RecordFields::Named(fs) => RecordFields::Named(
                fs.iter()
                    .map(|&(name, ty, is_mut, vis)| (self.key(name), self.ty(ty), is_mut, vis))
                    .collect(),
            ),
            RecordFields::Unnamed(fs) => RecordFields::Unnamed(
                fs.iter()
                    .map(|&(ty, is_mut, vis)| (self.ty(ty), is_mut, vis))
                    .collect(),
            ),
            RecordFields::Variants(vs) => RecordFields::Variants(
                vs.iter()
                    .map(|v| Variant {
                        name: self.key(v.name),
                        fields: self.tys(&v.fields),
                    })
                    .collect(),
            ),
            RecordFields::Opaque => RecordFields::Opaque,
        }
    }

    fn boxed(&self, expr: &Expr<'tcx>) -> Box<Expr<'tcx>> {
        Box::new(self.expr(expr))
    }

    fn exprs(&self, exprs: &[Expr<'tcx>]) -> Vec<Expr<'tcx>> {
        exprs.iter().map(|e| self.expr(e)).collect()
    }

    fn expr(&self, expr: &Expr<'tcx>) -> Expr<'tcx> {
        let kind = match &expr.kind {
            ExprKind::GlobalStr(_)
            | ExprKind::ConstChar(_)
            | ExprKind::ConstInt(_)
            | ExprKind::ConstFloat(_)
            | ExprKind::ConstBool(_)
            | ExprKind::Var(_)
            | ExprKind::Poison => expr.kind.clone(),
            ExprKind::Negate(x) => ExprKind::Negate(self.boxed(x)),
            ExprKind::Not(x) => ExprKind::Not(self.boxed(x)),
            ExprKind::Arith(l, op, r) => ExprKind::Arith(self.boxed(l), *op, self.boxed(r)),
            ExprKind::Cmp(l, op, r) => ExprKind::Cmp(self.boxed(l), *op, self.boxed(r)),
            ExprKind::Cast(x, ty) => ExprKind::Cast(self.boxed(x), self.ty(*ty)),
            ExprKind::If(c, t, f) => ExprKind::If(self.boxed(c), self.boxed(t), self.boxed(f)),
            ExprKind::RegionRun(x) => ExprKind::RegionRun(self.boxed(x)),
            ExprKind::Proj(base, path) => ExprKind::Proj(self.boxed(base), path.clone()),
            ExprKind::Match(scrut, tree) => ExprKind::Match(self.boxed(scrut), self.tree(tree)),
            ExprKind::Assign(dst, field, src) => {
                ExprKind::Assign(self.boxed(dst), *field, self.boxed(src))
            }
            ExprKind::Let {
                var,
                name,
                span,
                value,
            } => ExprKind::Let {
                var: *var,
                name: self.key(*name),
                span: *span,
                value: self.boxed(value),
            },
            ExprKind::Seq(items) => ExprKind::Seq(self.exprs(items)),
            ExprKind::FuncCall {
                target,
                ty_args,
                args,
                regional,
            } => ExprKind::FuncCall {
                target: self.def(*target),
                ty_args: self.tys(ty_args),
                args: self.exprs(args),
                regional: *regional,
            },
            ExprKind::TraitCall {
                trait_def,
                method,
                ty_args,
                args,
            } => ExprKind::TraitCall {
                trait_def: self.def(*trait_def),
                method: *method,
                ty_args: self.tys(ty_args),
                args: self.exprs(args),
            },
            ExprKind::CompoundCall {
                target,
                ty_args,
                args,
            } => ExprKind::CompoundCall {
                target: self.def(*target),
                ty_args: self.tys(ty_args),
                args: self.exprs(args),
            },
            ExprKind::VariantCall {
                target,
                ty_args,
                variant,
                args,
            } => ExprKind::VariantCall {
                target: self.def(*target),
                ty_args: self.tys(ty_args),
                variant: *variant,
                args: self.exprs(args),
            },
            ExprKind::NullableCall(inner) => {
                ExprKind::NullableCall(inner.as_ref().map(|x| self.boxed(x)))
            }
            ExprKind::Intrinsic { op, args } => ExprKind::Intrinsic {
                op: *op,
                args: self.exprs(args),
            },
            ExprKind::ArrayOp { op, args } => ExprKind::ArrayOp {
                op: *op,
                args: self.exprs(args),
            },
            ExprKind::Closure(closure) => ExprKind::Closure(ClosureExpr {
                captures: closure
                    .captures
                    .iter()
                    .map(|&(v, t)| (v, self.ty(t)))
                    .collect(),
                params: closure
                    .params
                    .iter()
                    .map(|&(v, t)| (v, self.ty(t)))
                    .collect(),
                body: self.boxed(&closure.body),
            }),
            ExprKind::ClosureCall { target, args } => ExprKind::ClosureCall {
                target: self.boxed(target),
                args: self.exprs(args),
            },
        };
        Expr {
            kind,
            ty: self.ty(expr.ty),
            span: expr.span,
            id: expr.id,
        }
    }

    fn tree(&self, tree: &DecisionTree<'tcx>) -> DecisionTree<'tcx> {
        match tree {
            DecisionTree::Uncovered => DecisionTree::Uncovered,
            DecisionTree::Unreachable => DecisionTree::Unreachable,
            DecisionTree::Leaf { body, bindings } => DecisionTree::Leaf {
                body: self.boxed(body),
                bindings: bindings.clone(),
            },
            DecisionTree::Guard {
                bindings,
                guard,
                success,
                failure,
            } => DecisionTree::Guard {
                bindings: bindings.clone(),
                guard: self.boxed(guard),
                success: Box::new(self.tree(success)),
                failure: Box::new(self.tree(failure)),
            },
            DecisionTree::Switch { scrutinee, cases } => DecisionTree::Switch {
                scrutinee: scrutinee.clone(),
                cases: self.cases(cases),
            },
        }
    }

    fn cases(&self, cases: &SwitchCases<'tcx>) -> SwitchCases<'tcx> {
        let boxed = |t: &DecisionTree<'tcx>| Box::new(self.tree(t));
        match cases {
            SwitchCases::Int { cases, default } => SwitchCases::Int {
                cases: cases.iter().map(|&(n, ref t)| (n, self.tree(t))).collect(),
                default: boxed(default),
            },
            SwitchCases::Bool { if_true, if_false } => SwitchCases::Bool {
                if_true: boxed(if_true),
                if_false: boxed(if_false),
            },
            SwitchCases::Char { cases, default } => SwitchCases::Char {
                cases: cases.iter().map(|&(c, ref t)| (c, self.tree(t))).collect(),
                default: boxed(default),
            },
            SwitchCases::Ctor(arms) => {
                SwitchCases::Ctor(arms.iter().map(|t| self.tree(t)).collect())
            }
            SwitchCases::String { cases, default } => SwitchCases::String {
                cases: cases.iter().map(|&(s, ref t)| (s, self.tree(t))).collect(),
                default: boxed(default),
            },
            SwitchCases::Nullable { non_null, null } => SwitchCases::Nullable {
                non_null: boxed(non_null),
                null: boxed(null),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reussir_syntax::Interner as _;
    use reussir_syntax::source::FileId;

    use super::*;
    use crate::semi::hir::print::Printer;
    use crate::semi::{PackageFile, elaborate_package};
    use crate::{surface, with_tcx};

    /// Elaborate `src` as a package rooted at `dep`, print its HIR, and
    /// re-parse — the same round trip `--extern` performs. (The reduction and
    /// header gating are the driver's business; declaration sees a `Parsed`.)
    fn dep_interface<'tcx>(tcx: &TyCtxt<'tcx>, src: &str) -> Parsed<'tcx> {
        let interner = Arc::new(reussir_syntax::new_threaded_interner());
        let mut keys = interner.clone();
        let dep = keys.get_or_intern("dep");
        let parse = reussir_syntax::parse_with_interner(src, interner.clone());
        assert!(parse.ok(), "dep parse errors: {:#?}", parse.errors);
        let prog = surface::program(&parse.root);
        let files = [PackageFile {
            file: FileId::ROOT,
            module: vec![dep],
            program: &prog,
        }];
        let elab = elaborate_package(tcx, &files, &interner);
        assert!(!elab.has_errors(), "dep elab errors: {:#?}", elab.reports);
        let strings = elab.strings.entries();
        let bounds = elab.bound_names();
        let (traits, impls) =
            crate::semi::hir::trait_texts(&elab.traits, &elab.defs, elab.resolver);
        let lang = elab.lang.declared_by_def();
        let text = Printer::new(&elab.defs, elab.resolver)
            .with_bounds(&bounds)
            .with_traits(&traits, &impls)
            .with_lang(&lang)
            .program(&elab.elaborated, &strings, &elab.records, &elab.trampolines);
        crate::semi::hir::build::parse_program(tcx, &text).expect("re-parse")
    }

    /// Declare `dep_src`'s printed interface, then elaborate `app_files`
    /// (module path, source) as a package rooted at `app` against it.
    fn check_with_dep(
        dep_src: &str,
        app_files: &[(&[&str], &str)],
        f: impl for<'a, 'tcx> FnOnce(&Elaborator<'a, 'tcx>),
    ) {
        with_tcx(|tcx| {
            let parsed = dep_interface(tcx, dep_src);
            let interner = Arc::new(reussir_syntax::new_threaded_interner());
            let mut keys = interner.clone();
            let app = keys.get_or_intern("app");
            let parses: Vec<_> = app_files
                .iter()
                .map(|(_, src)| {
                    let p = reussir_syntax::parse_with_interner(src, interner.clone());
                    assert!(p.ok(), "app parse errors for {src:?}: {:#?}", p.errors);
                    p
                })
                .collect();
            let programs: Vec<surface::Program> =
                parses.iter().map(|p| surface::program(&p.root)).collect();
            let files: Vec<PackageFile> = app_files
                .iter()
                .zip(&programs)
                .enumerate()
                .map(|(i, ((module, _), program))| {
                    let mut path = vec![app];
                    path.extend(module.iter().map(|seg| keys.get_or_intern(seg)));
                    PackageFile {
                        file: FileId::from_index(i as u32),
                        module: path,
                        program,
                    }
                })
                .collect();
            let mut elab = Elaborator::new(tcx, &interner);
            elab.declare_extern_package(
                &mut keys,
                &ExternPackage {
                    name: "dep",
                    parsed: &parsed,
                    files: &[],
                },
            );
            elab.run_package(&files);
            f(&elab);
        });
    }

    /// `#[lang]` markers travel through the interface: the consumer's
    /// registry binds the dependency's declaration, locations included.
    #[test]
    fn lang_items_reload_from_interfaces() {
        use crate::semi::lang::LangItem;
        check_with_dep(
            "#[lang(\"partial_eq\")]\npub trait PartialEq { fn eq(self: Self, other: Self) -> bool; }",
            &[(&[], "pub fn nop() -> i64 { 0 }")],
            |elab| {
                assert!(!elab.has_errors(), "{:#?}", elab.reports);
                let def = elab.lang.get(LangItem::PartialEq).expect("reloaded");
                assert_eq!(elab.sym(elab.defs.path(def).name()), "PartialEq");
                assert!(elab.lang_item_site(LangItem::PartialEq).is_some());
            },
        );
    }

    const TRAIT_DEP: &str = "pub trait Show { fn show(self: Self) -> i64; }\n\
         pub struct P { pub x: i64 }\n\
         impl Show for P { fn show(self: Self) -> i64 { self.x } }\n\
         pub fn use_show<T: Show>(x: T) -> i64 { x.show() }";

    /// Trait items reload across the interface: an extern trait dispatches
    /// on its extern type (dot and path form), bounds name it, and the
    /// dependency's own bounded generic accepts the extern type again.
    #[test]
    fn extern_traits_dispatch_across_packages() {
        check_with_dep(
            TRAIT_DEP,
            &[(
                &[],
                "pub fn direct(p: dep::P) -> i64 { p.show() }\n\
                 pub fn path_call(p: dep::P) -> i64 { dep::Show::show(p) }\n\
                 pub fn bounded<T: dep::Show>(x: T) -> i64 { x.show() }\n\
                 pub fn call_dep(p: dep::P) -> i64 { dep::use_show(p) }",
            )],
            |elab| {
                assert!(!elab.has_errors(), "{:#?}", elab.reports);
            },
        );
    }

    /// A consumer impl of an extern trait for a local type joins the same
    /// coherence world as the loaded impls: it is accepted, dispatches, and
    /// a both-extern impl or a duplicate is rejected by the same gates.
    #[test]
    fn consumer_impls_of_extern_traits_join_coherence() {
        check_with_dep(
            TRAIT_DEP,
            &[(
                &[],
                "pub struct Q { pub y: i64 }\n\
                 impl dep::Show for Q { fn show(self: Self) -> i64 { self.y } }\n\
                 pub fn go(q: Q) -> i64 { q.show() }",
            )],
            |elab| {
                assert!(!elab.has_errors(), "{:#?}", elab.reports);
            },
        );
        // Both trait and head extern: the orphan rule refuses.
        check_with_dep(
            TRAIT_DEP,
            &[(
                &[],
                "impl dep::Show for dep::P { fn show(self: Self) -> i64 { 9 } }",
            )],
            |elab| {
                assert!(
                    elab.reports.iter().any(|r| r
                        .message
                        .contains("the trait or the self type's head must be local")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
        // A consumer impl overlapping a loaded one: coherence refuses.
        check_with_dep(
            TRAIT_DEP,
            &[
                (
                    &[],
                    "pub struct Q { pub y: i64 }\n\
                 impl dep::Show for Q { fn show(self: Self) -> i64 { 1 } }\n\
                 mod b;",
                ),
                (
                    &["b"],
                    "impl dep::Show for super::Q { fn show(self: Self) -> i64 { 2 } }",
                ),
            ],
            |elab| {
                assert!(
                    elab.reports
                        .iter()
                        .any(|r| r.message.contains("conflicting implementations of trait")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    /// Loading the same package twice re-registers nothing: trait defs
    /// unify by path and impls dedupe by their method defs.
    #[test]
    fn twice_loaded_trait_packages_are_idempotent() {
        with_tcx(|tcx| {
            let parsed = dep_interface(tcx, TRAIT_DEP);
            let interner = Arc::new(reussir_syntax::new_threaded_interner());
            let mut keys = interner.clone();
            let mut elab = Elaborator::new(tcx, &interner);
            let ext = ExternPackage {
                name: "dep",
                parsed: &parsed,
                files: &[],
            };
            elab.declare_extern_package(&mut keys, &ext);
            let (traits, impls) = (elab.traits.traits_len(), elab.traits.impls_len());
            elab.declare_extern_package(&mut keys, &ext);
            assert_eq!(elab.traits.traits_len(), traits, "traits re-registered");
            assert_eq!(elab.traits.impls_len(), impls, "impls re-registered");
            assert!(!elab.has_errors(), "{:#?}", elab.reports);
        });
    }

    const DEP: &str = "pub struct Point { pub x: i64, pub y: i64 }\n\
                       impl Point {\n\
                           pub fn norm(self: Self) -> i64 { self.x + self.y }\n\
                           fn secret(self: Self) -> i64 { self.x }\n\
                       }\n\
                       pub struct Mixed { pub a: i64, b: i64 }\n\
                       struct Hidden { v: i64 }\n\
                       fn private_helper(h: Hidden) -> i64 { h.v }\n\
                       pub fn api<T : Num>(x: T) -> T { private_helper(Hidden { v: 1 }); x }\n\
                       pub fn narrow<T : Integral>(x: T) -> T { x }\n\
                       pub fn both<T : Num + Sync>(x: T) -> T { x }\n\
                       pub struct Holder<T : Num> { pub v: T }\n\
                       pub fn ground(x: i64) -> i64 { x + 1 }\n\
                       pub enum Opt { None, Some(i64) }";

    #[test]
    fn extern_method_dot_call_resolves() {
        check_with_dep(
            DEP,
            &[(
                &[],
                "fn a(p: dep::Point) -> i64 { p.norm() }\n\
                 fn b(p: dep::Point) -> i64 { dep::Point::norm(p) }",
            )],
            |elab| {
                assert!(!elab.has_errors(), "{:#?}", elab.reports);
            },
        );
    }

    /// The dot fallback's absolute lookup bypasses `resolve_extern`'s pub
    /// gate; the method path must re-apply it, keeping the is-private wording
    /// distinct from not-found.
    #[test]
    fn extern_private_method_gated_on_dot_call() {
        check_with_dep(
            DEP,
            &[(
                &[],
                "fn a(p: dep::Point) -> i64 { p.secret() }\n\
                 fn b(p: dep::Point) -> i64 { p.nope() }",
            )],
            |elab| {
                let messages: Vec<&str> = elab.reports.iter().map(|r| r.message.as_str()).collect();
                assert!(
                    messages.contains(&"function `secret` in package `dep` is private"),
                    "{messages:#?}"
                );
                assert!(
                    messages
                        .iter()
                        .any(|m| m.contains("no field or method `nope`")),
                    "{messages:#?}"
                );
            },
        );
    }

    #[test]
    fn impl_of_extern_package_type_rejected() {
        check_with_dep(
            DEP,
            &[(
                &[],
                "impl dep::Point { pub fn flip(p: dep::Point) -> i64 { p.x } }",
            )],
            |elab| {
                assert!(
                    elab.reports.iter().any(|r| r.message.contains(
                        "cannot define methods for `dep::Point` from extern package `dep`"
                    )),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn extern_bounds_reload() {
        check_with_dep(
            DEP,
            &[(&[], "fn go(x: i64) -> i64 { dep::api(x) }")],
            |elab| {
                assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
                let api = elab
                    .extern_functions
                    .iter()
                    .find(|f| elab.defs.path(f.def).display(elab.resolver) == "dep::api")
                    .expect("api carried");
                let (_, g) = api.generics[0];
                assert_eq!(
                    elab.generic_bounds(g),
                    [elab.lang.num],
                    "reloaded binder carries the serialized bound"
                );
            },
        );
    }

    /// A single-bound extern function checks at the consumer call site:
    /// the satisfied instantiation is clean, the violating one names the
    /// bound.
    #[test]
    fn extern_fn_bound_satisfied_and_violated() {
        check_with_dep(
            DEP,
            &[(&[], "fn ok(x: i64) -> i64 { dep::narrow(x) }")],
            |elab| assert!(!elab.has_errors(), "{:#?}", elab.reports),
        );
        check_with_dep(
            DEP,
            &[(&[], "fn bad(x: f64) -> f64 { dep::narrow(x) }")],
            |elab| {
                assert!(
                    elab.reports.iter().any(|r| r.message.contains("Integral")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    /// A multi-bound binder reloads with every TraitId, in order.
    #[test]
    fn extern_multi_bound_reloads_whole() {
        check_with_dep(
            DEP,
            &[(&[], "fn ok(x: i64) -> i64 { dep::both(x) }")],
            |elab| {
                assert!(!elab.has_errors(), "{:#?}", elab.reports);
                let both = elab
                    .extern_functions
                    .iter()
                    .find(|f| elab.defs.path(f.def).display(elab.resolver) == "dep::both")
                    .expect("both carried");
                let (_, g) = both.generics[0];
                assert_eq!(
                    elab.generic_bounds(g),
                    [elab.lang.num, elab.lang.sync],
                    "both bounds reload, in declaration order"
                );
            },
        );
    }

    /// A bounded extern *record* checks its bound where the consumer
    /// instantiates it — construction included.
    #[test]
    fn extern_record_bound_checks_at_construction() {
        check_with_dep(
            DEP,
            &[(&[], "fn ok() -> i64 { dep::Holder { v: 2 }.v }")],
            |elab| assert!(!elab.has_errors(), "{:#?}", elab.reports),
        );
        check_with_dep(
            DEP,
            &[(&[], "fn bad() -> bool { dep::Holder { v: true }.v }")],
            |elab| {
                assert!(
                    elab.reports.iter().any(|r| r.message.contains("Num")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn extern_bound_violation_reports_at_consumer_call_site() {
        check_with_dep(
            DEP,
            &[(&[], "fn bad(p: dep::Point) -> dep::Point { dep::api(p) }")],
            |elab| {
                assert!(
                    elab.reports.iter().any(|r| r.message.contains("Num")),
                    "expected a Num-bound violation at the consumer call: {:#?}",
                    elab.reports
                );
            },
        );
    }

    #[test]
    fn extern_unknown_bound_reports() {
        with_tcx(|tcx| {
            let text = "pub fn #dep::odd<$0 (T): NotATrait>(v0 (x): $0) -> $0 {\n    v0 : $0\n}\n";
            let parsed = crate::semi::hir::build::parse_program(tcx, text).expect("parse");
            let interner = Arc::new(reussir_syntax::new_threaded_interner());
            let mut keys = interner.clone();
            let mut elab = Elaborator::new(tcx, &interner);
            elab.declare_extern_package(
                &mut keys,
                &ExternPackage {
                    name: "dep",
                    parsed: &parsed,
                    files: &[],
                },
            );
            assert!(
                elab.reports
                    .iter()
                    .any(|r| r.message.contains("unknown trait bound `NotATrait`")),
                "{:#?}",
                elab.reports
            );
            // Declaration completes: the binder degrades to the resolvable
            // subset instead of aborting the load.
            assert_eq!(elab.extern_functions.len(), 1);
        });
    }

    #[test]
    fn extern_boundless_dump_still_loads() {
        with_tcx(|tcx| {
            let text = "pub fn #dep::id<$0 (T)>(v0 (x): $0) -> $0 {\n    v0 : $0\n}\n";
            let parsed = crate::semi::hir::build::parse_program(tcx, text).expect("parse");
            let interner = Arc::new(reussir_syntax::new_threaded_interner());
            let mut keys = interner.clone();
            let mut elab = Elaborator::new(tcx, &interner);
            elab.declare_extern_package(
                &mut keys,
                &ExternPackage {
                    name: "dep",
                    parsed: &parsed,
                    files: &[],
                },
            );
            assert!(elab.reports.is_empty(), "{:#?}", elab.reports);
            let id = &elab.extern_functions[0];
            let (_, g) = id.generics[0];
            assert!(elab.generic_bounds(g).is_empty());
        });
    }

    #[test]
    fn pub_items_resolve_through_the_package_head() {
        check_with_dep(
            DEP,
            &[(
                &[],
                "fn go(x: i64) -> i64 {\n\
                     let p = dep::Point { x: dep::api(x), y: dep::ground(x) };\n\
                     p.x + p.y\n\
                 }\n\
                 fn pick(x: i64) -> i64 {\n\
                     let o = dep::Opt::Some{x};\n\
                     match o {\n\
                         dep::Opt::Some(v) => v,\n\
                         dep::Opt::None => 0\n\
                     }\n\
                 }",
            )],
            |elab| {
                assert!(!elab.has_errors(), "elab errors: {:#?}", elab.reports);
                // The extern bodies are carried (remapped) for later
                // monomorphization, outside the consumer's own item list.
                let api = elab
                    .extern_functions
                    .iter()
                    .find(|f| elab.defs.path(f.def).display(elab.resolver) == "dep::api")
                    .expect("api carried");
                assert!(api.body.is_some(), "generic body ships and is carried");
                assert_eq!(elab.elaborated.len(), 2, "consumer items only");
                // Extern records join `records` (the checker reads layouts)
                // but not the dump's record set.
                assert!(!elab.records.is_empty());
                assert!(elab.local_records().is_empty(), "dump stays consumer-only");
                // Field visibility survives the `.rri` round trip and remap.
                let point = elab
                    .records
                    .values()
                    .find(|r| elab.sym(r.name) == "dep::Point")
                    .expect("extern Point carried");
                let crate::semi::ctxt::RecordFields::Named(fs) =
                    point.fields.as_ref().expect("populated")
                else {
                    panic!("named fields");
                };
                assert!(
                    fs.iter()
                        .all(|&(_, _, _, v)| v == crate::surface::Visibility::Public),
                    "{fs:#?}"
                );
            },
        );
    }

    #[test]
    fn private_extern_fields_are_rejected_distinctly() {
        check_with_dep(
            DEP,
            &[(
                &[],
                "fn a(m: dep::Mixed) -> i64 { m.a }\n\
                 fn b(m: dep::Mixed) -> i64 { m.b }\n\
                 fn c() -> dep::Mixed { dep::Mixed { a: 1, b: 2 } }\n\
                 fn d(m: dep::Mixed) -> i64 { m.nope }",
            )],
            |elab| {
                let messages: Vec<&str> = elab.reports.iter().map(|r| r.message.as_str()).collect();
                // The pub field elaborates; the private one is access
                // control, not absence — one level below the item-level
                // is-private/not-found split.
                assert!(
                    messages.contains(&"field `b` of record `dep::Mixed` is private"),
                    "{messages:#?}"
                );
                assert!(
                    messages.contains(
                        &"cannot construct `dep::Mixed` outside its module: \
                          field(s) `b` are private"
                    ),
                    "{messages:#?}"
                );
                assert!(messages.contains(&"no such field"), "{messages:#?}");
                assert_eq!(messages.len(), 3, "{messages:#?}");
            },
        );
    }

    #[test]
    fn private_extern_items_are_rejected_distinctly() {
        check_with_dep(
            DEP,
            &[(
                &[],
                "fn a(x: i64) -> i64 { dep::private_helper(x) }\n\
                 fn b(x: i64) -> i64 { dep::nope(x) }\n\
                 fn c(h: dep::Hidden) -> i64 { 0 }\n\
                 fn d() -> i64 { dep::Hidden { v: 1 }; 0 }",
            )],
            |elab| {
                let messages: Vec<&str> = elab.reports.iter().map(|r| r.message.as_str()).collect();
                // A private hit is access control, not absence …
                assert!(
                    messages.contains(&"function `private_helper` in package `dep` is private"),
                    "{messages:#?}"
                );
                assert!(
                    messages
                        .iter()
                        .filter(|m| m.contains("record `Hidden` in package `dep` is private"))
                        .count()
                        == 2,
                    "type and constructor positions both gate: {messages:#?}"
                );
                // … and absence stays not-found.
                assert!(
                    messages
                        .iter()
                        .any(|m| m.contains("unknown function `dep::nope`")),
                    "{messages:#?}"
                );
                assert!(
                    !messages
                        .iter()
                        .any(|m| m.contains("unknown function `dep::private_helper`")),
                    "private must not double-report as unknown: {messages:#?}"
                );
            },
        );
    }

    #[test]
    fn extern_head_and_local_modules_never_shadow() {
        // A local module named like the extern: `dep::…` resolves in the
        // extern package only (miss, not the local module), while the local
        // module stays reachable through `root::`.
        check_with_dep(
            DEP,
            &[
                (
                    &[],
                    "fn go(x: i64) -> i64 { dep::local(x) }\n\
                     fn ok(x: i64) -> i64 { root::dep::local(x) }",
                ),
                (&["dep"], "pub fn local(x: i64) -> i64 { x }"),
            ],
            |elab| {
                let messages: Vec<&str> = elab.reports.iter().map(|r| r.message.as_str()).collect();
                assert!(
                    messages
                        .iter()
                        .any(|m| m.contains("unknown function `dep::local`")),
                    "the extern head owns the path: {messages:#?}"
                );
                assert!(
                    !messages.iter().any(|m| m.contains("root::dep")),
                    "`root::` still reaches the local module: {messages:#?}"
                );
            },
        );
    }

    /// Declaring with a file map rewrites every imported item's `file` —
    /// functions (with and without bodies), prototypes, records, ffi imports
    /// and their preludes — into the consumer's cache id space; spans stay
    /// (byte offsets into the same content). Without a map (text-dump
    /// targets), ids pass through untouched.
    #[test]
    fn extern_item_files_remap_into_the_consumer_cache() {
        use reussir_syntax::source::SourceCache;

        let dep_src = "extern \"rust\" [{ use reussir_rt::collections::vec::Vec as RVec; }];\n\
                       #[ffi(rust = \"::reussir_rt::collections::vec::Vec\")]\n\
                       pub struct Vec<T>;\n\
                       #[ffi(import)]\n\
                       pub fn new<T>() -> Vec<T> [{ RVec::new() }];\n\
                       pub struct Point { x: i64, y: i64 }\n\
                       fn helper(p: Point) -> i64 { p.x }\n\
                       pub fn api<T : Num>(x: T) -> T { helper(Point { x: 1, y: 2 }); x }\n\
                       pub fn ground(x: i64) -> i64 { x + 1 }";
        with_tcx(|tcx| {
            // Elaborate the dep and print it in sources form, so the parsed
            // interface carries a file table and per-item `in <id>` files.
            let interner = Arc::new(reussir_syntax::new_threaded_interner());
            let mut keys = interner.clone();
            let dep = keys.get_or_intern("dep");
            let parse = reussir_syntax::parse_with_interner(dep_src, interner.clone());
            assert!(parse.ok(), "dep parse errors: {:#?}", parse.errors);
            let prog = surface::program(&parse.root);
            let files = [PackageFile {
                file: FileId::ROOT,
                module: vec![dep],
                program: &prog,
            }];
            let elab = elaborate_package(tcx, &files, &interner);
            assert!(!elab.has_errors(), "dep elab errors: {:#?}", elab.reports);
            let cache = SourceCache::single("lib.rr", dep_src);
            let strings = elab.strings.entries();
            let (traits, impls) =
                crate::semi::hir::trait_texts(&elab.traits, &elab.defs, elab.resolver);
            let text = Printer::with_sources(&elab.defs, elab.resolver, &cache)
                .with_ffi_metadata(&elab.ffi_preludes, &elab.ffi_imports)
                .with_traits(&traits, &impls)
                .program(&elab.elaborated, &strings, &elab.records, &elab.trampolines);
            let parsed = crate::semi::hir::build::parse_program(tcx, &text).expect("re-parse");
            assert_eq!(parsed.files, ["lib.rr"], "table travels");

            let with_declared = |files: &[FileId], check: &dyn Fn(&Elaborator)| {
                let interner = Arc::new(reussir_syntax::new_threaded_interner());
                let mut keys = interner.clone();
                let mut elab = Elaborator::new(tcx, &interner);
                elab.declare_extern_package(
                    &mut keys,
                    &ExternPackage {
                        name: "dep",
                        parsed: &parsed,
                        files,
                    },
                );
                check(&elab);
            };

            // The consumer's cache already holds its own files, so the dep's
            // single entry lands at some later id.
            let mapped = FileId::from_index(7);
            with_declared(&[mapped], &|consumer| {
                for func in &consumer.extern_functions {
                    assert_eq!(func.file, mapped, "function files remap");
                    assert!(func.span.is_some(), "spans survive the remap");
                }
                for proto in consumer.functions.values() {
                    assert_eq!(proto.file, mapped, "prototype files remap");
                }
                for record in consumer.records.values() {
                    assert_eq!(record.file, mapped, "record files remap");
                }
                assert!(
                    !consumer.extern_ffi_imports.is_empty(),
                    "ffi import carried"
                );
                for import in consumer.extern_ffi_imports.values() {
                    assert_eq!(import.file, mapped, "ffi import files remap");
                }
                assert!(!consumer.extern_ffi_preludes.is_empty(), "prelude carried");
                for prelude in &consumer.extern_ffi_preludes {
                    assert_eq!(prelude.file, mapped, "prelude files remap");
                }
            });

            // No map: ids pass through (nothing renders them).
            with_declared(&[], &|untouched| {
                for func in &untouched.extern_functions {
                    assert_eq!(func.file, FileId::ROOT);
                }
            });
        });
    }

    #[test]
    fn imports_cannot_bind_an_extern_head() {
        check_with_dep(
            DEP,
            &[(
                &[],
                "import core::intrinsic::math as dep;\nfn go() -> i64 { 0 }",
            )],
            |elab| {
                assert!(
                    elab.reports
                        .iter()
                        .any(|r| r.message.contains("`dep` names a loaded extern package")),
                    "{:#?}",
                    elab.reports
                );
            },
        );
    }
}
