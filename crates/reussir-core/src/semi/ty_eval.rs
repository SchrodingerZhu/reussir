//! Evaluating surface types into interned [`Ty`]s, including the capability
//! "coloring" of regional records.
//!
//! The coloring is deterministic and local: a record declared `[regional]`
//! starts at [`Capability::Regional`]; a `[flex]` annotation on a parameter,
//! return, or binding refines it to [`Capability::Flex`] (mutable) and its
//! absence to [`Capability::Rigid`] (read-only). Non-regional records are
//! [`Capability::Irrelevant`]. Generics are left uncolored — their modality is
//! resolved only at monomorphization.

use crate::semi::ty::{Capability, FpTy, IntTy, Ty, TyKind};
use crate::surface::{self, FpType, IntegralType, TypeKind};

use super::ctxt::{DefaultCap, Elaborator};

impl<'a, 'tcx> Elaborator<'a, 'tcx> {
    /// Evaluate a surface type, coloring concrete regional records as
    /// `Regional` (later refined by [`Elaborator::eval_type_flex`]).
    pub fn eval_type(&mut self, ty: &surface::Type) -> Ty<'tcx> {
        let kind = ty.kind();
        match &kind {
            TypeKind::TypeIntegral(it) => self.tcx.mk_int(match it {
                IntegralType::Signed(w) => IntTy::Signed(*w),
                IntegralType::Unsigned(w) => IntTy::Unsigned(*w),
            }),
            TypeKind::TypeFp(fp) => self.tcx.mk_fp(match fp {
                FpType::Ieee(w) => FpTy::Ieee(*w),
                FpType::BFloat16 => FpTy::BFloat16,
                FpType::Float8 => FpTy::Float8,
            }),
            TypeKind::TypeBool => self.tcx.mk_bool(),
            TypeKind::TypeStr => self.tcx.mk_str(),
            TypeKind::TypeUnit => self.tcx.mk_unit(),
            TypeKind::TypeArrow(args, ret) => {
                let args: Vec<Ty> = args.iter().map(|a| self.eval_type(a)).collect();
                let ret = self.eval_type(ret);
                self.tcx.mk_closure(&args, ret)
            }
            TypeKind::TypeExpr(path, args) => self.eval_type_expr(path, args, ty.span()),
        }
    }

    fn eval_type_expr(
        &mut self,
        path: &surface::Path,
        args: &[surface::Type],
        span: surface::Span,
    ) -> Ty<'tcx> {
        let name = self.sym(path.basename);

        // A bare generic parameter in scope.
        if path.segments.is_empty()
            && let Some(&generic) = self.generic_names.get(name)
        {
            if !args.is_empty() {
                self.error(
                    Some(span),
                    format!("generic `{name}` cannot take type arguments"),
                );
            }
            return self.tcx.mk_generic(generic);
        }

        // The built-in nullable type.
        if name == "Nullable" {
            return match args {
                [inner] => {
                    let inner = self.eval_type(inner);
                    self.tcx.mk_nullable(inner)
                }
                _ => {
                    self.error(Some(span), "`Nullable` takes exactly one type argument");
                    self.tcx.mk(TyKind::Bottom)
                }
            };
        }

        // A user record.
        let Some(default_cap) = self.records.get(name).map(|r| r.default_cap) else {
            self.error(Some(span), format!("unknown type `{name}`"));
            return self.tcx.mk(TyKind::Bottom);
        };
        let args: Vec<Ty> = args.iter().map(|a| self.eval_type(a)).collect();
        let flex = match default_cap {
            DefaultCap::Value | DefaultCap::Shared => Capability::Irrelevant,
            DefaultCap::Regional => Capability::Regional,
        };
        self.tcx.mk_record(name, &args, flex)
    }

    /// Evaluate a type that carries a `[flex]` flag (a parameter, return type,
    /// or `let` binding). A regional record refines to `Flex` or `Rigid`.
    pub fn eval_type_flex(&mut self, ty: &surface::Type, is_flex: bool) -> Ty<'tcx> {
        let t = self.eval_type(ty);
        self.refine_flex(t, is_flex)
    }

    /// Refine a regional record's flexivity by a `[flex]` flag.
    pub fn refine_flex(&self, t: Ty<'tcx>, is_flex: bool) -> Ty<'tcx> {
        let refined = if is_flex {
            Capability::Flex
        } else {
            Capability::Rigid
        };
        match t.kind() {
            TyKind::Record {
                path,
                args,
                flex: Capability::Regional,
            } => self.tcx.mk_record(path, args, refined),
            TyKind::Nullable(inner) => {
                if let TyKind::Record {
                    path,
                    args,
                    flex: Capability::Regional,
                } = inner.kind()
                {
                    let inner = self.tcx.mk_record(path, args, refined);
                    self.tcx.mk_nullable(inner)
                } else {
                    t
                }
            }
            _ => t,
        }
    }

    /// Freeze a value escaping a region: any regional record becomes `Rigid`.
    pub fn freeze_region(&self, t: Ty<'tcx>) -> Ty<'tcx> {
        match t.kind() {
            TyKind::Record {
                path,
                args,
                flex: Capability::Regional | Capability::Flex,
            } => self.tcx.mk_record(path, args, Capability::Rigid),
            _ => t,
        }
    }
}
