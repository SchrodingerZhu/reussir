use std::io::Write;

use crate::{
    Result,
    ir::{OutputValue, ValID},
};
use bitflags::bitflags;
use strum::IntoStaticStr;
use ustr::Ustr;

use crate::{CGContext, ir::Symbol, types::Type};

pub enum Instrinsic {
    Arith(ArithIntrinsic),
}

bitflags! {
    pub struct IntegerOverflowFlags: u8 {
        const NONE = 0;
        const NSW  = 1;
        const NUW  = 2;
    }

    pub struct FastMathFlags: u8 {
        const NONE = 0;
        const REASSOC = 1;
        const NNAN = 2;
        const NINF = 4;
        const NSZ = 8;
        const ARCP = 16;
        const CONTRACT = 32;
        const AFN = 64;
        const FAST = 127;
    }
}

#[derive(IntoStaticStr)]
pub enum CmpIPredicate {
    #[strum(serialize = "eq")]
    Eq,
    #[strum(serialize = "ne")]
    Ne,
    #[strum(serialize = "slt")]
    Slt,
    #[strum(serialize = "sle")]
    Sle,
    #[strum(serialize = "sgt")]
    Sgt,
    #[strum(serialize = "sge")]
    Sge,
    #[strum(serialize = "ult")]
    Ult,
    #[strum(serialize = "ule")]
    Ule,
    #[strum(serialize = "ugt")]
    Ugt,
    #[strum(serialize = "uge")]
    Uge,
}

pub enum CmpFPredicate {
    AlwaysFalse,
    Oeq,
    Ogt,
    Oge,
    Olt,
    Ole,
    One,
    Ord,
    Ueq,
    Ugt,
    Uge,
    Ult,
    Ule,
    Une,
    Uno,
    AlwaysTrue,
}

pub enum ArithIntrinsic {
    Addi(IntegerOverflowFlags),
    Subi(IntegerOverflowFlags),
    Addf(FastMathFlags),
    Subf(FastMathFlags),
    Cmpi(CmpIPredicate),
    Cmpf(CmpFPredicate),
    Divsi,
    Divui,
    Muli(IntegerOverflowFlags),
    Remsi,
    Remui,
}

impl<'a> ArithIntrinsic {
    fn dispatch(basename: &str, ty: &[&Type]) -> Option<Self> {
        match (basename, ty) {
            ("add", [t]) if t.is_integer() => Some(Self::Addi(IntegerOverflowFlags::NONE)),
            ("add", [t]) if t.is_float() => Some(Self::Addf(FastMathFlags::NONE)),
            ("sub", [t]) if t.is_integer() => Some(Self::Subi(IntegerOverflowFlags::NONE)),
            ("sub", [t]) if t.is_float() => Some(Self::Subf(FastMathFlags::NONE)),
            ("lt", [t]) if t.is_signed_integer() => Some(Self::Cmpi(CmpIPredicate::Slt)),
            ("lt", [t]) if t.is_unsigned_integer() => Some(Self::Cmpi(CmpIPredicate::Ult)),
            ("le", [t]) if t.is_signed_integer() => Some(Self::Cmpi(CmpIPredicate::Sle)),
            ("le", [t]) if t.is_unsigned_integer() => Some(Self::Cmpi(CmpIPredicate::Ule)),
            ("gt", [t]) if t.is_signed_integer() => Some(Self::Cmpi(CmpIPredicate::Sgt)),
            ("gt", [t]) if t.is_unsigned_integer() => Some(Self::Cmpi(CmpIPredicate::Ugt)),
            ("ge", [t]) if t.is_signed_integer() => Some(Self::Cmpi(CmpIPredicate::Sge)),
            ("ge", [t]) if t.is_unsigned_integer() => Some(Self::Cmpi(CmpIPredicate::Uge)),
            ("eq", [t]) if t.is_integer() => Some(Self::Cmpi(CmpIPredicate::Eq)),
            ("ne", [t]) if t.is_integer() => Some(Self::Cmpi(CmpIPredicate::Ne)),
            ("div", [t]) if t.is_signed_integer() => Some(Self::Divsi),
            ("div", [t]) if t.is_unsigned_integer() => Some(Self::Divui),
            ("mul", [t]) if t.is_integer() => Some(Self::Muli(IntegerOverflowFlags::NONE)),
            ("mod", [t]) if t.is_signed_integer() => Some(Self::Remsi),
            ("mod", [t]) if t.is_unsigned_integer() => Some(Self::Remui),
            _ => None,
        }
    }
}

impl<'a> Symbol<'a> {
    pub fn get_intrinsic(&self) -> Option<Instrinsic> {
        thread_local! {
            static ARITH_PREFIX :  [Ustr; 3] = ["core".into(), "intrinsics".into(), "arith".into()];
        }
        if self.path.segments().len() == 4
            && ARITH_PREFIX.with(|p| self.path.segments().starts_with(p))
        {
            return ArithIntrinsic::dispatch(self.path.basename().as_str(), self.type_params?)
                .map(Instrinsic::Arith);
        }
        None
    }
}

impl ArithIntrinsic {
    pub fn codegen<W: Write>(
        &self,
        ctx: &mut CGContext<W>,
        input: Option<&[(ValID, &Type)]>,
        result: Option<&OutputValue>,
    ) -> Result<()> {
        match self {
            ArithIntrinsic::Addi(flags) => {
                let input = input.unwrap();
                let out = result.unwrap();
                let (lhs, _) = input.get(0).unwrap();
                let (rhs, _) = input.get(1).unwrap();
                write!(ctx.output, "%{} = arith.addi %{lhs}, %{rhs} : ", out.value)?;
                if !flags.is_empty() {
                    unimplemented!()
                }
                out.ty.codegen(ctx)?;
            }
            ArithIntrinsic::Subi(flags) => {
                let input = input.unwrap();
                let out = result.unwrap();
                let (lhs, _) = input.get(0).unwrap();
                let (rhs, _) = input.get(1).unwrap();
                write!(ctx.output, "%{} = arith.subi %{lhs}, %{rhs} : ", out.value)?;
                if !flags.is_empty() {
                    unimplemented!()
                }
                out.ty.codegen(ctx)?;
            }
            ArithIntrinsic::Cmpi(flags) => {
                let input = input.unwrap();
                let out = result.unwrap();
                let (lhs, ty) = input.get(0).unwrap();
                let (rhs, _) = input.get(1).unwrap();
                write!(
                    ctx.output,
                    "%{} = arith.cmpi {}, %{lhs}, %{rhs} : ",
                    out.value,
                    Into::<&str>::into(flags)
                )?;
                ty.codegen(ctx)?;
            }
            ArithIntrinsic::Divsi => {
                let input = input.unwrap();
                let out = result.unwrap();
                let (lhs, _) = input.get(0).unwrap();
                let (rhs, _) = input.get(1).unwrap();
                write!(ctx.output, "%{} = arith.divsi %{lhs}, %{rhs} : ", out.value)?;
                out.ty.codegen(ctx)?;
            }
            ArithIntrinsic::Divui => {
                let input = input.unwrap();
                let out = result.unwrap();
                let (lhs, _) = input.get(0).unwrap();
                let (rhs, _) = input.get(1).unwrap();
                write!(ctx.output, "%{} = arith.divui %{lhs}, %{rhs} : ", out.value)?;
                out.ty.codegen(ctx)?;
            }
            ArithIntrinsic::Muli(flags) => {
                let input = input.unwrap();
                let out = result.unwrap();
                let (lhs, _) = input.get(0).unwrap();
                let (rhs, _) = input.get(1).unwrap();
                write!(ctx.output, "%{} = arith.muli %{lhs}, %{rhs} : ", out.value)?;
                if !flags.is_empty() {
                    unimplemented!()
                }
                out.ty.codegen(ctx)?;
            }
            ArithIntrinsic::Remsi => {
                let input = input.unwrap();
                let out = result.unwrap();
                let (lhs, _) = input.get(0).unwrap();
                let (rhs, _) = input.get(1).unwrap();
                write!(ctx.output, "%{} = arith.remsi %{lhs}, %{rhs} : ", out.value)?;
                out.ty.codegen(ctx)?;
            }
            ArithIntrinsic::Remui => {
                let input = input.unwrap();
                let out = result.unwrap();
                let (lhs, _) = input.get(0).unwrap();
                let (rhs, _) = input.get(1).unwrap();
                write!(ctx.output, "%{} = arith.remui %{lhs}, %{rhs} : ", out.value)?;
                out.ty.codegen(ctx)?;
            }
            _ => todo!(),
        }
        Ok(())
    }
}

impl Instrinsic {
    pub fn codegen<W: Write>(
        &self,
        ctx: &mut CGContext<W>,
        input: Option<&[(ValID, &Type)]>,
        result: Option<&OutputValue>,
    ) -> Result<()> {
        match self {
            Instrinsic::Arith(arity) => arity.codegen(ctx, input, result),
        }
    }
}
