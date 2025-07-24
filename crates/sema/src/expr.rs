use crate::builder::BlockBuilder;
use crate::{Result, builder::ValueDef};
use reussir_core::{
    ir::{OperationKind, Symbol, ValID},
    literal::{FloatLiteral, IntegerLiteral},
    path,
    types::{Primitive, Type},
};
use reussir_front::expr::{BinaryOp, CallTarget, CallTargetSegment, Expr, ExprBox};
use ustr::Ustr;

pub struct ExprValue<'a> {
    value: Option<ValID>,
    ty: &'a Type,
}

impl<'b, 'a: 'b> BlockBuilder<'b, 'a> {
    pub fn unify(&self, lhs: &'a Type, rhs: &'a Type) -> Result<'a, ()> {
        if core::ptr::eq(lhs, rhs) || lhs == rhs {
            return Ok(());
        } else {
            unimplemented!("cannot handle unification besides equality for now");
        }
    }
    // TODO: should not do short-circuiting here. All errors should be reported.
    pub fn add_expr(
        &self,
        expr: &ExprBox,
        used: bool,
        name: Option<Ustr>,
    ) -> Result<'a, ExprValue<'a>> {
        let mut operation_builder = self.new_operation();
        operation_builder.add_location(expr.location());
        match &***expr {
            Expr::Unit => {
                let unit = self.get_primitive_type(Primitive::Unit);
                if used {
                    operation_builder.add_output(unit, name);
                }
                let val = operation_builder.finish(OperationKind::Unit);
                Ok(ExprValue {
                    value: val,
                    ty: unit,
                })
            }
            Expr::Binary(lhs, op, rhs) => {
                let ExprValue {
                    value: Some(lhs_value),
                    ty: lhs_ty,
                } = self.add_expr(lhs, true, None)?
                else {
                    todo!("report internal error if lhs_value is None");
                };
                let ExprValue {
                    value: Some(rhs_value),
                    ty: rhs_ty,
                } = self.add_expr(rhs, true, None)?
                else {
                    todo!("report internal error if rhs_value is None");
                };

                macro_rules! arith_intrinsic_call {
                    (unify $target:ident) => {{
                        self.unify(lhs_ty, rhs_ty)?;
                        if !lhs_ty.is_float() && !lhs_ty.is_integer() {
                            todo!("report error for unsupported type for arithmetic operations");
                        }
                        let target_function =
                            path![stringify!(target), "core", "intrinsics", "arith"];
                        let type_params = self.alloc_slice_fill_iter([lhs_ty]);
                        let symbol = self.alloc(Symbol {
                            path: target_function,
                            type_params: Some(type_params),
                        });
                        let call = OperationKind::FnCall {
                            target: symbol,
                            args: Some(self.alloc_slice_fill_iter([lhs_value, rhs_value])),
                        };
                        operation_builder.add_output(lhs_ty, name);
                        Ok(ExprValue {
                            value: operation_builder.finish(call),
                            ty: lhs_ty,
                        })
                    }};
                }
                match &**op {
                    BinaryOp::Add => arith_intrinsic_call!(unify add),
                    BinaryOp::Sub => arith_intrinsic_call!(unify sub),
                    BinaryOp::Mod => arith_intrinsic_call!(unify mod),
                    BinaryOp::Mul => arith_intrinsic_call!(unify mul),
                    BinaryOp::Div => arith_intrinsic_call!(unify div),
                    BinaryOp::LAnd => todo!(),
                    BinaryOp::BAnd => todo!(),
                    BinaryOp::LOr => todo!(),
                    BinaryOp::BOr => todo!(),
                    BinaryOp::Xor => arith_intrinsic_call!(unify xor),
                    BinaryOp::Shr => todo!(),
                    BinaryOp::Shl => todo!(),
                    BinaryOp::Eq => todo!(),
                    BinaryOp::Ne => todo!(),
                    BinaryOp::Le => todo!(),
                    BinaryOp::Lt => todo!(),
                    BinaryOp::Ge => todo!(),
                    BinaryOp::Gt => todo!(),
                }
            }
            Expr::Unary(_, _) => todo!(),
            Expr::Boolean(value) => {
                let bool_type = self.get_primitive_type(Primitive::Bool);
                operation_builder.add_output(bool_type, name);
                let val = operation_builder.finish(OperationKind::ConstantBool(*value));
                Ok(ExprValue {
                    value: val,
                    ty: bool_type,
                })
            }
            Expr::Integer(lit) => {
                macro_rules! codegen_integer {
                    ($variant:ident) => {{
                        let int_type = self.get_primitive_type(Primitive::$variant);
                        let lit = self.parent.alloc(lit.clone());
                        operation_builder.add_output(int_type, name);
                        let val = operation_builder.finish(OperationKind::ConstInt(lit));
                        Ok(ExprValue {
                            value: val,
                            ty: int_type,
                        })
                    }};
                }
                match lit {
                    IntegerLiteral::I8(_) => codegen_integer!(I8),
                    IntegerLiteral::I16(_) => codegen_integer!(I16),
                    IntegerLiteral::I32(_) => codegen_integer!(I32),
                    IntegerLiteral::I64(_) => codegen_integer!(I64),
                    IntegerLiteral::I128(_) => codegen_integer!(I128),
                    IntegerLiteral::U8(_) => codegen_integer!(U8),
                    IntegerLiteral::U16(_) => codegen_integer!(U16),
                    IntegerLiteral::U32(_) => codegen_integer!(U32),
                    IntegerLiteral::U64(_) => codegen_integer!(U64),
                    IntegerLiteral::U128(_) => codegen_integer!(U128),
                }
            }
            Expr::Float(lit) => {
                macro_rules! codegen_float {
                    ($variant:ident) => {{
                        let float_type = self.get_primitive_type(Primitive::$variant);
                        let lit = self.alloc(lit.clone());
                        operation_builder.add_output(float_type, name);
                        let val = operation_builder.finish(OperationKind::ConstFloat(lit));
                        Ok(ExprValue {
                            value: val,
                            ty: float_type,
                        })
                    }};
                }
                match lit {
                    FloatLiteral::BF16(_) => codegen_float!(BF16),
                    FloatLiteral::F16(_) => codegen_float!(F16),
                    FloatLiteral::F32(_) => codegen_float!(F32),
                    FloatLiteral::F64(_) => codegen_float!(F64),
                    FloatLiteral::F128(_) => codegen_float!(F128),
                }
            }
            Expr::Variable(x) => {
                let Some((val, ValueDef { ty, .. })) = self.parent.lookup(&x.as_str().into())
                else {
                    todo!("report error for undefined variable {x}");
                };
                Ok(ExprValue {
                    value: Some(val),
                    ty,
                })
            }
            Expr::IfThenElse(_, _, _) => todo!(),
            Expr::Sequence(items) => todo!(),
            Expr::Let(x, ty, val) => {
                let ExprValue {
                    value: Some(val),
                    ty: expr_ty,
                } = self.add_expr(val, true, Some(x.as_str().into()))?
                else {
                    todo!("report internal error if value is None");
                };
                if let Some(ty) = ty {
                    self.unify(expr_ty, ty)?;
                }
                let unit_ty = self.get_primitive_type(Primitive::Unit);
                if used {
                    operation_builder.add_output(unit_ty, name);
                }
                Ok(ExprValue {
                    value: operation_builder.finish(OperationKind::Unit),
                    ty: unit_ty,
                })
            }
            Expr::Call(_, items) => todo!(),
            Expr::CtorCall(_, items) => todo!(),
            Expr::Match(match_expr) => todo!(),
            Expr::Lambda(lambda_expr) => todo!(),
            Expr::Cast(expr, primitive) => {
                let ExprValue {
                    value: Some(value),
                    ty: expr_ty,
                } = self.add_expr(expr, true, None)?
                else {
                    todo!("report internal error if value is None");
                };
                if !expr_ty.is_float() && !expr_ty.is_integer() {
                    todo!("report error for unsupported type for cast");
                }
                let target_type = self.get_primitive_type(*primitive);
                operation_builder.add_output(target_type, name);
                let target_function = path!["cast", "core", "intrinsics", "arith"];
                let type_params = self.alloc_slice_fill_iter([expr_ty, target_type]);
                let symbol = self.alloc(Symbol {
                    path: target_function,
                    type_params: Some(type_params),
                });
                operation_builder.add_output(target_type, name);
                let call = OperationKind::FnCall {
                    target: symbol,
                    args: Some(self.alloc_slice_fill_iter([value])),
                };
                Ok(ExprValue {
                    value: operation_builder.finish(call),
                    ty: target_type,
                })
            }
            Expr::Return(_) => todo!(),
            Expr::Yield(_) => todo!(),
        }
    }
}
