use crate::Result;
use crate::builder::BlockBuilder;
use reussir_core::{
    ir::{OperationKind, ValID},
    types::{Primitive, Type},
};
use reussir_front::expr::{Expr, ExprBox};
use ustr::Ustr;

pub struct ExprValue<'a> {
    value: Option<ValID>,
    ty: &'a Type,
}

impl<'b, 'a: 'b> BlockBuilder<'b, 'a> {
    pub fn unify(&self, lhs: &'a Type, rhs: &'a Type) -> Result<'a, ()> {
        unimplemented!("Type unification is not implemented yet");
    }
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
            Expr::Binary(_, _, _) => todo!(),
            Expr::Unary(_, _) => todo!(),
            Expr::Boolean(value) => {
                let bool_type = self.get_primitive_type(Primitive::Bool);
                let val = operation_builder.finish(OperationKind::ConstantBool(*value));
                Ok(ExprValue {
                    value: val,
                    ty: bool_type,
                })
            }
            Expr::Integer(integer_literal) => todo!(),
            Expr::Float(float_literal) => todo!(),
            Expr::Variable(_) => todo!(),
            Expr::IfThenElse(_, _, _) => todo!(),
            Expr::Sequence(items) => todo!(),
            Expr::Let(_, _, _) => todo!(),
            Expr::Call(_, items) => todo!(),
            Expr::CtorCall(_, items) => todo!(),
            Expr::Match(match_expr) => todo!(),
            Expr::Lambda(lambda_expr) => todo!(),
            Expr::Cast(_, primitive) => todo!(),
            Expr::Return(_) => todo!(),
            Expr::Yield(_) => todo!(),
        }
    }
}
