use crate::builder::ValueDef;
use crate::builder::{BlockBuilder, DiagnosticLevel};
use reussir_core::Location;
use reussir_core::{
    ir::{OperationKind, Symbol, ValID},
    literal::{FloatLiteral, IntegerLiteral},
    path,
    types::{Primitive, Type},
};
use reussir_front::expr::{BinaryOp, Expr, ExprBox};
use ustr::Ustr;

pub struct ExprValue<'a> {
    value: Option<ValID>,
    ty: &'a Type,
}

impl<'b, 'a: 'b> BlockBuilder<'b, 'a> {
    pub fn unify(&self, lhs: &'a Type, rhs: &'a Type) -> bool {
        if core::ptr::eq(lhs, rhs) || lhs == rhs {
            true
        } else {
            // Cannot unify besides pure equality for now
            false
        }
    }

    fn add_poison(&self, ty: &'a Type) -> ValID {
        let mut operation_builder = self.new_operation();
        operation_builder.add_output(ty, None);
        operation_builder.finish(OperationKind::Poison).unwrap()
    }

    fn add_poison_never(&self) -> ExprValue<'a> {
        let ty = self.get_primitive_type(Primitive::Never);
        ExprValue {
            value: Some(self.add_poison(ty)),
            ty,
        }
    }

    fn add_panic(&self, message: &'a str) -> (usize, &'a Type) {
        let mut operation_builder = self.new_operation();
        let never = self.get_primitive_type(Primitive::Never);
        operation_builder.add_output(never, None);
        let value = operation_builder.finish(OperationKind::Panic(message));
        (value.unwrap(), never)
    }

    pub fn add_expr(&self, expr: &ExprBox, used: bool, name: Option<Ustr>) -> ExprValue<'a> {
        let mut operation_builder = self.new_operation();
        operation_builder.add_location(expr.location());
        match &***expr {
            Expr::Unit => {
                let unit = self.get_primitive_type(Primitive::Unit);
                if used {
                    operation_builder.add_output(unit, name);
                }
                let val = operation_builder.finish(OperationKind::Unit);
                ExprValue {
                    value: val,
                    ty: unit,
                }
            }
            Expr::Binary(lhs, op, rhs) => {
                let ExprValue {
                    value: Some(lhs_value),
                    ty: lhs_ty,
                } = self.add_expr(lhs, true, None)
                else {
                    self.diagnostic(
                        DiagnosticLevel::Ice,
                        "LHS does not return a value properly",
                        expr.location(),
                    );
                    return self.add_poison_never();
                };
                let ExprValue {
                    value: Some(rhs_value),
                    ty: rhs_ty,
                } = self.add_expr(rhs, true, None)
                else {
                    self.diagnostic(
                        DiagnosticLevel::Ice,
                        "RHS does not return a value properly",
                        expr.location(),
                    );
                    return self.add_poison_never();
                };
                macro_rules! arith_intrinsic_call {
                    (unify $target:ident) => {{
                        if !self.unify(lhs_ty, rhs_ty) {
                            self.diagnostic(
                                DiagnosticLevel::Error,
                                "type mismatch in arithmetic operation",
                                expr.location(),
                            );
                        }
                        if !lhs_ty.is_float() && !lhs_ty.is_integer() {
                            todo!("report error for unsupported type for arithmetic operations");
                        }
                        let target_function =
                            path![stringify!($target), "core", "intrinsics", "arith"];
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
                        ExprValue {
                            value: operation_builder.finish(call),
                            ty: lhs_ty,
                        }
                    }};
                    (compare $target:ident) => {{
                        if !self.unify(lhs_ty, rhs_ty) {
                            self.diagnostic(
                                DiagnosticLevel::Error,
                                "type mismatch in arithmetic operation",
                                expr.location(),
                            );
                        }
                        if !lhs_ty.is_float() && !lhs_ty.is_integer() {
                            todo!("report error for unsupported type for arithmetic operations");
                        }
                        let target_function =
                            path![stringify!($target), "core", "intrinsics", "arith"];
                        let type_params = self.alloc_slice_fill_iter([lhs_ty]);
                        let symbol = self.alloc(Symbol {
                            path: target_function,
                            type_params: Some(type_params),
                        });
                        let call = OperationKind::FnCall {
                            target: symbol,
                            args: Some(self.alloc_slice_fill_iter([lhs_value, rhs_value])),
                        };
                        let boolean_type = self.get_primitive_type(Primitive::Bool);
                        operation_builder.add_output(boolean_type, name);
                        ExprValue {
                            value: operation_builder.finish(call),
                            ty: boolean_type,
                        }
                    }};
                }
                match &**op {
                    BinaryOp::Add => arith_intrinsic_call!(unify add),
                    BinaryOp::Sub => arith_intrinsic_call!(unify sub),
                    BinaryOp::Mod => arith_intrinsic_call!(unify mod),
                    BinaryOp::Mul => arith_intrinsic_call!(unify mul),
                    BinaryOp::Div => arith_intrinsic_call!(unify div),
                    BinaryOp::LAnd => ExprValue {
                        value: Some(self.short_circuit(false, lhs, rhs, expr.location())),
                        ty: self.get_primitive_type(Primitive::Bool),
                    },
                    BinaryOp::BAnd => arith_intrinsic_call!(unify and),
                    BinaryOp::LOr => ExprValue {
                        value: Some(self.short_circuit(true, lhs, rhs, expr.location())),
                        ty: self.get_primitive_type(Primitive::Bool),
                    },
                    BinaryOp::BOr => arith_intrinsic_call!(unify or),
                    BinaryOp::Xor => arith_intrinsic_call!(unify xor),
                    BinaryOp::Shr => todo!(),
                    BinaryOp::Shl => todo!(),
                    BinaryOp::Eq => arith_intrinsic_call!(compare eq),
                    BinaryOp::Ne => arith_intrinsic_call!(compare ne),
                    BinaryOp::Le => arith_intrinsic_call!(compare le),
                    BinaryOp::Lt => arith_intrinsic_call!(compare lt),
                    BinaryOp::Ge => arith_intrinsic_call!(compare ge),
                    BinaryOp::Gt => arith_intrinsic_call!(compare gt),
                }
            }
            Expr::Unary(_, _) => todo!(),
            Expr::Boolean(value) => {
                let bool_type = self.get_primitive_type(Primitive::Bool);
                operation_builder.add_output(bool_type, name);
                let val = operation_builder.finish(OperationKind::ConstantBool(*value));
                ExprValue {
                    value: val,
                    ty: bool_type,
                }
            }
            Expr::Integer(lit) => {
                macro_rules! codegen_integer {
                    ($variant:ident) => {{
                        let int_type = self.get_primitive_type(Primitive::$variant);
                        let lit = self.parent.alloc(lit.clone());
                        operation_builder.add_output(int_type, name);
                        let val = operation_builder.finish(OperationKind::ConstInt(lit));
                        ExprValue {
                            value: val,
                            ty: int_type,
                        }
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
                        ExprValue {
                            value: val,
                            ty: float_type,
                        }
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
                    self.diagnostic(
                        DiagnosticLevel::Ice,
                        "value not defined in value types",
                        expr.location(),
                    );
                    return self.add_poison_never();
                };
                ExprValue {
                    value: Some(val),
                    ty,
                }
            }
            Expr::IfThenElse(_, _, _) => todo!(),
            Expr::Sequence(items) => todo!(),
            Expr::Let(x, ty, val) => {
                let ExprValue {
                    value: Some(val),
                    ty: expr_ty,
                } = self.add_expr(val, true, Some(x.as_str().into()))
                else {
                    self.diagnostic(
                        DiagnosticLevel::Ice,
                        "value does not return a value properly",
                        expr.location(),
                    );
                    return self.add_poison_never();
                };
                let ty = self.alloc(ty.clone());
                if let Some(ty) = ty
                    && !self.unify(expr_ty, ty)
                {
                    self.diagnostic(
                        DiagnosticLevel::Error,
                        "type mismatch in let binding",
                        expr.location(),
                    );
                }
                let unit_ty = self.get_primitive_type(Primitive::Unit);
                if used {
                    operation_builder.add_output(unit_ty, name);
                }
                ExprValue {
                    value: operation_builder.finish(OperationKind::Unit),
                    ty: unit_ty,
                }
            }
            Expr::Call(_, items) => todo!(),
            Expr::CtorCall(_, items) => todo!(),
            Expr::Match(match_expr) => todo!(),
            Expr::Lambda(lambda_expr) => todo!(),
            Expr::Cast(expr, primitive) => {
                let target_type = self.get_primitive_type(*primitive);
                let ExprValue {
                    value: Some(value),
                    ty: expr_ty,
                } = self.add_expr(expr, true, None)
                else {
                    self.diagnostic(
                        DiagnosticLevel::Ice,
                        "value does not return a value properly",
                        expr.location(),
                    );
                    return ExprValue {
                        value: Some(self.add_poison(target_type)),
                        ty: target_type,
                    };
                };
                if !expr_ty.is_float() && !expr_ty.is_integer() {
                    self.diagnostic(
                        DiagnosticLevel::Error,
                        "cannot cast non-numeric type",
                        expr.location(),
                    );
                    return ExprValue {
                        value: Some(self.add_poison(target_type)),
                        ty: target_type,
                    };
                }
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
                ExprValue {
                    value: operation_builder.finish(call),
                    ty: target_type,
                }
            }
            Expr::Return(_) => todo!(),
            Expr::Yield(_) => todo!(),
            Expr::Cond(cond_arms) => todo!(),
        }
    }
    fn short_circuit(&self, is_or: bool, lhs: &ExprBox, rhs: &ExprBox, loc: Location) -> ValID {
        let boolean_type = self.get_primitive_type(Primitive::Bool);
        let never_type = self.get_primitive_type(Primitive::Never);
        let mut cond;
        match self.add_expr(lhs, true, None) {
            ExprValue {
                value: Some(value),
                ty,
            } if self.unify(ty, boolean_type) => {
                cond = value;
            }
            ExprValue { value: None, .. } => {
                self.diagnostic(
                    DiagnosticLevel::Ice,
                    "LHS does not return a value properly",
                    loc,
                );
                return self.add_poison(boolean_type);
            }
            _ => {
                self.diagnostic(
                    DiagnosticLevel::Error,
                    "type mismatch in short-circuit expression",
                    loc,
                );
                return self.add_poison(boolean_type);
            }
        };
        if !is_or {
            let mut operation_builder = self.new_operation();
            operation_builder.add_location(loc);
            operation_builder.add_output(boolean_type, None);
            let target_function = path!["not", "core", "intrinsics", "arith"];
            let type_params = self.alloc_slice_fill_iter([boolean_type]);
            let symbol = self.alloc(Symbol {
                path: target_function,
                type_params: Some(type_params),
            });
            let call = OperationKind::FnCall {
                target: symbol,
                args: Some(self.alloc_slice_fill_iter([cond])),
            };
            cond = operation_builder.finish(call).unwrap();
        }
        // now build the if-then-else, the first block returns constant true or false
        let then_region = {
            let block_builder = BlockBuilder::new(self.parent);
            let mut operation_builder = block_builder.new_operation();
            operation_builder.add_location(loc);
            operation_builder.add_output(boolean_type, None);
            let val = operation_builder
                .finish(OperationKind::ConstantBool(is_or))
                .unwrap();
            let mut operation_builder = block_builder.new_operation();
            operation_builder.add_location(loc);
            operation_builder.add_output(never_type, None);
            operation_builder.finish(OperationKind::Yield(Some(val)));
            self.alloc(block_builder.build())
        };
        let else_region = {
            let block_builder = BlockBuilder::new(self.parent);
            let ExprValue { value, ty } = block_builder.add_expr(rhs, true, None);
            if !self.unify(ty, boolean_type) {
                block_builder.diagnostic(
                    DiagnosticLevel::Error,
                    "type mismatch in short-circuit expression",
                    loc,
                );
            }
            let value = value.unwrap_or_else(|| block_builder.add_poison(boolean_type));
            let mut operation_builder = block_builder.new_operation();
            operation_builder.add_location(loc);
            operation_builder.add_output(never_type, None);
            operation_builder.finish(OperationKind::Yield(Some(value)));
            self.alloc(block_builder.build())
        };
        let mut operation_builder = self.new_operation();
        operation_builder.add_location(loc);
        operation_builder.add_output(boolean_type, None);
        operation_builder
            .finish(OperationKind::Condition {
                cond,
                then_region,
                else_region,
            })
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use reussir_front::lexer::Token;

    use crate::builder::IRBuilder;

    use super::*;

    macro_rules! test_expr_codegen {
        ($input:literal, $name:ident, $has_error:expr) => {
            #[test]
            fn $name() {
                use chumsky::prelude::*;
                let bump = bumpalo::Bump::new();
                let builder = IRBuilder::new(&bump);
                let mut parser_state = reussir_front::ParserState::new(path!("test"), "<stdin>");
                let expr_input = $input;
                let expr_parser = reussir_front::expr::expr();
                let token_stream = Token::stream(Ustr::from("<stdin>"), expr_input);
                let res = expr_parser
                    .parse_with_state(token_stream, &mut parser_state)
                    .unwrap();
                {
                    let blk_builder = BlockBuilder::new(&builder);
                    blk_builder.add_expr(&res, true, Some(Ustr::from("result")));
                    let blk = blk_builder.build();
                    for op in blk.0 {
                        println!("{:?}", op);
                    }
                }
                builder.report_errors(expr_input, Ustr::from("<stdin>"));
                if $has_error {
                    assert!(
                        builder.has_errors(),
                        "Expected errors during code generation"
                    );
                } else {
                    assert!(
                        !builder.has_errors(),
                        "Unexpected errors during code generation"
                    );
                }
            }
        };
    }

    test_expr_codegen!("1 + 2", test_addition_codegen, false);
    test_expr_codegen!(
        "1.0 * 2.0 + 3.0 * (114.512 + 1.0) + (5 * 5) as f32",
        test_complex_arithmetic_codegen,
        false
    );
    test_expr_codegen!("12 as f32 + () as f32", test_casting_unit_error, true);
    test_expr_codegen!("1.0 + 12 as f64 + 1.0f32", test_mismatched_types, true);

    test_expr_codegen!(
        "true || 1 >= 16 && false && (2 + 5 <= 3 * 8)",
        test_short_circuiting_codegen,
        false
    );
}
