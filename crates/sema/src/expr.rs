use crate::builder::ValueDef;
use crate::builder::{BlockBuilder, DiagnosticLevel};
use reussir_core::{Location, Path};
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

fn call_target_as_plain_path(target: &CallTarget) -> Option<Path> {
    let mut path_segments = Vec::with_capacity(target.len());
    for segment in target.iter() {
        match segment {
            CallTargetSegment::TurboFish(_) => return None,
            CallTargetSegment::Ident(name) => path_segments.push(*name),
        }
    }
    let last = path_segments.pop()?;
    Some(Path::new(last, path_segments))
}

impl<'b, 'a: 'b> BlockBuilder<'b, 'a> {
    pub fn unify(&self, lhs: &'a Type, rhs: &'a Type) -> bool {
        if core::ptr::eq(lhs, rhs) || lhs == rhs {
            true
        } else {
            // Cannot unify besides pure equality for now
            // TODO: add meta variables and solve variables during unification
            lhs.is_never() || rhs.is_never()
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

    fn add_unit(&self) -> usize {
        let ty = self.get_primitive_type(Primitive::Unit);
        let mut operation_builder = self.new_operation();
        operation_builder.add_output(ty, None);
        let value = operation_builder.finish(OperationKind::Unit);
        value.unwrap()
    }

    fn add_panic(&self, message: &'a str) -> (usize, &'a Type) {
        let mut operation_builder = self.new_operation();
        let never = self.get_primitive_type(Primitive::Never);
        operation_builder.add_output(never, None);
        let value = operation_builder.finish(OperationKind::Panic(message));
        (value.unwrap(), never)
    }

    pub fn add_expr_expect_value(
        &self,
        expr: &ExprBox,
        used: bool,
        name: Option<Ustr>,
    ) -> (ValID, &'a Type) {
        let ExprValue { value, ty } = self.add_expr(expr, used, name);
        if let Some(value) = value {
            (value, ty)
        } else {
            self.diagnostic(
                DiagnosticLevel::Ice,
                "expression does not return a value properly",
                expr.location(),
            );
            let never = self.get_primitive_type(Primitive::Never);
            (self.add_poison(never), never)
        }
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
                let (lhs_value, lhs_ty) = self.add_expr_expect_value(lhs, true, None);
                let (rhs_value, rhs_ty) = self.add_expr_expect_value(rhs, true, None);
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
                            self.diagnostic(
                                DiagnosticLevel::Error,
                                "unsupported type for arithmetic operations",
                                expr.location(),
                            );
                            return self.add_poison_never();
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
                            self.diagnostic(
                                DiagnosticLevel::Error,
                                "unsupported type for arithmetic operations",
                                expr.location(),
                            );
                            return self.add_poison_never();
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
                        DiagnosticLevel::Error,
                        "undefined variable",
                        expr.location(),
                    );
                    return self.add_poison_never();
                };
                ExprValue {
                    value: Some(val),
                    ty,
                }
            }
            Expr::IfThenElse(cond, then_branch, None) => {
                let unit_ty = self.get_primitive_type(Primitive::Unit);
                let boolean_type = self.get_primitive_type(Primitive::Bool);
                let (cond, ty) = self.add_expr_expect_value(cond, true, None);
                if !self.unify(ty, boolean_type) {
                    self.diagnostic(
                        DiagnosticLevel::Error,
                        "condition in if-then-else must be a boolean",
                        expr.location(),
                    );
                    return ExprValue {
                        value: Some(self.add_poison(boolean_type)),
                        ty: boolean_type,
                    };
                }
                if used {
                    operation_builder.add_output(unit_ty, name);
                }
                let then_region = self.alloc({
                    let block_builder = BlockBuilder::new(self.parent);
                    block_builder.add_expr(then_branch, false, None);
                    let mut op_builder = block_builder.new_operation();
                    op_builder.add_location(expr.location());
                    op_builder.finish(OperationKind::Yield(None));
                    block_builder.build()
                });
                ExprValue {
                    value: operation_builder.finish(OperationKind::Condition {
                        cond,
                        then_region,
                        else_region: None,
                    }),
                    ty,
                }
            }
            Expr::IfThenElse(cond, then_branch, Some(else_branch)) => {
                let boolean_type = self.get_primitive_type(Primitive::Bool);
                let (cond, ty) = self.add_expr_expect_value(cond, true, None);
                if !self.unify(ty, boolean_type) {
                    self.diagnostic(
                        DiagnosticLevel::Error,
                        "condition in if-then-else must be a boolean",
                        expr.location(),
                    );
                    return ExprValue {
                        value: Some(self.add_poison(boolean_type)),
                        ty: boolean_type,
                    };
                }
                let (then_ty, then_region) = {
                    let block_builder = BlockBuilder::new(self.parent);
                    let (val, ty) = block_builder.add_expr_expect_value(then_branch, true, None);
                    let mut operation_builder = block_builder.new_operation();
                    operation_builder.add_location(expr.location());
                    operation_builder.finish(OperationKind::Yield(Some(val)));
                    (ty, self.alloc(block_builder.build()))
                };
                let (else_ty, else_region) = self.alloc({
                    let block_builder = BlockBuilder::new(self.parent);
                    let (val, ty) = block_builder.add_expr_expect_value(else_branch, true, None);
                    let mut operation_builder = block_builder.new_operation();
                    operation_builder.add_location(expr.location());
                    operation_builder.finish(OperationKind::Yield(Some(val)));
                    (ty, block_builder.build())
                });
                if !self.unify(then_ty, else_ty) {
                    self.diagnostic(
                        DiagnosticLevel::Error,
                        "type mismatch in branches of if-then-else",
                        expr.location(),
                    );
                }
                let else_region = Some(else_region);
                ExprValue {
                    value: operation_builder.finish(OperationKind::Condition {
                        cond,
                        then_region,
                        else_region,
                    }),
                    ty,
                }
            }
            Expr::Sequence(items) => {
                for (idx, item) in items.iter().enumerate() {
                    let used = used && idx == items.len() - 1;
                    if !used {
                        self.add_expr(item, false, None);
                    } else {
                        let (value, expr_ty) = self.add_expr_expect_value(item, true, None);
                        return ExprValue {
                            value: Some(value),
                            ty: expr_ty,
                        };
                    }
                }
                let value = if used { Some(self.add_unit()) } else { None };
                let unit_ty = self.get_primitive_type(Primitive::Unit);
                ExprValue { value, ty: unit_ty }
            }
            Expr::Let(x, ty, val) => {
                let (_, expr_ty) = self.add_expr_expect_value(val, true, Some(x.as_str().into()));
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
                let value = if used { Some(self.add_unit()) } else { None };
                ExprValue { value, ty: unit_ty }
            }
            // TODO: tuple-like ctor can also be in this Call expression.
            // TODO: lambda call should also be handled here.
            Expr::Call(target, items) => {
                if let Some(path) = call_target_as_plain_path(target) {
                    // TODO: currently this is hard coded. Add a better way to resolve visiable aliases.
                    let prefixed_function = Path::new(
                        path.basename(),
                        self.ctx()
                            .module()
                            .segments()
                            .iter()
                            .copied()
                            .chain(path.prefix().iter().copied()),
                    );
                    if let Some(proto) = self
                        .functions()
                        .get(&prefixed_function)
                        .or_else(|| self.functions().get(&path))
                    {
                        let args = items
                            .iter()
                            .flatten()
                            .map(|item| {
                                let (value, ty) = self.add_expr_expect_value(item, true, None);
                                (value, ty)
                            })
                            .collect::<Vec<_>>();
                        let mut unification_failed = false;
                        let return_type = self.alloc(proto.return_type.clone());
                        for (_, ty) in args.iter() {
                            if !self.unify(return_type, ty) {
                                self.diagnostic(
                                    DiagnosticLevel::Error,
                                    "type mismatch in function call",
                                    expr.location(),
                                );
                                unification_failed = true;
                            }
                        }
                        if unification_failed {
                            let value = self.add_poison(return_type);
                            return ExprValue {
                                value: Some(value),
                                ty: return_type,
                            };
                        }
                        operation_builder.add_output(return_type, name);
                        operation_builder.add_location(expr.location());
                        let symbol = self.alloc(Symbol {
                            path,
                            type_params: None,
                        });
                        ExprValue {
                            value: operation_builder.finish(OperationKind::FnCall {
                                target: symbol,
                                args: Some(
                                    self.alloc_slice_fill_iter(args.iter().map(|(v, _)| *v)),
                                ),
                            }),
                            ty: return_type,
                        }
                    } else {
                        let mut builder = self.diagnostic(
                            DiagnosticLevel::Error,
                            "function not found",
                            target.location(),
                        );
                        for fuzzy_proto in self.functions().fuzzy_search(&path).into_iter().take(2)
                        {
                            builder.add_nested_label(
                                fuzzy_proto.name_location,
                                "A function with similar name is found",
                            );
                        }
                        self.add_poison_never()
                    }
                } else {
                    self.diagnostic(
                        DiagnosticLevel::Ice,
                        "polymorphic functions are not supported yet",
                        target.location(),
                    );
                    self.add_poison_never()
                }
            }
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
            operation_builder.finish(OperationKind::Yield(Some(value)));
            self.alloc(block_builder.build())
        };
        let mut operation_builder = self.new_operation();
        let else_region = Some(else_region);
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
                let ctx = reussir_core::Context::new(path!("test"));
                let builder = IRBuilder::new(&ctx);
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

    test_expr_codegen!(
        "{ let x = 1; let y = 2; x + y }",
        test_let_binding_codegen,
        false
    );

    test_expr_codegen!("test(1, 2, 3)", test_function_call_codegen, true);
}
