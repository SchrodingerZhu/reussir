//! Expression parsing, built around a Pratt loop.
//!
//! Operator table (tightest to loosest):
//!
//! | level | operators                                      | associativity |
//! |-------|------------------------------------------------|---------------|
//! | 6     | prefix `-` `!`; postfix `as T` or suffix chain | one prefix and one postfix application per term |
//! | 5     | `*` `/` `%`                                    | left          |
//! | 4     | `+` `-`                                        | left          |
//! | 3     | `==` `!=` `<` `>` `<=` `>=`                    | none          |
//! | 2     | `&&`                                           | left          |
//! | 1     | `\|\|`                                         | left          |
//! | 0     | `lhs -> field := rhs`                          | none          |
//!
//! ## The algorithm: precedence climbing
//!
//! Expressions are parsed by [`Parser::expr_bp`], a single loop implementing
//! *precedence climbing* (the loop form of Pratt parsing). `expr_bp(min_bp)`
//! parses the largest expression whose operators all bind at least as tightly
//! as `min_bp`: it reads one term, then repeatedly
//!
//! 1. peeks at the next operator and looks up its [`BindingPower`];
//! 2. if the operator's `left` power is `< min_bp`, stops — the operator is
//!    looser than this invocation is allowed to take, so it belongs to an
//!    enclosing call; the term parsed so far is returned;
//! 3. otherwise consumes the operator, parses the right operand by recursing
//!    with `min_bp = right`, folds `lhs op rhs` into the new `lhs` (via
//!    [`CompletedMarker::precede`], so the operator node wraps the
//!    already-parsed left side), and continues the loop.
//!
//! Associativity is therefore *not* special-cased; it falls out of the gap
//! between `left` and `right` (see [`BindingPower`]). A left-associative
//! operator parses its right side with a strictly higher minimum, so an equal
//! operator to the right is left for the loop to fold as a sibling.
//!
//! Worked example — `1 + 2 * 3`, entered at `min_bp = 0`. `*` (left 10)
//! outbinds `+` (left 8), so the recursion for `+`'s right operand swallows
//! the whole product before control returns:
//!
//! ```text
//!   expr_bp(0)
//!   │ term ........................... 1
//!   │ peek '+'  left 8 ≥ 0  → take; recurse for rhs at min_bp = 9
//!   │   expr_bp(9)
//!   │   │ term ....................... 2
//!   │   │ peek '*'  left 10 ≥ 9 → take; recurse for rhs at min_bp = 11
//!   │   │   expr_bp(11)
//!   │   │   │ term ................... 3
//!   │   │   │ peek <none>          → return 3
//!   │   │ fold (2 * 3); peek <none> → return (2 * 3)
//!   │ fold (1 + (2 * 3)); peek <eof> → return (1 + (2 * 3))
//! ```
//!
//! For a left-associative chain `a - b - c` the second `-` (left 8) is below
//! the right minimum (9) used to parse `b`, so `b` does not absorb it; the
//! loop folds `(a - b)` and re-enters, giving `((a - b) - c)`. A
//! non-associative `a < b < c` would likewise not nest, but the second `<` at
//! the same level trips the `non_assoc` chain check and is diagnosed instead.
//!
//! Details worth knowing:
//!
//! * a lambda is only recognized at full-expression entry points, never as
//!   an operand of a binary operator — `1 + |x| x` is not an expression;
//! * each term takes at most one prefix and one postfix application, and
//!   the postfix applies on top of the prefix: `- x as i32` is
//!   `Cast(i32, Negate x)`, while `x.f as i32` needs parentheses because a
//!   suffix chain and a cast are alternative postfixes;
//! * consecutive `.a.b` accesses group into a single access-chain node,
//!   while every call suffix creates its own node;
//! * struct-literal constructors are suppressed in scrutinee/condition
//!   position (`match x { ... }` does not read `x {` as a constructor) and
//!   re-enabled inside any bracketed subexpression;
//! * `a < b` is resolved against `a<b>(c)` by speculatively parsing the
//!   type-argument list and rolling back on failure.

use super::{CompletedMarker, Parser, STACK_GROW, STACK_RED_ZONE};
use crate::kind::SyntaxKind::{self, *};

/// The precedence and associativity of a binary operator, in the form the
/// Pratt loop in [`Parser::expr_bp`] consumes.
///
/// `left` and `right` are the operator's binding powers toward its left and
/// right operands. The loop compares an operator's `left` against the
/// caller's `min_bp` to decide whether to take it, then recurses for the
/// right operand using `right` as the new `min_bp`. Associativity is encoded
/// purely in the gap between the two:
///
/// * **left-associative** sets `right = left + 1`, so a following operator of
///   equal precedence fails the `left >= min_bp` test in the recursion and
///   binds as a sibling instead of nesting (`a - b - c` → `(a - b) - c`);
/// * **non-associative** comparisons additionally set `non_assoc`, which turns
///   a repeat at the same level into the "cannot be chained" diagnostic rather
///   than a (mis-grouped) tree.
#[derive(Clone, Copy)]
struct BindingPower {
    left: u8,
    right: u8,
    non_assoc: bool,
}

impl BindingPower {
    const fn left_assoc(left: u8) -> Self {
        BindingPower {
            left,
            right: left + 1,
            non_assoc: false,
        }
    }

    const fn non_assoc(left: u8) -> Self {
        BindingPower {
            left,
            right: left + 1,
            non_assoc: true,
        }
    }
}

/// The binding power of `kind` as an infix operator, or `None` if it is not
/// one. Levels are spaced by two so the ladder stays readable; relative
/// order is what matters, not the absolute numbers.
fn binary_bp(kind: SyntaxKind) -> Option<BindingPower> {
    Some(match kind {
        PipePipe => BindingPower::left_assoc(2),
        AmpAmp => BindingPower::left_assoc(4),
        EqEq | BangEq | LAngle | RAngle | LtEq | GtEq => BindingPower::non_assoc(6),
        Plus | Minus => BindingPower::left_assoc(8),
        Star | Slash | Percent => BindingPower::left_assoc(10),
        _ => return None,
    })
}

const ASSIGN_BP: u8 = 1;

impl Parser<'_> {
    /// A full expression (struct literals allowed).
    pub(crate) fn expr(&mut self) {
        self.expr_entry(true);
    }

    /// An expression in scrutinee/condition position, where `path {`
    /// is not a struct literal.
    pub(crate) fn expr_no_struct(&mut self) {
        self.expr_entry(false);
    }

    fn expr_entry(&mut self, allow_struct: bool) {
        // The lambda alternative is tried before the operator parser at
        // every expression entry.
        if self.at(Pipe) || self.at(PipePipe) {
            self.lambda_expr();
            return;
        }
        if self.expr_bp(0, allow_struct).is_none() {
            self.error(format!(
                "expected an expression, found {}",
                self.current().describe()
            ));
        }
    }

    fn expr_bp(&mut self, min_bp: u8, allow_struct: bool) -> Option<CompletedMarker> {
        let mut lhs = self.unary_term(allow_struct)?;
        let mut nonassoc_at: Option<u8> = None;
        loop {
            // `lhs -> access := rhs`: non-associative, loosest.
            if min_bp <= ASSIGN_BP
                && self.at(Arrow)
                && (self.nth(1).is_ident_like() || self.nth(1) == IntLit)
                && self.nth(2) == ColonEq
            {
                if nonassoc_at == Some(ASSIGN_BP) {
                    self.error("assignment cannot be chained");
                }
                let m = lhs.precede(self);
                let seg = self.start();
                self.bump(); // ->
                self.bump(); // field name or index
                seg.complete(self, AccessSeg);
                self.bump(); // :=
                if self.expr_bp(ASSIGN_BP + 1, allow_struct).is_none() {
                    self.error("expected an expression after `:=`");
                }
                lhs = m.complete(self, AssignExpr);
                nonassoc_at = Some(ASSIGN_BP);
                continue;
            }

            let Some(bp) = binary_bp(self.current()) else {
                break;
            };
            if bp.left < min_bp {
                break;
            }
            if bp.non_assoc && nonassoc_at == Some(bp.left) {
                self.error("comparison operators cannot be chained; use parentheses");
                break;
            }
            let m = lhs.precede(self);
            self.bump(); // the operator
            if self.expr_bp(bp.right, allow_struct).is_none() {
                self.error(format!(
                    "expected an expression after the operator, found {}",
                    self.current().describe()
                ));
            }
            lhs = m.complete(self, BinExpr);
            nonassoc_at = bp.non_assoc.then_some(bp.left);
        }
        Some(lhs)
    }

    /// One term: at most one prefix and one postfix application; the
    /// postfix applies to the prefixed atom.
    fn unary_term(&mut self, allow_struct: bool) -> Option<CompletedMarker> {
        stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW, || {
            let base = if self.at(Minus) || self.at(Bang) {
                let m = self.start();
                self.bump();
                if self.atom(allow_struct).is_none() {
                    self.error(format!(
                        "expected an expression after the unary operator, found {}",
                        self.current().describe()
                    ));
                }
                m.complete(self, PrefixExpr)
            } else {
                self.atom(allow_struct)?
            };
            Some(self.postfix_opt(base))
        })
    }

    /// One postfix application: either a cast (`as T`) or a chain of call /
    /// access suffixes.
    fn postfix_opt(&mut self, mut e: CompletedMarker) -> CompletedMarker {
        if self.at(AsKw) {
            let m = e.precede(self);
            self.bump();
            self.type_();
            return m.complete(self, CastExpr);
        }
        loop {
            if self.at(LParen) {
                let m = e.precede(self);
                self.arg_list();
                e = m.complete(self, CallExpr);
            } else if self.at(Dot) {
                let m = e.precede(self);
                while self.at(Dot) {
                    self.access_seg();
                }
                e = m.complete(self, AccessChain);
            } else {
                break;
            }
        }
        e
    }

    /// `.name` or `.0`. A float token like `0.1` after a dot denotes two
    /// consecutive numeric accesses (the lexer fuses `0.1` into one
    /// literal).
    fn access_seg(&mut self) {
        let m = self.start();
        self.bump(); // .
        if self.at_ident_like() || self.at(IntLit) {
            self.bump();
        } else if self.at(FloatLit)
            && self
                .current_text()
                .chars()
                .all(|c| c.is_ascii_digit() || c == '.')
        {
            self.bump();
        } else {
            self.error(format!(
                "expected a field name or tuple index after `.`, found {}",
                self.current().describe()
            ));
        }
        m.complete(self, AccessSeg);
    }

    /// `( expr, expr, ... )` call arguments.
    fn arg_list(&mut self) {
        let m = self.start();
        self.expect(LParen);
        while !self.at(RParen) && !self.at_eof() {
            self.expr();
            if self.at(Comma) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(RParen);
        m.complete(self, ArgList);
    }

    fn atom(&mut self, allow_struct: bool) -> Option<CompletedMarker> {
        match self.current() {
            LParen => {
                let m = self.start();
                self.bump();
                self.expr();
                self.expect(RParen);
                Some(m.complete(self, ParenExpr))
            }
            IfKw => Some(self.if_expr()),
            LetKw => Some(self.let_expr()),
            MatchKw => Some(self.match_expr()),
            RegionalKw if self.nth(1) == LBrace => {
                let m = self.start();
                self.bump();
                self.block_expr();
                Some(m.complete(self, RegionalExpr))
            }
            IntLit | FloatLit | StringLit | TrueKw | FalseKw => {
                let m = self.start();
                self.bump();
                Some(m.complete(self, LiteralExpr))
            }
            LBrace => Some(self.block_expr()),
            k if k.is_ident_like() => Some(self.path_based_expr(allow_struct)),
            _ => None,
        }
    }

    /// `if cond { ... } else { ... }` — the `else` branch is mandatory and
    /// the condition is parsed in no-struct mode.
    fn if_expr(&mut self) -> CompletedMarker {
        let m = self.start();
        self.bump(); // if
        self.expr_no_struct();
        self.block_expr();
        self.expect(ElseKw);
        self.block_expr();
        m.complete(self, IfExpr)
    }

    /// `let name (: [flex]? type)? = value`.
    fn let_expr(&mut self) -> CompletedMarker {
        let m = self.start();
        self.bump(); // let
        self.expect_ident("a binding name");
        if self.at(Colon) {
            self.bump();
            self.flex_flag_opt();
            self.type_();
        }
        self.expect(Eq);
        self.expr(); // the bound value always allows struct literals
        m.complete(self, LetExpr)
    }

    /// `match scrutinee { pat => expr, ... }` (no trailing comma).
    fn match_expr(&mut self) -> CompletedMarker {
        let m = self.start();
        self.bump(); // match
        self.expr_no_struct();
        self.expect(LBrace);
        while !self.at(RBrace) && !self.at_eof() {
            let arm = self.start();
            self.pattern();
            self.expect(FatArrow);
            self.expr();
            arm.complete(self, MatchArm);
            if self.at(Comma) {
                self.bump();
                if self.at(RBrace) {
                    self.error("match arms do not allow a trailing comma");
                    break;
                }
            } else {
                break;
            }
        }
        self.expect(RBrace);
        m.complete(self, MatchExpr)
    }

    /// `{ e1; e2; }` — semicolon-separated with an optional trailing
    /// semicolon.
    pub(crate) fn block_expr(&mut self) -> CompletedMarker {
        let m = self.start();
        self.expect(LBrace);
        while !self.at(RBrace) && !self.at_eof() {
            let before = self.snapshot();
            self.expr();
            if self.errored_since(&before) {
                // Best-effort recovery inside a block: skip to the next
                // semicolon or closing brace.
                while !self.at(Semicolon)
                    && !self.at(RBrace)
                    && !self.at_eof()
                    && !self.current().starts_stmt()
                {
                    let e = self.start();
                    self.bump();
                    e.complete(self, ErrorNode);
                }
                if self.current().starts_stmt() {
                    break;
                }
            }
            if self.at(Semicolon) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(RBrace);
        m.complete(self, BlockExpr)
    }

    /// `|x, y: T| (-> T)? body`. A `||` token is an empty parameter list
    /// (the lexer fuses the two pipes).
    fn lambda_expr(&mut self) {
        let m = self.start();
        let params = self.start();
        if self.at(PipePipe) {
            self.bump();
        } else {
            self.expect(Pipe);
            while !self.at(Pipe) && !self.at_eof() {
                let param = self.start();
                self.expect_ident("a lambda parameter");
                if self.at(Colon) {
                    self.bump();
                    self.type_();
                }
                param.complete(self, LambdaParam);
                if self.at(Comma) {
                    self.bump();
                } else {
                    break;
                }
            }
            self.expect(Pipe);
        }
        params.complete(self, LambdaParamList);
        if self.at(Arrow) {
            let r = self.start();
            self.bump();
            self.type_();
            r.complete(self, RetType);
        }
        self.expr();
        m.complete(self, LambdaExpr);
    }

    /// A path, speculative type arguments, then one of constructor call,
    /// function call, or plain variable.
    fn path_based_expr(&mut self, allow_struct: bool) -> CompletedMarker {
        let m = self.start();
        self.path();
        let mut has_ty_args = false;
        if self.at(LAngle) {
            has_ty_args = self.try_type_arg_list();
        }
        if self.at(LBrace) && allow_struct {
            self.ctor_arg_list();
            m.complete(self, CtorCallExpr)
        } else if self.at(LParen) {
            self.arg_list();
            m.complete(self, FuncCallExpr)
        } else if has_ty_args {
            m.complete(self, CtorCallExpr)
        } else {
            m.complete(self, VarExpr)
        }
    }

    /// Speculatively parse `<T, _, U>` type arguments; `a < b` must fall
    /// back to a comparison. Returns whether the arguments were committed.
    fn try_type_arg_list(&mut self) -> bool {
        let snapshot = self.snapshot();
        let m = self.start();
        self.bump(); // <
        loop {
            if self.at(Underscore) {
                let h = self.start();
                self.bump();
                h.complete(self, InferType);
            } else {
                self.type_();
            }
            if self.at(Comma) {
                self.bump();
            } else {
                break;
            }
        }
        let closed = self.at(RAngle);
        if closed {
            self.bump();
        }
        if !closed || self.errored_since(&snapshot) {
            self.rollback(snapshot);
            return false;
        }
        m.complete(self, TypeArgList);
        true
    }

    /// `{ name: expr, expr, ... }` constructor arguments. An identifier
    /// followed by a colon names the field; otherwise the whole argument is
    /// an expression.
    fn ctor_arg_list(&mut self) {
        let m = self.start();
        self.expect(LBrace);
        while !self.at(RBrace) && !self.at_eof() {
            let arg = self.start();
            if self.at_ident_like() && self.nth(1) == Colon {
                self.bump();
                self.bump();
                self.expr();
            } else {
                self.expr();
            }
            arg.complete(self, CtorArg);
            if self.at(Comma) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(RBrace);
        m.complete(self, CtorArgList);
    }
}
