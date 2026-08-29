//! Statement, type, and pattern grammar.
//!
//! Grammar summary:
//!
//! ```text
//! source-file ::= stmt*
//! stmt        ::= 'pub'? (fn-stmt | struct-stmt | enum-stmt | mod-stmt
//!                           | extern-stmt | transform-stmt | import-stmt)
//! transform-stmt ::= 'transform' RAW-MLIR-LITERAL ';'
//! import-stmt ::= 'import' path ('as' name)? ';'
//! fn-stmt     ::= 'regional'? 'fn' name generics? '(' params ')' ('->' '[flex]'? type)?
//!                     (block | ';' | RAW-MLIR-LITERAL ';')
//! trait-stmt  ::= 'trait' name generics? (':' path ('+' path)*)? '{' trait-method* '}'
//! trait-method ::= 'regional'? 'fn' name generics? '(' params ')' ('->' '[flex]'? type)? ';'
//! impl-stmt   ::= 'impl' generics? (type-atom 'for')? type-atom '{' (pub? regional? fn ...)* '}'
//! struct-stmt ::= 'struct' capability? name generics? (named-fields | unnamed-fields | ';')
//! enum-stmt   ::= 'enum' capability? name generics? '{' variant,* '}'
//! mod-stmt    ::= 'mod' name ';'
//! extern-stmt ::= 'extern' STRING ('trampoline' STRING '=' path ('<' type,* '>')? ';'
//!                                   | RAW-MLIR-LITERAL ';')
//! type        ::= segment ('->' type)?          (right associative)
//! segment     ::= '(' type,+ ')' | '[' type ';' expr,+ ']' | prim | path ('<' type,+ '>')?
//! pattern     ::= pattern-kind ('if' expr)?
//! ```
//!
//! Identifier positions accept keyword tokens as well (the language has no
//! reserved words; see [`SyntaxKind::is_ident_like`]), and contextual words
//! such as capability names and primitive type names are plain identifiers
//! matched by text.

use super::{CompletedMarker, Marker, Parser, STACK_GROW, STACK_RED_ZONE};
use crate::kind::SyntaxKind::{self, *};

/// The contextual primitive type names the parser recognizes. Re-exported at
/// the crate root: downstream consumers that match on primitive names (the
/// AST emitter here, `surface` lowering in `reussir-core`) must cover exactly
/// this set, and their tests iterate it so a new entry cannot silently miss
/// one of the mirrors.
pub const PRIM_TYPES: &[&str] = &[
    "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "f32", "f64", "bfloat16",
    "float8", "bool", "str", "char", "unit",
];

/// The contextual capability names, under the same contract as
/// [`PRIM_TYPES`].
pub const CAPABILITIES: &[&str] = &["shared", "value", "flex", "rigid", "field", "regional"];

impl Parser<'_> {
    pub(crate) fn source_file(&mut self) {
        let m = self.start();
        while !self.at_eof() {
            if self.current().starts_stmt()
                || self.at_transform_stmt()
                || self.at_impl_stmt()
                || self.at_trait_stmt()
            {
                self.stmt();
            } else {
                // Statement-level recovery: collect the unexpected tokens
                // into an error node until something can start a statement
                // again.
                let e = self.start();
                self.error(format!(
                    "expected a top-level item (fn, struct, enum, mod, extern, impl, trait, transform, or import), found {}",
                    self.current().describe()
                ));
                while !self.at_eof()
                    && !self.current().starts_stmt()
                    && !self.at_transform_stmt()
                    && !self.at_impl_stmt()
                    && !self.at_trait_stmt()
                {
                    self.bump();
                }
                e.complete(self, ErrorNode);
            }
        }
        m.complete(self, SourceFile);
    }

    /// Optional outer attributes, optional `pub`, then one of the item forms.
    fn stmt(&mut self) {
        let m = self.start();
        self.attrs_opt();
        if self.at(PubKw) {
            self.bump();
        }
        match self.current() {
            RegionalKw | FnKw => self.fn_stmt(m),
            StructKw => self.struct_stmt(m),
            EnumKw => self.enum_stmt(m),
            ModKw => self.mod_stmt(m),
            ExternKw => self.extern_trampoline_stmt(m),
            ImportKw => self.import_stmt(m),
            Ident if self.at_transform_stmt() => self.transform_stmt(m),
            Ident if self.at_impl_stmt() => self.impl_stmt(m),
            Ident if self.at_trait_stmt() => self.trait_stmt(m),
            _ => {
                self.error(format!(
                    "expected `fn`, `struct`, `enum`, `mod`, `extern`, `impl`, `trait`, `transform`, or `import` after visibility, found {}",
                    self.current().describe()
                ));
                // Consume nothing further; the outer loop recovers.
                m.complete(self, ErrorNode);
            }
        }
    }

    fn at_transform_stmt(&self) -> bool {
        self.at_ctx_kw("transform") && self.nth(1) == RawMlirLiteral
    }

    /// `impl` is contextual (the language has no reserved words): it heads an
    /// impl block only when followed by a generic-parameter list or a type
    /// name — `impl + 1` stays an expression over a variable named `impl`,
    /// and `impl as i64` a cast of one.
    fn at_impl_stmt(&self) -> bool {
        self.at_ctx_kw("impl")
            && (self.nth(1) == LAngle || (self.nth(1).is_ident_like() && self.nth(1) != AsKw))
    }

    /// `trait` is likewise contextual: it heads a declaration only when its
    /// name follows — `trait + 1` stays an expression over a variable named
    /// `trait`, `trait as i64` a cast of one. No generic-list case: the
    /// trait's name precedes its generics (unlike `impl<T>`).
    fn at_trait_stmt(&self) -> bool {
        self.at_ctx_kw("trait") && self.nth(1).is_ident_like() && self.nth(1) != AsKw
    }

    /// `trait Name generics? (':' path ('+' path)*)? '{' trait-method* '}'`.
    /// Supertraits are bare `+`-separated paths; members are signatures only.
    fn trait_stmt(&mut self, m: Marker) {
        self.bump(); // the `trait` ident
        self.expect_ident("a trait name");
        if self.at(LAngle) {
            self.generic_param_list();
        }
        if self.at(Colon) {
            self.bump();
            self.path_type();
            while self.at(Plus) {
                self.bump();
                self.path_type();
            }
        }
        self.expect(LBrace);
        while !self.at(RBrace) && !self.at_eof() {
            if self.at(RegionalKw) || self.at(FnKw) {
                let member = self.start();
                self.trait_method(member);
            } else {
                // Member-level recovery: `pub` is not a member starter —
                // trait methods take the trait's visibility.
                let e = self.start();
                self.error(format!(
                    "expected `fn` or `regional fn` in a `trait` body, found {}",
                    self.current().describe()
                ));
                while !self.at_eof() && !self.at(RBrace) && !self.at(RegionalKw) && !self.at(FnKw) {
                    self.bump();
                }
                e.complete(self, ErrorNode);
            }
        }
        self.expect(RBrace);
        m.complete(self, TraitStmt);
    }

    /// A trait method signature — `fn_stmt`'s body-less twin, completing an
    /// ordinary `FnStmt` node so downstream views project members uniformly.
    fn trait_method(&mut self, m: Marker) {
        if self.at(RegionalKw) {
            self.bump();
        }
        self.expect(FnKw);
        self.expect_ident("a method name");
        if self.at(LAngle) {
            self.generic_param_list();
        }
        self.param_list();
        if self.at(Arrow) {
            let r = self.start();
            self.bump();
            self.flex_flag_opt();
            self.type_();
            r.complete(self, RetType);
        }
        if self.at(Semicolon) {
            self.bump();
        } else if self.at(LBrace) {
            self.error(
                "trait methods are signatures; end the signature with `;`                  (default method bodies are not supported)",
            );
            // Consume the body losslessly so recovery continues at the next
            // member instead of cascading.
            self.block_expr();
            if self.at(Semicolon) {
                self.bump();
            }
        } else if self.at(RawMlirLiteral) {
            self.error("trait methods cannot have a foreign body");
            self.bump();
            if self.at(Semicolon) {
                self.bump();
            }
        } else {
            self.error("expected `;` after a trait method signature");
        }
        m.complete(self, FnStmt);
    }

    /// `impl generics? (Trait<args> 'for')? Type<args>? { (pub? (regional? fn ...))* }`.
    fn impl_stmt(&mut self, m: Marker) {
        self.bump(); // the `impl` ident
        if self.at(LAngle) {
            self.generic_param_list();
        }
        if PRIM_TYPES.contains(&self.current_text()) {
            let e = self.start();
            if self.nth_text(1) == "for" {
                self.error("the trait of an `impl` must be a named trait");
            } else {
                self.error("the target of `impl` must be a named type");
            }
            self.bump();
            e.complete(self, ErrorNode);
        } else if self.type_atom().is_none() {
            self.error("expected a type name as the `impl` target");
        }
        // `impl Trait<args> for Type`: a contextual `for` after the first
        // head makes it the trait reference and the next atom the target.
        // Greedy, single-shot: `impl for { }` is an inherent impl of a type
        // named `for` (the first atom consumed it), `impl for for X { }` a
        // trait named `for` implemented for `X`.
        if self.at_ctx_kw("for") {
            self.bump();
            // A primitive name is a legal trait-impl target (`impl PartialEq
            // for i64`): `core` declares the builtin impls this way. It
            // parses as an ordinary named atom; the semantic layer resolves
            // it to the scalar type.
            if self.type_atom().is_none() {
                self.error("expected a type name as the `impl` target after `for`");
            }
        }
        self.expect(LBrace);
        while !self.at(RBrace) && !self.at_eof() {
            if self.at(PubKw) || self.at(RegionalKw) || self.at(FnKw) {
                let member = self.start();
                if self.at(PubKw) {
                    self.bump();
                }
                if self.at(RegionalKw) || self.at(FnKw) {
                    self.fn_stmt(member);
                } else {
                    self.error(format!(
                        "expected `fn` or `regional fn` after visibility, found {}",
                        self.current().describe()
                    ));
                    member.complete(self, ErrorNode);
                }
            } else {
                // Member-level recovery: skip to something that can start a
                // member (or the closing brace) under an error node.
                let e = self.start();
                self.error(format!(
                    "expected `pub`, `fn`, or `regional fn` in an `impl` block, found {}",
                    self.current().describe()
                ));
                while !self.at_eof()
                    && !self.at(RBrace)
                    && !self.at(PubKw)
                    && !self.at(RegionalKw)
                    && !self.at(FnKw)
                {
                    self.bump();
                }
                e.complete(self, ErrorNode);
            }
        }
        self.expect(RBrace);
        m.complete(self, ImplStmt);
    }

    /// `transform [{ opaque MLIR }];`.
    fn transform_stmt(&mut self, m: Marker) {
        self.bump();
        self.expect(RawMlirLiteral);
        self.expect(Semicolon);
        m.complete(self, TransformStmt);
    }

    fn fn_stmt(&mut self, m: Marker) {
        if self.at(RegionalKw) {
            self.bump();
        }
        self.expect(FnKw);
        self.expect_ident("a function name");
        if self.at(LAngle) {
            self.generic_param_list();
        }
        self.param_list();
        if self.at(Arrow) {
            let r = self.start();
            self.bump();
            self.flex_flag_opt();
            self.type_();
            r.complete(self, RetType);
        }
        if self.at(Semicolon) {
            self.bump();
        } else if self.at(LBrace) {
            self.block_expr();
        } else if self.at(RawMlirLiteral) {
            // An opaque foreign body (`[{ Rust code }]`) for FFI functions.
            self.bump();
            self.expect(Semicolon);
        } else {
            self.error("expected a function body (`{ ... }`, `[{ ... }];`) or `;`");
        }
        m.complete(self, FnStmt);
    }

    /// `( name : [flex]? type, ... )`.
    fn param_list(&mut self) {
        let m = self.start();
        if !self.expect(LParen) {
            m.complete(self, ParamList);
            return;
        }
        while !self.at(RParen) && !self.at_eof() {
            let param = self.start();
            self.expect_ident("a parameter name");
            self.expect(Colon);
            self.flex_flag_opt();
            self.type_();
            param.complete(self, Param);
            if self.at(Comma) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(RParen);
        m.complete(self, ParamList);
    }

    /// `<T, U: Bound + Other>`.
    fn generic_param_list(&mut self) {
        let m = self.start();
        self.expect(LAngle);
        while !self.at(RAngle) && !self.at_eof() {
            let param = self.start();
            self.expect_ident("a generic parameter name");
            if self.at(Colon) {
                self.bump();
                self.path();
                while self.at(Plus) {
                    self.bump();
                    self.path();
                }
            }
            param.complete(self, GenericParam);
            if self.at(Comma) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(RAngle);
        m.complete(self, GenericParamList);
    }

    fn struct_stmt(&mut self, m: Marker) {
        self.expect(StructKw);
        self.capability_opt();
        self.expect_ident("a struct name");
        if self.at(LAngle) {
            self.generic_param_list();
        }
        if self.at(LBrace) {
            self.named_fields();
        } else if self.at(LParen) {
            self.unnamed_fields();
        } else if self.at(Semicolon) {
            // A field-less declaration (`struct V<T>;`): an opaque record,
            // only meaningful with an `#[ffi(...)]` attribute.
            self.bump();
        } else {
            self.error("expected `{` (named fields), `(` (unnamed fields), or `;`");
        }
        m.complete(self, StructStmt);
    }

    /// `{ pub? name: [field]? type, ... }`.
    fn named_fields(&mut self) {
        let m = self.start();
        self.expect(LBrace);
        while !self.at(RBrace) && !self.at_eof() {
            let f = self.start();
            // `pub` is also a legal field name (no reserved words); it is a
            // visibility marker only when not immediately followed by `:`.
            if self.at(PubKw) && self.nth(1) != Colon {
                let v = self.start();
                self.bump();
                v.complete(self, VisFlag);
            }
            self.expect_ident("a field name");
            self.expect(Colon);
            self.field_flag_opt();
            self.type_();
            f.complete(self, NamedField);
            if self.at(Comma) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(RBrace);
        m.complete(self, NamedFields);
    }

    /// `( pub? [field]? type, ... )`.
    fn unnamed_fields(&mut self) {
        let m = self.start();
        self.expect(LParen);
        while !self.at(RParen) && !self.at_eof() {
            let f = self.start();
            // `pub` is also a legal type name; it is a visibility marker only
            // when what follows cannot continue a type headed by `pub`.
            if self.at(PubKw) && !matches!(self.nth(1), Comma | RParen | PathSep | LAngle | Arrow) {
                let v = self.start();
                self.bump();
                v.complete(self, VisFlag);
            }
            self.field_flag_opt();
            self.type_();
            f.complete(self, UnnamedField);
            if self.at(Comma) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(RParen);
        m.complete(self, UnnamedFields);
    }

    fn enum_stmt(&mut self, m: Marker) {
        self.expect(EnumKw);
        self.capability_opt();
        self.expect_ident("an enum name");
        if self.at(LAngle) {
            self.generic_param_list();
        }
        let vl = self.start();
        self.expect(LBrace);
        while !self.at(RBrace) && !self.at_eof() {
            let v = self.start();
            self.expect_ident("a variant name");
            if self.at(LParen) {
                self.bump();
                while !self.at(RParen) && !self.at_eof() {
                    self.type_();
                    if self.at(Comma) {
                        self.bump();
                    } else {
                        break;
                    }
                }
                self.expect(RParen);
            }
            v.complete(self, Variant);
            if self.at(Comma) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(RBrace);
        vl.complete(self, VariantList);
        m.complete(self, EnumStmt);
    }

    /// `mod name;`.
    fn mod_stmt(&mut self, m: Marker) {
        self.expect(ModKw);
        self.expect_ident("a module name");
        self.expect(Semicolon);
        m.complete(self, ModStmt);
    }

    /// `import path (as name)?;`.
    fn import_stmt(&mut self, m: Marker) {
        self.expect(ImportKw);
        self.path();
        if self.at(AsKw) {
            self.bump();
            self.expect_ident("an import name");
        }
        self.expect(Semicolon);
        m.complete(self, ImportStmt);
    }

    /// `extern "ABI" trampoline "sym" = path<T,...>;` or an opaque foreign
    /// source block `extern "rust" [{ ... }];`.
    fn extern_trampoline_stmt(&mut self, m: Marker) {
        self.expect(ExternKw);
        self.expect(StringLit);
        if self.at(RawMlirLiteral) {
            self.bump();
            self.expect(Semicolon);
            m.complete(self, ExternSourceStmt);
            return;
        }
        if self.at_ctx_kw("trampoline") {
            self.bump();
        } else {
            self.error(format!(
                "expected `trampoline` or `[{{ ... }}]`, found {}",
                self.current().describe()
            ));
        }
        self.expect(StringLit);
        self.expect(Eq);
        self.path();
        if self.at(LAngle) {
            let args = self.start();
            self.bump();
            while !self.at(RAngle) && !self.at_eof() {
                self.type_();
                if self.at(Comma) {
                    self.bump();
                } else {
                    break;
                }
            }
            self.expect(RAngle);
            args.complete(self, TypeArgList);
        }
        self.expect(Semicolon);
        m.complete(self, ExternTrampolineStmt);
    }

    /// Zero or more outer attributes `#[ ... ]` preceding an item. Each is
    /// parsed as an `AttrList` node holding the bracketed contents verbatim
    /// (an identifier, optionally with a parenthesized argument list) so the
    /// surface layer can interpret e.g. `#[repr(fixed)]`. Unknown attributes
    /// parse fine here; the surface/HIR layers decide what they mean.
    fn attrs_opt(&mut self) {
        while self.at(Pound) {
            let m = self.start();
            self.bump();
            self.expect(LBracket);
            self.expect_ident("an attribute name");
            if self.at(LParen) {
                self.bump();
                while !self.at(RParen) && !self.at_eof() {
                    if self.at(Comma) {
                        self.bump();
                    } else if self.at_ident_like() {
                        // A bare argument, or a `key = "value"` pair.
                        self.bump();
                        if self.at(Eq) {
                            self.bump();
                            self.expect(StringLit);
                        }
                    } else if self.at(StringLit) {
                        // A bare string argument (`#[lang("partial_eq")]`).
                        self.bump();
                    } else {
                        self.error("expected an attribute argument");
                        break;
                    }
                }
                self.expect(RParen);
            }
            self.expect(RBracket);
            m.complete(self, AttrList);
        }
    }

    /// Optional `[shared]`, `[regional]`, ... on records.
    fn capability_opt(&mut self) {
        if !self.at(LBracket) {
            return;
        }
        let m = self.start();
        self.bump();
        if self.at(RegionalKw) || CAPABILITIES.iter().any(|c| self.at_ctx_kw(c)) {
            self.bump();
        } else {
            self.error(
                "expected a capability (`shared`, `value`, `flex`, `rigid`, `field`, or `regional`)",
            );
        }
        self.expect(RBracket);
        m.complete(self, Capability);
    }

    /// Optional `[field]` on record fields.
    fn field_flag_opt(&mut self) {
        if self.at(LBracket) && self.nth_text(1) == "field" {
            let m = self.start();
            self.bump();
            self.bump();
            self.expect(RBracket);
            m.complete(self, FieldFlag);
        }
    }

    /// Optional `[flex]` on parameters, return types, and let bindings.
    pub(crate) fn flex_flag_opt(&mut self) {
        if self.at(LBracket) && self.nth_text(1) == "flex" {
            let m = self.start();
            self.bump();
            self.bump();
            self.expect(RBracket);
            m.complete(self, FlexFlag);
        }
    }

    /// `ident (:: ident)*`.
    pub(crate) fn path(&mut self) {
        let m = self.start();
        self.expect_ident("a path segment");
        while self.at(PathSep) {
            self.bump();
            self.expect_ident("a path segment");
        }
        m.complete(self, Path);
    }

    // ===== Types =====

    /// A right-associative chain of arrow segments. A parenthesized list of
    /// two or more types denotes the argument tuple of a function type and
    /// is only meaningful directly to the left of an `->`.
    pub(crate) fn type_(&mut self) {
        stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW, || {
            let Some((seg, arity)) = self.type_segment() else {
                self.error(format!(
                    "expected a type, found {}",
                    self.current().describe()
                ));
                return;
            };
            if self.at(Arrow) {
                let m = seg.precede(self);
                self.bump();
                self.type_();
                m.complete(self, ArrowType);
            } else if arity > 1 {
                self.error(
                    "a parenthesized type list is only allowed to the left of `->`; \
                     use a single type here",
                );
            }
        })
    }

    /// One arrow segment: either `( T, U, ... )` or a type atom. Returns
    /// the completed node and the number of types it groups.
    fn type_segment(&mut self) -> Option<(CompletedMarker, usize)> {
        if self.at(LParen) {
            let m = self.start();
            self.bump();
            self.type_();
            let mut arity = 1;
            while self.at(Comma) {
                self.bump();
                self.type_();
                arity += 1;
            }
            self.expect(RParen);
            let done = m.complete(self, ParenTypeList);
            return Some((done, arity));
        }
        if self.at(LBracket) {
            return Some((self.array_type(), 1));
        }
        self.type_atom().map(|m| (m, 1))
    }

    /// A primitive type name (`i32`, `bool`, ...; matched by text) or a
    /// path with optional type arguments.
    fn type_atom(&mut self) -> Option<CompletedMarker> {
        if !self.at_ident_like() {
            return None;
        }
        if PRIM_TYPES.contains(&self.current_text()) {
            let m = self.start();
            self.bump();
            return Some(m.complete(self, PrimType));
        }
        Some(self.path_type())
    }

    /// A named-type reference — a qualified path with optional type
    /// arguments, completed as a `PathType` node. Shared by type atoms,
    /// the `impl` heads, and supertrait references (which accept the same
    /// shape; whether arguments are *admitted* there is semantics).
    fn path_type(&mut self) -> CompletedMarker {
        let m = self.start();
        self.path();
        if self.at(LAngle) {
            // In type position `<` always opens type arguments.
            let args = self.start();
            self.bump();
            self.type_();
            while self.at(Comma) {
                self.bump();
                self.type_();
            }
            self.expect(RAngle);
            args.complete(self, TypeArgList);
        }
        m.complete(self, PathType)
    }

    /// An array type: `[T; e1, e2, ...]` with one extent per dimension. An
    /// extent is an expression, or `_` for a dynamic dimension (its value is
    /// then a runtime operand of the constructing intrinsic). The grammar
    /// accepts any expression so the tree stays forward-compatible with
    /// constant expressions; what an extent may evaluate to is the
    /// elaborator's concern.
    fn array_type(&mut self) -> CompletedMarker {
        let m = self.start();
        self.bump();
        self.type_();
        self.expect(Semicolon);
        self.array_extent();
        while self.at(Comma) {
            self.bump();
            self.array_extent();
        }
        self.expect(RBracket);
        m.complete(self, ArrayType)
    }

    /// One extent position: `_` (a dynamic dimension, kept as an `InferType`
    /// node so it never enters the expression grammar) or an expression.
    fn array_extent(&mut self) {
        if self.at(Underscore) {
            let h = self.start();
            self.bump();
            h.complete(self, InferType);
        } else {
            self.expr();
        }
    }

    // ===== Patterns =====

    /// A pattern kind with an optional `if` guard.
    pub(crate) fn pattern(&mut self) {
        let m = self.start();
        self.pattern_kind();
        if self.at(IfKw) {
            let g = self.start();
            self.bump();
            self.expr();
            g.complete(self, PatGuard);
        }
        m.complete(self, Pattern);
    }

    pub(crate) fn pattern_kind(&mut self) {
        stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW, || match self.current() {
            Underscore => {
                let m = self.start();
                self.bump();
                m.complete(self, WildcardPat);
            }
            IntLit | FloatLit | StringLit | CharLit | TrueKw | FalseKw => {
                let m = self.start();
                self.bump();
                m.complete(self, ConstPat);
            }
            k if k.is_ident_like() => self.path_or_bind_pattern(),
            _ => {
                self.error(format!(
                    "expected a pattern, found {}",
                    self.current().describe()
                ));
            }
        })
    }

    /// After a path, `{...}` and `(...)` open constructor patterns. A bare
    /// path is a constructor when it is qualified or starts with an
    /// uppercase letter, and a fresh binding otherwise.
    fn path_or_bind_pattern(&mut self) {
        let m = self.start();
        let single_segment = self.nth(1) != PathSep;
        let first_char_upper = self
            .current_text()
            .chars()
            .next()
            .is_some_and(char::is_uppercase);
        self.path();
        if self.at(LBrace) {
            self.ctor_pat_args(LBrace, RBrace);
            m.complete(self, CtorPat);
        } else if self.at(LParen) {
            self.ctor_pat_args(LParen, RParen);
            m.complete(self, CtorPat);
        } else if single_segment && !first_char_upper {
            m.complete(self, BindPat);
        } else {
            m.complete(self, CtorPat);
        }
    }

    /// Constructor pattern arguments with an optional trailing `..` (and a
    /// tolerated trailing comma). The brace form takes named arguments
    /// (`name` or `name: pattern`), the paren form positional ones.
    fn ctor_pat_args(&mut self, open: SyntaxKind, close: SyntaxKind) {
        let named = open == LBrace;
        let m = self.start();
        self.expect(open);
        loop {
            if self.at(close) || self.at_eof() {
                break;
            }
            if self.at(DotDot) {
                let r = self.start();
                self.bump();
                r.complete(self, PatRest);
                break;
            }
            let arg = self.start();
            if named {
                self.expect_ident("a field name or pattern");
                if self.at(Colon) {
                    self.bump();
                    self.pattern_kind();
                }
            } else {
                self.pattern_kind();
            }
            arg.complete(self, PatArg);
            if self.at(Comma) {
                self.bump();
            } else {
                break;
            }
        }
        self.expect(close);
        m.complete(self, PatArgList);
    }
}
