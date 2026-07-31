
#import "/book.typ": book-page

#show: book-page.with(title: "Syntax")



= Syntax

The syntax is not stable yet and is subject to change. For example, type
bounds are not included, and the trait system is still under design.

== Syntax Family

Reussir adopts a traditional two-phase syntax. The input sequence is first
tokenized as a regular language and then parsed by a hand-written recursive
descent parser (with a precedence-climbing loop for expressions) into a
lossless concrete syntax tree.


== Lexical Tokens

=== Identifiers
#math.mono([$
    "IDENT" & eq.def "{XID_Start}" ~ "{XID_Continue}"
  $])
=== Symbols and Punctuations
#math.mono([$
       "PATHSEP" & eq.def "::" \
         "ARROW" & eq.def "->" \
      "FATARROW" & eq.def "=>" \
       "COLONEQ" & eq.def ":=" \
          "EQEQ" & eq.def "==" \
        "BANGEQ" & eq.def "!=" \
          "LTEQ" & eq.def "<=" \
          "GTEQ" & eq.def ">=" \
        "AMPAMP" & eq.def "&&" \
      "PIPEPIPE" & eq.def "||" \
        "DOTDOT" & eq.def ".." \
        "LPAREN" & eq.def "("  \
        "RPAREN" & eq.def ")"  \
        "LBRACE" & eq.def "{"  \
        "RBRACE" & eq.def "}"  \
      "LBRACKET" & eq.def "["  \
      "RBRACKET" & eq.def "]"  \
        "LANGLE" & eq.def "<"  \
        "RANGLE" & eq.def ">"  \
         "COLON" & eq.def ":"  \
     "SEMICOLON" & eq.def ";"  \
         "COMMA" & eq.def ","  \
           "DOT" & eq.def "."  \
            "EQ" & eq.def "="  \
          "PLUS" & eq.def "+"  \
         "MINUS" & eq.def "-"  \
          "STAR" & eq.def "*"  \
         "SLASH" & eq.def "/"  \
       "PERCENT" & eq.def "%"  \
          "BANG" & eq.def "!"  \
          "PIPE" & eq.def "|"  \
         "POUND" & eq.def "#"  \
    "UNDERSCORE" & eq.def "_"  \
  $])
=== Keywords

Only the words that introduce syntactic forms are keyword tokens:

#math.mono([$
        "FN" & eq.def "fn"       \
    "STRUCT" & eq.def "struct"   \
      "ENUM" & eq.def "enum"     \
       "PUB" & eq.def "pub"      \
       "MOD" & eq.def "mod"      \
       "LET" & eq.def "let"      \
        "IF" & eq.def "if"       \
      "ELSE" & eq.def "else"     \
     "MATCH" & eq.def "match"    \
  "REGIONAL" & eq.def "regional" \
    "EXTERN" & eq.def "extern"   \
        "AS" & eq.def "as"       \
      "TRUE" & eq.def "true"     \
     "FALSE" & eq.def "false"    \
    "IMPORT" & eq.def "import"   \
  $])

Contextual words are lexed as plain identifiers and matched by text in the
parser, so they stay usable as ordinary names:

- primitive type names: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`,
  `u64`, `f16`, `f32`, `f64`, `bfloat16`, `float8`, `bool`, `str`, `char`,
  and `unit`;
- reference capability names: `shared`, `value`, `flex`, `rigid`, `field`,
  and `regional`;
- other markers such as `trampoline`.

=== Literals

Integer literals are unsigned digit runs — negation is a prefix operator —
in decimal, hexadecimal, octal, or binary, with `_` digit separators.
Literals carry no type suffix; the type comes from the context or an
`as` cast. Float literals need a fractional part, an exponent, or both.

#math.mono([$
        "INTEGER" & eq.def "DECIMAL" | "BINARY" | "HEXADECIMAL" | "OCTAL"    \
        "DECIMAL" & eq.def "[0-9][0-9_]*"                                    \
         "BINARY" & eq.def "0[bB]_*[01][01_]*"                               \
    "HEXADECIMAL" & eq.def "0[xX]_*[0-9a-fA-F][0-9a-fA-F_]*"                 \
          "OCTAL" & eq.def "0[oO]_*[0-7][0-7_]*"                             \
          "FLOAT" & eq.def "[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9][0-9_]*)?" \
                  &      | "[0-9][0-9_]*[eE][+-]?[0-9][0-9_]*"
  $])

String literals are double-quoted and character literals are single-quoted,
both following Rust's escape grammar (`\n`, `\xNN`, `\u{...}`, ...). Raw
MLIR snippets for FFI bodies are written between `[{` and `}]`. Line
comments start with `//`; block comments are `/* ... */` and do not nest.

== Grammar

The normative grammar lives in the hand-written parser under
`crates/reussir-syntax/src/parser/`; its module documentation describes each
production. A source file is a module header followed by items: `import`
declarations, `mod` declarations, `struct`/`enum` definitions, `extern`
blocks, and (possibly `regional`) functions.

Expressions are parsed by precedence climbing with the following operator
table (tightest to loosest):

#table(
  columns: (auto, auto, auto),
  table.header([*level*], [*operators*], [*associativity*]),
  [6], [prefix `-` `!`; postfix `as T` or suffix chain], [one prefix and one postfix per term],
  [5], [`*` `/` `%`], [left],
  [4], [`+` `-`], [left],
  [3], [`==` `!=` `<` `>` `<=` `>=`], [none],
  [2], [`&&`], [left],
  [1], [`||`], [left],
  [0], [`lhs -> field := rhs`], [none],
)
