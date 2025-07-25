
#import "/book.typ": book-page

#show: book-page.with(title: "Syntax")



= Syntax

Currently, the syntax is not stable yet and subject change. For example, importing syntax are not provided and type bounds are not included.

== Syntax Family

Reussir language adopts a traditional two-phase syntax. The input sequence is first tokenized as a regular language and then parsed using parser expression grammar combinators.


== Lexical Tokens

=== Identifiers
#math.mono([$
    "IDENT" & eq.def "{XID_Start}" ~ "{XID_Continue}"
  $])
=== Symbols and Punctuations
#math.mono([$
        "LANGLE" & eq.def "<"  \
        "RANGLE" & eq.def ">"  \
        "LBRACE" & eq.def "{"  \
        "RBRACE" & eq.def "}"  \
        "LPAREN" & eq.def "("  \
        "RPAREN" & eq.def ")"  \
      "LBRACKET" & eq.def "["  \
      "RBRACKET" & eq.def "]"  \
         "COMMA" & eq.def ","  \
         "COLON" & eq.def ":"  \
     "SEMICOLON" & eq.def ";"  \
           "DOT" & eq.def "."  \
       "PATHSEP" & eq.def "::" \
         "ARROW" & eq.def "->" \
      "FATARROW" & eq.def "=>" \
            "EQ" & eq.def "="  \
          "EQEQ" & eq.def "==" \
         "NOTEQ" & eq.def "!=" \
        "LESSEQ" & eq.def "<=" \
     "GREATEREQ" & eq.def ">=" \
          "PLUS" & eq.def "+"  \
         "MINUS" & eq.def "-"  \
          "STAR" & eq.def "*"  \
         "SLASH" & eq.def "/"  \
       "PERCENT" & eq.def "%"  \
            "OR" & eq.def "or" \
        "ANDAND" & eq.def "&&" \
          "OROR" & eq.def "||" \
         "CARET" & eq.def "^"  \
          "BANG" & eq.def "!"  \
         "TILDE" & eq.def "~"  \
      "QUESTION" & eq.def "?"  \
     "AMPERSAND" & eq.def "&"  \
            "AT" & eq.def "@"  \
     "SHIFTLEFT" & eq.def "<<" \
    "SHIFTRIGHT" & eq.def ">>" \
          "UNIT" & eq.def "()" \
    "UNDERSCORE" & eq.def "_"  \
      "ELLIPSIS" & eq.def ".." \
  $])
=== Keywords
#math.mono([$
    "STRUCT" & eq.def "struct" \
      "ENUM" & eq.def "enum"   \
    "OPAQUE" & eq.def "opaque" \
       "LET" & eq.def "let"    \
        "IF" & eq.def "if"     \
      "ELSE" & eq.def "else"   \
     "MATCH" & eq.def "match"  \
      "COND" & eq.def "cond"   \
        "FN" & eq.def "fn"     \
       "PUB" & eq.def "pub"    \
    "REGION" & eq.def "reg"    \
        "AS" & eq.def "as"     \
    "RETURN" & eq.def "return" \
     "YIELD" & eq.def "yield"  \
      "TRUE" & eq.def "true"   \
     "FALSE" & eq.def "false"  \
        "I8" & eq.def "i8"     \
       "I16" & eq.def "i16"    \
       "I32" & eq.def "i32"    \
       "I64" & eq.def "i64"    \
      "I128" & eq.def "i128"   \
        "U8" & eq.def "u8"     \
       "U16" & eq.def "u16"    \
       "U32" & eq.def "u32"    \
       "U64" & eq.def "u64"    \
      "U128" & eq.def "u128"   \
      "BF16" & eq.def "bf16"   \
       "F16" & eq.def "f16"    \
       "F32" & eq.def "f32"    \
       "F64" & eq.def "f64"    \
      "F128" & eq.def "f128"   \
       "STR" & eq.def "str"    \
      "CHAR" & eq.def "char"   \
      "BOOL" & eq.def "bool"   \
  $])

=== Literals
#math.mono([$
        "INTEGER" & eq.def "DECIMAL" | "BINARY" | "HEXADECIMAL" | "OCTAL"                              \
        "DECIMAL" & eq.def "-?\d[\d_]*(i8|i16|i32|i64|i128|u8|u16|u32|u64|u128)?"                      \
         "BINARY" & eq.def "-?0b[01][01_]*(i8|i16|i32|i64|i128|u8|u16|u32|u64|u128)?"                  \
    "HEXADECIMAL" & eq.def "-?0x[0-9a-fA-F][0-9a-fA-F_]*(i8|i16|i32|i64|i128|u8|u16|u32|u64|u128)?"    \
          "OCTAL" & eq.def "-?0o[0-7][0-7_]*(i8|i16|i32|i64|i128|u8|u16|u32|u64|u128)?"                \
          "FLOAT" & eq.def "[+\-]?([\d]+(\.\d*)|[\d]+(\.\d*)?([eE][+\-]?\d+))(bf16|f16|f32|f64|f128)?"
  $])

== Grammar

TODO
