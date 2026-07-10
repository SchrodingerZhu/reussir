# Reussir debugger formatters

Reussir enums lower to a DWARF `DW_TAG_variant_part` (rustc-shaped: the
discriminant at offset 0 of the described composite, full-size case structs,
composite-relative field offsets) and the compile unit is tagged Rust, so
**gdb shows the active case automatically** — no setup needed:

```
$ gdb ./a.out
(gdb) print my_enum
$1 = Cons{f0: 5, f1: 0x...}
```

A boxed enum variable is presented directly as its cell: the variable's
`DIExpression` dereferences the rc pointer and lands on the tag-first view of
the box. A *field* of boxed type is a pointer to a thin `<name>$box` wrapper
whose `value` member is that view, so a pointer chase works in both debuggers:

```
(gdb) print *my_list.f1
$2 = _RC4List$box {value: Cons{f0: 6, f1: 0x...}}
```

**lldb** (without a language plugin) decodes the same DWARF but lists every
case under a synthetic `$variants$` node. Load `reussir_formatters.py` to
collapse that to the active case, matching gdb:

```
(lldb) command script import /path/to/tool/lldb/reussir_formatters.py
(lldb) frame variable my_enum
(_RC4List) my_enum = Cons {f0: 5, f1: 0x...} { f0 = 5, f1 = 0x... }
```

Add the `command script import` line to your `~/.lldbinit` to load it always.
Non-enum Reussir records (structs) are passed through unchanged. The
formatters are display sugar only — the underlying reads all come from the
DWARF, which the `tests/integration/debuginfo` lit suite pins end to end
under both debuggers (`%lldb` with these formatters, `%gdb` bare).
