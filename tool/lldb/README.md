# Reussir debugger formatters

Reussir enums lower to a DWARF `DW_TAG_variant_part` and the compile unit is
tagged Rust, so **gdb shows the active case automatically** — no setup needed:

```
$ gdb ./a.out
(gdb) print my_enum
$1 = Cons{f0: 5, f1: 0x...}
```

**lldb** (without a language plugin) instead lists every case under a synthetic
`$variants$` node. Load `reussir_formatters.py` to collapse that to the active
case, matching gdb:

```
(lldb) command script import /path/to/tool/lldb/reussir_formatters.py
(lldb) frame variable my_enum
(_RC4List) my_enum = Cons {f0: 5, f1: 0x...} { f0 = 5, f1 = 0x... }
```

Add the `command script import` line to your `~/.lldbinit` to load it always.
Non-enum Reussir records (structs) are passed through unchanged.
