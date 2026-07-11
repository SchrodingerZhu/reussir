"""lldb data formatters for Reussir types.

Reussir enums lower to a DWARF `DW_TAG_variant_part`. gdb collapses that to the
active case on its own; lldb (without a language plugin) instead shows every
case under a synthetic `$variants$` node. These formatters read the discriminant
and present only the live case — the same active-arm view gdb gives.

Load with:  command script import /path/to/reussir_formatters.py
"""

import re

import lldb


def _case_name(case):
    """The active case's display name, cleaned of lldb artifacts.

    lldb renders a variant member's type name with a bitfield-style ``:<n>``
    suffix (e.g. ``Cons:32``) and, for namespaced types, a scope prefix; strip
    both so the summary reads like gdb's (``Cons {...}``).
    """
    name = case.GetTypeName() or ""
    name = name.rsplit("::", 1)[-1]
    return re.sub(r":\d+$", "", name)


def _active(valobj):
    """Return (case_value, discr) for a variant_part value, or (None, None)."""
    # Read through the raw view: once the synthetic provider is installed, the
    # summary (and any re-entry) sees the *synthetic* children — the active
    # case's fields — and `$variants$` is no longer reachable there.
    raw = valobj.GetNonSyntheticValue()
    # Rebuild the value memory-backed at its own address before descending.
    # lldb gives the variant-part nodes (`$discr$`, each case's `value`) no
    # load address of their own: their bytes come from the ANCESTOR value's
    # host buffer. For a plain frame variable that buffer is right, but for
    # anything derived — a boxed field's pointer (buffer = the pointer bytes)
    # or a Dereference() result — the reads are garbage: the discriminant
    # comes out as the box refcount. CreateValueFromAddress always reads
    # target memory at the right base.
    if raw.GetType().IsPointerType():
        addr = raw.GetValueAsUnsigned()
        pointee = raw.GetType().GetPointeeType()
    else:
        addr = raw.GetLoadAddress()
        pointee = raw.GetType()
    if addr in (0, lldb.LLDB_INVALID_ADDRESS):
        return None, None
    target = raw.GetTarget()
    raw = target.CreateValueFromAddress(
        raw.GetName() or "value", lldb.SBAddress(addr, target),
        pointee).GetNonSyntheticValue()
    if not raw.IsValid():
        return None, None
    # A boxed member points at the box wrapper `{ value }`, whose `value`
    # member is the payload composite at its view offset — descend into it.
    value = raw.GetChildMemberWithName("value")
    if value.IsValid() and raw.GetNumChildren() == 1:
        raw = value.GetNonSyntheticValue()
    variants = raw.GetChildMemberWithName("$variants$")
    if not variants.IsValid() or variants.GetNumChildren() == 0:
        return None, None
    first = variants.GetChildAtIndex(0)
    discr_node = first.GetChildMemberWithName("$discr$")
    if not discr_node.IsValid():
        return None, None
    discr = discr_node.GetValueAsUnsigned()
    n = variants.GetNumChildren()
    idx = discr if discr < n else n - 1
    case = variants.GetChildAtIndex(idx)
    return case.GetChildMemberWithName("value"), discr


class VariantProvider:
    """Synthetic children: the active case's fields (passthrough otherwise)."""

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.target = valobj

    def update(self):
        case, _ = _active(self.valobj)
        # A non-variant Reussir record: show its own fields unchanged.
        self.target = case if (case and case.IsValid()) else self.valobj
        return False

    def num_children(self):
        return self.target.GetNumChildren()

    def get_child_at_index(self, index):
        return self.target.GetChildAtIndex(index)

    def get_child_index(self, name):
        return self.target.GetIndexOfChildWithName(name)

    def has_children(self):
        return self.target.GetNumChildren() > 0


def variant_summary(valobj, internal_dict):
    case, _ = _active(valobj)
    if case is None or not case.IsValid():
        return ""  # not a variant → fall back to the default rendering
    name = _case_name(case)
    fields = []
    for i in range(case.GetNumChildren()):
        child = case.GetChildAtIndex(i)
        text = child.GetValue() or child.GetSummary() or "..."
        fields.append("%s: %s" % (child.GetName(), text))
    return "%s {%s}" % (name, ", ".join(fields)) if fields else name


def __lldb_init_module(debugger, internal_dict):
    mod = __name__
    cat = "reussir"
    # Reussir record/enum types are mangled `_R…`; scalars are not, so this only
    # touches our composites. Non-variant records pass through unchanged.
    debugger.HandleCommand(
        'type synthetic add -x "^_R" --python-class %s.VariantProvider -w %s' % (mod, cat))
    debugger.HandleCommand(
        'type summary add -x "^_R" -e -F %s.variant_summary -w %s' % (mod, cat))
    debugger.HandleCommand("type category enable " + cat)
