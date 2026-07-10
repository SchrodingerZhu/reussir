# Companion command script for gdb_print.rr — see the RUN lines there.
break inspect_shape
break inspect_list
break inspect_snapshot
break inspect_packed
break observe_cell
break observe_rtag
break observe_fixed
break observe_ping
run
print s
continue
print s
continue
print s
continue
print l
print *l.f1
print (*l.f1).value.f0
continue
print snap
continue
print p
continue
print c
continue
print r
continue
print f
continue
print p
print *p.f0
continue
