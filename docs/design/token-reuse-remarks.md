# Token-reuse decision reports

The one-shot `TokenReuse` pass can report every fresh-allocation versus reuse
decision as structured JSON. The report is intended for source-aware tools: a
web playground can highlight the construction (the sink), jump to the dead
object that donated its storage (the source), and explain why the pass did or
did not reuse it.

Reporting is off by default. It is diagnostic output and never changes the
reuse result.

## Entry points

At the pass level, `reussir-token-reuse{emit-remarks=1}` emits standard MLIR
`Passed` and `Missed` remarks in category `TokenReuse`, subcategory `OneShot`.
This keeps the optimization available to MLIR tooling and custom remark
streamers. For example:

```sh
reussir-opt input.mlir \
  --reussir-token-reuse=emit-remarks=1 \
  --remarks-filter=TokenReuse \
  --remark-format=emitRemark
```

`rrc` installs Reussir's JSON streamer and enables the pass with:

```sh
rrc input.rr --emit mlir-llvm -o output.mlir \
  --token-reuse-remarks reuse.json
```

The requested output must run the lowering pipeline: `mlir-llvm`, LLVM IR,
assembly, an object, a library, or an executable. A plain `mlir` dump stops
before token reuse and is rejected instead of producing an empty, misleading
report. The report is target-independent; an explicit `--target-triple`,
including a WebAssembly target, follows the same path.

`rene` forwards the option to every artifact-producing compilation:

```sh
rene build --token-reuse-remarks
rene build --target wasm32-wasip1 --token-reuse-remarks
```

Each report is a cached sidecar beside its artifact. The name retains the
artifact's complete file name so target kinds cannot collide:

```text
dev/demo.token-reuse.json
dev/demo.wasm.token-reuse.json
dev/deps/libutil.a.token-reuse.json
```

Dependency-interface (`.rri`) compilations do not run token reuse and do not
receive a report. Deleting a sidecar invalidates only the product that owns it.
`rene inspect --commands --token-reuse-remarks` exposes the exact paths without
building.

## Schema v1

Every document has the schema tag `reussir.token-reuse.v1`:

```json
{
  "schema": "reussir.token-reuse.v1",
  "coordinates": {
    "line_base": 1,
    "column_base": 1,
    "column_unit": "utf-8-byte",
    "end": "exclusive"
  },
  "files": ["src/list.rr"],
  "locations": [
    {
      "kind": "file",
      "file": 0,
      "start": {"line": 8, "column": 25},
      "end": {"line": 11, "column": 10}
    },
    {
      "kind": "file",
      "file": 0,
      "start": {"line": 10, "column": 13},
      "end": {"line": 10, "column": 31}
    }
  ],
  "decisions": [
    {
      "kind": "reuse",
      "sink": 1,
      "source": 0,
      "strategy": "ensure",
      "score": 2,
      "available_tokens": 1,
      "compatible_tokens": 1,
      "instances": ["_RIC6updatelE", "_RIC6updatexE"],
      "occurrences": 2
    }
  ]
}
```

Coordinates match MLIR's source locations: lines and columns start at one,
columns count UTF-8 bytes, and a range's end is exclusive. `files` is a shared
string table; a file location's numeric `file` field indexes it.

Locations form another shared table. A decision's `source` and `sink` are
numeric indexes into that table. Supported nodes preserve MLIR's nested
location structure:

- `file`: a source range and file-table index;
- `name`: a name and `child` location;
- `callsite`: `callee` and `caller` locations;
- `fused`: a `locations` array;
- `unknown`;
- `opaque`: a lossless printed fallback for an unrecognized location kind.

For a reuse decision, `source` is the location of the token-producing dead
object and `sink` is the accepting construction. `strategy` is `ensure` for an
exact compatible token or `realloc` for a resizable donor. `score` is the
selection heuristic's score.

For an allocation decision, `source`, `strategy`, and `score` are absent.
`reason` is either `no-available-token` or `no-compatible-token`. Both decision
kinds record how many tokens were available and how many passed compatibility
scoring, so a UI can distinguish an empty reuse set from rejected candidates.

## Deduplication and determinism

Locations are structurally interned. A one-child `FusedLoc` that only carries
per-instantiation debug metadata is normalized to its child, preventing generic
monomorphizations from duplicating otherwise identical source ranges. Other
nested location structure remains visible.

Decisions with the same kind, source/sink, strategy or reason, score, and
candidate counts are grouped. Their function symbols appear once each in the
sorted `instances` array, while `occurrences` retains the total number of
remarks. This is lossless enough for a frontend to expand the grouping if it
wants per-instance annotations. Files, canonical locations, decisions, and
instances are sorted before serialization, so MLIR's parallel function-pass
scheduling and multiple codegen units do not perturb the JSON bytes.

## Cost model

The pass dispatches once per function to separate reporting-enabled and
reporting-disabled template instantiations. The disabled candidate loop has no
reporting branch or compatibility counter. Remark construction, string
conversion, location interning, locking, and JSON state do not run unless
reporting is enabled.

In the enabled streamer, closed state uses compact enums (`DecisionKind`,
`ReuseStrategy`, `AllocationReason`, and `LocationKind`) and numeric location
IDs. Strings are retained only for irreducible payloads—file paths, names,
opaque locations, and instance symbols—and for canonical location keys needed
by structural deduplication and deterministic ordering. A mutex protects the
single context-wide report while MLIR runs function passes in parallel.
