# Standard-library integration tests

Each top-level `.rr` file is both a lit test and the crate root of one
executable target in `rene.ncl`. Every test invokes the same package against
`%T/rene-build`; rene's build-directory lock queues concurrent lit workers,
so the first invocation builds all targets and the rest reuse its runtime,
dependencies, and artifacts.

Case-selector input files live under `Inputs/<target>/`, which lit excludes
from test discovery.
