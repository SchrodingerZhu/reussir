# Linear Recurrence Strength Reduction

Two LLVM-IR passes (`lib/LLVMPass/LinearRecurrence`) recognize
linear-recurrence computation in recursive and loop form and successively
strength-reduce it: exponential call trees become linear loops, and linear
affine loops become logarithmic-time matrix exponentiation. The flagship
consequence is that the naive doubly-recursive `fibonacci` compiles into an
O(log n) kernel at the default optimization level, but both passes are
general compiler transforms with precise applicability conditions rather than
a `fib` pattern-match.

## Pass 1: `reussir-recursion-linearization`

### Shape

A directly self-recursive function

```
f(x, c...) = combine(x, c..., f(x - d_1, c...), ..., f(x - d_k, c...))
```

where the `d_i` are constant positive offsets, every non-induction argument
is passed through unchanged, and `combine` — everything else the body does —
is an arbitrary side-effect-free, speculatable computation. No linearity is
required at this stage: modular sums, saturating/min/max combinations, and
floating-point recurrences all qualify, because the rewrite never
reassociates the user's arithmetic.

### Rewrite

The body is outlined into a private `alwaysinline` helper
`step(x, c..., y_1, ..., y_k)` in which the recursive call with offset `d`
is replaced by the parameter `y_d`. The function itself becomes a
dynamic-programming loop over a sliding window of the last `k` results:

```
f(x, c...):
  if (x < C)  return step(x, c..., poison...)        // call-free region
  w_j = step(C - j, c..., poison...)   for j = 1..k  // seeds, also < C
  for (v = C; ; ++v):
    cur = step(v, c..., w_1, ..., w_k)
    if (v == x) return cur
    w_1 = cur; w_j = w_{j-1}
```

`C` is the *recursion floor*: a constant lower bound of `x` at every
recursive call site, derived by intersecting `ConstantRange`s from dominating
`icmp x, constant` branch edges. The floor's lattice (signed or unsigned)
follows the predicates the source itself used, so signed base guards keep
negative inputs on the direct base path.

### Soundness

- **`x < C` implies the body is call-free.** Every call site is dominated by
  conditions placing `x` in a range whose minimum is at least `C`
  (contrapositive of the range computation). On those paths the `y`
  parameters are dead, so passing `poison` is safe; branch conditions cannot
  depend on a `y` value without a recursive call having executed first.
- **Seeds stay below the floor.** `C - k` is required not to wrap past the
  lattice minimum, so each seed evaluates the call-free region.
- **The window is exact.** By induction, at `v` the window holds
  `f(v-1) .. f(v-k)`; `step(v, ...)` therefore computes `f(v)` whether it
  takes a base or a recursive path.
- **Extra evaluation points cannot trap.** The loop visits every `v` in
  `[C, x]` even where the original call tree skipped points, so every
  instruction must be speculatable: the matcher whitelists effect-free
  instructions, requires an acyclic body CFG, and only admits division by
  provably benign constants.
- **Termination is structural.** The original recursion descends by at least
  1 per call under the same floor, so it terminates for exactly the same
  inputs; eliminating duplicated evaluations of a pure, terminating
  computation needs no `willreturn` attributes.

## Pass 2: `reussir-linear-recurrence-matexp`

### Shape

A single-block loop with a preheader, one exit, an exact
ScalarEvolution backedge-taken count, no side effects, whose loop-carried
integer state evolves as an affine map with compile-time constant
coefficients:

```
s_{i+1} = M * s_i + c      // all arithmetic wrapping, i.e. in Z/2^N
```

Every value used outside the loop must be an affine combination of the
header PHIs of one common integer type, and the closure of those PHIs under
their updates must stay affine. PHIs outside the closure — typically an exit
counter of a different width — die with the loop. Add/sub/mul-by-constant
and shl-by-constant are the recognized affine operations.

### Rewrite (Kitamasa)

With the augmented companion matrix `A = [[M, c], [0, 1]]` of dimension `D`,
the final state is `A^btc * s_0`. Rather than exponentiating the matrix
(`O(D^3)` per squaring), the pass works in the quotient ring
`(Z/2^N)[z] / chi_A(z)`:

1. `chi_A`, the characteristic polynomial of the compile-time constant `A`,
   is computed at compile time by division-free cofactor expansion mod 2^N
   (memoized on column subsets); it is monic by construction, so reduction
   mod `chi_A` needs no division either.
2. The emitted loop computes `p(z) = z^btc mod chi_A` by square-and-multiply
   over the exponent's bits — each step is a degree-`< D` polynomial
   multiply plus monic reduction, `O(D^2)` ring operations, with only `2D`
   loop-carried coefficients instead of `2D^2` matrix entries.
3. By Cayley–Hamilton (a polynomial identity over Z, hence valid in any
   commutative ring including Z/2^N), `A^btc s_0 = p(A) s_0 =
   sum_i p_i (A^i s_0)`. The Krylov vectors `A^i s_0` for `i < D` are
   emitted once as straight-line code with the (typically sparse, 0/±1)
   constant matrix entries folded away.

Total: `O(D^2 log n)` emitted ring operations — the FFT-free Kitamasa
bound. The FFT variant (`O(D log D log n)`) only wins for `D` in the dozens,
far beyond the `D <= 7` cap here, so plain schoolbook polynomial
multiplication is the right choice. Each live-out is then rebuilt from the
final state vector and the original loop is deleted.

### Soundness

Z/2^N is a commutative ring, so both the polynomial arithmetic and
Cayley–Hamilton are bit-exact for wrapping arithmetic; `nsw`/`nuw` flags on
the original operations only license *removing* poison, so dropping them in
the emitted code is a refinement. If the backedge-taken count SCEV relied on
no-wrap reasoning, the original loop already had the corresponding UB on the
offending inputs. Loops whose exact trip count SCEV cannot compute are
skipped.

## Pipeline placement

`registerLinearRecurrencePipelines` hooks both passes into the default
pipeline's extension points; both `runNPMOptimization` (bridge) and
`reussirRunBackendLLVMPipeline` (rrc/JIT) register it at the
speed-oriented levels (default/aggressive):

- **Pipeline start**: `mem2reg → recursion-linearization`. Linearization
  must run before the inliner tears the recursive shape apart.
- **Vectorizer start**: `loop-simplify → matexp → early-cse → instcombine`.
  This point is load-bearing in both directions. The loops are already
  canonicalized (rotated, indvar-simplified) — exactly what the inlined
  `step` helper of a linear recurrence collapses into — but the vectorizers
  have not run yet: placed *after* the whole pipeline instead, SLP
  vectorizes wider recurrence windows (order >= 4) into vector PHIs the
  affine matcher cannot see, and the rewrite silently never fires. And
  because the rewrite runs *inside* the pipeline, the emitted kernel gets
  the full late pipeline (instcombine, vectorizers, unrolling) on top of
  the immediate EarlyCSE, which folds the schoolbook squaring's commutative
  duplicate products (b_i*b_j vs b_j*b_i).

The `reussir-llvm-opt` flag `--linear-recurrence-pipeline=<level>` runs
this exact configuration, which is what the end-to-end lit tests use.

## Limits and future work

- Mutual recursion, multi-dimensional induction (`f(x-1, y+2)`), and
  non-constant descent such as `f(x / 2)` are out of scope (the latter is
  the divide-and-conquer class, a different transform).
- The matexp pass models Z/2^N only. An explicit-modulus recognizer
  (`urem` by a loop-invariant modulus at each step) and other semirings
  (min/max-plus) would slot into the same affine framework but need widened
  multiplies or a semiring-parameterized emitter.
- No runtime break-even guard is emitted: for tiny trip counts the
  exponentiation kernel does more work than the scalar loop it replaced.

Testing lives in `tests/integration/llvmpass/*.ll` (structure via FileCheck,
semantics via `lli`, including a naive-recursive `fib(10^15)` that only the
logarithmic path can answer) and
`tests/integration/frontend/naive_fib_linear_recurrence.rr` for the full
Reussir pipeline.
