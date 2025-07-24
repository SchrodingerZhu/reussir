#import "/book.typ": book-page
#import "@preview/cetz:0.4.0" as cetz: draw
#import "@preview/cetz-plot:0.1.2" as cetz-plot: smartart

#show: book-page.with(title: "Reference Capabilities")

== Compilation Process

=== Frontend Process
#let steps-rust = (
  [Program Input],
  [Parser],
  [Surface Syntax Tree],
  [Semantic Actions and Type Checking],
  [Middle-Level IR],
  [Codegen Phase 1],
)

#let colors = (
  blue,
  teal,
  green,
  lime,
  orange,
  red,
  purple,
  yellow,
  gray,
).map(c => c.lighten(35%))

#cetz.canvas({
  smartart.process.basic(
    steps-rust,
    step-style: colors,
    equal-length: true,
    dir: ltr,
    name: "fletcher-diagram",
  )
})

=== Backend Process
#let steps-backend = (
  [Codegen Phase 1],
  [Reussir MLIR Dialect],
  [Optimization],
  [Codegen Phase 2],
  [LLVM/Machine Code],
)

#let colors = (
  red,
  purple,
  yellow,
  red,
  gray,
).map(c => c.lighten(35%))

#cetz.canvas({
  smartart.process.basic(
    steps-backend,
    step-style: colors,
    equal-length: true,
    dir: ltr,
    name: "fletcher-diagram",
  )
})
