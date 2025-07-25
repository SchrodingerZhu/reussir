
#import "@preview/shiroa:0.2.3": *

#show: book

#book-meta(title: "Reussir Book", summary: [
  #prefix-chapter("chapters/about.typ")[About]
  = Language Reference
  - #chapter("chapters/language/syntax.typ")[Syntax]
  - #chapter("chapters/language/reference-capabilities.typ")[Reference Capabilities]
  = Development Guide
  - #chapter("chapters/development/compilation-process.typ")[Compilation Process]
])

// re-export page template
#import "/templates/page.typ": project
#let book-page = project
