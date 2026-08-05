# The built-in core package: the language-coupled layer of the standard
# library, compiled like any other dependency and injected into every build
# by rene. Items the compiler must know carry `#[lang("…")]` markers; their
# declarations here give every wired name a real source location.
{
  package = { name = "core", version = "0.1.0", no_core = true },
}
