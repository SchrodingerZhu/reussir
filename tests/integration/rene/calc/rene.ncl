# A small interpreter as a two-package build: the app owns the lexer,
# parser, and evaluator; all console traffic — reading the program text,
# printing values and error messages — goes through the `textio` polyffi
# package. `release` refines the built-in with a two-unit link so the
# optimized build also exercises the multi-unit path.
{
  package = {
    name | String = "calc",
    version = "1.0.0",
  },
  dependencies = {
    textio = { path = "vendor/textio", version = "^0.5" },
  },
  targets = {
    calc = { kind = 'executable },
  },
  profiles = {
    release = { codegen_units = 2 },
  },
}
