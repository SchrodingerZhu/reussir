{
  package = { name = "app", version = "0.1.0" },
  dependencies = {
    util = { path = "vendor/util", version = "^1.2" },
    logs = { path = "vendor/logs" },
  },
  targets = {
    demo = { kind = 'executable },
    shared = { kind = 'dynlib },
  },
}
