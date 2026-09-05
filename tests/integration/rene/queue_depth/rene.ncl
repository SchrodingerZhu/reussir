# A Hood-Melville functional queue churned deep enough that only the
# loop form of its tail-recursive driver fits the runner's stack.
{
  package = { name = "queue_depth", version = "0.1.0" },
  targets.queue_depth = { kind = 'executable },
  profiles = {
    release = { extra_flags = ["--target-features=-tail-call"] },
    optimized = {
      opt = "default",
      extra_flags = ["--target-features=-tail-call"],
    },
  },
}
