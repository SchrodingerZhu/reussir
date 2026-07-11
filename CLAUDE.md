# Notes for Claude

## Git workflow

- **PR stacking: every patch is based on `main`.** When splitting work into a
  PR stack, rebase each branch onto `main` as soon as its predecessor merges
  and retarget the PR's base to `main`. Never chain PR bases
  (`part-2 -> part-1`, `part-3 -> part-2`): predecessors are squash-merged, so
  chained bases duplicate already-merged history and show stale diffs.
