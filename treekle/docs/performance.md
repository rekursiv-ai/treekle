# Closing a speed gap

Diagnose the gap before changing code. Follow these steps when the goal is
"make treekle as fast as msgspec" or any "make X faster".

## 1. Find what bounds the gap

Count, inside the timed call only, for both libraries:

- core cycles,
- instructions,
- cache misses.

Read them from `perf_event_open` counters opened in the benchmark process.
Enable a counter just before the call and disable it just after.

| cycles ratio | instructions ratio | cache misses ratio | bound by |
|---|---|---|---|
| high | about the same | — | instructions |
| high | low | high | memory |
| — | — | — | collector or allocator, if the profile shows it |

- One cache miss costs about 150–200 cycles.
- 20k extra misses per pass is about 3.5M cycles, which can be the whole gap.

## 2. Compare the profiles

- Profile both libraries on the same workload with `perf record`.
- Attach `perf` to the process after its setup is done. Otherwise
  imports, BLAS threads and config building swamp the samples.
- Sort by DSO, then by symbol. Diff the two lists.
- Profile cache misses as well as cycles: `-e cache-misses:u`.

## 3. Ablate in one build

- Put each candidate component behind a bit of one env var.
  Example: `TK_KNOB=<bits>` skips the import path, the sort, the cache
  checks or the shape map.
- Build once and run every bit combination. Read off each component's
  share of cycles and misses.
- Remove the switches before gating.

## 4. Rewrite what bounds it

- A component the counters name gets redesigned for its dominant cost.
  For a memory bound, that means touching fewer cache lines.
- Trimming instructions from a memory-bound path changes nothing.
- Keep the Python contract: parity tests and every treekle test pass.

## 5. Measure with counters, not clocks

- Wall-clock time is unreliable on a shared machine.
- Counters count only this process's work.
- Use wall clock only for the final README tables, from
  `scripts/bench_configs.py`.

## 6. Stop rule

- Two changes that don't move the bound's counter mean your model of it
  is wrong.
- Go back to step 1. Do not try a third variant.
