# Amendments & Retry Log — Experiment A

This document records (a) **technical-failure retries** per protocol §23, and (b) design
amendments per §24. Every entry below is a **technical retry (§23)** — there is NO change
to the experimental design, hypothesis, splits, architectures, or evaluation criteria. The
test set was not unlocked to make any decision here.

---

## §23-A — Phase 6 (secondary benchmark), `original` split: empty val_loader

**Date recorded:** 2026-07-26
**Phase:** 6 (secondary benchmark, DESCRIPTIVE; not part of the confirmatory Phase 4-5).
**Failure class:** technical pipeline failure (not a design amendment).

### Symptom
9 runs for the `original` split (3 architectures × 3 seeds) failed with `rc=1`.
Traceback: `RuntimeError: torch.cat(): expected a non-empty list of Tensors` in
`engine.evaluate`.

### Root cause
The `original_split_deduplicated.csv` manifest labels validation rows with the split
value **`val`** (only 16 images — the original Kermany validation split, which is
genuinely small), while `train_benchmark.py` filters the validation loader using the
string **`"validation"`**. As a result, `val_loader` was empty and `engine.evaluate`
called `torch.cat([])`.

### Fix (changed_parameters) — per protocol §6.10
For the `original` split ONLY, `train_benchmark.py` now **skips validation evaluation
entirely**: the model is trained for a fixed `--original-epochs 9` epochs (9 = the
median `best_epoch` on the controlled clean patient-grouped condition, a frozen
decision), and the final-epoch model is saved. The 16-image validation set must not be
used for any selection anyway (protocol §6.10). The `patient_grouped` and `image_level`
splits are UNCHANGED (still use validation-based early stopping).

The change only touches the benchmark training control flow; there is no change to the
data, labels, architectures, seeds, or test evaluation metrics.

### Verification before rerun
- The `original` split smoke test was re-run → passed (no empty `torch.cat`).
- The old master log was backed up (`fase6_master.log.bak`); failed entries were
  cleared from the active log.

### Retry — 9 original runs, all rc=0
Timestamp source: `fase6_original_nohup.out`. All UTC, 2026-07-25.

| original_run_id                         | seed | retry_start (UTC)   | retry_end (UTC)     | rc |
|-----------------------------------------|------|---------------------|---------------------|----|
| benchmark__original__densenet121__s42   | 42   | 2026-07-25 14:49:47 | 2026-07-25 14:56:35 | 0  |
| benchmark__original__densenet121__s456  | 456  | 2026-07-25 14:56:35 | 2026-07-25 15:05:07 | 0  |
| benchmark__original__densenet121__s2026 | 2026 | 2026-07-25 15:05:07 | 2026-07-25 15:14:09 | 0  |
| benchmark__original__efficientnet_b0__s42   | 42   | 2026-07-25 15:14:09 | 2026-07-25 15:19:25 | 0  |
| benchmark__original__efficientnet_b0__s456  | 456  | 2026-07-25 15:19:25 | 2026-07-25 15:24:44 | 0  |
| benchmark__original__efficientnet_b0__s2026 | 2026 | 2026-07-25 15:24:44 | 2026-07-25 15:30:01 | 0  |
| benchmark__original__swin_tiny__s42     | 42   | 2026-07-25 15:30:01 | 2026-07-25 15:47:42 | 0  |
| benchmark__original__swin_tiny__s456    | 456  | 2026-07-25 15:47:42 | 2026-07-25 16:05:06 | 0  |
| benchmark__original__swin_tiny__s2026   | 2026 | 2026-07-25 16:05:06 | 2026-07-25 16:22:18 | 0  |

`n_train` for original = 5190; recorded `best_epoch` = 9 (fixed) for all 9 runs (see
`benchmark/analysis/benchmark_per_run.csv`).

### Impact on conclusions
None. The Phase 6 benchmark is DESCRIPTIVE (protocol §6.9/H6/§16); there is no
cross-split hypothesis test. The fix only allows the original runs to complete; the
original-split test AUROC values are reported as-is (in fact lower than
patient-grouped — a test-distribution effect, not leakage; see the §benchmark report).

---

## §24 — Design amendment (confirmatory)

**NONE.** No design amendment was made after the test set was unlocked (Phase 5). All
confirmatory analyses (Phase 7, §16.1-16.3) were run per the pre-registered protocol
v1.0. Any analysis proposed after this point MUST be labeled *exploratory* (protocol
§24, CLAUDE.md §8).
