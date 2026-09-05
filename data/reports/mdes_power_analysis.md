# MDES / A Priori Power Analysis - Experiment A (Common Anchor Test)

Frozen: 2026-07-24, before Phase 1 (controlled manifests) began.
Status: pre-registered - computed before seeing any model prediction data.

## 1. Purpose

Close the interpretive ambiguity of a "not significant" result: is it because the
leakage effect is genuinely small, or because the test set (n=279 patients) is
underpowered? This document freezes the Minimum Detectable Effect Size (MDES) before
training, so a null result can be given quantitative meaning ("able to detect delta >= X
at 80% power"), rather than just "not significant".

## 2. Setup

- Anchor test set (protocol section 6.2): 279 patients, 1 image/patient.
  n_pos (PNEUMONIA) = 169, n_neg (NORMAL) = 110.
- Primary comparison: ΔAUROC = AUROC_leaky - AUROC_clean, per architecture, **paired**
  design (identical test set for clean & leaky under the same architecture and seed).
- Correction family (protocol section 16.4): the 3 clean-vs-leaky comparisons (one per
  architecture) are Holm-Bonferroni corrected -> the conservative per-comparison alpha
  used here = 0.05 / 3 = 0.01667 (two-sided), z_crit = 2.394.

## 3. Method

SE(AUROC) is computed analytically with the Hanley-McNeil formula:

```
Q1 = AUC / (2 - AUC)
Q2 = 2*AUC^2 / (1 + AUC)
Var(AUC) = [AUC*(1-AUC) + (n_pos-1)*(Q1-AUC^2) + (n_neg-1)*(Q2-AUC^2)] / (n_pos*n_neg)
```

For the paired design (clean & leaky evaluated on the same patients), the variance of
the difference:

```
SE(delta) = sqrt(SE1^2 + SE2^2 - 2*rho*SE1*SE2)
```

`rho` is the correlation between the clean- and leaky-condition prediction scores on the
same patient. **No pilot data was available to estimate rho empirically before
training**, so rho is treated as a sensitivity parameter (not assumed to be a single
value) - varied from 0 (unpaired, a conservative lower bound) up to 0.95 (very high
correlation, realistic for two models sharing the same architecture and most of the
training data).

Power is computed from z = delta / SE(delta), then:

```
power = 1 - Phi(z_crit - z) + Phi(-z_crit - z)
```

MDES = the minimum delta such that power >= 80%, found via bisection.

Full code (run directly, not saved as a separate script since it is a one-off
pre-registration artifact, not part of the manifest pipeline):

```python
import math
from scipy.stats import norm

n_pos, n_neg = 169, 110
alpha_family = 0.05 / 3
z_crit = norm.ppf(1 - alpha_family / 2)

def se_auroc(auc, n_pos, n_neg):
    Q1 = auc / (2 - auc)
    Q2 = 2 * auc**2 / (1 + auc)
    var = (auc*(1-auc) + (n_pos-1)*(Q1-auc**2) + (n_neg-1)*(Q2-auc**2)) / (n_pos*n_neg)
    return math.sqrt(var)

def power_paired(auc_clean, delta, rho, n_pos, n_neg, z_crit):
    auc_leaky = min(auc_clean + delta, 0.999)
    se1 = se_auroc(auc_clean, n_pos, n_neg)
    se2 = se_auroc(auc_leaky, n_pos, n_neg)
    se_diff = math.sqrt(max(se1**2 + se2**2 - 2*rho*se1*se2, 1e-10))
    z = delta / se_diff
    return 1 - norm.cdf(z_crit - z) + norm.cdf(-z_crit - z)

def mdes(auc_clean, rho, n_pos, n_neg, z_crit, target_power=0.80):
    lo, hi = 0.0001, 0.30
    for _ in range(60):
        mid = (lo + hi) / 2
        p = power_paired(auc_clean, mid, rho, n_pos, n_neg, z_crit)
        lo, hi = (mid, hi) if p < target_power else (lo, mid)
    return hi
```

## 4. Results - MDES (delta AUROC, 80% power)

| AUROC_clean | rho=0.00 | rho=0.30 | rho=0.50 | rho=0.70 | rho=0.85 | rho=0.95 |
|---|---|---|---|---|---|---|
| 0.85 | 0.0860 | 0.0748 | 0.0651 | 0.0522 | 0.0381 | 0.0227 |
| 0.90 | 0.0672 | 0.0591 | 0.0519 | 0.0420 | 0.0309 | 0.0186 |
| 0.93 | 0.0537 | 0.0479 | 0.0425 | 0.0347 | 0.0258 | 0.0156 |
| 0.95 | 0.0433 | 0.0394 | 0.0353 | 0.0292 | 0.0219 | 0.0134 |

## 5. Results - actual power at several realistic deltas (AUROC_clean=0.90)

| delta | rho=0.00 | rho=0.30 | rho=0.50 | rho=0.70 | rho=0.85 | rho=0.95 |
|---|---|---|---|---|---|---|
| 0.01 | 0.03 | 0.03 | 0.04 | 0.05 | 0.09 | 0.27 |
| 0.02 | 0.06 | 0.08 | 0.11 | 0.18 | 0.38 | 0.86 |
| 0.03 | 0.13 | 0.19 | 0.27 | 0.45 | 0.77 | 0.99 |
| 0.04 | 0.26 | 0.38 | 0.52 | 0.75 | 0.96 | 1.00 |
| 0.05 | 0.45 | 0.61 | 0.76 | 0.93 | 1.00 | 1.00 |
| 0.06 | 0.66 | 0.82 | 0.92 | 0.99 | 1.00 | 1.00 |
| 0.08 | 0.95 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |

## 6. Results - McNemar (secondary discrete analysis, threshold 0.5)

Power depends on m (the number of discordant prediction pairs between clean & leaky) and
p (the proportion of discordant pairs favoring leaky, p=0.5 under the null).

| m (discordant) | p=0.60 | p=0.65 | p=0.70 | p=0.75 | p=0.80 |
|---|---|---|---|---|---|
| 10 | 0.04 | 0.07 | 0.11 | 0.17 | 0.27 |
| 15 | 0.05 | 0.10 | 0.18 | 0.30 | 0.47 |
| 20 | 0.06 | 0.14 | 0.25 | 0.43 | 0.64 |
| 30 | 0.09 | 0.22 | 0.41 | 0.65 | 0.87 |
| 40 | 0.12 | 0.30 | 0.56 | 0.81 | 0.96 |
| 60 | 0.19 | 0.47 | 0.78 | 0.96 | 1.00 |
| 80 | 0.27 | 0.62 | 0.90 | 0.99 | 1.00 |
| 279 (theoretical maximum) | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 |

## 7. Interpretation & frozen decisions

1. With n=279 and no pilot data, **the realistic MDES lies in the 0.02-0.05 AUROC
   range**, depending on how high the correlation is between clean/leaky predictions on
   the same patient. High correlation (rho>=0.7) - plausible since both models share the
   same architecture and >99% of the training data - makes this design reasonably
   powered for moderate effects (delta>=0.03). Low/unknown correlation makes small
   effects (delta<0.02) hard to detect.
2. McNemar (secondary endpoint) needs a fairly large number of discordant pairs (m>=40)
   and a clear asymmetry (p>=0.75) for adequate power - realistic only if the leakage
   effect is large enough. With small m, McNemar must NOT be used as the sole evidence
   of no effect.
3. **The reporting headline is effect size (ΔAUROC) + CI, not the p-value.** A null
   result must be reported as "the effect is below this design's detection threshold
   (~0.02-0.05 depending on rho)", not as "no leakage".
4. The empirical rho (not an assumption) will be computed from the actual prediction
   data once Phases 4-5 are complete, and reported as part of the realized-power
   analysis in the final results' methods section - compared against this table.
5. The numbers in this document are **frozen** - they must not be recomputed or changed
   to fit the results after the test set is unlocked (this would violate protocol
   section 24; amendments after unblinding are prohibited for the confirmatory
   analysis).
