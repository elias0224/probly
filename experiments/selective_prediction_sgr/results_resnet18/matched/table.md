# Matched comparisons at SGR's operating point (clean CIFAR-10)

Seeds 0 1 2 3 4 x 10 selection/test splits (5k/5k), delta 0.001. Reference criterion: `sr_base`. On every (unit, split) pair where SGR (Geifman and El-Yaniv 2017) certifies the reference at r*, its selection-half coverage and realized test risk define the operating point. `risk @ ref cov`: test risk of a CoverageSelector calibrated (label-free) to the reference's selection-half coverage on the criterion's own selection half. `cov @ ref risk`: largest test coverage of the criterion at risk <= the reference's realized test risk. `delta` is the paired difference to the reference (criterion minus reference) as mean +- std over the certified pairs; `better` is the share of pairs where the criterion is strictly better (lower risk, higher coverage). `own SGR cov` and `certified` refer to the criterion's own SGR on all splits (coverage 0 if it does not certify). The deep ensemble has one unit and is paired with unit 0 of every reference unit.

# E3: matched comparison

## r* = 0.01

Reference `sr_base` certified on 70% of the splits (35 of 50 pairs).

| criterion | own SGR cov | certified | risk @ ref cov | delta | better | cov @ ref risk | delta | better |
|---|---|---|---|---|---|---|---|---|
| SR, base model (`sr_base`) (ref) | 0.5292 +- 0.3502 | 70% | 0.0050 +- 0.0016 | 0.0000 +- 0.0000 | 0% | 0.7674 +- 0.0477 | 0.0000 +- 0.0000 | 0% |
| SR, dropout model (`sr_dropout`) | 0.5843 +- 0.2820 | 82% | 0.0053 +- 0.0016 | 0.0003 +- 0.0011 | 43% | 0.7264 +- 0.1167 | -0.0410 +- 0.0792 | 40% |
| MC, 1 - max mean prob (`mc_maxprob`) | 0.5248 +- 0.3303 | 72% | 0.0054 +- 0.0015 | 0.0004 +- 0.0011 | 40% | 0.7195 +- 0.1255 | -0.0479 +- 0.0868 | 37% |
| MC, total entropy (`mc_total`) | 0.5009 +- 0.3322 | 70% | 0.0055 +- 0.0015 | 0.0005 +- 0.0011 | 40% | 0.7125 +- 0.1281 | -0.0549 +- 0.0898 | 34% |
| MC, aleatoric (`mc_aleatoric`) | 0.5010 +- 0.3324 | 70% | 0.0055 +- 0.0015 | 0.0005 +- 0.0011 | 37% | 0.7129 +- 0.1271 | -0.0545 +- 0.0888 | 37% |
| MC, epistemic (`mc_epistemic`) | 0.4284 +- 0.3538 | 60% | 0.0057 +- 0.0015 | 0.0007 +- 0.0011 | 31% | 0.6904 +- 0.1453 | -0.0770 +- 0.1086 | 29% |
| MC, paper variance (`mc_variance`) | 0.4564 +- 0.3466 | 64% | 0.0056 +- 0.0015 | 0.0006 +- 0.0011 | 34% | 0.7020 +- 0.1377 | -0.0654 +- 0.1017 | 34% |
| Ensemble, 1 - max mean prob (`ens_maxprob`) | 0.8300 +- 0.0058 | 100% | 0.0023 +- 0.0011 | -0.0026 +- 0.0011 | 100% | 0.8355 +- 0.0216 | 0.0681 +- 0.0318 | 100% |
| Ensemble, total entropy (`ens_total`) | 0.8286 +- 0.0057 | 100% | 0.0024 +- 0.0010 | -0.0026 +- 0.0011 | 100% | 0.8355 +- 0.0215 | 0.0681 +- 0.0319 | 100% |
| Ensemble, aleatoric (`ens_aleatoric`) | 0.8282 +- 0.0076 | 100% | 0.0024 +- 0.0010 | -0.0026 +- 0.0011 | 100% | 0.8363 +- 0.0234 | 0.0689 +- 0.0315 | 100% |
| Ensemble, epistemic (`ens_epistemic`) | 0.8295 +- 0.0131 | 100% | 0.0026 +- 0.0012 | -0.0023 +- 0.0011 | 100% | 0.8306 +- 0.0320 | 0.0632 +- 0.0286 | 100% |
| DDU-style, maxprob (`ddu_maxprob`) | 0.3458 +- 0.3010 | 60% | 0.0080 +- 0.0027 | 0.0030 +- 0.0018 | 0% | 0.6385 +- 0.1106 | -0.1289 +- 0.0856 | 0% |
| DDU-style, density (`ddu_density`) | 0.2255 +- 0.2466 | 48% | 0.0143 +- 0.0042 | 0.0094 +- 0.0031 | 0% | 0.5107 +- 0.1034 | -0.2567 +- 0.0856 | 0% |

## r* = 0.02

Reference `sr_base` certified on 100% of the splits (50 of 50 pairs).

| criterion | own SGR cov | certified | risk @ ref cov | delta | better | cov @ ref risk | delta | better |
|---|---|---|---|---|---|---|---|---|
| SR, base model (`sr_base`) (ref) | 0.8808 +- 0.0108 | 100% | 0.0118 +- 0.0022 | 0.0000 +- 0.0000 | 0% | 0.8825 +- 0.0105 | 0.0000 +- 0.0000 | 0% |
| SR, dropout model (`sr_dropout`) | 0.8696 +- 0.0173 | 100% | 0.0132 +- 0.0022 | 0.0013 +- 0.0015 | 24% | 0.8685 +- 0.0191 | -0.0140 +- 0.0114 | 10% |
| MC, 1 - max mean prob (`mc_maxprob`) | 0.8699 +- 0.0175 | 100% | 0.0131 +- 0.0022 | 0.0013 +- 0.0014 | 20% | 0.8686 +- 0.0194 | -0.0138 +- 0.0117 | 12% |
| MC, total entropy (`mc_total`) | 0.8692 +- 0.0174 | 100% | 0.0133 +- 0.0022 | 0.0014 +- 0.0015 | 20% | 0.8682 +- 0.0193 | -0.0143 +- 0.0115 | 12% |
| MC, aleatoric (`mc_aleatoric`) | 0.8694 +- 0.0171 | 100% | 0.0133 +- 0.0022 | 0.0014 +- 0.0015 | 20% | 0.8682 +- 0.0192 | -0.0143 +- 0.0114 | 12% |
| MC, epistemic (`mc_epistemic`) | 0.8688 +- 0.0178 | 100% | 0.0132 +- 0.0023 | 0.0014 +- 0.0015 | 18% | 0.8675 +- 0.0203 | -0.0150 +- 0.0123 | 18% |
| MC, paper variance (`mc_variance`) | 0.8691 +- 0.0195 | 100% | 0.0132 +- 0.0022 | 0.0013 +- 0.0015 | 20% | 0.8677 +- 0.0203 | -0.0148 +- 0.0124 | 14% |
| Ensemble, 1 - max mean prob (`ens_maxprob`) | 0.9200 +- 0.0071 | 100% | 0.0084 +- 0.0014 | -0.0035 +- 0.0015 | 100% | 0.9125 +- 0.0120 | 0.0300 +- 0.0091 | 100% |
| Ensemble, total entropy (`ens_total`) | 0.9197 +- 0.0074 | 100% | 0.0085 +- 0.0015 | -0.0034 +- 0.0014 | 100% | 0.9116 +- 0.0115 | 0.0291 +- 0.0087 | 100% |
| Ensemble, aleatoric (`ens_aleatoric`) | 0.9098 +- 0.0060 | 100% | 0.0081 +- 0.0018 | -0.0037 +- 0.0013 | 100% | 0.9074 +- 0.0090 | 0.0249 +- 0.0079 | 100% |
| Ensemble, epistemic (`ens_epistemic`) | 0.9035 +- 0.0050 | 100% | 0.0091 +- 0.0019 | -0.0027 +- 0.0015 | 100% | 0.8989 +- 0.0084 | 0.0164 +- 0.0084 | 100% |
| DDU-style, maxprob (`ddu_maxprob`) | 0.8196 +- 0.0147 | 100% | 0.0206 +- 0.0033 | 0.0088 +- 0.0025 | 0% | 0.8174 +- 0.0223 | -0.0651 +- 0.0201 | 0% |
| DDU-style, density (`ddu_density`) | 0.7287 +- 0.0263 | 100% | 0.0295 +- 0.0036 | 0.0176 +- 0.0029 | 0% | 0.7264 +- 0.0350 | -0.1561 +- 0.0300 | 0% |

## r* = 0.03

Reference `sr_base` certified on 100% of the splits (50 of 50 pairs).

| criterion | own SGR cov | certified | risk @ ref cov | delta | better | cov @ ref risk | delta | better |
|---|---|---|---|---|---|---|---|---|
| SR, base model (`sr_base`) (ref) | 0.9261 +- 0.0067 | 100% | 0.0201 +- 0.0028 | 0.0000 +- 0.0000 | 0% | 0.9268 +- 0.0066 | 0.0000 +- 0.0000 | 0% |
| SR, dropout model (`sr_dropout`) | 0.9179 +- 0.0122 | 100% | 0.0219 +- 0.0032 | 0.0018 +- 0.0022 | 18% | 0.9182 +- 0.0120 | -0.0086 +- 0.0072 | 18% |
| MC, 1 - max mean prob (`mc_maxprob`) | 0.9179 +- 0.0120 | 100% | 0.0219 +- 0.0031 | 0.0018 +- 0.0020 | 18% | 0.9179 +- 0.0121 | -0.0089 +- 0.0073 | 12% |
| MC, total entropy (`mc_total`) | 0.9180 +- 0.0121 | 100% | 0.0219 +- 0.0031 | 0.0018 +- 0.0021 | 20% | 0.9179 +- 0.0122 | -0.0089 +- 0.0073 | 16% |
| MC, aleatoric (`mc_aleatoric`) | 0.9179 +- 0.0121 | 100% | 0.0219 +- 0.0030 | 0.0018 +- 0.0021 | 20% | 0.9179 +- 0.0121 | -0.0089 +- 0.0071 | 16% |
| MC, epistemic (`mc_epistemic`) | 0.9176 +- 0.0117 | 100% | 0.0221 +- 0.0033 | 0.0020 +- 0.0023 | 16% | 0.9174 +- 0.0123 | -0.0094 +- 0.0074 | 12% |
| MC, paper variance (`mc_variance`) | 0.9173 +- 0.0120 | 100% | 0.0221 +- 0.0032 | 0.0020 +- 0.0021 | 14% | 0.9176 +- 0.0121 | -0.0092 +- 0.0074 | 14% |
| Ensemble, 1 - max mean prob (`ens_maxprob`) | 0.9507 +- 0.0025 | 100% | 0.0142 +- 0.0020 | -0.0059 +- 0.0023 | 100% | 0.9472 +- 0.0069 | 0.0204 +- 0.0058 | 100% |
| Ensemble, total entropy (`ens_total`) | 0.9461 +- 0.0028 | 100% | 0.0146 +- 0.0022 | -0.0055 +- 0.0021 | 100% | 0.9437 +- 0.0062 | 0.0169 +- 0.0055 | 100% |
| Ensemble, aleatoric (`ens_aleatoric`) | 0.9405 +- 0.0036 | 100% | 0.0165 +- 0.0025 | -0.0036 +- 0.0022 | 98% | 0.9406 +- 0.0084 | 0.0138 +- 0.0070 | 98% |
| Ensemble, epistemic (`ens_epistemic`) | 0.9326 +- 0.0046 | 100% | 0.0191 +- 0.0022 | -0.0010 +- 0.0024 | 62% | 0.9303 +- 0.0083 | 0.0035 +- 0.0063 | 64% |
| DDU-style, maxprob (`ddu_maxprob`) | 0.8787 +- 0.0092 | 100% | 0.0316 +- 0.0035 | 0.0115 +- 0.0034 | 0% | 0.8789 +- 0.0136 | -0.0479 +- 0.0117 | 0% |
| DDU-style, density (`ddu_density`) | 0.8244 +- 0.0166 | 100% | 0.0403 +- 0.0030 | 0.0202 +- 0.0029 | 0% | 0.8234 +- 0.0239 | -0.1034 +- 0.0207 | 0% |

## r* = 0.05

Reference `sr_base` certified on 100% of the splits (50 of 50 pairs).

| criterion | own SGR cov | certified | risk @ ref cov | delta | better | cov @ ref risk | delta | better |
|---|---|---|---|---|---|---|---|---|
| SR, base model (`sr_base`) (ref) | 0.9731 +- 0.0048 | 100% | 0.0368 +- 0.0035 | 0.0000 +- 0.0000 | 0% | 0.9733 +- 0.0048 | 0.0000 +- 0.0000 | 0% |
| SR, dropout model (`sr_dropout`) | 0.9718 +- 0.0059 | 100% | 0.0383 +- 0.0034 | 0.0015 +- 0.0019 | 20% | 0.9697 +- 0.0069 | -0.0036 +- 0.0034 | 14% |
| MC, 1 - max mean prob (`mc_maxprob`) | 0.9720 +- 0.0058 | 100% | 0.0384 +- 0.0035 | 0.0016 +- 0.0019 | 16% | 0.9697 +- 0.0068 | -0.0036 +- 0.0033 | 16% |
| MC, total entropy (`mc_total`) | 0.9719 +- 0.0067 | 100% | 0.0380 +- 0.0036 | 0.0012 +- 0.0018 | 22% | 0.9704 +- 0.0067 | -0.0028 +- 0.0030 | 16% |
| MC, aleatoric (`mc_aleatoric`) | 0.9722 +- 0.0066 | 100% | 0.0381 +- 0.0035 | 0.0013 +- 0.0018 | 22% | 0.9704 +- 0.0068 | -0.0029 +- 0.0030 | 16% |
| MC, epistemic (`mc_epistemic`) | 0.9701 +- 0.0061 | 100% | 0.0387 +- 0.0036 | 0.0018 +- 0.0020 | 20% | 0.9690 +- 0.0072 | -0.0042 +- 0.0033 | 10% |
| MC, paper variance (`mc_variance`) | 0.9690 +- 0.0054 | 100% | 0.0391 +- 0.0039 | 0.0023 +- 0.0023 | 16% | 0.9679 +- 0.0065 | -0.0053 +- 0.0030 | 4% |
| Ensemble, 1 - max mean prob (`ens_maxprob`) | 0.9886 +- 0.0032 | 100% | 0.0293 +- 0.0031 | -0.0075 +- 0.0020 | 100% | 0.9899 +- 0.0045 | 0.0166 +- 0.0043 | 100% |
| Ensemble, total entropy (`ens_total`) | 0.9868 +- 0.0036 | 100% | 0.0309 +- 0.0029 | -0.0059 +- 0.0022 | 100% | 0.9881 +- 0.0055 | 0.0148 +- 0.0050 | 100% |
| Ensemble, aleatoric (`ens_aleatoric`) | 0.9872 +- 0.0039 | 100% | 0.0304 +- 0.0030 | -0.0064 +- 0.0021 | 100% | 0.9884 +- 0.0059 | 0.0152 +- 0.0050 | 100% |
| Ensemble, epistemic (`ens_epistemic`) | 0.9852 +- 0.0037 | 100% | 0.0325 +- 0.0027 | -0.0043 +- 0.0024 | 96% | 0.9856 +- 0.0059 | 0.0123 +- 0.0051 | 100% |
| DDU-style, maxprob (`ddu_maxprob`) | 0.9422 +- 0.0065 | 100% | 0.0499 +- 0.0042 | 0.0131 +- 0.0039 | 0% | 0.9419 +- 0.0091 | -0.0314 +- 0.0093 | 0% |
| DDU-style, density (`ddu_density`) | 0.9116 +- 0.0092 | 100% | 0.0540 +- 0.0036 | 0.0172 +- 0.0033 | 0% | 0.9104 +- 0.0146 | -0.0628 +- 0.0128 | 0% |

# E4: scores per family

Per family, every score against the family's `maxprob` (SR) score, at the reference's operating points above (same certified pairs as E3). AURC x1000 and AUGRC x1000 (Traub et al., 2024) are the mean +- std over unit x split test halves; the gate uses AURC only. A score beats SR if its mean delta risk @ ref cov is below 0 and it is better in more than half of the pairs at both r* 0.01 and 0.03, and its AURC is lower than the baseline's. Gate rule: a score is kept if it beats SR for at least one family on E3.

## Family `mc` (baseline `mc_maxprob`)

| score | AURC x1000 | AUGRC x1000 | d risk r*=0.01 | better | d cov r*=0.01 | better | d risk r*=0.03 | better | d cov r*=0.03 | better | beats SR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `mc_maxprob` | 6.442 +- 0.755 | 4.864 +- 0.458 | 0.0000 +- 0.0000 | 0% | 0.0000 +- 0.0000 | 0% | 0.0000 +- 0.0000 | 0% | 0.0000 +- 0.0000 | 0% | baseline |
| `mc_total` | 6.518 +- 0.770 | 4.900 +- 0.467 | 0.0001 +- 0.0002 | 37% | -0.0070 +- 0.0095 | 23% | -0.0001 +- 0.0006 | 52% | -0.0000 +- 0.0020 | 42% | no |
| `mc_aleatoric` | 6.508 +- 0.770 | 4.897 +- 0.467 | 0.0001 +- 0.0002 | 34% | -0.0067 +- 0.0084 | 20% | -0.0001 +- 0.0006 | 58% | -0.0000 +- 0.0020 | 42% | no |
| `mc_epistemic` | 6.740 +- 0.771 | 5.031 +- 0.452 | 0.0003 +- 0.0003 | 11% | -0.0292 +- 0.0369 | 23% | 0.0001 +- 0.0007 | 44% | -0.0005 +- 0.0019 | 36% | no |
| `mc_variance` | 6.702 +- 0.757 | 5.017 +- 0.455 | 0.0003 +- 0.0003 | 14% | -0.0176 +- 0.0331 | 20% | 0.0001 +- 0.0006 | 46% | -0.0004 +- 0.0013 | 30% | no |

Verdict: no score beats SR.

## Family `ens` (baseline `ens_maxprob`)

| score | AURC x1000 | AUGRC x1000 | d risk r*=0.01 | better | d cov r*=0.01 | better | d risk r*=0.03 | better | d cov r*=0.03 | better | beats SR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ens_maxprob` | 4.105 +- 0.519 | 3.184 +- 0.289 | 0.0000 +- 0.0000 | 0% | 0.0000 +- 0.0000 | 0% | 0.0000 +- 0.0000 | 0% | 0.0000 +- 0.0000 | 0% | baseline |
| `ens_total` | 4.210 +- 0.541 | 3.270 +- 0.308 | 0.0000 +- 0.0001 | 49% | -0.0000 +- 0.0025 | 51% | 0.0004 +- 0.0007 | 26% | -0.0035 +- 0.0022 | 2% | no |
| `ens_aleatoric` | 4.280 +- 0.554 | 3.318 +- 0.318 | 0.0000 +- 0.0002 | 43% | 0.0008 +- 0.0047 | 60% | 0.0023 +- 0.0014 | 4% | -0.0066 +- 0.0029 | 2% | no |
| `ens_epistemic` | 4.015 +- 0.525 | 3.409 +- 0.288 | 0.0003 +- 0.0004 | 23% | -0.0050 +- 0.0152 | 43% | 0.0049 +- 0.0012 | 0% | -0.0169 +- 0.0027 | 0% | no |

Verdict: no score beats SR.

## Gate

Kept: none.
