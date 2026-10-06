# Selective prediction vs. SGR (Table 1 of Geifman and El-Yaniv, 2017)

Reproduces the test side of Table 1 (Sec. 5.1) of "Selective Classification for Deep Neural Networks"
(arXiv 1705.08500, CIFAR-10) with an MC-dropout VGG-16 and `probly.selective_prediction.SelectivePredictor`
instead of SGR. A risk cannot be targeted, so for each of the paper's six test risks we read our coverage off our own
risk-coverage curve and compare it to the paper's test coverage. A second column picks the threshold on a 5k selection
half for the desired risk r* and reports risk and coverage on the other 5k half (the paper's protocol, without bounds).
No risk bounds are computed.

## Two stages per seed

1. **base**: the plain VGG-16 variant of Liu and Deng (2015), with conv-block dropout (0.3 after the first conv,
   0.4 after every other non-final conv of a block) and no dropout before the two Linear layers. 250 epochs, SGD
   momentum 0.9, lr 0.1, weight decay 5e-4, lr x0.5 every 25 epochs, batch size 128, flip / 10% shift / 15 degree
   rotation augmentation (border pixels repeated, like Keras' default `fill_mode="nearest"`), AMP on CUDA
   (channels-last is deliberately off, it was 4-5x slower here; `scripts/bench.py` measures it). The whole dataset is kept on the GPU and augmented there in batches, so no DataLoader workers are needed.
2. **dropout**: `probly.transformation.dropout(base, p=0.5, predictor_type="logit_classifier")` inserts a dropout
   layer in front of each of the two Linear layers; this model is fine-tuned for 50 epochs (lr 0.01, x0.5 every 10).
   `--finetune-epochs` and `--finetune-lr` on `run_all.py` (or `--epochs/--lr/--step-size` on `train.py`) change it.

MC sampling in probly forces every `nn.Dropout` into train mode (the conv-block ones too) while BatchNorm stays in
eval mode.

Artifacts per seed in `runs/seed{S}/`:

| file | content |
|---|---|
| `base.pt`, `base.json` | state dict of the plain model; config, seed, epochs, test acc, git commit |
| `dropout.pt`, `dropout.json` | state dict of the transformed model; same metadata |
| `last_base.pt`, `last_dropout.pt` | resume checkpoints (model, optimizer, scheduler, scaler, epoch) |
| `log_base.csv`, `log_dropout.csv` | epoch, train loss, test acc, lr, seconds |
| `predictions_base.npz` | `labels`, `softmax` |
| `predictions_dropout.npz` | `labels`, `softmax` (eval mode), `mc_probs` (float16), `criterion_probly`, plus `mean_probs` and `criterion_variance` (float32) |

## Run everything on Windows (RTX 2070 Super)

Requirements: `git`, `uv` (`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`) and an NVIDIA driver that
supports CUDA 12.6 (version 560 or newer, check with `nvidia-smi`). The branch lives on the fork:

```powershell
git remote add fork https://github.com/elias0224/probly.git   # once, if the remote does not exist yet
git fetch fork
git switch sgr-benchmark
cd experiments/selective_prediction_sgr
uv sync -p 3.13
uv run python scripts/run_all.py
```

On Windows, torch/torchvision come from the PyTorch CUDA 12.6 index (see `pyproject.toml`); elsewhere from PyPI.
`run_all.py` loops seeds 0..4 (train base, fine-tune dropout, dump predictions), skips finished stages, resumes an
interrupted stage from its checkpoint, and finally runs the evaluation (`--no-evaluate` to skip it). CIFAR-10 is
downloaded to `data/` on first use. Results land in `results/`: `table.md`, `table.csv`, `risk_coverage.png`.
Other options: `--seeds`, `--runs`, `--data-dir`, `--out`, `--num-samples`
(default 100), `--n-splits` (default 10), `--subset` (smoke tests).

Single steps:

```powershell
uv run python scripts/train.py --stage base --seed 0 --out runs
uv run python scripts/train.py --stage dropout --seed 0 --out runs
uv run python scripts/dump.py --seed 0 --runs runs --num-samples 100
uv run python scripts/evaluate.py --runs runs --n-splits 10 --out results
uv run pytest
```

## Evaluation

Criteria: `sr_base` (1 - max softmax of the base model), `sr_dropout` (same for the dropout model in eval mode),
`mc_probly` (the `SelectivePredictor` criterion, 1 - max mean MC probability) and `mc_paper_variance` (variance of
the predicted-class probability over MC samples). The risk-coverage helpers in `src/sgr_experiment/metrics.py` are
exact (every distinct criterion value is a threshold, ties are accepted together). Plots use Fira Sans if installed
and fall back to the default font otherwise.

## Selectors from probly

`SGRSelector` and `CoverageSelector` come from probly (`probly.selective_prediction`; the `selective-prediction` branch is
merged into this throwaway branch). `evaluate.py` adds an SGR table (`SGRSelector(r*, delta)` fitted on the selection
half: risk, coverage, violation share on the test half) and a coverage table (`CoverageSelector` calibrated on the
selection half for each paper test coverage: realized coverage and risk next to the paper's risk). In
`evaluate_shift.py` the `sgr` mode uses `SGRSelector`; `metrics.sgr_threshold` remains as a reference and the table
"SGR: probly SGRSelector vs the reference" counts how often the thresholds differ (they can differ by one instance
when almost everything is certifiable, since probly never tests the largest score, or with heavy ties) and how often
nothing is certified. The third mode `cov` is label-free: a `CoverageSelector` calibrated on the clean selection half to
the coverage the `emp` threshold reached there for the same r*; its risk and coverage on the test half, the shifted
sets and the OOD sets (share accepted) show how the coverage guarantee drifts under shift.

## Fixed thresholds

`scripts/fixed_threshold.py` reports risk and coverage of `ThresholdSelector(c)` for a grid of fixed c and the c each
paper risk r* needs, raw and temperature scaled (fitted on the selection half): `uv run python scripts/fixed_threshold.py`.

## Distribution shift and OOD

Compares selection criteria on clean CIFAR-10, under covariate shift and on OOD data. Criteria (lower = more
confident): `sr_base`, `sr_dropout`; MC dropout via probly: `mc_maxprob` (1 - max mean prob), `mc_total` (entropy of
the mean), `mc_aleatoric` (expected entropy), `mc_epistemic` (mutual information), `mc_variance` (paper variance);
deep ensemble of the 5 base models (one per seed, no extra training): `ens_maxprob`, `ens_total`, `ens_aleatoric`,
`ens_epistemic`. There is only one ensemble, so its spread comes from the random splits only. The entropy
decomposition goes through `probly.quantification.quantify`.

Datasets: clean CIFAR-10 test, SVHN test, CIFAR-100 test (OOD), and the CIFAR-10 test set under gaussian noise,
gaussian blur, contrast and pixelate at severities 1, 3, 5 (CIFAR-10-C parameters, `sgr_experiment/shift.py`).

1. `scripts/dump_shift.py` writes `runs/seed{S}/shift/{dataset}.npz` (labels, both softmaxes, MC mean, the five MC
   criteria; no raw samples) and skips existing files. Options: `--seeds`, `--runs`, `--data-dir`, `--num-samples`,
   `--datasets`, `--subset`, `--batch-size`.
2. `scripts/evaluate_shift.py` uses the protocol of `evaluate.py` (10 random 5k/5k splits of the clean test set).
   Thresholds are always chosen on the clean selection half for each paper r*, two ways: `emp` (empirical risk <= r*)
   and `sgr` (`metrics.sgr_threshold`, Algorithm 1 of Geifman and El-Yaniv, delta 0.001). It reports (a) AURC and
   E-AURC plus risk, coverage, violation share and bound on the ID test half, (b) OOD AUROC, share of OOD accepted
   and a mixed CIFAR-10 + OOD set, (c) risk, coverage and accuracy per corruption and severity.

```powershell
uv run python scripts/dump_shift.py
uv run python scripts/evaluate_shift.py
uv run python scripts/run_all.py --shift     # everything, including the shift stages
```

Artifacts in `results/shift/`: `table.md`, one csv per table (`id_aurc`, `id_thresholds`, `ood_*`, `mixed_*`,
`shift_*`), `id_risk_coverage.png`, `ood_acceptance.png`, `shift_risk.png`.

Runtime (inference only): about 156k images (10k clean, 26k SVHN, 10k CIFAR-100, 12 x 10k corrupted) x 100 MC samples
per seed, i.e. roughly 15.6M VGG-16 forward passes per seed; expect on the order of half an hour to an hour per seed on
the RTX 2070 Super. The evaluation takes about a minute.

If the CIFAR-100 or SVHN download fails with an SSL "certificate has expired" error, fetch the files by hand
(PowerShell) and re-run; torchvision then only verifies the MD5:

```powershell
curl.exe -L -k --retry 10 --retry-all-errors -C - -o data\cifar-100-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
curl.exe -L -k --retry 10 --retry-all-errors -C - -o data\test_32x32.mat http://ufldl.stanford.edu/housenumbers/test_32x32.mat
```

Expected MD5: `eb9058c3a382ffc7106e4002c42a8d85` (CIFAR-100 archive), `eb5a983be6a315427106f1b164d9cef3` (SVHN test).

## Post-training methods

Further methods built on each seed's `base.pt` with probly's public API, compared with the earlier criteria in
`evaluate_shift.py` (criteria `{method}_{quantity}` in all tables and plots; methods without dumps are skipped).

| method | what it does | budget |
|---|---|---|
| `finetune` | control: the plain base model fine-tuned like the dropout stage, criterion 1 - max softmax | 50 epochs, lr 0.01, x0.5 every 10 |
| `swag` | `swag(base, max_rank=20, scale=0.5)`, `collect_swag` once per epoch from epoch 5 on (the last epoch is always collected), 30 samples via the probly representer; the SWA-mean deterministic softmax is stored as well (`swa_maxprob`) | 20 epochs, constant lr 0.01, momentum 0.9, wd 5e-4 |
| `laplace` | `laplace.Laplace(model, "classification", subset_of_weights="last_layer", hessian_structure="kron")`, fitted on the un-augmented training set, prior precision optimized by marglik, 100 samples via the probly representer | none |
| `gda` | probly's `GaussianMixtureHead` fitted on the 512-dim features in front of the last Linear layer of the base model; criterion = `negative_log_density`, predictions stay the base softmax. A density/OOD score, not a confidence | none |
| `ddu` | `ddu(base, sn_coeff=3.0)` fine-tuned, then the density head is fitted on training features; criteria 1 - max softmax and negative log density | 20 epochs, lr 0.01, x0.5 every 10, fp32 |
| `vbll` | `vbll(base, parameterization="dense")` (a fresh dense variational last layer), trained with `vbll_loss`, kl_weight 1/50000, 100 samples via the probly representer | 20 epochs, whole network SGD lr 0.01 (no weight decay on the VBLL layer), fp32 |
| `sngp` | `sngp(base, norm_multiplier=6.0, random_feature_init_std=0.05, momentum=-1.0)` (spectral normalization, random-feature GP last layer), trained with cross entropy on the GP mean logits; the precision matrix is reset at the start of every epoch. Criteria 1 - max softmax of the mean logits and the Dempster-Shafer epistemic score (`quantify(predict(model, x)).epistemic`) | 20 epochs, lr 0.01, x0.5 every 10, fp32 |

Criteria: `maxprob` (1 - max mean probability), and for the sampling methods `total`, `aleatoric`, `epistemic` from
`probly.quantification.quantify`; `density` for `gda` and `ddu`; `ds` (Dempster-Shafer) for `sngp`.

Notes:

- The VGG has no residual connections, which the DDU paper relies on, so `ddu` is DDU-style only (probly warns about it).
- VBLL: the whole-network SGD run was checked for stability on a small subset (finite, decreasing loss, accuracy kept);
  head-only Adam was not needed.
- Samplers force every `nn.Dropout` into train mode. For SWAG the dropout `p` is therefore set to 0 at inference
  (`model.disable_dropout`), so the samples carry only the SWAG randomness.
- SWAG and BatchNorm: sampled weights do not match the BatchNorm running statistics, and probly does not recompute them.
  `--swag-bn-update per_sample` (dump_methods.py) does it through a forward pre-hook on the wrapped model: for every
  weight sample the BN statistics are recomputed on a fixed 5k training subset (`--swag-bn-samples`). That makes every
  sampled forward pass cost (batch + 5k) images instead of batch, so use a large `--batch-size` (e.g. 2000) with it.
  The default is `none`.
- `swag.pt` stores the weights and the SWAG statistics (about 1.4 GB per seed, `last_swag.pt` the same again).
- Laplace and the density heads are refitted on the training set whenever `dump_methods.py` has something left to
  dump for that method and seed (a few seconds to a minute), instead of being saved.

```powershell
uv run python scripts/train.py --stage finetune --seed 0 --out runs      # also: swag, ddu, vbll, sngp
uv run python scripts/dump_methods.py                                   # all methods, all seeds, resumable
uv run python scripts/evaluate_shift.py                                 # needs dump_shift.py first
uv run python scripts/run_all.py --methods                              # everything above for seeds 0..4
```

`dump_methods.py` writes `runs/seed{S}/shift/{method}/{dataset}.npz` (labels, `mean_probs`, the criteria). The
evaluation adds figures per group: `*_methods_maxprob.png` (confidence scores) and `*_methods_uncertainty.png`
(epistemic and density scores) for the ID risk-coverage curves, the OOD acceptance and the shift risk.

Runtime estimate (RTX 2070 Super, about 9 s per training epoch, about 15k img/s per forward pass): training takes about
(50 + 20 + 20 + 20) x 9 s = 16.5 min per seed, 1.4 h for 5 seeds. Dumping 156k images per seed: finetune, gda, ddu
(one forward pass each) about 10 s per method; laplace and vbll (one pass of the network, 100 cheap last-layer
samples) about 15 s each plus the fit (about a minute for Laplace); swag 30 forward passes, about 5 min (about 10 min with
`--swag-bn-update per_sample`). Roughly 10 to 15 min per seed, 1 h for all seeds. These are estimates, not measurements.

## Reusing the trained models

```python
from sgr_experiment.loaders import load_base, load_dropout

base = load_base("runs/seed0/base.pt")                 # plain VGG, eval mode
mc_model = load_dropout("runs/seed0/dropout.pt", p=0.5)  # transformed model, pass to probly (representer, ...)
```
