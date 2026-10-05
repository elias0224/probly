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
   rotation augmentation, AMP on CUDA.
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

```powershell
git pull
cd experiments/selective_prediction_sgr
uv sync -p 3.13
uv run python scripts/run_all.py
```

On Windows, torch/torchvision come from the PyTorch CUDA 12.6 index (see `pyproject.toml`); elsewhere from PyPI.
`run_all.py` loops seeds 0..4 (train base, fine-tune dropout, dump predictions), skips finished stages, resumes an
interrupted stage from its checkpoint, and finally runs the evaluation (`--no-evaluate` to skip it). CIFAR-10 is
downloaded to `data/` on first use. Results land in `results/`: `table.md`, `table.csv`, `risk_coverage.png`.
Other options: `--seeds`, `--runs`, `--data-dir`, `--out`, `--workers` (default min(8, CPU count)), `--num-samples`
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

## Reusing the trained models

```python
from sgr_experiment.loaders import load_base, load_dropout

base = load_base("runs/seed0/base.pt")                 # plain VGG, eval mode
mc_model = load_dropout("runs/seed0/dropout.pt", p=0.5)  # transformed model, pass to probly (representer, ...)
```
