"""Tests of the ImageNet (Sec. 5.3) helpers: top-k losses and criteria, the paper's tables and the split protocol."""

from __future__ import annotations

import numpy as np
import torch

from sgr_experiment.imagenet import (
    PAPER_TABLES,
    evaluate_unit,
    paper_bound_check,
    rules_with_sweep,
    splits,
    topk_losses,
    torch_topk_complement,
)
from sgr_experiment.uncertainty import one_minus_max


def test_topk_losses() -> None:
    top5 = np.array([[3, 1, 2, 0, 4], [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    labels = np.array([3, 4, 1])
    np.testing.assert_array_equal(topk_losses(top5, labels, 1), [0, 1, 1])
    np.testing.assert_array_equal(topk_losses(top5, labels, 5), [0, 0, 1])


def test_topk_complement() -> None:
    probs = torch.softmax(torch.randn(50, 1000, generator=torch.Generator().manual_seed(0)), dim=-1)
    top = probs.double().topk(5, dim=-1).values
    np.testing.assert_allclose(torch_topk_complement(probs, 5).numpy(), 1 - top.sum(-1).numpy(), atol=1e-6)
    np.testing.assert_allclose(torch_topk_complement(probs, 1).numpy(), one_minus_max(probs.numpy()), atol=1e-12)


def test_paper_tables_are_monotone() -> None:
    assert len(PAPER_TABLES) == 4
    for rows in PAPER_TABLES.values():
        r = np.array(rows)
        assert np.all(np.diff(r[:, 0]) > 0)  # r*
        assert np.all(np.diff(r[:, 2]) > 0)  # train coverage
        assert np.all(np.diff(r[:, 4]) > 0)  # test coverage
        assert np.all(r[:, 5] <= r[:, 0] + 1e-4)  # bound at most r*


def test_paper_bound_check_row() -> None:
    row = paper_bound_check("vgg16", "top1")[0]  # r* = 0.02, train risk 0.0161, train coverage 0.2355
    assert row["accepted"] == 5888
    assert row["errors"] == 95
    assert row["bound_split"] > row["bound_full"] > row["paper_bound"]


def test_evaluate_unit() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    crit = rng.random(n)
    loss = (rng.random(n) < 0.3 * crit).astype(np.float64)  # error rate grows with the criterion
    rows = PAPER_TABLES["vgg16", "top5"][:3]
    halves = splits(n, 3, seed=0)
    assert all(len(s) == len(t) == n // 2 and not set(s) & set(t) for s, t in halves)
    acc = evaluate_unit(crit, loss, rows, halves, delta=0.001)
    for r, *_ in rows:
        for rule in ("sgr", "emp"):
            assert len(acc[rule, r, "test_cov"]) == 3
            assert all(0 <= c <= 1 for c in acc[rule, r, "test_cov"])
        bounds = np.array(acc["sgr", r, "bound"])
        assert np.all(np.isnan(bounds) | (bounds <= r + 1e-12))
        assert np.all(np.isnan(acc["emp", r, "bound"]))
        assert len(acc[r, "cov_at_paper"]) == len(acc[r, "risk_at_paper"]) == 3


def test_delta_sweep_buys_coverage() -> None:
    rng = np.random.default_rng(1)
    n = 6000
    crit = rng.random(n)
    loss = (rng.random(n) < 0.2 * crit).astype(np.float64)
    rules = rules_with_sweep([0.1])
    assert list(rules) == ["sgr", "emp", "sgr_delta_0.1"]
    acc = evaluate_unit(crit, loss, PAPER_TABLES["vgg16", "top5"][2:3], splits(n, 2, seed=0), 0.001, rules)
    r = PAPER_TABLES["vgg16", "top5"][2][0]
    sgr_cov, loose_cov, emp_cov = (np.mean(acc[k, r, "sel_cov"]) for k in ("sgr", "sgr_delta_0.1", "emp"))
    assert sgr_cov <= loose_cov <= emp_cov
