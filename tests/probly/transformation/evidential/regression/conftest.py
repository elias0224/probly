import os
import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")

@pytest.fixture(scope="session")
def rng_seed():
    return 1337

@pytest.fixture(autouse=True)
def _set_seed(rng_seed):
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)

@pytest.fixture(scope="session")
def device():
    return "cpu"

@pytest.fixture
def toy_regression_data():
    # y = 3x + 1 + noise
    x = torch.linspace(-2, 2, 128).unsqueeze(1)
    noise = 0.1 * torch.randn_like(x)
    y = 3 * x + 1 + noise
    return x, y

@pytest.fixture
def base_backbone():
    # 你们库里如果要求 backbone 输出 1 维均值，再由 transform 包装成 evidential 的 4 参数，
    # 就用这段。若要求 4 维直接由 backbone 输出，自己把 out_features=4。
    import torch.nn as nn
    return nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
