"""Seeds, devices and run directories."""

import datetime as _dt
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy and PyTorch so runs repeat exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_run_dir(name: str, config: dict | None = None, root: str | Path = "results") -> Path:
    """Create results/<timestamp>_<name>/ and save the run's config there."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    if config is not None:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    return run_dir
