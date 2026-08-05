"""Shared bookkeeping for the PHVE experiment scripts."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys

import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def hardware_record():
    """Everything needed to reproduce or reinterpret a timing."""
    return {
        "cpu": _cpu_model(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "git_rev": _git_rev(),
        "perf_counters_available": _perf_available(),
    }


def _perf_available():
    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as f:
            level = int(f.read().strip())
        return {"perf_event_paranoid": level, "usable": level <= 2}
    except OSError:
        return {"perf_event_paranoid": None, "usable": False}


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def save_json(obj, name):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_Enc)
    print(f"[saved] {path}")
    return path


def load_json(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)
