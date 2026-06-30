"""Compatibility helpers for importing numbered revision scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def import_benchmark_classes():
    path = Path(__file__).resolve().parent / "07_train_benchmarks.py"
    spec = importlib.util.spec_from_file_location("revision_train_benchmarks", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import benchmark script from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MLPClassifier, module.GRUClassifier
