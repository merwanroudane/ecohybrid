"""Collect all lab specs lazily (heavy libraries are imported only when run)."""

from __future__ import annotations

import importlib

MODULES = ["specs_causal", "specs_panel", "specs_granger", "specs_var", "specs_vol", "specs_kernel"]


def all_specs() -> list:
    out = []
    for m in MODULES:
        out.extend(importlib.import_module(f"core.lab.{m}").SPECS)
    return out


def by_family(family: str) -> list:
    return [s for s in all_specs() if s.family == family]
