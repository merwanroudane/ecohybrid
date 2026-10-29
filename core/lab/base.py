"""Shared building blocks of the applied lab: specs, roles, options, results."""

from __future__ import annotations

import contextlib
import io
import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FAMILIES = {
    "causal": "السببية و DML",
    "panel": "البانل + ML",
    "granger": "سببية غرانجر واكتشاف السببية",
    "var": "VAR والسلاسل العصبية",
    "vol": "التقلب والمخاطر",
    "kernel": "النواة والموجات والكمّيات",
}


@dataclass
class Role:
    """A variable the user maps to a column. kind: num | nums | col."""
    key: str
    label: str
    kind: str = "num"
    optional: bool = False
    help: str = ""


@dataclass
class Opt:
    """An estimation option. kind: int | float | select | bool | text."""
    key: str
    label: str
    kind: str
    default: Any
    options: list | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    help: str = ""


@dataclass
class LabResult:
    text: str = ""
    metrics: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    figs: list = field(default_factory=list)      # (title, plotly-or-matplotlib figure)
    notes: list = field(default_factory=list)


@dataclass
class LabSpec:
    key: str
    pip: str
    name: str
    family: str
    reference: str
    roles: list
    opts: list
    run: Callable[[pd.DataFrame, dict, dict], LabResult]
    demo: Callable[[], tuple]
    theory: str
    guide: str
    code: str
    data_kind: str = "ts"          # ts | panel | xy
    runtime: str = "ثوانٍ"
    demo_label: str = "بيانات تجريبية"


def execute(spec: LabSpec, df: pd.DataFrame, roles: dict, opts: dict) -> tuple[LabResult | None, str, str, float]:
    """Run a spec capturing stdout, warnings and errors."""
    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = spec.run(df.copy(), roles, opts)
        return res, buf.getvalue(), "", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return None, buf.getvalue(), f"{type(e).__name__}: {e}\n\n{traceback.format_exc(limit=6)}", time.time() - t0


# --------------------------------------------------------------- helpers ----
def num(df: pd.DataFrame, cols) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def clean(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    cols = [c for c in dict.fromkeys(cols) if c]
    out = df[cols].copy()
    for c in cols:
        if c in out and out[c].dtype == object:
            out[c] = pd.to_numeric(out[c], errors="ignore")
    return out.dropna()


def capture(fn, *a, **k) -> str:
    """Return whatever a print-style summary method writes (or returns)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    s = buf.getvalue()
    if isinstance(r, str):
        s = (s + "\n" + r).strip()
    elif r is not None and not s:
        s = str(r)
    return s


def to_df(obj) -> pd.DataFrame | None:
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    if isinstance(obj, dict):
        try:
            return pd.DataFrame(obj)
        except Exception:
            return pd.Series(obj, dtype=object).to_frame("value")
    try:
        return pd.DataFrame(obj)
    except Exception:
        return None


def lag_matrix(y: np.ndarray, X: np.ndarray | None, ky: int, kx: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack ky lags of y and kx lags of X (columns) to predict y_t."""
    T = len(y)
    m = max(ky, kx)
    cols = [y[m - j - 1:T - j - 1] for j in range(ky)]
    if X is not None and X.size:
        for j in range(kx):
            cols.append(X[m - j - 1:T - j - 1])
    U = np.column_stack(cols)
    return U, y[m:]
