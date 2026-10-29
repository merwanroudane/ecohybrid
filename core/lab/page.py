"""Rendering of the applied lab: one page per family, each library with theory, guide and live runner."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui
from core.lab.base import FAMILIES, LabSpec, execute
from core.lab.registry import by_family

FAMILY_COLOR = {"causal": ("violet", "indigo"), "panel": ("emerald", "teal"), "granger": ("rose", "pink"),
                "var": ("sky", "cyan"), "vol": ("orange", "amber"), "kernel": ("teal", "lime")}


@st.cache_data(show_spinner="تحميل البيانات التجريبية للمكتبة...", ttl=3600, max_entries=40)
def _demo(key: str):
    spec = next(s for f in FAMILIES for s in by_family(f) if s.key == key)
    df, roles, over = spec.demo()
    return df, roles, over


def user_data() -> tuple[pd.DataFrame | None, str]:
    return st.session_state.get("lab_df"), st.session_state.get("lab_df_name", "")


def _role_widgets(spec: LabSpec, df: pd.DataFrame, defaults: dict, prefix: str) -> dict:
    cols_all = list(df.columns)
    num_cols = [c for c in cols_all if pd.api.types.is_numeric_dtype(df[c])]
    roles = {}
    c1, c2 = st.columns(2)
    for i, rl in enumerate(spec.roles):
        box = c1 if i % 2 == 0 else c2
        key = f"{prefix}_role_{rl.key}"
        dv = defaults.get(rl.key)
        if rl.kind == "col":
            opts = cols_all
            idx = opts.index(dv) if dv in opts else 0
            roles[rl.key] = box.selectbox(rl.label, opts, index=idx, key=key, help=rl.help or None)
        elif rl.kind == "num":
            opts = (["— بلا —"] if rl.optional else []) + num_cols
            idx = opts.index(dv) if dv in opts else (0 if rl.optional else min(i, len(opts) - 1))
            v = box.selectbox(rl.label, opts, index=idx, key=key, help=rl.help or None)
            roles[rl.key] = None if v == "— بلا —" else v
        else:
            dvl = [c for c in (dv or []) if c in num_cols]
            roles[rl.key] = box.multiselect(rl.label, num_cols, default=dvl, key=key, help=rl.help or None)
    return roles


def _opt_widgets(spec: LabSpec, roles: dict, over: dict, prefix: str) -> dict:
    vals = {}
    cols = st.columns(3)
    for i, o in enumerate(spec.opts):
        box = cols[i % 3]
        key = f"{prefix}_opt_{o.key}"
        dv = over.get(o.key, o.default)
        h = o.help or None
        if o.kind == "int":
            vals[o.key] = box.number_input(o.label, value=int(dv), min_value=int(o.min) if o.min is not None else None,
                                           max_value=int(o.max) if o.max is not None else None, step=int(o.step or 1), key=key, help=h)
        elif o.kind == "float":
            vals[o.key] = box.number_input(o.label, value=float(dv), min_value=float(o.min) if o.min is not None else None,
                                           max_value=float(o.max) if o.max is not None else None, step=float(o.step or 0.01), format="%.4f", key=key, help=h)
        elif o.kind == "select":
            opts = list(o.options)
            vals[o.key] = box.selectbox(o.label, opts, index=opts.index(dv) if dv in opts else 0, key=key, help=h)
        elif o.kind == "bool":
            vals[o.key] = box.toggle(o.label, value=bool(dv), key=key, help=h)
        elif o.kind == "multi":
            vals[o.key] = box.multiselect(o.label, list(o.options), default=[x for x in dv if x in o.options], key=key, help=h)
        elif o.kind == "subset":
            src = roles.get(o.options[0]) or []
            vals[o.key] = box.multiselect(o.label, list(src), default=[x for x in (dv or []) if x in src], key=key, help=h)
        else:
            vals[o.key] = box.text_input(o.label, value=str(dv), key=key, help=h)
    return vals


def _validate(spec: LabSpec, roles: dict) -> list[str]:
    errs = []
    for rl in spec.roles:
        v = roles.get(rl.key)
        if not rl.optional and (v is None or (isinstance(v, list) and len(v) == 0)):
            errs.append(f"اختر: {rl.label}")
    return errs


def _render_result(spec: LabSpec, payload: dict) -> None:
    res, log, err, dt = payload["res"], payload["log"], payload["err"], payload["dt"]
    if err:
        st.error("تعذّر التنفيذ. راجع المتغيرات والخيارات (مثل طول العينة أو نوع البيانات).", icon=":material/error:")
        with st.expander("تفاصيل الخطأ"):
            st.code(err, language="text")
        if log:
            with st.expander("مخرجات المكتبة"):
                st.code(log[-6000:], language="text")
        return
    st.success(f"اكتمل التنفيذ في {dt:.1f} ثانية — البيانات: {payload['data_label']}", icon=":material/check_circle:")
    if res.metrics:
        mcols = st.columns(min(4, len(res.metrics)))
        for i, (k, v) in enumerate(res.metrics.items()):
            mcols[i % len(mcols)].metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    for n in res.notes:
        st.info(n, icon=":material/info:")
    t_names = list(res.tables)
    f_names = [f[0] for f in res.figs]
    sub = st.tabs((["الأشكال"] if res.figs else []) + (["الجداول"] if res.tables else []) + (["ملخص المكتبة"] if (res.text or log) else []))
    i = 0
    if res.figs:
        with sub[i]:
            for title, fig in res.figs:
                if fig is None:
                    continue
                if isinstance(fig, go.Figure):
                    ui.style_fig(fig, fig.layout.title.text or title, height=fig.layout.height or 420)
                    ui.show(fig)
                else:
                    st.pyplot(fig, clear_figure=False)
        i += 1
    if res.tables:
        with sub[i]:
            for name in t_names:
                tdf = res.tables[name]
                if tdf is None or (hasattr(tdf, "empty") and tdf.empty):
                    continue
                st.markdown(f"**{name}**")
                st.dataframe(tdf, height=min(420, 38 + 35 * len(tdf)))
                st.download_button("تنزيل CSV", tdf.to_csv().encode("utf-8-sig"), file_name=f"{spec.key}_{name[:30]}.csv",
                                   key=f"dl_{spec.key}_{name}", icon=":material/download:")
        i += 1
    if res.text or log:
        with sub[i]:
            st.code((res.text or "") + ("\n\n---- log ----\n" + log[-4000:] if log and log.strip() else ""), language="text")


def render_lab(spec: LabSpec) -> None:
    prefix = f"lab_{spec.key}"
    udf, uname = user_data()
    src_opts = ["بيانات تجريبية مدمجة"] + (["بياناتي المرفوعة"] if udf is not None else [])
    src = st.segmented_control("مصدر البيانات", src_opts, default=src_opts[0], key=f"{prefix}_src")
    if udf is None:
        st.caption("لم ترفع بيانات بعد. ارفعها من صفحة «مدير البيانات» ثم عد إلى هنا.")
        ui.link("app_pages/lab_data.py", "فتح مدير البيانات", ":material/upload_file:")
    if src == "بياناتي المرفوعة" and udf is not None:
        df, defaults, over, label = udf, {}, {}, f"ملف المستخدم: {uname}"
    else:
        try:
            df, defaults, over = _demo(spec.key)
        except Exception as e:  # noqa: BLE001
            st.error(f"تعذّر تحميل البيانات التجريبية: {e}")
            return
        label = spec.demo_label
    with st.expander(f"معاينة البيانات — {label} ({df.shape[0]} × {df.shape[1]})"):
        st.dataframe(df.head(200), height=260)
    ui.section("1) اختيار المتغيرات", "Variables", FAMILY_COLOR[spec.family][0])
    roles = _role_widgets(spec, df, defaults, prefix)
    ui.section("2) الخيارات", "Options", FAMILY_COLOR[spec.family][1])
    with st.form(f"{prefix}_form", border=True):
        opts = _opt_widgets(spec, roles, over, prefix)
        go_ = st.form_submit_button("تنفيذ", type="primary", icon=":material/play_arrow:")
    st.caption(f"الزمن المتوقع على البيانات التجريبية: {spec.runtime}.")
    if go_:
        errs = _validate(spec, roles)
        if errs:
            for e in errs:
                st.warning(e)
        else:
            with st.status(f"تشغيل {spec.pip} ...", expanded=False) as status:
                res, log, err, dt = execute(spec, df, roles, opts)
                status.update(label="انتهى" if not err else "فشل التنفيذ", state="complete" if not err else "error")
            st.session_state[f"{prefix}_result"] = {"res": res, "log": log, "err": err, "dt": dt, "data_label": label}
    payload = st.session_state.get(f"{prefix}_result")
    if payload:
        ui.section("3) النتائج", "Results", "slate")
        _render_result(spec, payload)


def render_family(family: str) -> None:
    c1, c2 = FAMILY_COLOR[family]
    specs = by_family(family)
    ui.hero(f"المختبر التطبيقي — {FAMILIES[family]}", "Applied lab for Dr Merwan Roudane's libraries — upload data, choose variables and options, run",
            "لكل مكتبة: <b>شرح نظري</b> للمنهجية ومعادلاتها، <b>دليل تطبيقي</b> للخيارات والقراءة، و<b>مختبر حي</b> يشغّل المكتبة الحقيقية على بياناتك "
            "أو على بيانات تجريبية مدمجة، ويعرض النتائج والأشكال والجداول فورًا مع إمكانية التنزيل.", c1, c2)
    names = [s.name.split(" — ")[0] for s in specs]
    pick = st.pills("اختر المكتبة", names, default=names[0], key=f"lab_pick_{family}")
    spec = specs[names.index(pick)] if pick in names else specs[0]
    st.markdown(f"### {spec.name}")
    st.caption(f"`pip install {spec.pip}`  ·  المرجع: {spec.reference}")
    t1, t2, t3 = st.tabs([":material/menu_book: الشرح النظري", ":material/school: الدليل التطبيقي", ":material/science: المختبر"])
    with t1:
        st.markdown(spec.theory)
    with t2:
        st.markdown(spec.guide)
        st.markdown("**مثال كود Python:**")
        st.code(spec.code, language="python")
        st.markdown("**الأدوار (المتغيرات) المطلوبة:**")
        st.table(pd.DataFrame([{"الدور": r.label, "النوع": {"num": "متغير رقمي واحد", "nums": "متغيرات رقمية متعددة", "col": "عمود"}[r.kind],
                                "إلزامي": "لا" if r.optional else "نعم"} for r in spec.roles]))
        st.markdown("**الخيارات:**")
        st.table(pd.DataFrame([{"الخيار": o.label, "الافتراضي": str(o.default), "ملاحظة": o.help} for o in spec.opts]))
    with t3:
        render_lab(spec)


# ------------------------------------------------------------ data manager --
def read_upload(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    if name.endswith(".dta"):
        return pd.read_stata(file)
    if name.endswith(".parquet"):
        return pd.read_parquet(file)
    raw = file.getvalue()
    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(raw))


TRANSFORMS = {
    "لوغاريتم ln(x)": lambda s, g: np.log(s.where(s > 0)),
    "فرق أول Δx": lambda s, g: s.groupby(g).diff() if g is not None else s.diff(),
    "معدل نمو لوغاريتمي 100·Δln(x)": lambda s, g: 100 * (np.log(s.where(s > 0)).groupby(g).diff() if g is not None else np.log(s.where(s > 0)).diff()),
    "نسبة تغير %": lambda s, g: 100 * (s.groupby(g).pct_change() if g is not None else s.pct_change()),
    "تعيير (z-score)": lambda s, g: (s - s.mean()) / s.std(),
    "فجوة أولى x(t−1)": lambda s, g: s.groupby(g).shift(1) if g is not None else s.shift(1),
}
