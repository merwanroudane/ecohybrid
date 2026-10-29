"""Lab adapters — Granger causality & causal discovery: drgct, shapneurgc, castor-causal, nonparmtvcsq."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.lab.base import LabResult, LabSpec, Opt, Role, capture, clean, to_df


def _heat(M, labels_x, labels_y, title, scale="Reds"):
    fig = go.Figure(go.Heatmap(z=M, x=labels_x, y=labels_y, colorscale=scale, text=np.round(M, 3), texttemplate="%{text}"))
    fig.update_layout(height=max(320, 40 * len(labels_y) + 140), title=title, xaxis_title="السبب", yaxis_title="النتيجة")
    return fig


# ====================================================================== drgct
def _drgct_demo():
    import drgct
    s = drgct.simulate_dgp("P1", n=500, lag=1, rng=0)
    return pd.DataFrame({"X": s.x, "Y": s.y}), {"x": "X", "y": "Y"}, {}


def _drgct_run(df, r, o):
    import drgct
    d = clean(df, [r["x"], r["y"]])
    x, y = d[r["x"]].to_numpy(float), d[r["y"]].to_numpy(float)
    kw = dict(lag=int(o["lag"]), B=int(o["B"]), doubly_robust=bool(o["dr"]), seed=0,
              mlp={"epochs": int(o["epochs"])}, mdn={"epochs": int(o["epochs"])})
    out = LabResult()
    if o["both"]:
        res = drgct.drgc_both_directions(x, y, x_name=r["x"], y_name=r["y"], **kw)
        rows = []
        for k, v in res.items():
            out.text += capture(v.summary) + "\n\n"
            rows.append({"الاتجاه": v.direction, "إحصاءة KS": v.ks_stat, "p-value": v.pvalue, "رفض عدم السببية": v.reject})
        out.tables["الاختبار في الاتجاهين"] = pd.DataFrame(rows)
        main = res["x_to_y"]
    else:
        main = drgct.drgc_test(x, y, x_name=r["x"], y_name=r["y"], **kw)
        out.text = capture(main.summary)
        out.tables["النتيجة"] = to_df(main.to_frame())
    out.metrics = {"إحصاءة KS": float(main.ks_stat), "p-value (bootstrap)": float(main.pvalue), "القرار": "سببية" if main.reject else "لا سببية"}
    bs = np.asarray(main.boot_stats)
    fig = go.Figure(go.Histogram(x=bs, nbinsx=40, marker_color="#e11d48", opacity=.7, name="bootstrap"))
    fig.add_vline(x=float(main.ks_stat), line_dash="dash", annotation_text="الإحصاءة المشاهدة")
    fig.update_layout(height=340, title="توزيع إحصاءة KS تحت العدم (multiplier bootstrap)")
    out.figs.append(("توزيع الـ bootstrap", fig))
    return out


DRGCT = LabSpec(
    key="drgct", pip="drgct", name="drgct — اختبار غرانجر المزدوج المتانة بالتعلم العميق", family="granger", runtime="ثوانٍ معدودة",
    reference="Hui, Liu & Song (2025) — arXiv 2509.15798",
    roles=[Role("x", "السبب المحتمل X", "num"), Role("y", "النتيجة Y", "num")],
    opts=[Opt("lag", "الفجوة", "int", 1, min=1, max=10), Opt("B", "تكرارات الـ bootstrap", "int", 300, min=100, max=2000, step=100),
          Opt("epochs", "حقب تدريب الشبكات", "int", 150, min=50, max=800, step=50), Opt("dr", "الصيغة المزدوجة المتانة", "bool", True),
          Opt("both", "الاختبار في الاتجاهين", "bool", False)],
    run=_drgct_run, demo=_drgct_demo, demo_label="محاكاة P1 من الورقة (X يسبب Y بفجوة 1، n=500)",
    theory=r"""
**فرضية عدم السببية** غير الخطية: $H_0:\ Y_t\perp X_{t-1:t-q}\mid Y_{t-1:t-p}$ — أي أن التوزيع الشرطي لـ $Y_t$ لا يتغير بإضافة ماضي $X$.

**الاختبار**: إحصاءة كولموغوروف–سميرنوف على عملية تجريبية موزونة تبني على **دالتين مساعدتين** تُقدَّران بالتعلم العميق: المتوسط الشرطي $m(\cdot)$ بشبكة MLP، والكثافة الشرطية $\phi(\cdot)$ بشبكة مزيج كثافات (MDN). **الصيغة المزدوجة المتانة** تجعل الإحصاءة صالحة إن أُحسن تقدير أحدهما، والقيم الحرجة من **multiplier bootstrap** (رادماخر). تتفوق على اختبارات التنعيم بالنواة (NHKJ) حين تكبر أبعاد الفجوات.
""",
    guide="""
1. اختر X (السبب المحتمل) و Y (النتيجة) — سلاسل **مستقرة** (عوائد/نسب تغير).
2. الفجوة 1–5؛ مسح عدة فجوات يكشف أفق الأثر.
3. B = 300 للتجربة، 1000 للنشر؛ زد الحقب إن كانت العلاقة معقدة.
4. فعّل **الاتجاهين** لاختبار X→Y و Y→X معًا.
""",
    code="""import drgct
res = drgct.drgc_test(x, y, lag=1, B=1000, seed=0)
print(res.summary())
both = drgct.drgc_both_directions(x, y, lag=1, x_name="P", y_name="V")""",
)


# ================================================================ shapneurgc
def _srngc_demo():
    rng = np.random.default_rng(1)
    T, k = 500, 5
    X = np.zeros((T, k))
    for t in range(2, T):
        X[t, 0] = 0.5 * X[t - 1, 0] + rng.normal(0, .5)
        X[t, 1] = 0.3 * X[t - 1, 1] + 0.8 * np.tanh(X[t - 1, 0]) + rng.normal(0, .5)
        X[t, 2] = 0.3 * X[t - 1, 2] + 0.6 * X[t - 2, 1] ** 2 / (1 + X[t - 2, 1] ** 2) + rng.normal(0, .5)
        X[t, 3] = 0.4 * X[t - 1, 3] + rng.normal(0, .5)
        X[t, 4] = 0.2 * X[t - 1, 4] + 0.5 * np.sin(X[t - 1, 3]) + rng.normal(0, .5)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4", "x5"])
    return df, {"X": list(df.columns)}, {}


def _srngc_run(df, r, o):
    from shapneurgc import SRNGC
    d = clean(df, list(r["X"]))
    m = SRNGC(lag=int(o["lag"]), penalty=o["penalty"], hidden_dim=int(o["hidden"]), lam_ind=float(o["lam"]),
              max_epochs=int(o["epochs"]), verbose=0, seed=2025).fit(d.to_numpy(float), names=list(r["X"]))
    out = LabResult(text=capture(m.summary))
    G = m.graph_frame()
    out.tables["مصفوفة الأهمية السببية (السبب ← النتيجة)"] = G
    out.tables["أقوى الروابط"] = to_df(m.to_frame(top=20))
    out.figs.append(("الرسم السببي", _heat(G.values, [str(c) for c in G.columns], [str(i) for i in G.index], "أهمية شابلي المعلوماتية لكل رابط")))
    try:
        A = m.discover(q=float(o["q"]))
        out.tables[f"الحواف المختارة (q = {o['q']})"] = pd.DataFrame(A, index=G.index, columns=G.columns).astype(int)
    except Exception as e:  # noqa: BLE001
        out.notes.append(f"تعذّر اختيار الحواف: {e}")
    return out


SHAPNEURGC = LabSpec(
    key="shapneurgc", pip="shapneurgc", name="shapneurgc — غرانجر العصبي المنظّم بشابلي", family="granger", runtime="ثوانٍ معدودة",
    reference="Yang, Zhu, Tian, Gao & Li (2026), Shapley Regularized Neural Granger Causality, ICML — PMLR 306",
    roles=[Role("X", "السلاسل الزمنية (متعددة المتغيرات)", "nums")],
    opts=[Opt("lag", "الفجوات", "int", 3, min=1, max=10), Opt("penalty", "العقوبة", "select", "shap", ["shap", "fshap", "jacob_l1", "jacob_f"]),
          Opt("hidden", "حجم الطبقة الخفية", "int", 64, min=8, max=256), Opt("lam", "شدة التنظيم λ", "float", 0.001, min=0.0, max=0.1, step=0.0005),
          Opt("epochs", "الحقب القصوى", "int", 600, min=100, max=3000, step=100), Opt("q", "مستوى اختيار الحواف q", "float", 0.1, min=0.01, max=0.3, step=0.01)],
    run=_srngc_run, demo=_srngc_demo, demo_label="محاكاة: x1→x2→x3 و x4→x5 بعلاقات غير خطية",
    theory=r"""
**غرانجر العصبي** (Tank et al.) يكشف السببية عبر أوزان الطبقة الأولى؛ لكن الأوزان مؤشر هش للأهمية. **SRNGC** يستخدم **قيمة شابلي المعلوماتية (Info-Shap)** كمقياس عالمي لإسناد الميزات، ويدخلها كتنظيم في دالة الخسارة:
$$\min_\theta\ \mathcal L_{pred}(\theta)+\lambda\sum_{i}\big\|\text{InfoShap}_i(\theta)\big\|$$
فتنطفئ مساهمات الفجوات غير السببية. الأهمية النهائية لكل رابط (سبب، نتيجة، فجوة) تُلخَّص في مصفوفة، ثم **قاعدة اختيار مُتحقق منها** (Knockoffs/الاستقرار) تحدد الحواف — وتوثق المكتبة أن «قاعدة الاختيار أهم من المقدّر».
""",
    guide="""
1. اختر 3–15 سلسلة **مستقرة** (فروق/معدلات نمو).
2. الفجوات 3 كبداية؛ العقوبة **shap** هي مقترح الورقة.
3. الحقب 600 للتجربة؛ زدها للبيانات الطويلة.
4. اقرأ المصفوفة (أعمدة = أسباب، صفوف = نتائج) ثم الحواف المختارة عند q.
""",
    code="""from shapneurgc import SRNGC
m = SRNGC(lag=3, penalty="shap", max_epochs=2000).fit(X, names=names)
print(m.summary()); m.graph_frame(); m.discover(q=0.1)""",
)


# ===================================================================== castor
def _castor_demo():
    from castor.datasets import simulate_regime_mts
    data = simulate_regime_mts(n_regimes=2, n_nodes=5, n_samples=[400, 400], lag=1, random_state=0)
    df = pd.DataFrame(np.asarray(data.X), columns=[f"x{i + 1}" for i in range(np.asarray(data.X).shape[1])])
    return df, {"X": list(df.columns)}, {}


def _castor_run(df, r, o):
    from castor import CASTOR
    d = clean(df, list(r["X"]))
    m = CASTOR(lag=int(o["lag"]), window=int(o["window"]), min_regime_duration=int(o["min_dur"]), max_iter=int(o["max_iter"]),
               functional_form=o["form"], edge_threshold=float(o["thr"]), random_state=0).fit(d.to_numpy(float), var_names=list(r["X"]))
    out = LabResult()
    rt = m.regime_table()
    out.tables["الأنظمة المكتشفة"] = rt
    out.metrics = {"عدد الأنظمة": int(len(m.regime_intervals()))}
    names = list(r["X"])
    for k in sorted(m.regime_intervals())[:4]:
        try:
            g = m.get_graph(k, weighted=True)
            W = np.asarray(g.W if hasattr(g, "W") else g.to_matrix(), dtype=float)
            if W.ndim == 3:   # (lag, i, j) -> sum over lags
                W = np.abs(W).sum(0)
            out.figs.append((f"النظام {k}", _heat(np.abs(W).T, names, names, f"الرسم السببي المتأخر للنظام {k}", "Purples")))
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"تعذّر رسم النظام {k}: {e}")
    out.text = "\n".join(f"Regime {k}: {v}" for k, v in m.regime_intervals().items())
    return out


CASTOR = LabSpec(
    key="castor-causal", pip="castor-causal", name="castor — بنى سببية متعددة الأنظمة", family="granger", runtime="نحو 75 ثانية",
    reference="Rahmani & Frossard (2025), Causal Temporal Regime Structure Learning, AISTATS — PMLR 258",
    roles=[Role("X", "السلاسل الزمنية (متعددة المتغيرات)", "nums")],
    opts=[Opt("lag", "الفجوة", "int", 1, min=1, max=5), Opt("window", "طول النافذة الأولية", "int", 200, min=50, max=1000, step=10),
          Opt("min_dur", "أدنى مدة للنظام", "int", 100, min=20, max=1000, step=10), Opt("max_iter", "تكرارات EM", "int", 8, min=2, max=30),
          Opt("form", "الشكل الدالي", "select", "linear", ["linear", "nonlinear"]), Opt("thr", "عتبة الحواف", "float", 0.4, min=0.05, max=1.0, step=0.05)],
    run=_castor_run, demo=_castor_demo, demo_label="محاكي CASTOR الرسمي: نظامان × 400 مشاهدة، 5 متغيرات",
    theory=r"""
معظم طرق الاكتشاف السببي الزمني تفترض **نظامًا مستقرًا واحدًا**؛ حين تتغير الآلية في منتصف العينة تعطي رسمًا متوسطًا لا يصف أي نظام.

**CASTOR** يتعلم معًا: **عدد الأنظمة** $K$، **حدود كل نظام**، و**رسمًا سببيًا زمنيًا لكل نظام** (حواف لحظية $W$ ومتأخرة $A$) بخوارزمية شبيهة بـ EM:
- خطوة E: احتمالات الانتماء للأنظمة عبر محاذاة زمنية ناعمة (softmax).
- خطوة M: تقدير رسم لكل نظام بقيد اللادورية المستمر $h(W)=\operatorname{tr}(e^{W\circ W})-d=0$ (أسلوب NOTEARS) بصيغة خطية أو عصبية.
""",
    guide="""
1. اختر سلاسل متعددة طويلة نسبيًا (مئات المشاهدات).
2. **النافذة** حجم القطع الأولية؛ **أدنى مدة** تمنع أنظمة قصيرة جدًا.
3. ابدأ بالشكل **الخطي**؛ الشكل العصبي أبطأ.
4. اقرأ جدول الأنظمة (البداية/النهاية) ورسم كل نظام.
""",
    code="""from castor import CASTOR
m = CASTOR(lag=1, window=200, min_regime_duration=100).fit(X, var_names=names)
m.regime_table(); m.get_graph(0)""",
)


# ================================================================ nonparmtvcsq
def _np_demo():
    rng = np.random.default_rng(5)
    T = 400
    x = rng.standard_t(5, T) * 0.8
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.2 * y[t - 1] + rng.normal(0, 0.5 + 0.6 * abs(x[t - 1]))
    return pd.DataFrame({"x": x, "y": y}), {"x": "x", "y": "y"}, {}


def _np_run(df, r, o):
    import nonparmtvcsq as nq
    d = clean(df, [r["x"], r["y"]])
    x, y = d[r["x"]].to_numpy(float), d[r["y"]].to_numpy(float)
    q = np.round(np.arange(float(o["q_min"]), float(o["q_max"]) + 1e-9, float(o["q_step"])), 3)
    out = LabResult()
    if o["test"] == "NPCQ":
        res = nq.np_quantile_causality(x, y, type=o["type"], q=q, method=o["method"])
        out.text = capture(res.summary)
        fr = to_df(res.to_frame())
        out.tables["الإحصاءة عبر الكمّيات"] = fr
        fig = go.Figure(go.Scatter(x=res.quantiles, y=res.statistic, mode="lines+markers", line=dict(color="#0d9488", width=3)))
        fig.add_hline(y=res.cv, line_dash="dash", annotation_text="القيمة الحرجة 5%")
        fig.update_layout(height=360, title=f"السببية في الكمّيات ({o['type']})", xaxis_title="الكمّي τ", yaxis_title="الإحصاءة")
        out.figs.append(("NPCQ", fig))
        out.metrics = {"كمّيات دالة": int(np.sum(np.asarray(res.statistic) > res.cv))}
    elif o["test"] == "TVNCQ":
        res = nq.tv_np_quantile_causality(x, y, type=o["type"], q=q, window=int(o["window"]), step=int(o["step"]))
        S = np.asarray(res.statistic)
        if S.shape[0] != len(q):
            S = S.T
        fig = go.Figure(go.Heatmap(z=S, x=list(res.window_end), y=list(res.quantiles), colorscale="Viridis"))
        fig.update_layout(height=420, title="السببية في الكمّيات المتغيرة زمنيًا (نهاية النافذة × الكمّي)", xaxis_title="نهاية النافذة", yaxis_title="τ")
        out.figs.append(("TVNCQ", fig))
        out.tables["نسبة الرفض"] = to_df(res.reject_share()) if callable(getattr(res, "reject_share", None)) else pd.DataFrame()
    else:
        data = pd.DataFrame({r["y"]: y, r["x"]: x})
        res = nq.tvgc(data, depvar=r["y"], causes=[r["x"]], p=int(o["p"]), window=int(o["window"]), boot=int(o["boot"]), seed=0)
        out.text = capture(res.summary)
        out.tables["إحصاءات Wald المتغيرة زمنيًا"] = to_df(res.to_frame())
        try:
            sf = to_df(res.series_frame())
            fig = go.Figure([go.Scatter(y=sf[c], name=str(c), mode="lines") for c in sf.columns])
            fig.update_layout(height=360, title="Wald: forward / rolling / recursive")
            out.figs.append(("TVGC", fig))
        except Exception:
            pass
    return out


NONPARM = LabSpec(
    key="nonparmtvcsq", pip="nonparmtvcsq", name="nonparmtvcsq — السببية اللامعلمية في الكمّيات والمتغيرة زمنيًا", family="granger", runtime="نحو 30 ثانية (NPCQ)",
    reference="Balcilar et al. (2018); Jeong, Härdle & Song (2012); Shi, Phillips & Hurn; Otero & Baum (tvgc)",
    roles=[Role("x", "السبب X", "num"), Role("y", "النتيجة Y", "num")],
    opts=[Opt("test", "الاختبار", "select", "NPCQ", ["NPCQ", "TVNCQ", "TVGC"]), Opt("type", "في المتوسط أم التباين", "select", "variance", ["mean", "variance"]),
          Opt("method", "طريقة التقدير (NPCQ)", "select", "sm", ["sm", "lp"], help="sm أسرع؛ lp يطابق حزمة R الأصلية بدقة الآلة"),
          Opt("q_min", "أدنى كمّي", "float", 0.1, min=0.05, max=0.5, step=0.05), Opt("q_max", "أعلى كمّي", "float", 0.9, min=0.5, max=0.95, step=0.05),
          Opt("q_step", "خطوة الكمّيات", "float", 0.1, min=0.05, max=0.2, step=0.05), Opt("window", "النافذة (TVNCQ/TVGC)", "int", 150, min=50, max=600, step=10),
          Opt("step", "خطوة النافذة (TVNCQ)", "int", 25, min=1, max=100), Opt("p", "فجوات VAR (TVGC)", "int", 2, min=1, max=8),
          Opt("boot", "bootstrap (TVGC)", "int", 99, min=49, max=999, step=50)],
    run=_np_run, demo=_np_demo, demo_label="محاكاة: X يسبب تباين Y (سببية في التباين لا في المتوسط)",
    theory=r"""
**NPCQ** (Jeong, Härdle & Song 2012؛ Balcilar et al. 2018): اختبار لا معلمي بالنواة لعدم السببية **عند كل كمّي** $\tau$:
$$H_0:\ \Pr\{Q_\tau(y_t\mid y_{t-1},x_{t-1})=Q_\tau(y_t\mid y_{t-1})\}=1$$
وعلى المربعات لاختبار السببية **في التباين**. **TVNCQ**: النسخة المتدحرجة ⇒ سطح زمن × كمّي. **TVGC** (Shi, Phillips & Hurn): إحصاءات Wald للسببية بنوافذ أمامية ومتدحرجة وتكرارية متطورة مع قيم حرجة بالـ bootstrap.
""",
    guide="""
1. اختر X و Y المستقرين.
2. **NPCQ** لاختبار عبر الكمّيات؛ اختر **variance** لكشف السببية في التقلب.
3. **TVNCQ** لسطح متغير زمنيًا (اختر النافذة والخطوة).
4. **TVGC** لسببية غرانجر المتغيرة زمنيًا بثلاثة مخططات.
""",
    code="""import nonparmtvcsq as nq
res = nq.np_quantile_causality(x, y, type="variance", q=np.arange(0.1, 0.95, 0.05))
res.summary(); nq.tv_np_quantile_causality(x, y, window=150); nq.tvgc(df, depvar="y", causes=["x"], p=2)""",
)

SPECS = [DRGCT, SHAPNEURGC, CASTOR, NONPARM]
