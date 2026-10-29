"""Lab adapters — panel data + ML: dynpanelai, mllpanel, deeppanel, mlpaneldata, panelxai."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.lab.base import LabResult, LabSpec, Opt, Role, capture, to_df

PANEL_ROLES = [Role("unit", "معرّف الوحدة (الفرد/الشركة/الدولة)", "col"), Role("time", "متغير الزمن", "col")]


def _panel(df, r, cols):
    keep = [r["unit"], r["time"], *cols]
    d = df[list(dict.fromkeys(keep))].dropna().copy()
    return d.sort_values([r["unit"], r["time"]]).reset_index(drop=True)


def _bar(series: pd.Series, title: str, color="#059669"):
    s = series.sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=[str(i) for i in s.index], orientation="h", marker_color=color))
    fig.update_layout(height=max(300, 28 * len(s) + 120), title=title)
    return fig


# ================================================================= dynpanelai
def _dpa_demo():
    import dynpanelai as dp
    df = dp.datasets.load_abond_employment()
    return df, {"unit": "id", "time": "year", "y": "n", "d": "w", "X": ["w", "k"]}, {}


DPA_EST = ["Fixed effects", "Debiased FE", "Anderson–Hsiao", "Difference GMM", "Panel Lasso (desparsified)", "DML dynamic panel"]


def _dpa_run(df, r, o):
    import dynpanelai as dp
    X = list(r["X"])
    d = _panel(df, r, [r["y"], *X, *([r["d"]] if r.get("d") else [])])
    panel = dp.PanelData(d, unit=r["unit"], time=r["time"])
    L = int(o["lags"])
    res, out = {}, LabResult()
    for name in o["estimators"]:
        try:
            if name == "Fixed effects":
                res[name] = dp.fixed_effects(panel, r["y"], lags=L, x=X)
            elif name == "Debiased FE":
                res[name] = dp.debiased_fe(panel, r["y"], lags=L, x=X)
            elif name == "Anderson–Hsiao":
                res[name] = dp.anderson_hsiao(panel, r["y"], lags=1, exogenous=X)
            elif name == "Difference GMM":
                pre = [r["d"]] if r.get("d") and r["d"] in X else []
                exo = [c for c in X if c not in pre]
                res[name] = dp.diff_gmm(panel, r["y"], lags=L, predetermined=pre, exogenous=exo, gmm_lags=(2, 4),
                                        collapse=bool(o["collapse"]), steps=int(o["steps"]))
            elif name == "Panel Lasso (desparsified)":
                res[name] = dp.PanelLasso(r["y"], lags=L, x=X).fit(panel)
            elif name == "DML dynamic panel":
                if not r.get("d"):
                    out.notes.append("DML يحتاج متغير معالجة d."); continue
                xc = [c for c in X if c != r["d"]]
                res[name] = dp.DMLDynamicPanel(r["y"], r["d"], xc or None, y_lags=L, learner=o["learner"]).fit(panel)
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"تعذّر تقدير {name}: {type(e).__name__}: {e}")
    if not res:
        raise RuntimeError("لم ينجح أي مقدّر — تحقق من المتغيرات وطول البانل.")
    out.tables["جدول المقارنة"] = dp.comparison_table(res)
    out.text = "\n\n".join(capture(v.summary) for v in res.values())
    out.metrics = {"عدد الوحدات": int(d[r["unit"]].nunique()), "عدد الفترات": int(d[r["time"]].nunique()), "المقدّرات الناجحة": len(res)}
    if d[r["time"]].nunique() < 15:
        out.notes.append("T قصير (< 15): المقدّرات الآلية (DML، Panel Lasso) تحتاج √N/T → 0؛ فضّل Difference GMM أو تصحيحات التحيز.")
    return out


DYNPANELAI = LabSpec(
    key="dynpanelai", pip="dynpanelai", name="dynpanelai — البانل الديناميكي بتعلم الآلة", family="panel", data_kind="panel",
    reference="Arellano & Bond (1991); Chernozhukov, Fernández-Val, Huang & Wang (2024); Semenova et al. (2023); Kock & Tang (2019); Sneller (2026)",
    roles=PANEL_ROLES + [Role("y", "المتغير التابع", "num"), Role("d", "متغير المعالجة/المحدد مسبقًا (اختياري)", "num", optional=True),
                         Role("X", "المفسرات", "nums")],
    opts=[Opt("estimators", "المقدّرات المطلوب مقارنتها", "multi", DPA_EST[:4], DPA_EST),
          Opt("lags", "فجوات المتغير التابع", "int", 1, min=1, max=3), Opt("collapse", "طيّ الأدوات (collapse)", "bool", True),
          Opt("steps", "خطوات GMM", "select", 2, [1, 2]), Opt("learner", "متعلم DML", "select", "lasso", ["lasso", "rf"])],
    run=_dpa_run, demo=_dpa_demo, demo_label="بيانات Arellano–Bond: 140 شركة بريطانية × 9 سنوات",
    theory=r"""
$$y_{it}=\sum_{l=1}^{L}\rho_l y_{i,t-l}+x_{it}'\beta+\alpha_i+\varepsilon_{it}$$
الآثار الثابتة تتفاعل مع المتغير التابع المتأخر ⇒ **تحيز نيكل** $O(1/T)$. الحل الكلاسيكي (GMM بأدوات المستويات المتأخرة) يولّد عددًا من الشروط يتزايد كـ $T^2$ فيعيد التحيز بالإفراط في التوفيق. والتطبيقات الحديثة تضيف مئات الضوابط حيث الاستدلال الساذج بعد لاسو باطل.

| الوحدة | الطريقة | المصدر |
|---|---|---|
| gmm | Difference GMM، Anderson–Hsiao | Arellano & Bond (1991) |
| biascorr | تصحيحات تحليلية، جاكنايف مقسّم | Hahn & Kuersteiner (2002); Dhaene & Jochmans (2015) |
| ablasso | اختيار الشروط العزمية بلاسو | Chernozhukov et al. (2024) |
| dml | DML بطيات زمنية محجوبة | Sneller (2026) |
| ortho | لاسو متعامد لـ CATE عالي الأبعاد | Semenova et al. (2023) |
| hdpanel | استدلال موحّد، آثار ثابتة متناثرة ضعيفًا | Kock & Tang (2019) |
| neural | AC-GATE لاكتشاف الفجوات | Xu (2026) |
""",
    guide="""
1. حدد **الوحدة** و**الزمن** و**y** و**المفسرات**؛ و **d** إن أردت DML أو معاملته كمحدد مسبقًا في GMM.
2. اختر المقدّرات للمقارنة: الجدول يضعها جنبًا إلى جنب (معاملات، أخطاء معيارية، تشخيصات Hansen و AR).
3. **قاعدة الاختيار**: T قصير ⇒ Difference GMM مطويّ أو تصحيحات التحيز؛ ضوابط كثيرة/لا خطية ⇒ DML أو Panel Lasso.
4. Difference GMM يطابق xtabond2 بفارق أقل من 0.2% وفق توثيق المكتبة.
""",
    code="""import dynpanelai as dp
panel = dp.PanelData(df, unit="id", time="year")
res = dp.diff_gmm(panel, y="n", lags=2, predetermined=["w"], exogenous=["k"], gmm_lags=(2, 4), collapse=True, steps=2)
print(res.summary())""",
)


# =================================================================== mllpanel
def _mll_demo():
    from mllpanel.datasets import load_snmesp
    return load_snmesp(), {"unit": "firm", "time": "year", "y": "y", "X": ["n", "k"]}, {}


def _mll_run(df, r, o):
    from mllpanel import MLLPanel
    d = _panel(df, r, [r["y"], *r["X"]])
    m = MLLPanel(learner=o["learner"], spec=o["spec"], vcov=o["vcov"], n_folds=int(o["n_folds"]), random_state=0).fit(
        d, y=r["y"], x=list(r["X"]), unit=r["unit"], time=r["time"])
    out = LabResult(text=capture(m.summary))
    out.tables["المعاملات β (حدود المتوسطات)"] = to_df(m.summary_frame())
    for v in r["X"][:3]:
        try:
            pdp = m.partial_dependence(v)
            xcol, ycol = pdp.columns[0], pdp.columns[-1]
            fig = go.Figure(go.Scatter(x=pdp[xcol], y=pdp[ycol], mode="lines", line=dict(color="#059669", width=3)))
            fig.update_layout(height=320, title=f"الاعتماد الجزئي للدالة غير المعلمية f على {v}", xaxis_title=v, yaxis_title="f̂")
            out.figs.append((f"f({v})", fig))
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"تعذّر الاعتماد الجزئي لـ {v}: {e}")
    return out


MLLPANEL = LabSpec(
    key="mllpanel", pip="mllpanel", name="mllpanel — بانل بتعلم الآلة مع آثار ثابتة وعوامل مشتركة", family="panel", data_kind="panel",
    reference="Yang, Long & Cai (2024) — Machine Learning Based Panel Data Models; Mundlak (1978); Pesaran (2006)",
    roles=PANEL_ROLES + [Role("y", "المتغير التابع", "num"), Role("X", "المفسرات (الدالة المجهولة f)", "nums")],
    opts=[Opt("learner", "المتعلم", "select", "lasso", ["lasso", "rf", "nn", "ols"]),
          Opt("spec", "المواصفة", "select", "crfe", ["pooled", "fe", "crfe"], help="crfe = متوسطات مقطعية + آثار ثابتة"),
          Opt("vcov", "الأخطاء المعيارية", "select", "twoway", ["twoway", "cluster_unit", "cluster_time", "dk", "theorem1"]),
          Opt("n_folds", "عدد الطيات (على الزمن)", "int", 5, min=2, max=10)],
    run=_mll_run, demo=_mll_demo, demo_label="بيانات SNMESP الحقيقية: 738 شركة صناعية إسبانية × 8 سنوات",
    theory=r"""
$$y_{it}=f(x_{it})+\alpha_i+\gamma_i'\lambda_t+\varepsilon_{it}$$
$f$ مجهولة، والآثار الثابتة $\alpha_i$ مرتبطة بالمفسرات، والعوامل المشتركة $\lambda_t$ قد ترتبط بها أيضًا. **الحيلة**: المتغيران الكامنان تمتد عليهما متوسطات مشاهدة — الزمنية تمتص $\alpha_i$ (Mundlak) والمقطعية تمتص $\lambda_t$ (Pesaran):
$$y_{it}=f(x_{it})+\beta'z_{it}+e_{it},\qquad z_{it}=\big(\bar y_t-\bar y,\ \bar x_t-\bar x,\ \bar y_i-\bar y,\ \bar x_i-\bar x\big)$$
فيصبح النموذج شبه خطي يُقدَّر بـ DML: إزالة $g=\mathbb E[y|x]$ و $m=\mathbb E[z|x]$ بأي متعلم، **توفيق متقاطع على الزمن** (كل الوحدات في الفترة t في الطيّة نفسها كي لا يتسرب $\lambda_t$)، حل الشرط المتعامد، ثم استرجاع $f$ من انحدار $y-\hat\beta'z$ على $x$. **بلا متغير بديل ولا أداة.**
""",
    guide="""
1. حدد الوحدة والزمن و y والمفسرات (مثلًا العمل ورأس المال لدالة إنتاج).
2. **crfe** هي المواصفة الكاملة (عوامل + آثار ثابتة)؛ قارن بـ fe و pooled لترى أثر المتغيرات الكامنة.
3. المتعلم: rf لدالة إنتاج غير معلمية، lasso أسرع.
4. الأخطاء المعيارية **twoway** عنقودية بالوحدة والزمن.
5. اقرأ منحنيات الاعتماد الجزئي: شكل $f$ دون فرض كوب–دوغلاس.
""",
    code="""from mllpanel import MLLPanel
m = MLLPanel(learner="rf", spec="crfe", vcov="twoway", random_state=0).fit(df, y="y", x=["n","k"], unit="firm", time="year")
print(m.summary())""",
)


# ================================================================== deeppanel
def _dp_demo():
    from deeppanel.datasets import simulate_pooled_panel
    sim = simulate_pooled_panel(N=15, T=60, p=3, seed=0)
    P = sim.panel
    N, T, p = P.X.shape
    rows = [{"unit": f"u{i + 1}", "time": t, "y": P.y[i, t], **{f"x{k + 1}": P.X[i, t, k] for k in range(p)}} for i in range(N) for t in range(T)]
    return pd.DataFrame(rows), {"unit": "unit", "time": "time", "y": "y", "X": [f"x{k + 1}" for k in range(p)]}, {}


def _dp_run(df, r, o):
    from deeppanel import DeepPooledPanel, PanelData, TrainConfig
    d = _panel(df, r, [r["y"], *r["X"]])
    panel = PanelData.from_long(d, unit=r["unit"], time=r["time"], y=r["y"], x=list(r["X"]))
    model = DeepPooledPanel(horizon=int(o["horizon"]), depth=int(o["depth"]), width=int(o["width"]), idiosyncratic=bool(o["idio"]),
                            config=TrainConfig(max_epochs=int(o["epochs"]), lr=float(o["lr"]), batch_size=int(o["batch"]), seed=0), seed=0)
    model.fit(panel)
    out = LabResult()
    yhat = model.forecast(panel)
    out.tables["التنبؤ المباشر h خطوة لكل وحدة"] = pd.DataFrame({"الوحدة": panel.units, "ŷ": yhat})
    grad = model.partial_derivatives(panel.X[:, -1, :])
    g = pd.DataFrame(grad, columns=list(r["X"]), index=panel.units)
    out.tables["المشتقات الجزئية ∂g/∂x (آخر فترة)"] = g
    out.figs.append(("متوسط الأثر الحدي", _bar(g.mean(), "متوسط المشتقات الجزئية عبر الوحدات")))
    try:
        test = model.poolability_test(panel)
        out.metrics = {"إحصاءة قابلية التجميع": float(test.statistic), "p-value": float(test.p_value)}
        out.notes.append("رفض فرضية قابلية التجميع ⇒ أضف المكوّن الخاص بكل وحدة (idiosyncratic).")
    except Exception as e:  # noqa: BLE001
        out.notes.append(f"تعذّر اختبار قابلية التجميع: {e}")
    out.text = f"Architecture: {getattr(model, 'architecture', '')}\nN={panel.y.shape[0]}, T={panel.y.shape[1]}, p={panel.X.shape[2]}"
    return out


DEEPPANEL = LabSpec(
    key="deeppanel", pip="deeppanel", name="deeppanel — الشبكات العميقة للبانل", family="panel", data_kind="panel", runtime="نحو 15 ثانية",
    reference="Chronopoulos, Chrysikou, Kapetanios, Mitchell & Raftapostolos (2023; Econometric Reviews 2026); Gao, Sun, Wang, Liu & Hsiao (2026)",
    roles=PANEL_ROLES + [Role("y", "المتغير التابع", "num"), Role("X", "المفسرات", "nums")],
    opts=[Opt("horizon", "أفق التنبؤ المباشر h", "int", 1, min=1, max=8), Opt("depth", "عدد الطبقات L", "int", 3, min=1, max=6),
          Opt("width", "عرض الطبقة M", "int", 20, min=4, max=128), Opt("idio", "مكوّن خاص بكل وحدة", "bool", False),
          Opt("epochs", "الحقب القصوى", "int", 300, min=50, max=3000, step=50), Opt("lr", "معدل التعلم", "float", 0.01, min=0.0001, max=0.1, step=0.001),
          Opt("batch", "حجم الدفعة", "int", 32, min=4, max=512)],
    run=_dp_run, demo=_dp_demo, demo_label="محاكاة بانل مجمّع من deeppanel (15 وحدة × 60 فترة)",
    theory=r"""
$$y_{i,t+h}=h(x_{it})+h_i(x_{it})+\varepsilon_{i,t+h}$$
**المكوّن المشترك** $h$ شبكة ReLU عميقة مقدّرة بخسارة MSE المجمّعة عبر كل الوحدات (مع عقوبة لاسو على الطبقة الأولى $\lambda=c\sqrt{\log p/NT}$)، و**المكوّن الخاص** $h_i$ اختياري. التجميع يعوّض قِصر السلاسل الفردية.

**قابلية التفسير**: المشتقات الجزئية الدقيقة $\partial \hat g/\partial x$ بالتفاضل الآلي. **اختبار قابلية التجميع** غير الخطي لـ $h_i=h$. وتضيف المكتبة نموذج **LDPM** الذي يدمج مخرجات نماذج اللغة للتنبؤ الإقليمي بالتضخم مع فترات مطابقة.
""",
    guide="""
1. البانل يجب أن يكون **متوازنًا** (كل وحدة لها كل الفترات).
2. h = 1 للتنبؤ بالفترة القادمة؛ العمق 3 والعرض 20 كما في الورقة.
3. فعّل **المكوّن الخاص** إن رفض اختبار قابلية التجميع.
4. النتائج: تنبؤ لكل وحدة، ومشتقات جزئية (آثار حدية)، واختبار قابلية التجميع.
""",
    code="""from deeppanel import PanelData, DeepPooledPanel, TrainConfig
panel = PanelData.from_long(df, unit="country", time="year", y="inflation", x=["unemp","core","energy"])
m = DeepPooledPanel(horizon=4, depth=3, width=20, config=TrainConfig(max_epochs=1000, lr=0.01, seed=0)).fit(panel)
m.forecast(panel); m.partial_derivatives(panel.X[:, -1, :]); m.poolability_test(panel)""",
)


# ================================================================ mlpaneldata
def _mlp_demo():
    from mlpaneldata import simulate_panel
    df = simulate_panel(n_units=30, n_periods=40, n_features=5, seed=0)
    X = [c for c in df.columns if c.startswith("x")]
    return df, {"unit": "unit", "time": "time", "y": "y", "X": X}, {}


def _mlp_run(df, r, o):
    from mlpaneldata import HybridPanel, full_pretest_suite, regression_table
    d = _panel(df, r, [r["y"], *r["X"]])
    out = LabResult()
    if o["pretests"]:
        try:
            pre = full_pretest_suite(d, y=r["y"], X=list(r["X"]), unit=r["unit"], time=r["time"])
            out.text += "==== Pre-estimation tests ====\n" + capture(pre.summary) + "\n\n"
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"تعذّرت الاختبارات القبلية: {e}")
    m = HybridPanel(linear_part=o["linear"], nn_part=o["nn"], nn_kwargs={"epochs": int(o["epochs"])})
    m.fit(d, r["y"], list(r["X"]), unit=r["unit"], time=r["time"])
    out.text += "==== Hybrid model ====\n" + capture(m.summary) + "\n\n" + regression_table([m])
    yhat = np.asarray(m.predict(d)).ravel()
    yv = d[r["y"]].to_numpy()
    if len(yhat) == len(yv):
        out.metrics = {"R² داخل العينة": float(1 - np.var(yv - yhat) / np.var(yv)), "RMSE": float(np.sqrt(np.mean((yv - yhat) ** 2)))}
        fig = go.Figure(go.Scatter(x=yv, y=yhat, mode="markers", marker=dict(size=4, color="#059669", opacity=.5)))
        fig.update_layout(height=360, title="الفعلي مقابل المقدّر (الهجين)", xaxis_title="y", yaxis_title="ŷ")
        out.figs.append(("الملاءمة", fig))
    return out


MLPANELDATA = LabSpec(
    key="mlpaneldata", pip="mlpaneldata", name="mlpaneldata — الهجين الخطي–العصبي للبانل", family="panel", data_kind="panel", runtime="عشرات الثواني",
    reference="Yang, Zheng & E (2020) arXiv 2010.05311; Chronopoulos et al. (2023) arXiv 2305.19921",
    roles=PANEL_ROLES + [Role("y", "المتغير التابع", "num"), Role("X", "المفسرات", "nums")],
    opts=[Opt("linear", "الجزء الخطي", "select", "within", ["within", "pooled", "re", "mundlak", "fd"]),
          Opt("nn", "الجزء العصبي على البواقي", "select", "dnn_panel", ["dnn_panel", "inn"]),
          Opt("epochs", "الحقب", "int", 150, min=20, max=2000, step=10), Opt("pretests", "الاختبارات القبلية (Hausman, CD, CIPS...)", "bool", True)],
    run=_mlp_run, demo=_mlp_demo, demo_label="محاكاة بانل غير خطي (30 وحدة × 40 فترة)",
    theory=r"""
**مقدّر هجين ذو خطوتين**: نموذج بانل خطي (آثار ثابتة/عشوائية/مندلاك/فروق) يلتقط البنية المعلمية، ثم مكوّن عصبي يتعلم **البواقي** غير الخطية:
$$y_{it}=x_{it}'\beta+\alpha_i+\underbrace{\phi_\omega(x_{it})}_{\text{NN on residuals}}+\varepsilon_{it}$$
ومنهجيتان عصبيتان: **الشبكات القابلة للتفسير بمرشحات التغير المستمر** (Yang, Zheng & E, 2020) و**شبكة البانل العميقة** بالمكوّن المشترك والخاص (Chronopoulos et al., 2023)، مع بطارية كاملة من الاختبارات القبلية والبعدية.
""",
    guide="""
1. فعّل **الاختبارات القبلية** لترى Hausman، F للآثار الثابتة، Pesaran CD، جذور الوحدة في البانل، VIF...
2. اختر الجزء الخطي (within افتراضيًا) والجزء العصبي (dnn_panel).
3. قارن المقدّر الهجين بالخطي وحده بنسبة R² وجدول الانحدار.
""",
    code="""from mlpaneldata import HybridPanel, full_pretest_suite, regression_table
print(full_pretest_suite(df, y="y", X=X, unit="unit", time="time").summary())
m = HybridPanel(linear_part="within", nn_part="dnn_panel", nn_kwargs={"epochs": 200}).fit(df, "y", X, unit="unit", time="time")
print(regression_table([m]))""",
)


# =================================================================== panelxai
def _pxai_demo():
    import panelxai
    df = panelxai.simulate_panel_ts(n_units=25, n_periods=40, n_features=4, seed=0)
    X = [c for c in df.columns if c not in ("unit", "time", "y") and not str(c).startswith(("regime", "f_", "factor"))]
    return df, {"unit": "unit", "time": "time", "y": "y", "X": X[:4]}, {}


def _pxai_run(df, r, o):
    import panelxai
    d = _panel(df, r, [r["y"], *r["X"]])
    cls = panelxai.HybridPanel if o["model"] == "HybridPanel" else panelxai.GBPanel
    model = cls(lags=int(o["lags"]), csa=bool(o["csa"]), n_estimators=int(o["n_estimators"])).fit(d, y=r["y"], X=list(r["X"]), unit=r["unit"], time=r["time"])
    out = LabResult()
    try:
        out.text = capture(model.summary)
    except Exception:
        pass
    try:
        out.metrics = {k: float(v) for k, v in model.score().items() if isinstance(v, (int, float))}
    except Exception:
        pass
    ex = panelxai.StructuredExplainer(model, background=int(o["background"]))
    vi = ex.variable_importance()
    out.tables["أهمية المتغيرات (SHAP مهيكل)"] = vi
    out.tables["أهمية الفجوات"] = ex.lag_importance()
    out.tables["الخاص مقابل العوامل المشتركة"] = ex.own_vs_factor()
    try:
        s = vi.set_index(vi.columns[0])[vi.columns[1]] if vi.shape[1] >= 2 else vi.iloc[:, 0]
        out.figs.append(("أهمية المتغيرات", _bar(s.astype(float), "متوسط |SHAP| حسب المتغير")))
    except Exception:
        pass
    try:
        vl = ex.variable_lag_matrix()
        fig = go.Figure(go.Heatmap(z=vl.values, x=[str(c) for c in vl.columns], y=[str(i) for i in vl.index], colorscale="Viridis"))
        fig.update_layout(height=360, title="مصفوفة المتغير × الفجوة")
        out.figs.append(("المتغير × الفجوة", fig))
    except Exception:
        pass
    return out


PANELXAI = LabSpec(
    key="panelxai", pip="panelxai", name="panelxai — تفسير الذكاء الاصطناعي في البانل", family="panel", data_kind="panel", runtime="عشرات الثواني",
    reference="منهجية المؤلف — تبني على SHAP (Lundberg & Lee, 2017) و CCE (Pesaran, 2006)",
    roles=PANEL_ROLES + [Role("y", "المتغير التابع", "num"), Role("X", "المفسرات", "nums")],
    opts=[Opt("model", "النموذج", "select", "HybridPanel", ["HybridPanel", "GBPanel"], help="HybridPanel = نواة قياسية خطية + بواقي ML"),
          Opt("lags", "عدد الفجوات", "int", 2, min=1, max=6), Opt("csa", "متوسطات مقطعية (CCE)", "bool", True),
          Opt("n_estimators", "عدد الأشجار", "int", 300, min=50, max=1500, step=50), Opt("background", "حجم عينة الخلفية لـ SHAP", "int", 150, min=50, max=1000, step=50)],
    run=_pxai_run, demo=_pxai_demo, demo_label="محاكاة بانل زمني بفجوات وعوامل مشتركة وانكسار نظام",
    theory=r"""
قيمة SHAP الخام مفهرسة بـ (مشاهدة، ميزة) فقط. **panelxai** يعيد فهرستها على **البنية الاقتصادية للبانل الديناميكي**:
$$\text{variable}\times\text{lag}\times\text{unit}\times\text{time}\quad+\quad\text{own vs. factor (CSD) decomposition}$$
- **أهمية حسب الفجوة** و**مصفوفة متغير × فجوة**.
- **فصل مساهمة الديناميكيات الخاصة عن العوامل المشتركة** (متوسطات مقطعية كبدائل للعوامل).
- **انجراف التفسير بين الأنظمة** وانقلاب إشارات الأثر.
- **فترات bootstrap عنقودية** لأهميات المتغيرات، و**بدائل مضادة للواقع مقيدة**.
- نموذج **هجين**: نواة قياسية خطية + بواقي تعلم آلة.
""",
    guide="""
1. حدد الوحدة والزمن و y والمفسرات؛ تُبنى الفجوات والمتوسطات المقطعية آليًا مع تحويل داخلي للآثار الثابتة.
2. **HybridPanel** يعطي جدولًا خطيًا قابلًا للتفسير + مكوّن ML؛ **GBPanel** تعزيز كامل.
3. اقرأ أهمية المتغيرات، أهمية الفجوات، ونسبة الخاص مقابل العوامل المشتركة.
""",
    code="""import panelxai
m = panelxai.HybridPanel(lags=2, csa=True).fit(df, y="y", X=X, unit="unit", time="time")
ex = panelxai.StructuredExplainer(m)
ex.variable_importance(); ex.lag_importance(); ex.own_vs_factor()""",
)

SPECS = [DYNPANELAI, MLLPANEL, DEEPPANEL, MLPANELDATA, PANELXAI]
