"""Lab adapters — causal inference & DML: pragdml, ardldml, tsdml, oracle-varx."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.lab.base import LabResult, LabSpec, Opt, Role, capture, clean, to_df


def _v(x):
    return x() if callable(x) else x


def _mpl(obj):
    """Return a matplotlib Figure from a figure/axes/tuple returned by a plot helper."""
    if obj is None:
        return None
    if isinstance(obj, tuple):
        obj = obj[0]
    return getattr(obj, "figure", obj)


# ===================================================================== pragdml
def _prag_demo():
    from pragdml.datasets import PENSION_COVARIATES, load_pension
    Y, D, X = load_pension(return_X_y=True)
    df = pd.DataFrame(X, columns=list(PENSION_COVARIATES))
    df.insert(0, "e401", D)
    df.insert(0, "net_tfa", Y)
    return df, {"Y": "net_tfa", "D": "e401", "X": list(PENSION_COVARIATES)}, {}


def _prag_run(df, r, o):
    from pragdml import PragmaticDML
    d = clean(df, [r["Y"], r["D"], *r["X"]])
    Y, D, X = d[r["Y"]].to_numpy(float), d[r["D"]].to_numpy(float), d[r["X"]].to_numpy(float)
    res = PragmaticDML(functional=o["functional"], n_folds=int(o["n_folds"]), n_rep=int(o["n_rep"]), seed=int(o["seed"])).fit(Y, D, X)
    out = LabResult(text=capture(res.summary))
    out.metrics = {"θ̂": float(_v(res.theta)), "الخطأ المعياري": float(_v(res.se)), "p-value": float(_v(res.pvalue)), "n": int(_v(res.n))}
    out.tables["النتيجة"] = to_df(res.to_frame())
    if o["sensitivity"]:
        sa = res.sensitivity()
        out.text += "\n\n" + capture(sa.summary)
        try:
            out.metrics["قيمة المتانة RV"] = float(sa.robustness_value())
        except Exception:
            pass
        try:
            fig = _mpl(res.plot_sensitivity_contour())
            if fig is not None:
                out.figs.append(("منحنيات الحساسية للمعلومات المحذوفة", fig))
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"تعذّر رسم منحنى الحساسية: {e}")
    out.notes.append("الفترة تخص الهدف المبني على التمثيل θ_λ؛ عند فقدان معلومات جوهري اعتمد منطقة الحساسية وقيمة المتانة.")
    return out


PRAGDML = LabSpec(
    key="pragdml", pip="pragdml", name="pragdml — Pragmatic DML", family="causal", data_kind="xy",
    reference="Aradillas Fernández, Chernozhukov, Cinelli, Klaassen, Newey, Spindler et al. (2026); Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (REStat 2026)",
    roles=[Role("Y", "المتغير التابع Y", "num"), Role("D", "المعالجة D", "num", help="ثنائية للـ ATE/ATT أو مستمرة لمتوسط المشتقة"),
           Role("X", "الضوابط / التمثيل X", "nums", help="متغيرات أو أبعاد تضمين (Embedding)")],
    opts=[Opt("functional", "المعلمة الهدف", "select", "ate", ["ate", "att", "average_derivative"]),
          Opt("n_folds", "عدد الطيات K", "int", 5, min=2, max=10),
          Opt("n_rep", "تكرار التقسيم", "int", 1, min=1, max=10),
          Opt("sensitivity", "تحليل الحساسية وقيمة المتانة", "bool", True),
          Opt("seed", "البذرة", "int", 0, min=0, max=10_000)],
    run=_prag_run, demo=_prag_demo, demo_label="بيانات 401(k) الحقيقية (SIPP 1991، 9915 أسرة)",
    theory=r"""
**المشكلة**: نضغط متغيرًا غنيًا (نص، صورة، خصائص كثيرة) إلى تمثيل $X_\lambda$ بنموذج أساس ونستخدمه كضابط. هذا الضغط يغيّر **الهدف السكاني** نفسه.

**النتيجة المركزية**: الفجوة بين الهدف الحقيقي والهدف المبني على التمثيل هي **حاصل ضرب** خطأي التمثيل:
$$\theta^*-\theta_\lambda=\mathbb E\big[(g^*-g_\lambda)(\alpha^*-\alpha_\lambda)\big]$$
حيث $g$ انحدار النتيجة و $\alpha$ ممثّل ريس (وزن التوازن). يكفي أن يحفظ التمثيل **أحدهما**.

**الدالة المتعامدة**: $\psi = m(O,g)+\alpha(W)\{Y-g(W)\}-\theta$ مع توفيق متقاطع؛ والتمثيل يمكن تعلمه داخل الطيات دون شروط استقرار.

**عند فقدان معلومات جوهري**: منطقة تعرّف لـ $\theta^*$ مع استدلال بجذر n لحدودها، و**قيمة متانة** تقول كم يجب أن تكون قوة المعلومات المحذوفة لقلب الإشارة.
""",
    guide="""
1. اختر **Y** (النتيجة) و **D** (المعالجة) و **X** (الضوابط أو أبعاد التضمين).
2. اختر المعلمة: **ATE** للمعالجة الثنائية، **ATT** للأثر على المعالَجين، **average_derivative** للمعالجة المستمرة.
3. عدد الطيات 5 افتراضيًا؛ زد **تكرار التقسيم** لتقليل حساسية النتيجة للتقسيم.
4. فعّل **تحليل الحساسية** لتحصل على **قيمة المتانة**: نسبة التباين المتبقي التي يجب أن تفسرها المعلومات المحذوفة (في النتيجة والمعالجة) لجعل الحد الأدنى للفترة صفرًا.
5. اقرأ الجدول: التقدير، الخطأ المعياري، الفترة — وتذكّر أنها تخص $\\theta_\\lambda$.
""",
    code="""from pragdml import PragmaticDML
res = PragmaticDML(functional="ate", n_folds=5, n_rep=5, seed=0).fit(Y, D, X)
print(res.summary())
sa = res.sensitivity(); print(sa.summary()); print(sa.robustness_value())""",
)


# ===================================================================== ardldml
def _ardl_demo():
    from ardldml import CONTROLS, DEFAULT_INTEGRATED, load_passthrough
    df = load_passthrough(regime="1999-2007").reset_index()
    return df, {"y": "cpi", "d": "neer", "W": list(CONTROLS)}, {"integrated": [c for c in DEFAULT_INTEGRATED if c in CONTROLS]}


def _ardl_run(df, r, o):
    from ardldml import DMLBounds, plot_bootstrap_null
    d = clean(df, [r["y"], r["d"], *r["W"]]).reset_index(drop=True)
    integ = [c for c in (o.get("integrated") or []) if c in r["W"]] or None
    res = DMLBounds(d[r["y"]], d[r["d"]], d[r["W"]], lags=int(o["lags"]), case=int(o["case"]), n_blocks=int(o["n_blocks"]),
                    buffer=int(o["buffer"]), integrated=integ).fit()
    if int(o["B"]) > 0:
        res = res.bootstrap(B=int(o["B"]), seed=int(o["seed"]))
    out = LabResult(text=capture(res.summary))
    out.metrics = {"إحصاءة F (DML-Bounds)": float(res.stat), "θ̂ (تصحيح الخطأ)": float(res.theta), "SE": float(res.theta_se)}
    try:
        out.metrics["القرار"] = str(res.decision())
    except Exception:
        pass
    out.tables["النتائج"] = to_df(res.to_frame())
    if res.boot is not None:
        try:
            out.figs.append(("توزيع الإحصاءة تحت فرضية العدم (Wild bootstrap نظامي مقيّد)", _mpl(plot_bootstrap_null(res))))
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"تعذّر رسم توزيع الـ bootstrap: {e}")
    out.notes.append("المعلمة الهدف شرطية: هل يتكامل Y و D **بشرط W**؟ عند الشك في امتصاص الاتجاه شغّل trend_absorption بإسقاط الضوابط المشتبه بها.")
    return out


ARDLDML = LabSpec(
    key="ardldml", pip="ardldml", name="ardldml — DML-Bounds", family="causal",
    reference="DML-Bounds — امتداد لاختبار الحدود لـ Pesaran, Shin & Smith (2001)",
    roles=[Role("y", "المتغير التابع (بالمستويات)", "num"), Role("d", "المتغير المحوري (بالمستويات)", "num"),
           Role("W", "الضوابط عالية الأبعاد W", "nums")],
    opts=[Opt("lags", "عدد الفجوات", "int", 4, min=1, max=12), Opt("case", "حالة PSS (الحد الثابت/الاتجاه)", "select", 3, [1, 2, 3, 4, 5]),
          Opt("n_blocks", "عدد الكتل K", "int", 5, min=2, max=10), Opt("buffer", "الفاصل h بين الكتل", "int", 6, min=0, max=24),
          Opt("integrated", "الضوابط المتكاملة I(1)", "subset", [], options=["W"], help="اتركها فارغة ليصنّفها الاختبار آليًا بـ ADF"),
          Opt("B", "تكرارات الـ bootstrap (0 = بلا)", "int", 199, min=0, max=2000, step=50), Opt("seed", "البذرة", "int", 20260625)],
    run=_ardl_run, demo=_ardl_demo, demo_label="بيانات FRED-MD الشهرية: انتقال سعر الصرف إلى الأسعار 1999–2007",
    theory=r"""
اختبار الحدود الكلاسيكي (Pesaran, Shin & Smith, 2001) يحصر مجهولًا واحدًا: هل المفسرات I(0) أم I(1)؟ مع **عشرات الضوابط المستمرة** يظهر مجهول ثانٍ: بعض الضوابط قد تشارك الاتجاه العشوائي، فيمتص إسقاطها جزءًا من العلاقة المختبرة. يحكم توزيع العدم حينها **العدد الفعلي للمتكاملات**:
$$\tilde k=k-r,\qquad \tilde k=k\Rightarrow\text{الحد الأعلى } I(1),\quad 0<\tilde k<k\Rightarrow\text{داخل الحصر},\quad \tilde k=0\Rightarrow\text{الحد الأدنى } I(0)$$
**ثلاثة مكونات**: (1) مرحلة أولى متوازنة — $\Delta Y_t$ يُسقط على الضوابط المستقرة بالمستويات والمتكاملة بالفروق، والمستويات المتأخرة $Z_{t-1}$ على مستويات الضوابط؛ (2) **توفيق متقاطع بكتل h** مع فاصل؛ (3) **Wild bootstrap نظامي مقيّد** بوزن رادماخر واحد على البواقي المكدسة يحفظ الارتباط بين المعادلتين. لا جداول: الباقي $O_p(s\log d/\sqrt T)$ ليس مهملًا في العينات التطبيقية.
""",
    guide="""
1. أدخل **y** و **d** بالمستويات (لوغاريتمات عادة)، واختر **W** (الضوابط).
2. حدد الضوابط **المتكاملة I(1)** إن كنت تعرفها (وإلا تُصنّف آليًا).
3. الفجوات 4 للشهري؛ الكتل 5 والفاصل 6 أشهر كبداية.
4. استخدم **bootstrap** بـ 199 للتجربة السريعة و 999 للنشر.
5. القرار يقارن الإحصاءة بالقيمة الحرجة المحسوبة من الـ bootstrap لا من جداول PSS.
""",
    code="""from ardldml import DMLBounds, load_passthrough, CONTROLS
df = load_passthrough(regime="1999-2007")
res = DMLBounds(df["cpi"], df["neer"], df[CONTROLS], lags=4, n_blocks=5, buffer=6,
                integrated=["m2","ip","oil","gs10","baa","ffr"]).fit().bootstrap(B=999, seed=1)
print(res.summary())""",
)


# ======================================================================= tsdml
def _tsdml_demo():
    import tsdml
    sim = tsdml.simulate_plr(T=240, p=40, theta=1.0, rho=0.7, seed=3)
    X = pd.DataFrame(sim["X"], columns=[f"x{i + 1}" for i in range(sim["X"].shape[1])])
    df = pd.concat([pd.DataFrame({"y": sim["y"], "d": sim["d"]}), X], axis=1)
    return df, {"y": "y", "d": "d", "X": list(X.columns)}, {}


def _tsdml_learner(o):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, LassoCV
    if o["learner"] == "Lasso":
        return Lasso(alpha=float(o["alpha"]), max_iter=100_000)
    if o["learner"] == "LassoCV":
        return LassoCV(cv=5, max_iter=50_000)
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=-1)


def _tsdml_run(df, r, o):
    import tsdml
    from sklearn.base import clone
    from sklearn.linear_model import Lasso
    d = clean(df, [r["y"], r["d"], *r["X"]])
    X, y, dd = d[r["X"]].to_numpy(float), d[r["y"]].to_numpy(float), d[r["d"]].to_numpy(float)
    K = int(o["n_blocks"])
    bsl = None
    out = LabResult()
    if o["goldilocks"] and o["learner"] == "Lasso":
        grid = {"alpha": list(np.linspace(1e-4, 0.4, int(o["grid"]))), "max_iter": [100_000]}
        cal = tsdml.Calibrator(metric="goldilocks_zone", n_blocks=K)
        cal.calibrate(X, y, dd, outcome_learner_class=Lasso, outcome_param_grid=grid, treatment_learner_class=Lasso, treatment_param_grid=grid)
        bsl = cal.block_specific_learners_
        out.notes.append("الضبط بمنطقة غولديلوكس: عقوبة لكل طيّة من منطقة مستقرة في منحنى خطأ التحقق.")
    base = _tsdml_learner(o)
    rows, models = [], {}
    rcf = tsdml.ReverseCrossFitting(clone(base), clone(base), n_blocks=K, block_specific_learners=bsl,
                                    hac_kernel=o["hac_kernel"]).fit(X, y, dd)
    models["RCF (Reverse Cross-Fitting)"] = rcf
    if o["compare_nlo"] and K >= 4:
        models["NLO (Neighbors-Left-Out)"] = tsdml.NLOCrossFitting(clone(base), clone(base), n_blocks=K, hac_kernel=o["hac_kernel"]).fit(X, y, dd)
    for name, m in models.items():
        res = m.results_
        rows.append({"الطريقة": name, "θ̂": res["coef"], "SE (HAC)": res["std_error"], "t": res["t_stat"], "p": res["p_value"],
                     "CI أدنى": res["ci_lower"], "CI أعلى": res["ci_upper"]})
    out.tables["التقديرات"] = pd.DataFrame(rows)
    out.text = capture(rcf.summary)
    r0 = rcf.results_
    out.metrics = {"θ̂ (RCF)": r0["coef"], "SE": r0["std_error"], "p": r0["p_value"]}
    bc = np.asarray(r0["block_coefs"])
    fig = go.Figure(go.Bar(x=[f"B{k + 1}" for k in range(len(bc))], y=bc, marker_color="#7c3aed"))
    fig.add_hline(y=r0["coef"], line_dash="dash", annotation_text="θ̂ المجمّع")
    fig.update_layout(height=360, title="تقديرات θ في كل كتلة (RCF)", xaxis_title="الكتلة", yaxis_title="θ̂_k")
    out.figs.append(("تقديرات الكتل", fig))
    out.tables["استخدام العينة: RCF مقابل NLO"] = tsdml.sample_use_table([4, 5, 6, 8, 10, 12])
    return out


TSDML = LabSpec(
    key="tsdml", pip="tsdml", name="tsdml — DML للسلاسل الزمنية", family="causal",
    reference="Ciganovic, D'Amario & Tancioni (2026), The Econometrics Journal — arXiv 2603.10999",
    roles=[Role("y", "المتغير التابع y_t", "num"), Role("d", "متغير السياسة d_t", "num"), Role("X", "الضوابط X_t", "nums")],
    opts=[Opt("n_blocks", "عدد الكتل K", "int", 6, min=3, max=12),
          Opt("learner", "متعلم الدوال المساعدة", "select", "Lasso", ["Lasso", "LassoCV", "Random forest"]),
          Opt("alpha", "عقوبة لاسو α (إن لم يُستخدم الضبط)", "float", 0.05, min=0.0001, max=1.0, step=0.01),
          Opt("goldilocks", "ضبط منطقة غولديلوكس (Lasso)", "bool", False),
          Opt("grid", "حجم شبكة α للضبط", "int", 25, min=10, max=100),
          Opt("compare_nlo", "قارن بـ NLO", "bool", True),
          Opt("hac_kernel", "نواة HAC", "select", "bartlett", ["bartlett", "qs", "parzen", "ewc"])],
    run=_tsdml_run, demo=_tsdml_demo, demo_label="محاكاة النموذج شبه الخطي الزمني من tsdml (T=240، p=40، θ=1)",
    theory=r"""
$$y_t=\theta_0 d_t+g_0(X_t)+\varepsilon_t,\qquad d_t=m_0(X_t)+\xi_t$$
DML القياسي يتطلب توفيقًا متقاطعًا **عشوائيًا** يمزّق البنية الزمنية، والحل السابق **NLO** يحذف كتلًا فاصلة مكلفة في العينات القصيرة.

**التوفيق المتقاطع العكسي (RCF)**: الطيات الأولى تتعلم على الكتل اليمنى **معكوسة زمنيًا**، والأخيرة على اليسرى للأمام، والوسطى (K فردي) من الجانبين. المبرر: العمليات المستقرة الغاوسية لها التوزيعات نفسها للأمام والخلف. **لا فاصل محذوف**؛ الصلاحية من **الاستقرار الشرطي**:
$$\mathbb E[\xi_t\mid X_t,\mathcal F_{aux,k}]=0,\qquad \mathbb E[\varepsilon_t\mid X_t,\mathcal F_{aux,k}]=0$$
استخدام العينة: $u_{RCF}=\frac{3K-2}{4K}$ (K زوجي) مقابل $u_{NLO}=\frac{(K-1)(K-2)}{K^2}$.

**منطقة غولديلوكس**: العقوبة التي تدني خطأ التنبؤ ليست التي تدني تحيز الدالة السببية؛ نختارها من منطقة **مستقرة محليًا** في منحنى RMSE ⇒ خفض التحيز في العينات الصغيرة بنحو 24%.
""",
    guide="""
1. البيانات **مرتبة زمنيًا** (لا تخلطها). اختر y و d و X.
2. K = 6 كما في تطبيق الورقة؛ RCF يستخدم عينة أكبر من NLO حتى K≈10.
3. اختر المتعلم: Lasso بعقوبة ثابتة (سريع)، LassoCV، أو غابة عشوائية.
4. فعّل **غولديلوكس** لاختيار عقوبة لكل طيّة (أبطأ).
5. الاستدلال **HAC** على سلسلة الدوال المكدسة؛ قارن RCF بـ NLO في الجدول ورسم تقديرات الكتل.
""",
    code="""import tsdml
from sklearn.linear_model import Lasso
cal = tsdml.Calibrator(metric="goldilocks_zone", n_blocks=6)
cal.calibrate(X, y, d, outcome_learner_class=Lasso, outcome_param_grid=grid,
              treatment_learner_class=Lasso, treatment_param_grid=grid)
m = tsdml.ReverseCrossFitting(n_blocks=6, block_specific_learners=cal.block_specific_learners_).fit(X, y, d)
m.summary()""",
)


# ================================================================== oracle-varx
def _ovx_demo():
    rng = np.random.default_rng(7)
    T = 520
    w = np.zeros(T); Y = np.zeros((T, 3))
    for t in range(2, T):
        w[t] = 0.8 * w[t - 1] + rng.normal()
        Y[t, 0] = 0.4 * Y[t - 1, 0] + 0.5 * np.tanh(w[t - 1]) + rng.normal(0, .6)
        Y[t, 1] = 0.3 * Y[t - 1, 1] + 0.45 * Y[t - 2, 0] + 0.4 * np.tanh(w[t - 1]) + rng.normal(0, .6)
        Y[t, 2] = 0.3 * Y[t - 1, 2] + 0.35 * Y[t - 1, 1] + rng.normal(0, .6)
    df = pd.DataFrame(Y, columns=["A", "B", "C"])
    df["W_confounder"] = w
    return df, {"Y": ["A", "B", "C"], "W": ["W_confounder"]}, {"ols_window": 160, "p_max": 4}


def _ovx_run(df, r, o):
    from oracle_varx import oracle_varx
    cols = list(r["Y"]) + list(r.get("W") or [])
    d = clean(df, cols)
    W = d[r["W"]] if r.get("W") else None
    res = oracle_varx(d[r["Y"]], W, p_max=int(o["p_max"]), method=o["method"], learner=o["learner"], ols_window=int(o["ols_window"]),
                      block=int(o["block"]), validation_days=int(o["validation_days"]), q=float(o["q"]), seed=0)
    out = LabResult(text=capture(res.summary))
    out.tables["الحواف السببية"] = to_df(res.to_frame())
    try:
        g = res.graph()
        adj = g.adjacency_frame()
        fig = go.Figure(go.Heatmap(z=adj.values, x=list(adj.columns), y=list(adj.index), colorscale="Purples", text=np.round(adj.values, 3), texttemplate="%{text}"))
        fig.update_layout(height=380, title="مصفوفة الحواف (السبب ← النتيجة)")
        out.figs.append(("الرسم السببي", fig))
    except Exception as e:  # noqa: BLE001
        out.notes.append(f"تعذّر بناء الرسم: {e}")
    try:
        ls = to_df(res.lag_series())
        fig = go.Figure([go.Scatter(y=ls[c], name=str(c), mode="lines") for c in ls.columns])
        fig.update_layout(height=340, title="الفجوة المختارة عبر الزمن")
        out.figs.append(("تتبع الفجوة", fig))
    except Exception:
        pass
    return out


ORACLEVARX = LabSpec(
    key="oracle-varx", pip="oracle-varx", name="oracle-varx — اكتشاف الفجوات السببية", family="causal", runtime="نحو 50 ثانية",
    reference="Tan, Zanoria, Chen, Lyu & Cucuringu (2026) — arXiv 2610.05618",
    roles=[Role("Y", "السلاسل محل الدراسة", "nums"), Role("W", "المرْبكات W (اختياري)", "nums", optional=True)],
    opts=[Opt("p_max", "أقصى فجوة", "int", 5, min=1, max=15), Opt("method", "الطريقة", "select", "oracle-varx", ["oracle-varx", "or-varx", "acle-varx", "varx", "var"]),
          Opt("learner", "المتعلم للمرْبكات", "select", "extra_trees", ["extra_trees", "random_forest", "linear"]),
          Opt("ols_window", "نافذة التقدير المتدحرجة", "int", 200, min=60, max=1000, step=20),
          Opt("block", "طول الكتلة", "int", 20, min=5, max=100), Opt("validation_days", "فترة التحقق", "int", 20, min=5, max=100),
          Opt("q", "معدل الاكتشاف الزائف FDR", "float", 0.05, min=0.01, max=0.2, step=0.01)],
    run=_ovx_run, demo=_ovx_demo, demo_label="محاكاة: A ← فجوة 2 → B ← فجوة 1 → C مع مُربك غير خطي W",
    theory=r"""
**السؤال**: من يسبب من؟ وبأي فجوة؟ ومتى؟ في سلاسل متعددة يربكها متغير مشترك $W$ بشكل غير خطي.

**ORACLE-VARX**: لكل حافة مرشحة $x_i(t-\ell)\to x_j(t)$ يُقدَّر المعامل بـ **DML متعامد** يزيل أثر $W$ بمتعلم مرن (أشجار)، مع **اختيار تكيفي للفجوة** (معيار ACLE) في نوافذ متدحرجة، ثم اختبار كل الحواف مع **ضبط معدل الاكتشاف الزائف** (Benjamini–Hochberg):
$$\hat\beta_{ij\ell}=\frac{\sum_t \tilde x_{i,t-\ell}\,\tilde x_{j,t}}{\sum_t \tilde x_{i,t-\ell}^2},\qquad \tilde x=x-\hat{\mathbb E}[x\mid W,\text{past}]$$
النتيجة: لكل حافة معامل وخطأ معياري وفترة ثقة وقيمة q.
""",
    guide="""
1. اختر **السلاسل** (2–6 سلاسل) و**المرْبكات** إن وُجدت.
2. **p_max** أقصى فجوة مرشحة؛ **النافذة** لتقدير متدحرج (يجب أن تكون أقصر من العينة بوضوح).
3. **الطريقة**: oracle-varx (الأكمل)، أو أبسط منها للمقارنة (var، varx).
4. **q** مستوى FDR للحواف.
5. اقرأ جدول الحواف ومصفوفة الرسم السببي وتتبع الفجوة عبر الزمن.
""",
    code="""from oracle_varx import oracle_varx
res = oracle_varx(Y, W, p_max=5, method="oracle-varx", learner="extra_trees", q=0.05)
print(res.summary()); res.to_frame()""",
)

SPECS = [PRAGDML, ARDLDML, TSDML, ORACLEVARX]
