"""Lab adapters — VAR & neural time series: samovar-var, gpvar, tvwnn, hybridmetrics, bartardl."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.lab.base import LabResult, LabSpec, Opt, Role, capture, clean, lag_matrix, to_df


def _var_sim(T=600, k=3, seed=0, nonlin=True):
    rng = np.random.default_rng(seed)
    Y = np.zeros((T, k))
    A1 = np.array([[0.5, 0.1, 0.0], [0.3, 0.4, 0.0], [0.0, 0.25, 0.5]])[:k, :k]
    A2 = np.array([[-0.2, 0.0, 0.1], [0.0, -0.1, 0.0], [0.1, 0.0, -0.15]])[:k, :k]
    for t in range(2, T):
        Y[t] = A1 @ Y[t - 1] + A2 @ Y[t - 2] + rng.normal(0, .5, k)
        if nonlin:
            Y[t, 1] += 0.3 * np.tanh(2 * Y[t - 1, 0])
    return pd.DataFrame(Y, columns=["gdp", "infl", "rate"][:k])


def _lines(df_or_arr, names, title, x=None):
    arr = np.asarray(df_or_arr)
    fig = go.Figure([go.Scatter(x=x, y=arr[:, j], name=str(n), mode="lines") for j, n in enumerate(names)])
    fig.update_layout(height=360, title=title)
    return fig


# =================================================================== samovar
def _sam_demo():
    df = _var_sim(700)
    return df, {"Y": list(df.columns)}, {}


def _sam_run(df, r, o):
    from samovar import SaMoVAR
    d = clean(df, list(r["Y"]))
    m = SaMoVAR(horizon=int(o["horizon"]), input_length=int(o["input_length"]), n_layers=int(o["n_layers"]), epochs=int(o["epochs"]),
                batch_size=int(o["batch"]), verbose=0, seed=0).fit(d)
    out = LabResult(text=capture(m.summary))
    try:
        ev = m.evaluate()
        out.metrics = {k: float(v) for k, v in ev.items() if isinstance(v, (int, float))}
    except Exception:
        pass
    try:
        out.tables["مقارنة بـ VAR/AR على عينة الاختبار"] = to_df(m.compare())
    except Exception as e:  # noqa: BLE001
        out.notes.append(f"تعذّرت المقارنة: {e}")
    try:
        fc = m.forecast(as_frame=True)
        out.tables["التنبؤ"] = fc
        out.figs.append(("التنبؤ", _lines(fc.values, list(fc.columns), "التنبؤ متعدد الآفاق")))
    except Exception as e:  # noqa: BLE001
        out.notes.append(f"تعذّر التنبؤ: {e}")
    try:
        vc = m.var_coefficients(as_frame=True, max_lags=int(o["max_lags"]))
        out.tables["معاملات VAR المسترجعة من المحوّل"] = to_df(vc)
    except Exception as e:  # noqa: BLE001
        out.notes.append(f"تعذّر استرجاع المعاملات: {e}")
    try:
        irf = np.asarray(m.impulse_response(horizon=12, max_lags=int(o["max_lags"])))
        names = list(r["Y"])
        if irf.ndim == 3:   # (h, response, shock)
            fig = go.Figure([go.Scatter(y=irf[:, i, 0], name=f"{names[i]} ← صدمة {names[0]}") for i in range(irf.shape[1])])
            fig.update_layout(height=360, title="الاستجابات النبضية الضمنية (صدمة في المتغير الأول)")
            out.figs.append(("الاستجابة النبضية", fig))
    except Exception as e:  # noqa: BLE001
        out.notes.append(f"تعذّرت الاستجابة النبضية: {e}")
    return out


SAMOVAR = LabSpec(
    key="samovar-var", pip="samovar-var", name="SaMoVAR — محوّل خطي يكافئ VAR", family="var", runtime="نحو 25 ثانية",
    reference="Lu & Yang (2025), Linear Transformers as VAR Models, ICML — arXiv 2502.07244",
    roles=[Role("Y", "السلاسل الزمنية (2 فأكثر)", "nums")],
    opts=[Opt("horizon", "أفق التنبؤ", "int", 4, min=1, max=24), Opt("input_length", "طول المدخل (فجوات)", "int", 24, min=4, max=128),
          Opt("n_layers", "عدد الطبقات", "int", 2, min=1, max=6), Opt("epochs", "الحقب", "int", 40, min=5, max=300, step=5),
          Opt("batch", "حجم الدفعة", "int", 32, min=8, max=256), Opt("max_lags", "فجوات VAR المعروضة", "int", 4, min=1, max=12)],
    run=_sam_run, demo=_sam_demo, demo_label="محاكاة VAR(2) ثلاثي مع لاخطية خفيفة (T=700)",
    theory=r"""
**النتيجة النظرية**: طبقة انتباه **خطية** ذاتية الانحدار تكتب التنبؤ كمجموع موزون للماضي — أي أنها **نموذج VAR ديناميكي** بمعاملات تعتمد على السياق:
$$\hat{\mathbf y}_{t+1}=\sum_{l=1}^{L}\mathbf A_{l,t}\,\mathbf y_{t+1-l},\qquad \mathbf A_{l,t}=f_{attn}(\mathbf y_{1:t})$$
**SaMoVAR** (Structural Aligned Mixture of VAR) يعيد ترتيب مكونات المحوّل (المدخلات، الانتباه، الطبقات) ليتوافق مع هدف التنبؤ الذاتي الانحدار، فيصبح ممكنًا **استرجاع مصفوفات الفجوات والاستجابات النبضية ومسارات التأثير** من الشبكة المدرّبة، ومقارنتها مع VAR/AR الكلاسيكي واختبار Diebold–Mariano — مثال على **النظرية داخل المعمارية (L5)**.
""",
    guide="""
1. اختر سلسلتين أو أكثر **مستقرتين** (فروق/معدلات نمو).
2. طول المدخل = عدد الفجوات التي يراها المحوّل؛ الأفق = عدد خطوات التنبؤ.
3. الحقب 40 للتجربة السريعة؛ البيانات تُقسم آليًا إلى تدريب/تحقق/اختبار.
4. اقرأ: مقاييس الاختبار، مقارنة VAR/AR، المعاملات المسترجعة، والاستجابات النبضية.
""",
    code="""from samovar import SaMoVAR
m = SaMoVAR(horizon=4, input_length=24, epochs=100).fit(df)
m.evaluate(); m.compare(); m.var_coefficients(as_frame=True); m.impulse_response(horizon=12)""",
)


# ===================================================================== gpvar
def _gp_demo():
    from gpvar import simulate_replication_dgp
    sd = simulate_replication_dgp(T=150, M=3, p=2, seed=1)
    Y = sd.Y.reset_index(drop=True)
    return Y, {"Y": list(Y.columns)}, {}


def _gp_run(df, r, o):
    from gpvar import GPVAR, predict
    d = clean(df, list(r["Y"])).reset_index(drop=True)
    model = GPVAR(d, p=int(o["p"]), var_names=list(r["Y"]), sv=bool(o["sv"]))
    res = model.fit(nburn=int(o["nburn"]), nsave=int(o["nsave"]), seed=0, verbose=False)
    out = LabResult()
    out.metrics = {"زمن المعاينة (ث)": float(res.runtime)}
    try:
        out.tables["ملخص المعلمات الفائقة"] = to_df(res.hyper_summary())
    except Exception:
        pass
    try:
        vol = np.asarray(res.volatility())
        med = vol[1] if vol.ndim == 3 else vol
        out.figs.append(("التقلب العشوائي", _lines(med, list(r["Y"]), "التقلب الشرطي (الوسيط البعدي)")))
    except Exception as e:  # noqa: BLE001
        out.notes.append(f"تعذّر رسم التقلب: {e}")
    try:
        fc = predict(res, horizon=int(o["horizon"]), n_draws=min(200, int(o["nsave"])), seed=0)
        q = fc.quantiles()
        out.tables["التنبؤ الكثافي (كمّيات)"] = q
    except Exception as e:  # noqa: BLE001
        out.notes.append(f"تعذّر التنبؤ: {e}")
    if o["girf"]:
        try:
            from gpvar import girf
            g = girf(res, shock_var=r["Y"][0], horizon=12, n_draws=50, n_sim=3, verbose=False)
            avg = np.asarray(g.average())
            arr = avg if avg.ndim == 2 else avg.reshape(avg.shape[0], -1)
            out.figs.append(("GIRF", _lines(arr, list(r["Y"]), f"الاستجابة النبضية المعممة لصدمة في {r['Y'][0]}")))
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"تعذّرت GIRF: {e}")
    return out


GPVAR = LabSpec(
    key="gpvar", pip="gpvar", name="gpvar — VAR بالعمليات الغاوسية والتقلب العشوائي", family="var", runtime="نحو 25 ثانية",
    reference="Hauzenberger, Huber, Marcellino & Petz (2022), Gaussian Process VARs and Macroeconomic Uncertainty — JBES",
    roles=[Role("Y", "السلاسل (2–6 متغيرات)", "nums")],
    opts=[Opt("p", "الفجوات", "int", 2, min=1, max=6), Opt("sv", "تقلب عشوائي", "bool", True),
          Opt("nburn", "سحب الإحماء MCMC", "int", 300, min=100, max=5000, step=100), Opt("nsave", "السحب المحفوظة", "int", 300, min=100, max=5000, step=100),
          Opt("horizon", "أفق التنبؤ", "int", 8, min=1, max=24), Opt("girf", "حساب GIRF (أبطأ)", "bool", False)],
    run=_gp_run, demo=_gp_demo, demo_label="محاكي التكرار الرسمي للورقة (T=150، 3 متغيرات، p=2)",
    theory=r"""
$$\mathbf y_t=\mathbf f(\mathbf y_{t-1},\dots,\mathbf y_{t-p})+\boldsymbol\varepsilon_t,\qquad f_j\sim\mathcal{GP}\big(0,k_{\kappa,\xi}\big),\qquad \varepsilon_{jt}\sim N(0,\omega_{jt}),\ \log\omega_{jt}\sim AR(1)$$
VAR **لا معلمي**: كل معادلة دالة غير معروفة للفجوات بقبْلي عملية غاوسية مع فصل دوال الفجوات الذاتية عن الأخرى، وقبليات مقيسة بالتقلب تجعل النموذج **مترافقًا**، وشبكة منفصلة للمعلمات الفائقة حول **قاعدة الوسيط**، ومصفوفة متزامنة بانكماش **horseshoe**. اللاخطية تجعل الاستجابات **معممة (GIRF)** تعتمد على التاريخ والإشارة والحجم — الرسالة التجريبية الأساسية حول صدمات عدم اليقين.
""",
    guide="""
1. اختر 2–6 سلاسل (معدلات نمو، عدم يقين، فائدة...)؛ تُعيَّر آليًا.
2. الفجوات 2 للتجربة؛ الورقة تستخدم 5 للفصلي.
3. MCMC: 300+300 للتجربة، 1000+1000 للنشر.
4. فعّل GIRF لاستجابات معممة (تتطلب وقتًا أطول).
""",
    code="""from gpvar import GPVAR, predict, girf
res = GPVAR(Y, p=5, sv=True).fit(nburn=1000, nsave=1000, seed=0)
predict(res, horizon=8).quantiles(); girf(res, shock_var="UNC", size=1.0, horizon=16)""",
)


# ===================================================================== tvwnn
def _tv_demo():
    rng = np.random.default_rng(2)
    T = 360
    x = np.cumsum(rng.normal(0, .3, (T, 3)), 0) * 0.1 + rng.normal(0, 1, (T, 3))
    beta = np.linspace(0.2, 1.2, T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.3 * y[t - 1] + beta[t] * np.tanh(x[t - 1, 0]) + 0.4 * x[t - 1, 1] + rng.normal(0, .4)
    df = pd.DataFrame(x, columns=["x1", "x2", "x3"])
    df.insert(0, "y", y)
    return df, {"y": "y", "X": ["x1", "x2", "x3"]}, {}


def _tv_run(df, r, o):
    from tvwnn import TVNN, RidgeARX
    d = clean(df, [r["y"], *r["X"]])
    y = d[r["y"]].to_numpy(float)
    X = d[r["X"]].to_numpy(float) if r["X"] else None
    U, yt = lag_matrix(y, X, int(o["ky"]), int(o["kx"]))
    n = len(yt); cut = int(n * float(o["train"]))
    m = TVNN(d=int(o["d"]), hidden_sizes=(int(o["hidden"]),), n_outer=int(o["n_outer"]), random_state=0).fit(U[:cut], yt[:cut])
    fit_in = np.ravel(m.predict())
    preds = []
    for t in range(cut, n):
        preds.append(float(np.ravel(m.forecast(U[t], gap=t - cut + 1))[0]))
    preds = np.array(preds)
    rr = RidgeARX().fit(U[:cut], yt[:cut])
    rp = np.asarray(rr.forecast(U[cut:]) if hasattr(rr, "forecast") else rr.predict(U[cut:])).ravel()
    rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    out = LabResult()
    out.metrics = {"RMSE خارج العينة — TVNN": rm(yt[cut:], preds), "RMSE — Ridge ARX": rm(yt[cut:], rp), "نسبة TVNN/Ridge": rm(yt[cut:], preds) / rm(yt[cut:], rp)}
    fig = go.Figure([go.Scatter(y=yt, name="الفعلي", line=dict(color="#94a3b8")),
                     go.Scatter(y=fit_in, name="TVNN داخل العينة", line=dict(color="#0284c7")),
                     go.Scatter(x=np.arange(cut, n), y=preds, name="TVNN خارج العينة", line=dict(color="#e11d48", width=3)),
                     go.Scatter(x=np.arange(cut, n), y=rp, name="Ridge ARX", line=dict(color="#d97706", dash="dot"))])
    fig.add_vline(x=cut, line_dash="dash")
    fig.update_layout(height=400, title="TVNN: أوزان متغيرة زمنيًا فوق أساس عصبي")
    out.figs.append(("التنبؤ", fig))
    try:
        out.text = "\n".join(f"{k}: {v}" for k, v in m.summary().items())
    except Exception:
        pass
    out.notes.append("التنبؤ خارج العينة يستخدم الحالة الأخيرة المرشّحة مع gap = عدد الخطوات منذ نهاية التدريب، كما يوصي التوثيق.")
    return out


TVWNN = LabSpec(
    key="tvwnn", pip="tvwnn", name="tvwnn — شبكات عصبية بأوزان متغيرة زمنيًا", family="var",
    reference="Rudd, Bondell & Silver (2026), Augmenting Neural Networks With Time-Varying Weights, Journal of Forecasting 45(1)",
    roles=[Role("y", "المتغير المستهدف", "num"), Role("X", "المتنبئات", "nums", optional=True)],
    opts=[Opt("ky", "فجوات y", "int", 2, min=1, max=12), Opt("kx", "فجوات X", "int", 2, min=0, max=12),
          Opt("d", "أبعاد الأساس d", "int", 1, min=1, max=5), Opt("hidden", "وحدات الطبقة الخفية", "int", 32, min=4, max=128),
          Opt("n_outer", "دورات EM/Adam", "int", 15, min=3, max=60), Opt("train", "نسبة التدريب", "float", 0.8, min=0.5, max=0.95, step=0.05)],
    run=_tv_run, demo=_tv_demo, demo_label="محاكاة: أثر لا خطي معامله يتزايد عبر الزمن",
    theory=r"""
$$y_t=\boldsymbol\beta_t'\,\mathbf z_t+e_t,\qquad \mathbf z_t=f_w(\mathbf u_t),\qquad \boldsymbol\beta_t=\boldsymbol\beta_{t-1}+\boldsymbol\eta_t,\quad \boldsymbol\eta_t\sim N(0,\Omega)$$
شبكة أمامية تتعلم **أساسًا منخفض الأبعاد** $\mathbf z_t$ من الفجوات، والأوزان فوقه **ليست ثابتة** بل تتبع مسيرة عشوائية تُقدَّر بمرشح/مملّس كالمان داخل خوارزمية **EM** تتناوب مع خطوات **Adam** على الشبكة. النتيجة: لا خطية الشبكة + قدرة نماذج المعاملات المتغيرة (TVP) على تتبع **التغير البنيوي**.
""",
    guide="""
1. اختر y والمتنبئات؛ تُبنى الفجوات آليًا (ky لـ y و kx لـ X).
2. d = 1 أو 2 كافٍ غالبًا؛ دورات EM 15 للتجربة.
3. القسم الأخير للاختبار: نقارن TVNN بـ Ridge ARX خارج العينة.
""",
    code="""from tvwnn import TVNN
m = TVNN(d=1, hidden_sizes=(32,), n_outer=30).fit(U_train, y_train)
m.forecast(u_origin, gap=h)""",
)


# ============================================================== hybridmetrics
def _hm_demo():
    rng = np.random.default_rng(4)
    T = 300
    x = np.cumsum(rng.normal(0, 1, T))
    z = np.cumsum(rng.normal(0, 1, T))
    y = 1.0 + 0.6 * x - 0.3 * z + rng.normal(0, 1, T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + 0.5 * (1.0 + 0.6 * x[t] - 0.3 * z[t]) + rng.normal(0, .6)
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    return df, {"y": "y", "X": ["x", "z"]}, {}


def _hm_run(df, r, o):
    import hybridmetrics as hm
    d = clean(df, [r["y"], *r["X"]]).reset_index(drop=True)
    out = LabResult()
    model = o["model"]
    if model == "ARDL":
        m = hm.ARDL(p=int(o["p"]), q=int(o["q"])).fit(d[r["y"]], d[r["X"]])
        out.text = capture(m.summary)
        try:
            out.tables["اختبار الحدود"] = to_df({k: [v] for k, v in m.bounds_test().items() if np.ndim(v) == 0})
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"تعذّر اختبار الحدود: {e}")
        try:
            out.tables["المضاعفات طويلة الأجل"] = to_df(m.long_run_multipliers())
        except Exception:
            pass
    elif model == "Neural Granger causality":
        m = hm.NeuralGrangerCausality(method=o["ngc"], lag=int(o["p"]), epochs=int(o["epochs"])).fit(d)
        C = m.causality_matrix()
        out.tables["مصفوفة السببية"] = C
        fig = go.Figure(go.Heatmap(z=C.values, x=[str(c) for c in C.columns], y=[str(i) for i in C.index], colorscale="Reds"))
        fig.update_layout(height=380, title="سببية غرانجر العصبية")
        out.figs.append(("NGC", fig))
    elif model in ("DeepTVAR", "DeepVARwT"):
        cls = hm.DeepTVAR if model == "DeepTVAR" else hm.DeepVARwT
        m = cls(p=int(o["p"]), epochs=int(o["epochs"])).fit(d)
        fc = m.forecast(h=int(o["h"]))
        out.tables["التنبؤ"] = fc
        out.figs.append(("التنبؤ", _lines(fc.values, list(fc.columns), f"{model}: التنبؤ")))
        try:
            lh = np.asarray(m.loss_history())
            fig = go.Figure(go.Scatter(y=lh, mode="lines", line=dict(color="#7c3aed")))
            fig.update_layout(height=300, title="منحنى خسارة التدريب")
            out.figs.append(("الخسارة", fig))
        except Exception:
            pass
    else:
        out.tables["بطارية جذور الوحدة"] = hm.run_unit_root_battery(d)
    if o["unit_root"] and model != "بطارية جذور الوحدة":
        try:
            out.tables["بطارية جذور الوحدة (ADF, KPSS, PP, ZA)"] = hm.run_unit_root_battery(d)
        except Exception:
            pass
    return out


HYBRIDMETRICS = LabSpec(
    key="hybridmetrics", pip="hybridmetrics", name="hybridmetrics — الهجين القياسي × التعلم العميق", family="var",
    reference="Chalkiadakis, Peters & Ames (2023), Digital Finance; Zhang (2003)",
    roles=[Role("y", "المتغير التابع / الأول", "num"), Role("X", "المفسرات / بقية السلاسل", "nums")],
    opts=[Opt("model", "النموذج", "select", "ARDL", ["ARDL", "Neural Granger causality", "DeepTVAR", "DeepVARwT", "بطارية جذور الوحدة"]),
          Opt("p", "فجوات y / الفجوة", "int", 2, min=1, max=10), Opt("q", "فجوات X (ARDL)", "int", 2, min=0, max=10),
          Opt("ngc", "طريقة NGC", "select", "cmlp", ["cmlp", "clstm"]), Opt("epochs", "الحقب", "int", 200, min=20, max=2000, step=20),
          Opt("h", "أفق التنبؤ", "int", 10, min=1, max=36), Opt("unit_root", "أضف بطارية جذور الوحدة", "bool", True)],
    run=_hm_run, demo=_hm_demo, demo_label="محاكاة: y متكامل مشتركًا مع x و z",
    theory=r"""
مكتبة موحّدة للهجين القياسي–العميق مع بطارية اختبارات قبلية وبعدية:
- **ARDL** واختبار الحدود والمضاعفات طويلة الأجل — المكوّن القياسي.
- **ARDL-MIDAS-Transformer** (Chalkiadakis, Peters & Ames, 2023): تقدير خماسي المراحل بـ 2SLS/IV يدمج ترددات مختلطة ومحوّلًا، و**ARDL-MIDAS-DNN** على مبدأ الهجين الجمعي لتشانغ (2003).
- **DeepTVAR / DeepVARwT**: VAR بمعاملات متغيرة زمنيًا تولّدها شبكة متكررة (مع/دون اتجاه).
- **سببية غرانجر العصبية** (cMLP / cLSTM بعقوبة المجموعات).
$$y_t=\underbrace{\alpha+\sum_i\phi_iy_{t-i}+\sum_j\beta_j'x_{t-j}}_{\text{ARDL}}+\underbrace{g_\omega(\cdot)}_{\text{DNN / Transformer}}+\varepsilon_t$$
""",
    guide="""
1. **ARDL**: y التابع و X المفسرات؛ يعرض الملخص واختبار الحدود والمضاعفات طويلة الأجل.
2. **Neural Granger**: كل المتغيرات معًا؛ المصفوفة تظهر قوة السببية.
3. **DeepTVAR/DeepVARwT**: تنبؤ متعدد المتغيرات بمعاملات متغيرة زمنيًا.
4. بطارية جذور الوحدة (ADF، KPSS، PP، Zivot–Andrews) قبل أي تقدير.
""",
    code="""import hybridmetrics as hm
m = hm.ARDL(p=2, q=2).fit(y, X); m.summary(); m.bounds_test()
hm.NeuralGrangerCausality(lag=5).fit(df).causality_matrix()""",
)


# =================================================================== bartardl
def _bart_demo():
    from bartardl import load_macro
    df = load_macro(source="auto").reset_index(drop=True)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    tgt = "GDP" if "GDP" in num_cols else num_cols[0]
    return df[num_cols], {"target": tgt, "others": [c for c in num_cols if c != tgt]}, {}


def _bart_run(df, r, o):
    from bartardl import ARDLBART, BART, ElasticNet, Lasso
    cols = [r["target"], *r["others"]]
    d = clean(df, cols).reset_index(drop=True)
    est = {"BART": BART, "Lasso": Lasso, "ElasticNet": ElasticNet}[o["estimator"]]
    kw = {"n_trees": int(o["n_trees"]), "n_draws": int(o["n_draws"]), "n_burn": int(o["n_draws"]), "random_state": 0} if o["estimator"] == "BART" else {}
    res = ARDLBART(n_lags=int(o["lags"]), estimator=est, estimator_kwargs=kw).fit(d, r["target"])
    out = LabResult()
    imp = res.importance_frame()
    out.tables["أهمية المتغيرات المتأخرة"] = imp
    rmse = float(np.sqrt(np.mean(np.asarray(res.resid) ** 2)))
    out.metrics = {"RMSE داخل العينة": rmse, "عدد الميزات": len(res.feature_names)}
    fig = go.Figure([go.Scatter(y=res.y_train, name="الفعلي", line=dict(color="#94a3b8")),
                     go.Scatter(y=res.fitted, name=f"ARDL-{o['estimator']}", line=dict(color="#0284c7", width=2))])
    fig.update_layout(height=360, title="الفعلي مقابل الموفَّق")
    out.figs.append(("الملاءمة", fig))
    try:
        top = res.top_features(k=min(12, len(res.feature_names)))
        s = top.set_index(top.columns[0])[top.columns[1]] if top.shape[1] >= 2 else top.iloc[:, 0]
        s = s.astype(float).sort_values()
        f2 = go.Figure(go.Bar(x=s.values, y=[str(i) for i in s.index], orientation="h", marker_color="#0284c7"))
        f2.update_layout(height=420, title="أهم المتغيرات المتأخرة")
        out.figs.append(("الأهمية", f2))
    except Exception:
        pass
    return out


BARTARDL = LabSpec(
    key="bartardl", pip="bartardl", name="bartardl — ARDL لا معلمي بأشجار BART", family="var", runtime="ثوانٍ معدودة",
    reference="Mahdavi, Ehsani, Ahelegbey & Mohammadpour (2024), Measuring Causal Effect with ARDL-BART, Applied Mathematics 15",
    roles=[Role("target", "المتغير المستهدف", "num"), Role("others", "المتغيرات الأخرى", "nums")],
    opts=[Opt("estimator", "المقدّر", "select", "BART", ["BART", "Lasso", "ElasticNet"]), Opt("lags", "عدد الفجوات", "int", 2, min=1, max=8),
          Opt("n_trees", "عدد الأشجار (BART)", "int", 100, min=20, max=400, step=20), Opt("n_draws", "سحب MCMC (والإحماء)", "int", 300, min=100, max=2000, step=100)],
    run=_bart_run, demo=_bart_demo, demo_label="لوحة الاقتصاد الكلي الأمريكي بثمانية متغيرات (مضمّنة أو اصطناعية)",
    theory=r"""
$$y_t=f\big(y_{t-1},\dots,y_{t-p},\ x_{t-1},\dots,x_{t-p}\big)+\varepsilon_t,\qquad f=\sum_{m=1}^{M}g(\cdot;T_m,\mu_m)$$
ARDL يُقدَّر متوسطه الشرطي **لا معلميًا** بمجموع أشجار صغيرة بقبليات Chipman–George–McCulloch (2010): قبْلي الشجرة $\alpha(1+d)^{-\beta}$، أوراق مترافقة طبيعية، وقبْلي خطأ معاير بـ $(\nu,q)$. يلتقط التفاعلات واللاخطية، ونسب **تضمين المتغيرات** في الأشجار تعطي مقياسًا للأهمية. يقارَن بلاسو والشبكة المرنة وشبكة بايزية (spike-and-slab). المعاين مكتوب من الصفر بـ NumPy.
""",
    guide="""
1. اختر المتغير المستهدف والمتغيرات الأخرى (بعد تحويلات الاستقرارية).
2. اختر BART أو لاسو/الشبكة المرنة للمقارنة.
3. 100 شجرة و 300 سحبًا للتجربة؛ الورقة: 200 شجرة.
4. اقرأ أهمية المتغيرات المتأخرة والملاءمة.
""",
    code="""from bartardl import ARDLBART, BART, load_macro
res = ARDLBART(n_lags=2, estimator=BART, estimator_kwargs={"n_trees": 200}).fit(load_macro(), "GDP")
res.importance_frame()""",
)

SPECS = [SAMOVAR, GPVAR, TVWNN, HYBRIDMETRICS, BARTARDL]
