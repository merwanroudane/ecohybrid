"""Lab adapters — volatility & risk: garchai, hybridecon, harlstmgarch, peltwtcn."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.lab.base import LabResult, LabSpec, Opt, Role, capture, clean, to_df


def _garch_sim(T=1200, seed=0):
    rng = np.random.default_rng(seed)
    r, h = np.zeros(T), np.zeros(T)
    h[0] = 1.0
    for t in range(1, T):
        h[t] = 0.05 + 0.08 * r[t - 1] ** 2 + 0.07 * (r[t - 1] < 0) * r[t - 1] ** 2 + 0.85 * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.standard_t(6) / np.sqrt(1.5)
    return r


def _ret_demo():
    try:
        import garchai
        s = garchai.datasets.load_returns("sp500").dropna()
        s = s.iloc[-1500:]
        return pd.DataFrame({"date": s.index, "sp500": s.values}), {"r": "sp500"}, {}
    except Exception:
        return pd.DataFrame({"returns": _garch_sim()}), {"r": "returns"}, {}


# ==================================================================== garchai
GA_MODELS = ["GARCH", "GJRGARCH", "EGARCH", "GARCHNN", "StructuralGARCHLSTM", "NeuralGARCH", "SigmaLSTM", "ANNGARCH", "RECH"]


def _ga_run(df, r, o):
    import garchai
    y = clean(df, [r["r"]])[r["r"]].to_numpy(float)
    cfg = garchai.ModelConfig(distribution=o["dist"], seed=0)
    tc = garchai.TrainConfig(max_epochs=int(o["epochs"]), lr=float(o["lr"]))
    out = LabResult()
    rows, vols = [], {}
    for name in o["models"]:
        try:
            m = getattr(garchai, name)(config=cfg)
            fr = m.fit(y, train=tc)
            vols[name] = np.asarray(fr.conditional_volatility).ravel()
            k = fr.n_params
            ll = float(fr.loglikelihood)
            n = fr.n_obs
            try:
                fc_vol = float(np.ravel(m.forecast(horizon=int(o["h"])).volatility)[-1])
            except Exception as e:  # noqa: BLE001
                fc_vol = np.nan
                out.notes.append(f"{name}: نجح التقدير وتعذّر التنبؤ ({type(e).__name__}: {e}).")
            rows.append({"النموذج": name, "LogLik": ll, "AIC": -2 * ll + 2 * k, "BIC": -2 * ll + k * np.log(n), "عدد المعلمات": k,
                         f"تقلب h={o['h']}": fc_vol})
            if name == o["models"][0]:
                out.text = capture(m.summary)
                try:
                    var = np.asarray(m.var(alpha=float(o["alpha"]))).ravel()
                    out.metrics[f"VaR {int(o['alpha'] * 100)}% (آخر مشاهدة)"] = float(var[-1])
                except Exception:
                    pass
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"تعذّر تقدير {name}: {type(e).__name__}: {e}")
    if not rows:
        raise RuntimeError("لم ينجح أي نموذج.")
    out.tables["مقارنة النماذج"] = pd.DataFrame(rows).sort_values("AIC")
    fig = go.Figure([go.Scatter(y=np.abs(y), name="|r|", line=dict(color="#cbd5e1"))] +
                    [go.Scatter(y=v, name=k, mode="lines") for k, v in vols.items()])
    fig.update_layout(height=400, title="التقلب الشرطي المقدّر")
    out.figs.append(("التقلب الشرطي", fig))
    return out


GARCHAI = LabSpec(
    key="garchai", pip="garchai", name="garchai — نماذج GARCH-AI مقيّدة قياسيًا وقابلة للتفاضل", family="vol", runtime="ثوانٍ لـ GARCH/GJR، ونحو دقيقة لكل نموذج عصبي",
    reference="24 ورقة GARCH-AI منشورة (لكل نموذج سجل مصدر Provenance)",
    roles=[Role("r", "العوائد (نسبة مئوية)", "num")],
    opts=[Opt("models", "النماذج", "multi", ["GARCH", "GJRGARCH"], GA_MODELS, help="النماذج العصبية مثل GARCHNN تستغرق نحو دقيقة لكل منها"), Opt("dist", "التوزيع", "select", "student-t", ["normal", "student-t", "skew-t", "ged"]),
          Opt("epochs", "الحقب القصوى", "int", 150, min=20, max=2000, step=10), Opt("lr", "معدل التعلم", "float", 0.01, min=0.0001, max=0.1, step=0.001),
          Opt("h", "أفق التنبؤ", "int", 5, min=1, max=60), Opt("alpha", "مستوى VaR", "float", 0.05, min=0.01, max=0.1, step=0.01)],
    run=_ga_run, demo=_ret_demo, demo_label="عوائد S&P 500 اليومية (آخر 1500 مشاهدة، مضمّنة في garchai)",
    theory=r"""
$$r_t=\mu_t+\sigma_t z_t,\qquad \sigma_t^2=\underbrace{\omega+\alpha\varepsilon_{t-1}^2+\gamma\,\mathbb 1\{\varepsilon_{t-1}<0\}\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2}_{\text{econometric core}}+\underbrace{g_\theta(\varepsilon_{t-1},\sigma_{t-1}^2,h_{t-1})}_{\text{neural component}}$$
مكتبة PyTorch واحدة لـ **24 نموذج GARCH-AI منشور**: GARCH/GJR/EGARCH الكلاسيكية (مطابقة لحزمة arch بدقة 4.4e-15)، GARCH-NN، GARCH-LSTM البنيوي، Neural GARCH، Sigma-LSTM، ANN-GARCH، RECH، Realized/Deep RGARCH، Neural BEKK... كل مسار تباين **قابل للتفاضل**، والقيود القياسية (الإيجابية، الاستقرارية) مفروضة داخل المعمارية، وكل نموذج **يصرّح حين لا يتفوق على خط أساسه**.
""",
    guide="""
1. أدخل سلسلة **عوائد** (بالنسبة المئوية أفضل).
2. اختر عدة نماذج للمقارنة: الجدول يرتبها بمعيار AIC.
3. التوزيع student-t مناسب للذيول الثقيلة.
4. النتائج: التقلب الشرطي لكل نموذج، تنبؤ التقلب، وقيمة المخاطرة VaR.
""",
    code="""import garchai
cfg = garchai.ModelConfig(distribution="student-t")
m = garchai.GJRGARCH(config=cfg); fr = m.fit(returns)
m.forecast(horizon=5).volatility; m.var(alpha=0.05); print(m.summary())""",
)


# ================================================================== hybridecon
HE_MODELS = ["GARCH", "GARCHLSTM", "GARCHGRU", "GARCHBiLSTM", "ARIMALSTM", "DeepQuantileVaR"]


def _he_run(df, r, o):
    import hybridecon.hybrids as hh
    y = clean(df, [r["r"]])[r["r"]].to_numpy(float)
    out = LabResult()
    cfg = hh.TrainConfig(epochs=int(o["epochs"]))
    rows, fcs = [], {}
    for name in o["models"]:
        try:
            if name == "GARCH":
                m = hh.GARCH(vol=o["vol"])
            elif name == "DeepQuantileVaR":
                m = hh.DeepQuantileVaR(alpha=float(o["alpha"]), lookback=int(o["lookback"]), train_cfg=cfg)
            else:
                m = getattr(hh, name)(lookback=int(o["lookback"]), hidden=int(o["hidden"]), train_cfg=cfg)
            m.fit(y)
            row = {"النموذج": getattr(m, "name", name)}
            row.update({k: v for k, v in (getattr(m, "metrics_", {}) or {}).items() if np.ndim(v) == 0})
            rows.append(row)
            try:
                fcs[name] = np.ravel(m.forecast(horizon=int(o["h"])))
            except Exception:
                pass
            if not out.text:
                out.text = capture(m.summary)
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"تعذّر {name}: {type(e).__name__}: {e}")
    if not rows:
        raise RuntimeError("لم ينجح أي نموذج.")
    out.tables["مقاييس الأداء (عينة الاختبار)"] = pd.DataFrame(rows)
    if fcs:
        fig = go.Figure([go.Scatter(y=v, name=k, mode="lines+markers") for k, v in fcs.items()])
        fig.update_layout(height=360, title=f"التنبؤ لـ {o['h']} خطوات")
        out.figs.append(("التنبؤ", fig))
    return out


HYBRIDECON = LabSpec(
    key="hybridecon", pip="hybridecon", name="hybridecon — هجين التعلم العميق والقياس للتقلب والمخاطر", family="vol", runtime="ثوانٍ معدودة",
    reference="Engle, Ghysels & Sohn (2013) GARCH-MIDAS; Corsi (2009) HAR; Engle & Kroner (1995) BEKK؛ وأدبيات الهجين 2023–2026",
    roles=[Role("r", "العوائد", "num")],
    opts=[Opt("models", "النماذج", "multi", ["GARCH", "GARCHLSTM"], HE_MODELS), Opt("vol", "نوع GARCH", "select", "GARCH", ["GARCH", "EGARCH", "FIGARCH", "HARCH"]),
          Opt("lookback", "النافذة (lookback)", "int", 20, min=5, max=120), Opt("hidden", "الوحدات الخفية", "int", 32, min=8, max=256),
          Opt("epochs", "الحقب", "int", 20, min=5, max=300, step=5), Opt("h", "أفق التنبؤ", "int", 5, min=1, max=60),
          Opt("alpha", "مستوى VaR (DeepQuantileVaR)", "float", 0.05, min=0.01, max=0.1, step=0.01)],
    run=_he_run, demo=_ret_demo, demo_label="عوائد S&P 500 اليومية (من garchai)",
    theory=r"""
واجهة موحّدة لكل نماذج الهجين المسحية في أدبيات التقلب والمخاطر والتنبؤ الكلي (2023–2026): كل نموذج يتبع السلسلة
`fit → forecast → diagnose → plot → report`.
- **المكوّن القياسي**: GARCH بأنواعه، GARCH-MIDAS، HAR-RV، GAS، BEKK القطري، ARIMA/ARFIMA، VAR.
- **الهجين**: GARCH-LSTM/GRU/BiLSTM/CNN-BiLSTM/Attention، ARIMA-LSTM، HAR-LSTM-GARCH، MIDAS-LSTM، KAN-GARCH-MIDAS، Deep Quantile VaR (خسارة pinball)، LSTM-GARCH الواعي بالأنظمة (HMM).
- **التقييم**: مقاييس على عينة اختبار ومجموعة ثقة النماذج (MCS).
$$\hat\sigma^2_{t+1}=f_{LSTM}\big(r_{t-L+1:t},\ \hat\sigma^{2,GARCH}_{t-L+1:t}\big)$$
""",
    guide="""
1. أدخل العوائد؛ اختر GARCH كمرجع وهجينًا أو أكثر.
2. lookback 20 والحقب 20 للتجربة السريعة.
3. اقرأ جدول المقاييس على عينة الاختبار ورسم التنبؤ.
""",
    code="""import hybridecon.hybrids as hh
m = hh.GARCHLSTM(lookback=20, hidden=64).fit(returns); m.forecast(horizon=5); print(m.summary())
hh.run_benchmark(returns, models=[hh.GARCHLSTM(), hh.GARCHGRU()])""",
)


# ================================================================ harlstmgarch
def _hl_demo():
    df, roles, _ = _ret_demo()
    r = df[roles["r"]].to_numpy(float)
    rv = pd.Series(r).abs().rolling(5).mean().dropna().reset_index(drop=True) + 1e-3
    return pd.DataFrame({"RV": rv}), {"rv": "RV"}, {}


def _hl_run(df, r, o):
    import harlstmgarch as hl
    rv = clean(df, [r["rv"]])[r["rv"]].reset_index(drop=True)
    H = hl.har_components(rv).dropna()
    tr, va, te = hl.chronological_split(H, train=float(o["train"]), val=0.15)
    harm = hl.HARModel().fit(tr)
    m = hl.HARLSTMGARCH(lstm_kwargs={"lookback": int(o["lookback"]), "units": int(o["units"]), "epochs": int(o["epochs"]), "patience": 10}).fit(tr, va)
    fc = m.forecast(te)
    yt = te["RV"].to_numpy()
    pt, ph = np.asarray(fc.point).ravel(), np.asarray(fc.har).ravel()
    n = min(len(yt), len(pt))
    yt, pt, ph = yt[-n:], pt[-n:], ph[-n:]
    rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    out = LabResult()
    out.metrics = {"RMSE — HAR-LSTM-GARCH": rm(yt, pt), "RMSE — HAR": rm(yt, ph), "التحسن %": 100 * (1 - rm(yt, pt) / rm(yt, ph))}
    try:
        out.tables["معاملات HAR"] = to_df(harm.coefficients())
    except Exception:
        pass
    lo, up = np.asarray(fc.lower).ravel()[-n:], np.asarray(fc.upper).ravel()[-n:]
    fig = go.Figure([go.Scatter(y=up, line=dict(width=0), showlegend=False), go.Scatter(y=lo, fill="tonexty", fillcolor="rgba(225,29,72,.15)", line=dict(width=0), name="فترة 95%"),
                     go.Scatter(y=yt, name="RV الفعلي", line=dict(color="#475569")), go.Scatter(y=ph, name="HAR", line=dict(color="#d97706", dash="dot")),
                     go.Scatter(y=pt, name="HAR-LSTM-GARCH", line=dict(color="#e11d48", width=2))])
    fig.update_layout(height=400, title="تنبؤ التقلب المحقق على عينة الاختبار")
    out.figs.append(("التنبؤ", fig))
    return out


HARLSTMGARCH = LabSpec(
    key="harlstmgarch", pip="harlstmgarch", name="harlstmgarch — HAR-LSTM-GARCH ثلاثي المراحل", family="vol", runtime="10–20 ثانية",
    reference="Ben Romdhane & Boubaker (2026), A Hybrid HAR-LSTM-GARCH Model for Forecasting Volatility in Energy Markets, JRFM 19",
    roles=[Role("rv", "التقلب المحقق RV", "num")],
    opts=[Opt("lookback", "نافذة LSTM", "int", 20, min=5, max=60), Opt("units", "وحدات LSTM", "int", 32, min=8, max=128),
          Opt("epochs", "الحقب", "int", 40, min=5, max=300, step=5), Opt("train", "نسبة التدريب", "float", 0.7, min=0.5, max=0.85, step=0.05)],
    run=_hl_run, demo=_hl_demo, demo_label="تقلب محقق تقريبي من عوائد S&P 500 (متوسط |r| لخمسة أيام)",
    theory=r"""
**ثلاث مراحل**:
1. **HAR** (Corsi, 2009): مرشح خطي للديناميكيات المستمرة متعددة المقاييس: $RV_t=\beta_0+\beta_dRV_{t-1}+\beta_wRV^{(w)}_{t-1}+\beta_mRV^{(m)}_{t-1}+e_t$.
2. **LSTM على البواقي** $e_t$ (مضبوط بالتحسين البايزي) لالتقاط اللاخطية المتبقية.
3. **GARCH على البواقي المعيارية** لتجمّع التقلب وبناء فترات تنبؤ.
$$\widehat{RV}_{t+1}=\widehat{RV}^{HAR}_{t+1}+\hat e^{LSTM}_{t+1},\qquad \text{PI}=\widehat{RV}_{t+1}\pm z\,\hat\sigma^{GARCH}_{t+1}$$
مع معايير مقارنة اقتصادية قياسية وتعلم آلي واختبار **مجموعة ثقة النماذج**.
""",
    guide="""
1. أدخل سلسلة **التقلب المحقق** (يومية).
2. تُبنى مكونات HAR (يومي، أسبوعي، شهري) والتقسيم الزمني آليًا.
3. الحقب 40 للتجربة؛ زدها وفعّل الضبط البايزي في الاستخدام البحثي.
4. قارن RMSE الهجين بـ HAR ورسم الفترات.
""",
    code="""import harlstmgarch as hl
H = hl.har_components(rv); tr, va, te = hl.chronological_split(H)
m = hl.HARLSTMGARCH().fit(tr, va); fc = m.forecast(te)  # fc.point, fc.lower, fc.upper""",
)


# =================================================================== peltwtcn
def _pw_demo():
    from peltwtcn import load_paper_dataset
    df = load_paper_dataset().reset_index()
    date_col = df.columns[0]
    num_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    price = "Carbon_Price" if "Carbon_Price" in num_cols else num_cols[0]
    return df, {"date": date_col, "price": price, "exog": [c for c in num_cols if c != price][:4]}, {}


def _pw_run(df, r, o):
    import pywt
    from peltwtcn import TrainConfig, run_experiment
    cols = [r["price"], *(r.get("exog") or [])]
    d = df[[r["date"], *cols]].dropna().copy()
    d[r["date"]] = pd.to_datetime(d[r["date"]])
    d = d.set_index(r["date"]).sort_index()
    models = {f"PELT-WT-{k.upper()}": {"detector": o["detector"], "model": k, "multivariate": bool(r.get("exog")),
                                       "window": int(o["window"]), "train": TrainConfig(epochs=int(o["epochs"]), patience=5)}
              for k in o["models"]}
    # pandas >= 3 (copy-on-write) hands pywt read-only views; give it writable copies
    orig = pywt.wavedec
    pywt.wavedec = lambda x, *a, **k: orig(np.array(x, dtype=float), *a, **k)
    try:
        res = run_experiment(d, price_col=r["price"], models=models, mode=o["mode"], include_random_walk=True, verbose=False)
    finally:
        pywt.wavedec = orig
    out = LabResult()
    out.tables["جدول المقارنة (عينة الاختبار)"] = res.table
    fig = go.Figure([go.Scatter(x=res.index, y=res.actual, name="الفعلي", line=dict(color="#475569"))] +
                    [go.Scatter(x=res.index, y=v, name=k, mode="lines") for k, v in res.predictions.items()])
    fig.update_layout(height=420, title="تنبؤ السعر على عينة الاختبار")
    out.figs.append(("التنبؤ", fig))
    try:
        out.text = capture(next(iter(res.pipelines.values())).summary)
    except Exception:
        pass
    try:
        rw = float(res.table.loc["Random walk", "RMSE"])
        worse = [m for m in res.table.index if m != "Random walk" and float(res.table.loc[m, "RMSE"]) > rw]
        if worse:
            out.notes.append(f"تنبيه: {', '.join(worse)} أسوأ من السير العشوائي (RMSE). جرّب حقبًا أكثر أو نافذة مختلفة قبل تفسير النتيجة.")
    except Exception:
        pass
    out.notes.append("الوضع causal يحسب التنقية الموجية والانكسارات دون معلومات مستقبلية؛ وضع paper يطابق الورقة.")
    return out


PELTWTCN = LabSpec(
    key="peltwtcn", pip="peltwtcn", name="peltwtcn — PELT + موجات + تعلم عميق لتنبؤ الأسعار", family="vol", runtime="دقيقة ونصف تقريبًا",
    reference="Ren et al. (2025), A Hybrid Deep Learning based Carbon Price Forecasting Framework with Structural Breakpoints Detection and Signal Denoising — arXiv 2511.04988",
    roles=[Role("date", "عمود التاريخ", "col"), Role("price", "السعر المستهدف", "num"), Role("exog", "متغيرات خارجية (اختياري)", "nums", optional=True)],
    opts=[Opt("models", "النماذج العميقة", "multi", ["tcn"], ["lstm", "gru", "tcn"]), Opt("detector", "كاشف الانكسارات", "select", "pelt", ["pelt", "bp_icss"]),
          Opt("window", "نافذة الإدخال", "int", 30, min=5, max=120), Opt("epochs", "الحقب", "int", 10, min=2, max=200, step=2),
          Opt("mode", "الوضع", "select", "causal", ["causal", "paper"])],
    run=_pw_run, demo=_pw_demo, demo_label="بيانات سوق الكربون الأوروبي EU ETS الحقيقية (6113 مشاهدة يومية 2007–2024)",
    theory=r"""
**خط أنابيب ثلاثي**: (1) **كشف الانكسارات البنيوية** بخوارزمية PELT (Killick et al., 2012) أو Bai–Perron/ICSS، وتحويلها إلى متغيرات أنظمة؛ (2) **تنقية موجية** للسعر (عتبات شاملة ناعمة على معاملات التفصيل)؛ (3) **تعلم عميق** (LSTM/GRU/TCN) على نوافذ منزلقة مع المتغيرات الخارجية ومتغيرات الأنظمة.
$$\text{price}\ \xrightarrow{\text{PELT}}\ \text{regimes}\ \ \oplus\ \ \xrightarrow{\text{wavelet denoise}}\ \tilde p_t\ \xrightarrow{\text{LSTM/GRU/TCN}}\ \hat p_{t+1}$$
المكتبة تضيف **تدقيق التكرار**: وضع causal يمنع تسرب المستقبل عبر المرشحات، ومقارنة دائمة بالمسيرة العشوائية.
""",
    guide="""
1. حدد عمود **التاريخ** و**السعر** والمتغيرات الخارجية إن وُجدت.
2. ابدأ بـ TCN و 10 حقب في الوضع **causal** (الصادق زمنيًا).
3. الجدول يقارن كل نموذج بالمسيرة العشوائية على عينة الاختبار.
""",
    code="""from peltwtcn import load_paper_dataset, run_experiment
res = run_experiment(load_paper_dataset(), price_col="Carbon_Price", mode="causal")
res.table""",
)

SPECS = [GARCHAI, HYBRIDECON, HARLSTMGARCH, PELTWTCN]
