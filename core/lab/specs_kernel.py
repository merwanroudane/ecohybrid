"""Lab adapters — kernel, wavelet & quantile hybrids: wkrls, qqkrls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.lab.base import LabResult, LabSpec, Opt, Role, capture, clean, to_df

PARULA = [[0.0, "#352a87"], [0.15, "#0363e1"], [0.3, "#1485d4"], [0.45, "#06a7c6"], [0.6, "#38b99e"], [0.75, "#92bf73"],
          [0.85, "#d9ba56"], [0.93, "#fcce2e"], [1.0, "#f9fb0e"]]


# ====================================================================== wkrls
def _wk_demo():
    rng = np.random.default_rng(8)
    T = 256
    t = np.arange(T)
    x1 = np.sin(2 * np.pi * t / 64) + 0.5 * rng.normal(size=T)
    x2 = np.sin(2 * np.pi * t / 6) + 0.5 * rng.normal(size=T)
    x3 = rng.normal(size=T)
    y = 1.2 * np.tanh(x1) - 0.8 * x2 + 0.3 * x3 ** 2 + 0.4 * rng.normal(size=T)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3}), {"y": "y", "X": ["x1", "x2", "x3"]}, {}


def _wk_run(df, r, o):
    from wkrls import wkrls
    d = clean(df, [r["y"], *r["X"]]).reset_index(drop=True)
    if len(d) > 700:
        d = d.iloc[-700:]
    res = wkrls(d[r["y"]].to_numpy(float), d[r["X"]], wavelet=o["wavelet"], J=int(o["J"]), bands=bool(o["bands"]),
                freq=o["freq"], dep_name=r["y"], var_names=list(r["X"]), verbose=False)
    out = LabResult(text=capture(res.summary))
    out.tables["متوسط الآثار الحدية حسب النطاق (AME)"] = to_df(res.ame)
    out.tables["ملخص الإشارات"] = to_df(res.sign_summary())
    for v in list(r["X"])[:3]:
        try:
            M = res.to_matrix(v)
            fig = go.Figure(go.Heatmap(z=M.values, x=[str(c) for c in M.columns], y=[str(i) for i in M.index], colorscale=PARULA))
            fig.update_layout(height=380, title=f"الأثر الحدي لـ {v}: النطاق الترددي × المشاهدات (مرتبة حسب y)", xaxis_title="المشاهدة", yaxis_title="النطاق")
            out.figs.append((f"خريطة {v}", fig))
        except Exception as e:  # noqa: BLE001
            out.notes.append(f"تعذّرت خريطة {v}: {e}")
    if len(df) > 700:
        out.notes.append("استُخدمت آخر 700 مشاهدة لأن KRLS تكلفته O(n³) لكل نطاق.")
    return out


WKRLS = LabSpec(
    key="wkrls", pip="wkrls", name="WKRLS — الآثار الحدية غير الخطية حسب التردد", family="kernel", runtime="ثوانٍ معدودة",
    reference="تركيب المؤلف: MODWT (Percival & Walden, 2000) × KRLS (Hainmueller & Hazlett, 2014)",
    roles=[Role("y", "المتغير التابع", "num"), Role("X", "المفسرات", "nums")],
    opts=[Opt("wavelet", "الموجة", "select", "la8", ["la8", "haar", "d4", "la16"]), Opt("J", "مستويات التفكيك J", "int", 4, min=2, max=7),
          Opt("bands", "تجميع في نطاقات (قصير/متوسط/طويل)", "bool", True), Opt("freq", "تواتر البيانات", "select", "quarterly", ["monthly", "quarterly", "annual", "daily"])],
    run=_wk_run, demo=_wk_demo, demo_label="محاكاة: أثر x1 في التردد المنخفض و x2 في التردد المرتفع",
    theory=r"""
الانحدار العادي يعطي **رقمًا واحدًا** لكل متغير، وهو متوسط لشيئين غير ثابتين: **الأفق الزمني** و**الموقع في التوزيع**. WKRLS يحلهما معًا:
1. **MODWT** يفكك كل سلسلة إلى مكونات $x^{(j)}$ تنتمي لآفاق زمنية مختلفة.
2. **KRLS** يقدّر متوسطًا شرطيًا مرنًا لكل نطاق ويعطي **أثرًا حديًا لكل مشاهدة**:
$$\hat f_j(x)=\sum_i c_i\,k(x,x_i),\qquad k(x,x')=e^{-\|x-x'\|^2/\sigma^2},\qquad \frac{\partial\hat f_j}{\partial x_k}\Big|_{x_i}=-\frac{2}{\sigma^2}\sum_l c_l\,k(x_i,x_l)(x_{ik}-x_{lk})$$
النتيجة: خريطة حرارية **نطاق ترددي × مشاهدة** للأثر الحدي دون افتراض الخطية أو الجمعية.
""",
    guide="""
1. اختر y والمفسرات (سلاسل مستقرة يفضّل أن يكون طولها قوة للعدد 2 تقريبًا).
2. J = 4–5 مستويات؛ la8 موجة شائعة.
3. اقرأ AME حسب النطاق، وملخص الإشارات، وخرائط الأثر الحدي.
""",
    code="""from wkrls import wkrls
res = wkrls(y, X, wavelet="la8", J=5); res.ame; res.sign_summary(); res.heatmap("x1")""",
)


# ===================================================================== qqkrls
def _qq_demo():
    rng = np.random.default_rng(9)
    n = 300
    x = rng.normal(size=n)
    y = 0.5 * x + 0.6 * x * (x > 0.5) + (0.3 + 0.4 * np.abs(x)) * rng.standard_t(5, n)
    return pd.DataFrame({"y": y, "x": x}), {"y": "y", "x": "x"}, {}


def _qq_run(df, r, o):
    from qqkrls import qqkrls
    d = clean(df, [r["y"], r["x"]])
    q = np.round(np.arange(float(o["q_min"]), float(o["q_max"]) + 1e-9, float(o["q_step"])), 3)
    res = qqkrls(d[r["y"]].to_numpy(float), d[r["x"]].to_numpy(float), y_quantiles=q, x_quantiles=q, n_boot=int(o["n_boot"]), verbose=False)
    out = LabResult(text=capture(res.summary))
    C = np.asarray(res.to_matrix("coefficient"), dtype=float)
    out.tables["المعاملات (صفوف: كمّيات Y، أعمدة: كمّيات X)"] = pd.DataFrame(C, index=res.y_quantiles, columns=res.x_quantiles)
    try:
        out.tables["الدلالة 5%"] = pd.DataFrame(res.significance_matrix(0.05), index=res.y_quantiles, columns=res.x_quantiles)
    except Exception:
        pass
    fig = go.Figure(go.Heatmap(z=C, x=list(res.x_quantiles), y=list(res.y_quantiles), colorscale=PARULA, colorbar=dict(title="β")))
    fig.update_layout(height=420, title="QQKRLS: الأثر عند كمّي X (أفقي) على كمّي Y (عمودي)", xaxis_title="τ(X)", yaxis_title="θ(Y)")
    out.figs.append(("الخريطة الحرارية", fig))
    f3 = go.Figure(go.Surface(z=C, x=list(res.x_quantiles), y=list(res.y_quantiles), colorscale=PARULA))
    f3.update_layout(height=520, title="السطح ثلاثي الأبعاد للمعاملات", scene=dict(xaxis_title="τ(X)", yaxis_title="θ(Y)", zaxis_title="β", aspectmode="manual",
                                                                         aspectratio=dict(x=1, y=1, z=0.7), camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))),
                     margin=dict(l=0, r=0, t=50, b=0))
    out.figs.append(("السطح ثلاثي الأبعاد", f3))
    out.metrics = {"عدد المشاهدات": int(res.n_obs), "متوسط β": float(np.nanmean(C))}
    return out


QQKRLS = LabSpec(
    key="qqkrls", pip="qqkrls", name="QQKRLS — الكمّي على الكمّي بالمربعات الصغرى المنظّمة بالنواة", family="kernel", runtime="ثوانٍ معدودة",
    reference="Hainmueller & Hazlett (2014) Political Analysis; Sim & Zhou (2015) JBF; Adebayo, Ozkan & Eweade (2024) J. Cleaner Production 440",
    roles=[Role("y", "المتغير التابع Y", "num"), Role("x", "المتغير المفسر X", "num")],
    opts=[Opt("q_min", "أدنى كمّي", "float", 0.1, min=0.05, max=0.4, step=0.05), Opt("q_max", "أعلى كمّي", "float", 0.9, min=0.6, max=0.95, step=0.05),
          Opt("q_step", "خطوة الكمّيات", "float", 0.1, min=0.05, max=0.2, step=0.05), Opt("n_boot", "تكرارات الـ bootstrap", "int", 50, min=10, max=500, step=10)],
    run=_qq_run, demo=_qq_demo, demo_label="محاكاة: أثر غير متماثل يشتد في الكمّيات العليا",
    theory=r"""
**الانحدار الكمّي على الكمّي** (Sim & Zhou, 2015) يقدّر أثر كمّي $\tau$ من $X$ على كمّي $\theta$ من $Y$ عبر انحدار كمّي محلي موزون بالنواة حول $x^\tau$. **QQKRLS** (Adebayo et al., 2024) يستبدل الخطي المحلي بـ **KRLS** (Hainmueller & Hazlett, 2014) فيلتقط اللاخطية والتفاعلات:
$$\beta(\theta,\tau)=\frac{\partial\,Q_\theta(Y\mid X)}{\partial X}\Big|_{X=x^\tau},\qquad \hat f=\arg\min_{f\in\mathcal H_k}\sum_i\big(y_i-f(x_i)\big)^2+\lambda\|f\|^2_{\mathcal H_k}$$
والناتج سطح $\beta(\theta,\tau)$ بخرائط حرارية وسطوح ثلاثية الأبعاد بألوان Parula، مع دلالة بالـ bootstrap.
""",
    guide="""
1. اختر Y و X (يفضّل مستقرين).
2. شبكة الكمّيات 0.1–0.9 بخطوة 0.1 (81 خلية).
3. الـ bootstrap 50 للتجربة و200+ للنشر.
4. اقرأ الخريطة الحرارية والسطح ومصفوفة الدلالة.
""",
    code="""from qqkrls import qqkrls
res = qqkrls(y, x, y_quantiles=np.arange(.1,.95,.05), x_quantiles=np.arange(.1,.95,.05), n_boot=200)
res.summary(); res.to_matrix("coefficient")""",
)

SPECS = [WKRLS, QQKRLS]
