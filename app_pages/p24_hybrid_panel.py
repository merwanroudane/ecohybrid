import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from core import ui

ui.hero(
    "الخوارزميات الهجينة للبانل والتصاميم السببية",
    "Hybrid Panel & Causal-Design Algorithms — DML-IV, Deep IV, DML-DiD, DML-RDD, Panel DML, Quantile & Spatial ML",
    "كيف يدخل تعلم الآلة داخل كل تصميم تعرّف كلاسيكي دون أن يغيّر منطقه: المتغيرات الأداتية، الفرق في الفروق، الانقطاع الانحداري، "
    "التحكم التركيبي، بيانات البانل بآثار ثابتة وتفاعلية، الكمّيات، الوساطة، الانتقاء، والبيانات المكانية والشبكية.",
    "violet", "emerald",
)

tabs = st.tabs([
    ":material/map: الخريطة",
    ":material/call_split: IV + ML",
    ":material/difference: DiD والتحكم التركيبي",
    ":material/vertical_split: RDD + ML",
    ":material/grid_view: البانل + ML",
    ":material/stacked_bar_chart: الكمّيات والتوزيعات",
    ":material/alt_route: الوساطة والانتقاء والجرعات",
    ":material/public: المكاني والشبكي",
    ":material/table_view: جدول شامل",
])

spl = lambda: SplineTransformer(n_knots=6, degree=3)

with tabs[0]:
    ui.section("ML داخل كل تصميم تعرّف", "Where ML enters each identification design", "violet")
    st.mermaid_chart(
        """
flowchart TB
    D[Identification design fixed by the researcher] --> IV[IV: first stage and NPIV]
    D --> DID[DiD: propensity + outcome-change models]
    D --> RDD[RDD: covariate adjustment near the cutoff]
    D --> SC[Synthetic control: weights, matrix completion]
    D --> PAN[Panel: FE with nonlinear nuisance]
    IV --> M1[DML-PLIV, Deep IV, Kernel IV, AGMM, DeepGMM]
    DID --> M2[DML-DiD, DR-DiD, staggered DR with ML]
    RDD --> M3[Flexible covariate adjustment, RD causal trees]
    SC --> M4[Elastic-net SC, MC-NNM, SDID, conformal SC]
    PAN --> M5[DML-FE, DML-IFE, MERF, GPBoost, dynamic-panel DML]
"""
    )
    ui.note("الخوارزمية **لا تخلق** مجموعة مقارنة ولا تجعل **الاتجاهات المتوازية** صحيحة ولا تحوّل الارتباط إلى صدمة خارجية. "
            "دورها: تقدير الأجزاء المساعدة بمرونة داخل تصميم صالح مسبقًا. اختبارات التوازن، والدواء الوهمي (Placebo)، والاتجاهات المسبقة، وفوق التعرّف تبقى ضرورية.",
            "القاعدة", "violet", "📏")
    ui.note("تطبيقات Python للبانل والتصاميم السببية في مكتبات د. مروان رودان على PyPI: **dynpanelai** (GMM، AB-LASSO، DML، لاسو متعامد، AC-GATE)، **mllpanel** (Yang, Long & Cai)، "
            "**deeppanel**، **mlpaneldata**، **panelxai**، و **pragdml** للتمثيلات المتعلَّمة.", "مكتبات جاهزة", "fuchsia", "📦")
    ui.link("app_pages/p30_author_libs.py", "عرض صفحة المكتبات الهجينة بالتفصيل", ":material/deployed_code:")

with tabs[1]:
    ui.section("المتغيرات الأداتية مع تعلم الآلة", "Instrumental variables meet ML", "orange")
    st.latex(r"\text{PLIV: }\ Y=\theta D+g(X)+U,\ \ Z=m(X)+V,\ \ \mathbb E[U\mid Z,X]=0;\qquad \text{NPIV: }\ \mathbb E\big[Y-h(D,X)\mid Z,X\big]=0")
    ui.cards([
        {"title": "DML-PLIV", "en": "Chernozhukov et al. (2018)", "color": "orange", "body": "ضوابط كثيرة بمرونة مع أداة؛ الدالة المتعامدة (Y−ℓ(X)−θ(D−r(X)))(Z−m(X))."},
        {"title": "لاسو للمرحلة الأولى", "en": "Belloni, Chen, Chernozhukov & Hansen (2012, Econometrica)", "color": "amber", "body": "أدوات كثيرة: اختيار الأدوات المثلى بلاسو مع استدلال صالح."},
        {"title": "Deep IV", "en": "Hartford, Lewis, Leyton-Brown & Taddy (2017, ICML)", "color": "rose", "body": "شبكتان: الأولى لتوزيع D|Z,X (كثافة شرطية)، والثانية لدالة الاستجابة بخسارة تتكامل على ذلك التوزيع."},
        {"title": "Kernel IV", "en": "Singh, Sahani & Gretton (2019)", "color": "violet", "body": "تعميم الغربال إلى فضاء هيلبرت النواتي المولّد (RKHS) بقواعد لا نهائية."},
        {"title": "Adversarial GMM", "en": "Lewis & Syrgkanis (2018)", "color": "teal", "body": "تعلم تمثيلات عميقة لنماذج معرّفة بقيود لحظية شرطية عبر لعبة تنافسية؛ تفوق على Deep IV في تجاربهم."},
        {"title": "DeepGMM / DFIV", "en": "Bennett, Kallus & Schnabel (2019); Xu et al. (2021)", "color": "sky", "body": "GMM تنافسي بمصفوفة أوزان مثلى؛ ميزات عميقة متعلَّمة لـ IV — مع قلق الاستقرار مقارنة بـ 2SLS."},
    ], cols=3)
    ui.note("**مشكلة سوء الطرح (Ill-posedness)** في NPIV: حتى لو قدّرنا التوقع الشرطي جيدًا، فإن استرجاع $h$ يضخّم الأخطاء. الأدوات الضعيفة تبقى ضعيفة مع أي شبكة، "
            "وشرط الاستبعاد (Exclusion) افتراض اقتصادي لا يُختبر بالخوارزمية.", "محاذير", "orange", "⚠️")

with tabs[2]:
    ui.section("الفرق في الفروق والتحكم التركيبي مع ML", "DiD & synthetic control with ML", "teal")
    ui.cards([
        {"title": "DML للفرق في الفروق", "en": "Chang (2020, Econometrics Journal)", "color": "teal", "body": "مقدّر DiD شبه معلمي مزدوج المتانة بالمعدلات مع متعلمات ML لضوابط عالية الأبعاد."},
        {"title": "DiD مزدوج المتانة", "en": "Sant'Anna & Zhao (2020, J. Econometrics)", "color": "violet", "body": "ATT صالح إن صح نموذج درجة الميل أو نموذج تغير النتيجة؛ متوافق مع DML."},
        {"title": "التبني المتدرج + ML", "en": "Callaway & Sant'Anna (2021); Hatamyar, Kreif, Rocha & Huber (2023)", "color": "orange", "body": "ATT(g,t) بالمجموعات والفترات مع تقدير ML لعدم التجانس في التبني المتدرج."},
        {"title": "DR متدرج مع DML", "en": "Sun, Xie & Zhang (2025)", "color": "rose", "body": "متوسط مرجح لمقدّرات DR-DiD عبر المجموعات بتوافق مع DML."},
        {"title": "التحكم التركيبي المنظّم", "en": "Doudchenko & Imbens (2016); Abadie & L'Hour (2021)", "color": "sky", "body": "أوزان بالشبكة المرنة بدل القيود الصارمة، وعقوبات لتقليل التحيز التداخلي."},
        {"title": "إكمال المصفوفات", "en": "Athey, Bayati, Doudchenko, Imbens & Khosravi (2021, JASA)", "color": "emerald", "body": "النتائج المضادة للواقع كمصفوفة ناقصة منخفضة الرتبة تُكمل بعقوبة المعيار النووي (MC-NNM)."},
        {"title": "DiD التركيبي", "en": "Arkhangelsky et al. (2021, AER)", "color": "amber", "body": "SDID: أوزان للوحدات والفترات معًا — يجمع مزايا DiD والتحكم التركيبي."},
        {"title": "الاستدلال المطابق للتحكم التركيبي", "en": "Chernozhukov, Wüthrich & Zhu (2021, JASA)", "color": "fuchsia", "body": "فترات واختبارات صالحة بالتبديل المطابق لأي مقدّر للنتائج المضادة."},
    ], cols=4)
    ui.section("مختبر: اتجاهات متوازية مشروطة بـ X غير خطيًا", "Lab — naive DiD vs. DR-DiD with flexible learners", "pink")
    st.markdown("المعالجة أرجح لوحدات ذات $X$ مرتفع، والاتجاه الزمني دالة **لا خطية** في $X$ ⇒ الاتجاهات متوازية **فقط بشرط $X$**. "
                "نقارن DiD الساذج (بلا ضوابط)، DiD بضابط خطي، و **DR-DiD** (سانتانا وتشاو) بمتعلمات مرنة (splines) وتوفيق متقاطع. الأثر الحقيقي ATT = 1.")

    @st.cache_data(show_spinner="محاكاة...")
    def did_lab(reps: int = 200, n: int = 1500):
        rng = np.random.default_rng(8)
        out = {"DiD ساذج": [], "DiD + X خطي": [], "DR-DiD (ML)": []}
        for _ in range(reps):
            X = rng.uniform(-2, 2, n)
            e = 1 / (1 + np.exp(-1.2 * X)); D = rng.binomial(1, e)
            trend = 1.5 * np.sin(1.5 * X) + 0.5 * X ** 2
            dY = trend + D * 1.0 + rng.normal(0, 1, n)
            out["DiD ساذج"].append(dY[D == 1].mean() - dY[D == 0].mean())
            Z = np.column_stack([np.ones(n), D, X])
            out["DiD + X خطي"].append(np.linalg.lstsq(Z, dY, rcond=None)[0][1])
            idx = rng.permutation(n); f = np.array_split(idx, 2); psi_num = np.zeros(n)
            ps = np.zeros(n); mu0 = np.zeros(n)
            for k in range(2):
                te, tr = f[k], f[1 - k]
                ps[te] = np.clip(make_pipeline(spl(), LogisticRegression(max_iter=500)).fit(X[tr, None], D[tr]).predict_proba(X[te, None])[:, 1], .02, .98)
                c = tr[D[tr] == 0]
                mu0[te] = make_pipeline(spl(), Ridge(alpha=1.0)).fit(X[c, None], dY[c]).predict(X[te, None])
            w1 = D / D.mean(); w0 = ps * (1 - D) / (1 - ps); w0 = w0 / w0.mean()
            out["DR-DiD (ML)"].append(np.mean((w1 - w0) * (dY - mu0)))
        return {k: np.array(v) for k, v in out.items()}

    dd = did_lab()
    fig = go.Figure([go.Box(y=v, name=k, marker_color=c, boxmean=True) for (k, v), c in zip(dd.items(), [ui.COLORS["rose"], ui.COLORS["amber"], ui.COLORS["teal"]])])
    fig.add_hline(y=1.0, line_dash="dash", annotation_text="ATT الحقيقي = 1")
    ui.style_fig(fig, "توزيع المقدّرات عبر 200 محاكاة", height=440, legend_h=False)
    ui.show(fig)
    st.dataframe(pd.DataFrame({"المقدّر": list(dd), "المتوسط": [v.mean() for v in dd.values()], "التحيز": [v.mean() - 1 for v in dd.values()],
                               "الانحراف المعياري": [v.std() for v in dd.values()]}).style.format({"المتوسط": "{:.3f}", "التحيز": "{:+.3f}", "الانحراف المعياري": "{:.3f}"}),
                 hide_index=True)

with tabs[3]:
    ui.section("الانقطاع الانحداري مع تعلم الآلة", "Regression discontinuity with ML", "sky")
    st.latex(r"\hat\tau_{RD}(\hat\eta)=\text{local-linear RD on } \ M_i=Y_i-\hat\eta(Z_i),\qquad \eta^*(z)=\tfrac12\Big(\mathbb E[Y\mid R=c^+,Z=z]+\mathbb E[Y\mid R=c^-,Z=z]\Big)")
    ui.cards([
        {"title": "تعديل مرن للمتغيرات المصاحبة", "en": "Noack, Olma & Rothe (2021–2025)", "color": "sky",
         "body": "نطرح دالة مقدَّرة للمتغيرات المصاحبة من النتيجة ثم نجري RD محليًا خطيًا. يمكن تقديرها بلاسو أو غابات أو شبكات أو تجميعات <b>دون</b> تغيير الخصائص من الدرجة الأولى للمقدّر — مع مكاسب كفاءة (أكثر من 35% في طول فترات الثقة في بعض الحالات المعاد تحليلها)."},
        {"title": "المتغيرات المصاحبة الكثيرة", "en": "Calonico, Cattaneo, Farrell & Titiunik (2019); Kreiss & Rothe (2023)", "color": "violet",
         "body": "إدراج خطي للمتغيرات المسبقة في RD، وتعميمه للأبعاد العالية بلاسو مع تصحيح الاستدلال."},
        {"title": "الأثر غير المتجانس في RD", "en": "Reguly (2021) RD causal trees", "color": "orange",
         "body": "أشجار سببية معدّلة لتصاميم الانقطاع لاكتشاف من يتأثر أكثر عند العتبة."},
        {"title": "ما لا تفعله ML", "en": "What ML cannot fix", "color": "rose",
         "body": "التلاعب بالمتغير الجاري (اختبار McCrary/الكثافة)، عدم الاستمرارية في المتغيرات المصاحبة، اختيار عرض النطاق — كلها تبقى قضايا تصميم."},
    ], cols=2)
    ui.section("مختبر: تعديل المتغيرات المصاحبة يضيّق الخطأ المعياري لـ RD", "Lab — ML covariate adjustment in RD", "pink")

    @st.cache_data(show_spinner="محاكاة...")
    def rd_lab(reps: int = 200, n: int = 2000, h: float = 0.4):
        rng = np.random.default_rng(10)
        res = {"RD بدون متغيرات مصاحبة": [], "RD مع تعديل مرن (ML)": []}
        for _ in range(reps):
            R = rng.uniform(-1, 1, n); Zc = rng.normal(size=n)
            Y = 0.5 + 1.0 * (R >= 0) + 0.8 * R + 1.2 * np.sin(2 * Zc) + 0.6 * Zc ** 2 + rng.normal(0, .5, n)
            idx = rng.permutation(n); f = np.array_split(idx, 2); eta = np.zeros(n)
            for k in range(2):
                te, tr = f[k], f[1 - k]
                eta[te] = make_pipeline(spl(), Ridge(alpha=1.0)).fit(np.column_stack([Zc[tr]]), Y[tr]).predict(np.column_stack([Zc[te]]))
            w = np.abs(R) < h
            for key, outc in (("RD بدون متغيرات مصاحبة", Y), ("RD مع تعديل مرن (ML)", Y - eta)):
                Xd = np.column_stack([np.ones(w.sum()), (R[w] >= 0), R[w], R[w] * (R[w] >= 0)])
                b = np.linalg.lstsq(Xd, outc[w], rcond=None)[0]
                res[key].append(b[1])
        return {k: np.array(v) for k, v in res.items()}

    rr = rd_lab()
    fig = go.Figure([go.Histogram(x=v, name=k, opacity=.6, marker_color=c, xbins=dict(start=0, end=2, size=0.025))
                     for (k, v), c in zip(rr.items(), [ui.COLORS["amber"], ui.COLORS["sky"]])])
    fig.update_layout(barmode="overlay"); fig.add_vline(x=1.0, line_dash="dash", annotation_text="τ = 1")
    sd0, sd1 = [v.std() for v in rr.values()]
    ui.style_fig(fig, f"الانحراف المعياري: {sd0:.3f} بدون تعديل ← {sd1:.3f} مع التعديل (انخفاض {1 - sd1 / sd0:.0%})", height=420)
    ui.show(fig)
    st.caption("لأن المتغيرات المصاحبة مسبقة (لا تتأثر بالمعالجة)، يبقى المقدّر غير متحيز في الحالتين؛ التعديل يزيل جزءًا من الضجيج فقط.")

with tabs[4]:
    ui.section("بيانات البانل مع تعلم الآلة", "Panel data meets ML", "emerald")
    st.latex(r"Y_{it}=\theta D_{it}+g(X_{it})+\alpha_i+U_{it}\ \Longrightarrow\ \Delta Y_{it}=\theta\,\Delta D_{it}+\big[g(X_{it})-g(X_{i,t-1})\big]+\Delta U_{it}")
    ui.cards([
        {"title": "DML بآثار ثابتة", "en": "Clarke & Polselli (2026, Econometrics Journal)", "color": "emerald",
         "body": "تعميم نموذج روبنسون للبانل: امتدادات لمقدّرات الآثار العشوائية المترابطة (CRE)، والتحويل الداخلي، والفروق الأولى؛ يوصون بالفروق الأولى لأقل القيود على الآثار الثابتة. حزمة R: xtdml."},
        {"title": "DML بآثار ثابتة تفاعلية", "en": "Chen, Polselli & Clarke (2026, preprint)", "color": "teal",
         "body": "عوامل مشتركة غير مشاهدة بأحمال فردية مع دوال ضبط لا خطية متعلَّمة."},
        {"title": "البانل الديناميكي عالي الأبعاد", "en": "Semenova, Goldman, Chernozhukov & Taddy (2023, QE)", "color": "violet",
         "body": "CATE في بانل ديناميكي: توفيق متقاطع يترك الجيران زمنيًا، ثم متعلم لاسو للأثر؛ مرونات أسعار البقالة من بيانات الماسحات."},
        {"title": "لاسو البانل", "en": "Belloni, Chernozhukov, Hansen & Kozbur (2016, JBES)", "color": "orange",
         "body": "الاستدلال في نماذج بانل عالية الأبعاد بآثار ثابتة وأخطاء عنقودية؛ تطبيق على قوانين الأسلحة."},
        {"title": "DML والبانل: الوعود والمزالق", "en": "Fuhr, Berens & Papies (2024, arXiv 2409.01266)", "color": "rose",
         "body": "مراجعة عملية لما يحدث عند تطبيق DML ساذجًا على البانل وكيف نصمّم الطيات والتحويلات."},
        {"title": "ML بآثار مختلطة", "en": "MERF (Hajjem et al., 2014); GPBoost (Sigrist, 2022, JMLR)", "color": "sky",
         "body": "غابات/تعزيز مع آثار عشوائية مجمّعة؛ في بانل أجور ~4700 امرأة شابة تفوق GPBoost ثم MERF على بدائل ML بلا آثار."},
        {"title": "بانل MIDAS بتعلم الآلة", "en": "Babii, Ball, Ghysels & Striaukas (2023)", "color": "amber",
         "body": "sg-LASSO لانحدارات بانل بتواترات مختلطة — تنبؤ بأرباح الشركات."},
        {"title": "الغابات السببية في البانل", "en": "panel causal forests", "color": "fuchsia",
         "body": "إزالة الآثار الثابتة أولًا (تحويل داخلي/فروق) ثم غابة سببية مع طيات حسب الوحدة."},
    ], cols=4)
    ui.note("**قواعد البانل الذهبية مع ML**: (1) الطيات **حسب الوحدة** لا حسب المشاهدة، (2) أزل الآثار الثابتة بتحويل مناسب قبل/داخل التعلم، "
            "(3) أخطاء معيارية **عنقودية** بالوحدة (وبالزمن عند الاعتماد المقطعي)، (4) انتبه لتحيز نيكل (Nickell) في البانل الديناميكي قصير T.",
            "قواعد", "emerald", "📋")

with tabs[5]:
    ui.section("الكمّيات والتوزيعات مع تعلم الآلة", "Quantile & distributional ML", "orange")
    ui.cards([
        {"title": "غابات الانحدار الكمّي", "en": "Meinshausen (2006, JMLR)", "color": "orange", "body": "الغابة تحمل معلومة التوزيع الشرطي كاملًا ⇒ كمّيات شرطية متسقة في أبعاد عالية. حزمة quantregForest."},
        {"title": "آثار المعالجة على الكمّيات", "en": "Belloni, Chernozhukov, Fernández-Val & Hansen (2017, Econometrica)", "color": "violet", "body": "تقييم البرامج والاستدلال السببي بالبيانات عالية الأبعاد: آثار توزيعية وكمّية بدوال متعامدة."},
        {"title": "DML المحلي للكمّيات", "en": "Kallus, Mao & Uehara (localized debiased ML)", "color": "teal", "body": "تقدير كمّيات المعالجة بمعادلات تقدير غير ملساء مع تصحيح التحيز محليًا."},
        {"title": "الشبكات الكمّية", "en": "Quantile neural networks", "color": "rose", "body": "تدريب الشبكات بخسارة pinball لكل كمّي؛ قيود لمنع تقاطع الكمّيات."},
        {"title": "الكمّي على الكمّي + النواة", "en": "QQR × KRLS", "color": "sky", "body": "دمج QQR (Sim & Zhou, 2015) مع KRLS (Hainmueller & Hazlett, 2014) لعلاقات لا خطية عبر كمّيات المتغيرين."},
        {"title": "الغابات العشوائية المعممة", "en": "GRF quantile & instrumental forests", "color": "emerald", "body": "إطار موحد لأي معلمة محلية بشرط لحظي — كمّيات، أدوات، معاملات متغيرة."},
    ], cols=3)
    ui.note("ميل الانحدار عند كمّي معين ليس بالضرورة أثرًا على **الفرد نفسه** ولا أثرًا سببيًا؛ والكمّي الشرطي يختلف عن الأثر على الكمّي غير الشرطي (UQPE / RIF).",
            "تنبيه تفسيري", "orange", "⚠️")

with tabs[6]:
    ui.section("الوساطة والانتقاء والمعالجات المستمرة والديناميكية", "Mediation, selection, continuous and dynamic treatments", "rose")
    ui.ladder([
        ("الوساطة السببية", "Farbmacher, Huber, Lafférs, Langen & Spindler (2022, Econometrics Journal)", "الأثر غير المباشر عبر وسيط والأثر المباشر بدوال كفؤة متعددة المتانة + DML؛ تطبيق على التأمين الصحي (أثر لا يمر عبر الفحوص الدورية)."),
        ("الانتقاء والتسرب", "Bia, Huber & Lafférs (2023, JBES)", "DML لنماذج الانتقاء في العينة والتسرب مع معالجات ثنائية أو متعددة؛ حزمة causalweight."),
        ("المعالجات المستمرة", "Colangelo & Lee (2020–2023)", "منحنى الجرعة–الاستجابة بالنواة مع DML؛ Auto-DML للمعالجات المستمرة."),
        ("المعالجات الديناميكية", "Lewis & Syrgkanis (2021, NeurIPS); Bodory, Huber & Lafférs (2022)", "تسلسلات معالجة عبر الزمن بتقدير g-estimation متعامد."),
        ("DiD مع الوساطة", "Huber & Oberhänsli (2026)", "إطار فرق في الفروق مع وساطة لمعالجات منفصلة أو مستمرة بمقدّرات DML."),
    ])

with tabs[7]:
    ui.section("البيانات المكانية والشبكية", "Spatial & network econometrics with ML", "indigo")
    st.latex(r"\text{SAR: }\ \mathbf y=\rho W\mathbf y+X\boldsymbol\beta+\boldsymbol\varepsilon\qquad\text{vs.}\qquad \text{GNN: }\ h_i^{(l+1)}=\sigma\Big(\sum_{j\in\mathcal N(i)}\alpha_{ij}\,W^{(l)}h_j^{(l)}\Big)")
    ui.cards([
        {"title": "غابات بفجوات مكانية", "en": "Credit (2022, Geographical Analysis)", "color": "indigo", "body": "الغابة تتفوق قليلًا على النماذج المكانية في كثافة التوظيف؛ إضافة الفجوات المكانية تحسّن الدقة."},
        {"title": "ميزات الترشيح المكاني", "en": "ISPRS (2022) spatial lag + eigenvector filtering", "color": "teal", "body": "تخفض خطأ التنبؤ والارتباط الذاتي المكاني في البواقي وتتفوق على GWR."},
        {"title": "GNN و SAR", "en": "graph convolution as spatial autoregression", "color": "violet", "body": "شبكات التفاف رسومية تعمم SAR وتتجاوز قيد مصفوفة الأوزان المحددة مسبقًا."},
        {"title": "SAR عصبي", "en": "neural SAR with asymptotics", "color": "orange", "body": "إضافة مكوّن شبكي لنموذج SAR مع إثبات الاتساق والتوزيع الطبيعي التقاربي."},
        {"title": "اقتصاد الشبكات القياسي", "en": "Mátyás & Chan (2022), ch. 6 & 8", "color": "rose", "body": "تقليل أبعاد الرسوم، أخذ عينات الشبكات، النماذج الرسومية في المال والاقتصاد."},
        {"title": "المحاذير", "en": "reflection problem, endogenous links", "color": "amber", "body": "تمرير الرسائل في الشبكة ينقل ارتباطات لا آثارًا سببية؛ مشكلة الانعكاس (Manski, 1993)، الروابط الداخلية، التداخل، والطيات المكانية المحجوبة."},
    ], cols=3)

with tabs[8]:
    ui.section("جدول شامل للخوارزميات الهجينة في البانل والتصاميم السببية", "Comprehensive table", "slate")
    big = pd.DataFrame([
        ("PDS-Lasso", "OLS سببي", "لاسو اختيار مزدوج", "ATE بضوابط كثيرة", "Belloni et al. (2014)", "R hdm؛ Stata pdslasso"),
        ("DML-PLR / IRM", "شبه خطي / AIPW", "أي متعلم + توفيق متقاطع", "ATE/ATT", "Chernozhukov et al. (2018)", "DoubleML؛ ddml"),
        ("DML-PLIV / IIVM", "IV", "متعلمات للدوال المساعدة", "أثر IV / LATE", "Chernozhukov et al. (2018)", "DoubleML"),
        ("Deep IV", "NPIV", "شبكتان عميقتان", "دالة استجابة", "Hartford et al. (2017)", "EconML"),
        ("AGMM / DeepGMM", "GMM شرطي", "تدريب تنافسي", "دالة سببية", "Lewis & Syrgkanis (2018); Bennett et al. (2019)", "Python"),
        ("DML-DiD", "DiD", "متعلمات + DR", "ATT", "Chang (2020)", "DoubleML (DID)"),
        ("DR-DiD", "DiD", "درجة ميل + نموذج نتيجة", "ATT", "Sant'Anna & Zhao (2020)", "R DRDID؛ did"),
        ("MC-NNM", "بانل/تحكم تركيبي", "إكمال مصفوفة منخفضة الرتبة", "مضاد للواقع", "Athey et al. (2021)", "R/ Python"),
        ("SDID", "DiD + SC", "أوزان منظمة", "ATT", "Arkhangelsky et al. (2021)", "R synthdid"),
        ("RD + ML adjustment", "RDD", "أي متعلم للمتغيرات المصاحبة", "أثر محلي أدق", "Noack, Olma & Rothe", "R"),
        ("DML-FE (xtdml)", "بانل FE", "متعلمات + CRE/WG/FD", "θ هيكلي", "Clarke & Polselli (2026)", "R xtdml"),
        ("DML-IFE", "بانل بعوامل تفاعلية", "متعلمات لا خطية", "θ مع عوامل", "Chen, Polselli & Clarke (2026)", "—"),
        ("Dynamic-panel CATE", "بانل ديناميكي", "لاسو + ترك الجيران", "CATE", "Semenova et al. (2023)", "—"),
        ("MERF / GPBoost", "آثار مختلطة", "غابة / تعزيز + RE", "تنبؤ بانل", "Hajjem et al.; Sigrist (2022)", "Python merf؛ gpboost"),
        ("QRF", "انحدار كمّي", "غابة", "كمّيات شرطية", "Meinshausen (2006)", "R quantregForest؛ grf"),
        ("DML mediation", "وساطة", "DML متعدد المتانة", "أثر مباشر/غير مباشر", "Farbmacher et al. (2022)", "R causalweight"),
        ("DML selection", "انتقاء العينة", "DML", "ATE مع تسرب", "Bia, Huber & Lafférs (2023)", "R causalweight"),
        ("Spatial RF / GNN-SAR", "SAR/SDM", "غابة/شبكة رسومية", "تنبؤ مكاني", "Credit (2022)", "Python PyG"),
    ], columns=["الهجين", "المكوّن القياسي", "المكوّن الذكي", "الهدف", "المرجع", "البرمجيات"])
    st.dataframe(big, hide_index=True, height=620)
    ui.refs("Chang (2020); Sant'Anna & Zhao (2020); Callaway & Sant'Anna (2021); Athey et al. (2021); Arkhangelsky et al. (2021); Noack, Olma & Rothe (2025); "
            "Clarke & Polselli (2026); Semenova et al. (2023); Belloni et al. (2016, 2017); Farbmacher et al. (2022); Bia, Huber & Lafférs (2023); Sigrist (2022).")
