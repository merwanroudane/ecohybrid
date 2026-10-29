import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

from core import ui

ui.hero(
    "الاتجاه المعاكس: القياس الاقتصادي في خدمة الذكاء الاصطناعي",
    "Econometrics for AI — Experiments, Off-policy Evaluation, Interpretability, Fairness, Shift and the Economics of AI",
    "الدمج طريق ذو اتجاهين: أدوات الاقتصاديين في التعرّف والتجارب والاستدلال أصبحت ضرورية لشركات التقنية ولتقييم أنظمة الذكاء الاصطناعي نفسها "
    "وقياس أثرها على الإنتاجية وسوق العمل.",
    "teal", "violet",
)

tabs = st.tabs([
    ":material/science: التجارب و A/B",
    ":material/recommend: التقييم خارج الخط",
    ":material/visibility: التفسير ≠ السببية",
    ":material/balance: العدالة الخوارزمية",
    ":material/swap_calls: التحول التوزيعي",
    ":material/insights: قياس أثر الذكاء الاصطناعي",
])

with tabs[0]:
    ui.section("تجارب A/B: الاقتصاد القياسي في قلب شركات التقنية", "Online experiments", "teal")
    st.markdown("تجري شركات التقنية آلاف التجارب العشوائية سنويًا. التحديات **قياسية بامتياز**: تقليل التباين، التداخل الشبكي، الآثار طويلة الأجل، الاختبار المتسلسل، والتعدد.")
    ui.cards([
        {"title": "CUPED", "en": "Deng et al. (2013) — regression adjustment", "color": "teal", "body": "استخدام متغير ما قبل التجربة كضابط لخفض التباين = تعديل الانحدار في التجارب (Lin, 2013)."},
        {"title": "التداخل الشبكي", "en": "Interference / SUTVA violations", "color": "violet", "body": "في الأسواق ثنائية الجانب والشبكات، معالجة فرد تؤثر في غيره ⇒ تصاميم عنقودية وتبديلية (Switchback)."},
        {"title": "الآثار طويلة الأجل", "en": "Surrogate indices (Athey, Chetty, Imbens & Kang)", "color": "orange", "body": "مؤشرات بديلة قصيرة الأجل تتنبأ بالنتيجة طويلة الأجل لتقييم سريع."},
        {"title": "التجارب التكيفية", "en": "Bandits & adaptive designs", "color": "rose", "body": "تخصيص ديناميكي للمعالجات (انظر صفحة التعلم المعزز) مع استدلال يصحح التكيف."},
    ], cols=4)
    ui.section("مختبر: كم يوفّر CUPED من حجم العينة؟", "CUPED variance reduction", "emerald")
    rho = st.slider("الارتباط بين المقياس قبل التجربة وبعدها ρ", 0.0, 0.95, 0.7, 0.05, key="cuped_rho")
    red = 1 - rho ** 2
    c1, c2, c3 = st.columns(3)
    c1.metric("نسبة التباين المتبقي", f"{red:.0%}")
    c2.metric("عرض فترة الثقة مقارنة بالأصل", f"{np.sqrt(red):.0%}")
    c3.metric("العينة اللازمة لنفس الدقة", f"{red:.0%} من الأصل")
    st.latex(r"\hat\tau_{CUPED}=\big(\bar Y_1-\theta\bar X_1\big)-\big(\bar Y_0-\theta\bar X_0\big),\quad \theta=\frac{\operatorname{Cov}(X,Y)}{\operatorname{Var}(X)},\quad \operatorname{Var}\downarrow\times(1-\rho^2)")

with tabs[1]:
    ui.section("تقييم سياسات جديدة من بيانات قديمة", "Off-policy evaluation (OPE)", "violet")
    st.markdown("نظام توصيات جمع بيانات تحت سياسة قديمة $\\pi_0$؛ نريد تقدير قيمة سياسة جديدة $\\pi$ **دون** تجربتها. هذه مسألة استدلال سببي بحتة:")
    st.latex(r"\hat V_{IPS}(\pi)=\frac1n\sum_i\frac{\pi(a_i\mid x_i)}{\pi_0(a_i\mid x_i)}\,r_i,\qquad "
             r"\hat V_{DR}(\pi)=\frac1n\sum_i\Big[\hat q(x_i,\pi)+\frac{\pi(a_i\mid x_i)}{\pi_0(a_i\mid x_i)}\big(r_i-\hat q(x_i,a_i)\big)\Big]")
    ui.note("مقدّر **IPS** = أوزان الاحتمال العكسي (Horvitz–Thompson)، و **DR** = نفس منطق AIPW في الاقتصاد القياسي (Dudík, Langford & Li, 2011). "
            "الشرط نفسه: **الدعم/التداخل** — لا يمكن تقييم أفعال لم تُجرَّب قط في البيانات.", "الصلة بالقياس", "violet", "🔗")

with tabs[2]:
    ui.section("مختبر: الاعتماد الجزئي قد يضلّل عندما تترابط المتغيرات", "Partial dependence vs. true marginal effects", "orange")
    st.markdown("النتيجة الحقيقية: $y = 2x_1 + 0\\cdot x_2 + \\varepsilon$ — المتغير $x_2$ **لا أثر له**، لكنه مترابط مع $x_1$. ندرّب نموذج تعزيز ونحسب **الاعتماد الجزئي (PDP)** "
                "و**أهمية المتغيرات**. ارفع الارتباط لترى كيف «يسند» النموذج جزءًا من الأثر إلى $x_2$ عبر الأشجار — التفسير التنبؤي ليس أثرًا سببيًا.")
    r = st.slider("الارتباط بين x₁ و x₂", 0.0, 0.98, 0.9, 0.02, key="pdp_r")

    @st.cache_data(show_spinner=False)
    def pdp_lab(r: float):
        rng = np.random.default_rng(0)
        n = 2000
        x1 = rng.normal(size=n); x2 = r * x1 + np.sqrt(1 - r ** 2) * rng.normal(size=n)
        y = 2 * x1 + rng.normal(0, 1, n)
        X = np.column_stack([x1, x2])
        gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(X, y)
        grid = np.linspace(-2, 2, 40)
        pd1, pd2 = [], []
        for g in grid:
            Xa = X.copy(); Xa[:, 0] = g; pd1.append(gb.predict(Xa).mean())
            Xb = X.copy(); Xb[:, 1] = g; pd2.append(gb.predict(Xb).mean())
        return grid, np.array(pd1), np.array(pd2), gb.feature_importances_

    grid, pd1, pd2, imp = pdp_lab(r)
    fig = go.Figure([go.Scatter(x=grid, y=pd1 - pd1.mean(), name="PDP لـ x₁", line=dict(color=ui.COLORS["orange"], width=4)),
                     go.Scatter(x=grid, y=pd2 - pd2.mean(), name="PDP لـ x₂ (أثره الحقيقي صفر)", line=dict(color=ui.COLORS["violet"], width=4)),
                     go.Scatter(x=grid, y=2 * grid, name="الأثر الحقيقي لـ x₁", line=dict(color="#111827", dash="dot"))])
    ui.style_fig(fig, f"أهمية المتغيرات في النموذج: x₁ = {imp[0]:.0%}، x₂ = {imp[1]:.0%}", height=440)
    ui.show(fig)
    tbl = pd.DataFrame([
        ("SHAP", "Lundberg & Lee (2017)", "إسناد مساهمة تنبؤية بقيم شابلي من نظرية الألعاب", "يعتمد على توزيع خلفي؛ ليس أثر تدخل"),
        ("PDP", "Friedman (2001)", "متوسط التنبؤ عند تثبيت متغير", "يستقرئ إلى مناطق غير مشاهدة مع الترابط"),
        ("ICE", "Goldstein et al. (2015)", "منحنى لكل مشاهدة", "يكشف عدم التجانس التنبؤي"),
        ("ALE", "Apley & Zhu (2020)", "آثار محلية متراكمة", "أكثر متانة للترابط من PDP"),
        ("الأثر الحدي القياسي", "Marginal effect", "∂E[Y|X]/∂x_j في نموذج معرّف", "سببي فقط بتصميم تعرّف"),
    ], columns=["الأداة", "المرجع", "ماذا تقيس؟", "المحذور"])
    st.table(tbl)

with tabs[3]:
    ui.section("العدالة الخوارزمية بعين اقتصادية", "Algorithmic fairness", "rose")
    ui.cards([
        {"title": "خوارزمية الرعاية الصحية", "en": "Obermeyer, Powers, Vogeli & Mullainathan (2019, Science)", "color": "rose",
         "body": "خوارزمية تستهدف المرضى بالتنبؤ بـ <b>التكاليف</b> بدل <b>الحاجة الصحية</b> ⇒ تحيز عرقي لأن الإنفاق على المرضى السود أقل لنفس المرض. درس في <b>خطأ قياس المتغير المستهدف</b>."},
        {"title": "استحالة الجمع", "en": "Kleinberg, Mullainathan & Raghavan (2017)", "color": "violet",
         "body": "لا يمكن تحقيق المعايرة وتساوي معدلات الخطأ معًا بين مجموعات مختلفة في معدلات الأساس إلا في حالات خاصة."},
        {"title": "الخوارزميات والقضاة", "en": "Kleinberg et al. (2018, QJE)", "color": "teal",
         "body": "الخوارزمية يمكن أن تخفض الجريمة أو الاحتجاز مع تقليل التفاوت العرقي — مع معالجة مشكلة «الوسوم الانتقائية»."},
        {"title": "العدالة في المسائل العكسية", "en": "Mátyás & Chan (2022), ch. 7", "color": "orange",
         "body": "إدخال قيود العدالة في تقدير المتغيرات الأداتية والمسائل العكسية — تقاطع مباشر بين العدالة والقياس."},
    ], cols=2)
    ui.section("محاكاة: عتبة واحدة، معدلات خطأ مختلفة", "One threshold, unequal error rates", "pink")
    thr = st.slider("عتبة القرار على الدرجة المتنبأ بها", 0.1, 0.9, 0.5, 0.05, key="fair_thr")
    rng = np.random.default_rng(1)
    rows = []
    for g, base in (("المجموعة أ", 0.3), ("المجموعة ب", 0.5)):
        y = rng.binomial(1, base, 20000)
        s = np.clip(0.25 + 0.5 * y + rng.normal(0, .18, 20000) + (0.05 if g == "المجموعة ب" else 0), 0, 1)
        pred = s > thr
        rows.append({"المجموعة": g, "معدل الأساس": y.mean(), "الإيجابيات الكاذبة (FPR)": pred[y == 0].mean(),
                     "السلبيات الكاذبة (FNR)": 1 - pred[y == 1].mean(), "نسبة المختارين": pred.mean()})
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.1%}" for c in ["معدل الأساس", "الإيجابيات الكاذبة (FPR)", "السلبيات الكاذبة (FNR)", "نسبة المختارين"]}),
                 hide_index=True)
    st.caption("بيانات مصطنعة للتوضيح: حتى مع درجة «عادلة» ظاهريًا، اختلاف معدلات الأساس وتوزيع الدرجات يولد معدلات خطأ مختلفة بين المجموعات.")

with tabs[4]:
    ui.section("التحول التوزيعي: نقد لوكاس في ثوب جديد", "Distribution shift, performativity & Goodhart", "amber")
    ui.cards([
        {"title": "تحول المتغيرات", "en": "Covariate shift", "color": "amber", "body": "توزيع X يتغير بينما P(Y|X) ثابت ⇒ إعادة الترجيح بنسبة الكثافات."},
        {"title": "تحول المفهوم", "en": "Concept drift", "color": "rose", "body": "P(Y|X) نفسها تتغير — ما يعادل الانكسار البنيوي في السلاسل الزمنية."},
        {"title": "التنبؤ الأدائي", "en": "Performative prediction (Perdomo et al., 2020)", "color": "violet", "body": "التنبؤ يغيّر السلوك الذي يتنبأ به (درجة الائتمان، التسعير) — نقد لوكاس حرفيًا."},
        {"title": "قانون غودهارت", "en": "Goodhart's law", "color": "teal", "body": "عندما يصبح المؤشر هدفًا يتوقف عن كونه مؤشرًا جيدًا — مؤشرات AI المستخدمة في السياسات معرضة للتلاعب."},
        {"title": "التعميم خارج التوزيع", "en": "OOD generalization", "color": "sky", "body": "الثبات (Invariance) عبر البيئات كبديل للسببية (Peters, Bühlmann & Meinshausen, 2016)."},
        {"title": "الصدق الخارجي", "en": "External validity", "color": "emerald", "body": "نقل الآثار المقدّرة من مجتمع لآخر يتطلب نموذجًا لعدم التجانس — حيث تساعد الغابات السببية."},
    ], cols=3)

with tabs[5]:
    ui.section("القياس الاقتصادي لأثر الذكاء الاصطناعي نفسه", "Measuring the economic impact of AI", "indigo")
    imp = pd.DataFrame([
        ("Brynjolfsson, Li & Raymond (2025, QJE)", "مساعد ذكاء توليدي لموظفي مراكز الاتصال", "إنتاجية +14% في المتوسط و+34% للمبتدئين؛ أثر ضئيل على الأكثر خبرة", "تبنٍّ متدرج (شبه تجريبي)"),
        ("Noy & Zhang (2023, Science)", "ChatGPT لمهام كتابة مهنية", "انخفاض الوقت ~40% وارتفاع الجودة ~18%؛ تقلص التفاوت بين العمال", "تجربة عشوائية"),
        ("Eloundou, Manning, Mishkin & Rock (2024, Science)", "التعرض المهني لنماذج اللغة", "~80% من القوى العاملة الأمريكية قد يتأثر 10% على الأقل من مهامها", "قياس تعرض بمساعدة LLM"),
        ("Acemoglu (2025, Economic Policy)", "الاقتصاد الكلي البسيط للذكاء الاصطناعي", "مكاسب إنتاجية كلية متواضعة (أقل من 1% خلال عقد) وفق إطار هولتن", "نمذجة كلية بالمهام"),
        ("Calvano et al. (2020, AER)", "خوارزميات تسعير بالتعلم المعزز", "تتعلم أسعارًا فوق تنافسية دون تواصل صريح", "محاكاة احتكار القلة"),
    ], columns=["الدراسة", "الموضوع", "النتيجة الرئيسية", "تصميم التعرّف"])
    st.table(imp)
    ui.note("هنا ينعكس الدمج: **الذكاء الاصطناعي هو المعالجة (Treatment)** والقياس الاقتصادي هو أداة التقييم — بالتجارب العشوائية والفرق في الفروق والمتغيرات الأداتية.",
            "الدمج المعكوس", "indigo", "🔁")
    ui.refs("Deng et al. (2013); Dudík, Langford & Li (2011); Lundberg & Lee (2017); Obermeyer et al. (2019) Science; Kleinberg, Mullainathan & Raghavan (2017); "
            "Perdomo et al. (2020); Brynjolfsson, Li & Raymond (2025) QJE; Noy & Zhang (2023) Science; Eloundou et al. (2024) Science; Acemoglu (2025).")
