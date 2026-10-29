import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "مستويات الدمج وتصنيفاته",
    "Levels of Integration (L0–L7) and Five Further Classification Axes",
    "سلّم من ثمانية مستويات يتدرج من «الذكاء الاصطناعي كمساعد بحث» إلى «النظرية داخل الخوارزمية» و«النماذج التأسيسية». "
    "هو إطار تعليمي منظم — لا سلّم معياري متفق عليه — ويمكن لمشروع واحد أن ينتمي لأكثر من مستوى.",
    "fuchsia", "sky",
)

LEVELS = [
    ("L0", "مساندة البحث", "AI Copilot / Coding Agent",
     "تنظيم الأدبيات، كتابة الأكواد، الفحوص الآلية، التلخيص — تحت رقابة بشرية كاملة.",
     "LLMs، وكلاء البرمجة، أدوات البحث الدلالي", "Korinek (2023, 2025)؛ Galiani, López & Sosa (2026)",
     "أقل درجة بنيوية: الذكاء الاصطناعي خارج المقدّر تمامًا", "هلوسة المراجع، أخطاء برمجية صامتة"),
    ("L1", "دمج مرحلة البيانات", "AI-Assisted Preprocessing",
     "التعويض عن القيم المفقودة، كشف القيم الشاذة، مطابقة الكيانات، التصنيف الأولي للنصوص، التعديل الموسمي.",
     "Autoencoders، RF، KNN، NLP، مطابقة الأسماء", "Peña & Tsay (2021)؛ Dell (2025)",
     "مساعد خارج المقدّر يغيّر البيانات الداخلة", "تسرب معلومات الاختبار، تغيير الصدمات"),
    ("L2", "الدمج التنبؤي", "Forecasting Hybrid / Ensemble",
     "نماذج هجينة للتنبؤ: ARIMA–LSTM، GARCH–ML، تعلم البواقي، التكديس، مزيج الخبراء.",
     "Zhang (2003) ARIMA–ANN، Stacking، MRF، BART-VAR، TSFM", "Zhang (2003)؛ Goulet Coulombe et al. (2022)",
     "تكامل داخل دالة التنبؤ", "تعلم البواقي داخل العينة، تقسيم زمني خاطئ"),
    ("L3", "الدمج الاستدلالي", "Causal ML / Orthogonal ML",
     "ML داخل مقدّر قياسي سببي: DML، DR-Learner، الغابات السببية، Deep IV، ML-DiD، ML-RDD.",
     "DML، GRF، Meta-learners، AGMM، DR-DiD", "Belloni et al. (2014)؛ Chernozhukov et al. (2018)؛ Wager & Athey (2018)",
     "ML داخل مقدّر قياسي مع حماية الاستدلال", "انتهاك التداخل، تعرّف غير صالح، تقسيم يتجاهل البنية"),
    ("L4", "الدمج القياسي المفاهيمي", "AI-Generated Economic Measurement",
     "الذكاء الاصطناعي ينتج متغيرًا جديدًا: معنويات، توقعات، وسم العمل عن بعد، مؤشرات من الأقمار.",
     "LLMs، BERT، CNN، PPI، AI-PI، ValidMLInference", "Battaglia et al. (2024)؛ Dell & Rambachan (2026)؛ Ludwig et al. (2025)",
     "AI يصنع المتغير الذي يدخل الانحدار", "خطأ قياس غير كلاسيكي، اختلاف النماذج والمطالبات"),
    ("L5", "الدمج النظري", "Theory- / Economics-Informed ML",
     "النظرية داخل التعلم: قيود رتابة، بواقي أويلر في دالة الخسارة، شروط عدم المراجحة، نماذج الكتل الجمعية.",
     "SKINNs، BAM، SDF-nets، قيود الشكل، Teaching Economics to Machines", "Chen, Pelger & Zhu (2024)؛ Cao et al. (2026)؛ Buckmann & Potjagailo (2025)",
     "النظرية داخل دالة الخسارة أو المعمارية", "قيد نظري خاطئ يولّد تحيزًا منظمًا"),
    ("L6", "الدمج الهيكلي الحسابي", "Neural Structural Econometrics",
     "الشبكات تحل النماذج الهيكلية وتقدّرها: شبكات التوازن العميق، PINN لمعادلات HJB، المحاكيات، التقدير العصبي والتنافسي.",
     "DEQN، PINN، Deep surrogates، Adversarial estimation، Neural estimators", "Maliar et al. (2021)؛ Azinovic et al. (2022)؛ Kaji et al. (2023)؛ Kase et al. (2025)",
     "AI يساعد في الحل والتقدير الهيكلي", "خطأ التقريب خارج نطاق التدريب، خلط خطأ الحل بخطأ التقدير"),
    ("L7", "الدمج التأسيسي/العام", "Foundation Models / Agents",
     "نماذج مسبقة التدريب متعددة الوظائف ووكلاء: BISTRO، MACROCAST، Chronos، TabPFN، وكلاء البحث المدقَّقون.",
     "TSFM، LLM agents، Tabular FMs", "BIS (2026)؛ Carriero et al. (2026)؛ Hollmann et al. (2025)",
     "نموذج مسبق التدريب ووظائف متعددة", "تلوث التدريب المسبق، مراجعات البيانات، درجات حرية خفية"),
]

tabs = st.tabs([
    ":material/stairs: السلّم L0–L7",
    ":material/zoom_in: مستكشف المستويات",
    ":material/donut_large: الخريطة الشمسية",
    ":material/grid_on: المستويات × البيانات",
    ":material/schema: معماريات الدمج",
    ":material/category: محاور التصنيف الخمسة",
    ":material/linear_scale: عمق المعرفة الاقتصادية",
])

with tabs[0]:
    ui.section("سلّم المستويات الثمانية", "The eight-level ladder", "fuchsia")
    ui.ladder([(f"{l[0]} — {l[1]}", l[2], f"{l[3]}<br><b>الأدوات:</b> {l[4]}") for l in LEVELS], start=0)
    ui.note("هذا التصنيف التعليمي مأخوذ من **موسوعة د. مروان رودان (2026)**، ويُستخدم لتنظيم المحاضرة. ليس سلّم جودة: المستوى الأعلى ليس «أفضل» بالضرورة؛ "
            "فمشروع L2 بسيط قد يكون أنسب علميًا من مشروع L6 معقّد.", "ملاحظة تصنيفية مهمة", "amber", "📌")

with tabs[1]:
    ui.section("استكشف أي مستوى بالتفصيل", "Level explorer", "sky")
    pick = st.segmented_control("اختر المستوى", [l[0] for l in LEVELS], default="L3", key="lvl_pick")
    L = next(l for l in LEVELS if l[0] == pick)
    idx = [l[0] for l in LEVELS].index(pick)
    color = ui.color_at(idx)
    with ui.tinted(color):
        st.markdown(f"### {L[0]} — {L[1]}  \n*{L[2]}*")
        st.markdown(f"**الوصف:** {L[3]}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**الأدوات والخوارزميات:** {L[4]}")
            st.markdown(f"**موضع الذكاء الاصطناعي:** {L[6]}")
        with c2:
            st.markdown(f"**مراجع مفتاحية:** {L[5]}")
            st.markdown(f"**المخاطر الرئيسية:** {L[7]}")
    fig = go.Figure(go.Funnel(
        y=[f"{l[0]} {l[1]}" for l in LEVELS][::-1], x=[8 - i for i in range(8)][::-1],
        marker=dict(color=[ui.PLOT_COLORS[i % 12] if l[0] != pick else "#111827" for i, l in enumerate(LEVELS)][::-1]),
        textinfo="label", hoverinfo="y",
    ))
    ui.style_fig(fig, "موقع المستوى المختار في هرم الدمج (اللون الداكن)", height=470, legend_h=False)
    fig.update_yaxes(showticklabels=False)
    ui.show(fig)

with tabs[2]:
    ui.section("الخريطة الشمسية: المستويات وأدواتها", "Sunburst of levels → methods", "violet")
    labels, parents, colors = ["AI × Econometrics"], [""], ["#ffffff"]
    for i, l in enumerate(LEVELS):
        node = f"{l[0]} {l[2]}"
        labels.append(node); parents.append("AI × Econometrics"); colors.append(ui.PLOT_COLORS[i % 12])
        for tool in [t.strip() for t in l[4].replace("،", ",").split(",")][:6]:
            labels.append(f"{tool} ({l[0]})"); parents.append(node); colors.append(ui.rgba(ui.PLOT_COLORS[i % 12], 0.6))
    fig = go.Figure(go.Sunburst(labels=labels, parents=parents, marker=dict(colors=colors), insidetextorientation="radial", maxdepth=2))
    ui.style_fig(fig, "انقر أي قطاع للتكبير", height=680, legend_h=False)
    ui.show(fig)

with tabs[3]:
    ui.section("ملاءمة كل مستوى لكل نوع من البيانات", "Levels × data structures (qualitative)", "teal")
    data_types = ["مقطعية", "سلاسل زمنية", "بانل", "مكانية", "شبكات", "تواتر مختلط", "نص/صورة", "مالية عالية التواتر"]
    M = [
        [3, 3, 3, 3, 3, 3, 3, 3],
        [3, 2, 3, 2, 2, 3, 3, 2],
        [2, 3, 2, 2, 1, 3, 1, 3],
        [3, 2, 3, 2, 1, 1, 2, 1],
        [2, 2, 2, 2, 2, 2, 3, 2],
        [2, 2, 1, 1, 1, 1, 1, 3],
        [1, 2, 1, 1, 1, 1, 1, 2],
        [2, 3, 2, 1, 1, 3, 3, 2],
    ]
    fig = go.Figure(go.Heatmap(z=M, x=data_types, y=[f"{l[0]} {l[1]}" for l in LEVELS], colorscale=[[0, "#fef3c7"], [.5, "#a7f3d0"], [1, "#7c3aed"]],
                               text=[[["منخفض", "متوسط", "مرتفع"][v - 1] for v in row] for row in M],
                               texttemplate="%{text}", showscale=False))
    fig.update_yaxes(autorange="reversed")
    ui.style_fig(fig, "تقدير نوعي لكثافة الأدبيات/الملاءمة (حكم تعليمي لا قياس كمي)", height=520, legend_h=False)
    fig.update_layout(margin=dict(l=210))
    ui.show(fig)
    ui.note("هذه المصفوفة **حكم نوعي** لمساعدة الباحث على توقع حجم الأدبيات والأدوات الجاهزة؛ الخلايا «المنخفضة» هي غالبًا **فجوات بحثية** وفرص لتطوير خوارزميات جديدة "
            "(مثل الدمج النظري في البانل أو الشبكات الموجّهة بالنظرية للبيانات المكانية).", "كيف تقرأها؟", "teal", "🔎")

with tabs[4]:
    ui.section("معماريات الدمج: كيف نركّب المكوّنات؟", "Integration architectures", "orange")
    arch = st.segmented_control("المعمارية", ["تتابعية", "متوازية", "بواقي", "تحسين مشترك", "معلّم–طالب", "تنافسية", "حلقة مغلقة"], default="بواقي", key="arch")
    diagrams = {
        "تتابعية": "flowchart LR\n A[Raw data] --> B[ML step: measure / select / embed] --> C[Econometric model] --> D[Inference]",
        "متوازية": "flowchart LR\n A[Data] --> B[Econometric forecast]\n A --> C[ML forecast]\n B --> D[Weights / stacking]\n C --> D --> E[Combined forecast]",
        "بواقي": "flowchart LR\n A[Data] --> B[Econometric model e.g. ARIMA, VAR, ARDL] --> C[Residuals out-of-sample] --> D[ML learns residual pattern] --> E[Forecast = econ + ML correction]\n B --> E",
        "تحسين مشترك": "flowchart LR\n A[Data] --> B[Joint loss: L_data + lambda L_econ]\n B --> C[Network weights w]\n B --> D[Structural parameters theta]\n C --> E[Fitted, theory-consistent model]\n D --> E",
        "معلّم–طالب": "flowchart LR\n A[Structural model teacher] --> B[Synthetic data] --> C[Pretrain network] --> D[Fine-tune on real data] --> E[Student model]",
        "تنافسية": "flowchart LR\n A[Structural model with theta] --> B[Simulated data]\n C[Real data] --> D{Discriminator}\n B --> D --> E[Update theta to fool discriminator] --> A",
        "حلقة مغلقة": "flowchart LR\n A[Policy / agent] --> B[Economy or simulator] --> C[Outcomes and rewards] --> D[Learning update] --> A",
    }
    notes = {
        "تتابعية": "الأكثر شيوعًا: AI يقيس أو يختار ثم يأتي النموذج القياسي. **الخطر**: تجاهل خطأ المرحلة الأولى (المتغيرات المولّدة).",
        "متوازية": "توليف التنبؤات (Forecast combination) — قديم في الاقتصاد (Bates & Granger, 1969). **الخطر**: ضبط الأوزان داخل العينة نفسها.",
        "بواقي": "نموذج تشانغ (2003) الهجين: الجزء الخطي للقياس، والجزء اللاخطي للشبكة على البواقي. **الخطر**: تعلم بواقي داخل العينة يضخم النتائج.",
        "تحسين مشترك": "SKINNs والشبكات الموجّهة بالنظرية: الشبكة والمعلمات الاقتصادية تُقدَّر معًا. **الخطر**: اختيار λ يغيّر الاستنتاج.",
        "معلّم–طالب": "Teaching Economics to the Machines (2026): التدريب المسبق على عالم نظري ثم التكييف. **الخطر**: انتقال أخطاء النظرية.",
        "تنافسية": "التقدير التنافسي (Kaji et al., 2023) وDeepGMM/AGMM. **الخطر**: عدم استقرار الألعاب التنافسية.",
        "حلقة مغلقة": "التعلم المعزز والتجارب التكيفية. **الخطر**: الأداء في المحاكاة لا يضمن الأداء في الاقتصاد الحقيقي.",
    }
    st.mermaid_chart(diagrams[arch])
    ui.note(notes[arch], f"المعمارية: {arch}", "orange", "🏗️")

with tabs[5]:
    ui.section("خمسة محاور أخرى لتصنيف أي مشروع دمج", "Five further axes", "indigo")
    ui.cards([
        {"title": "حسب البنية", "en": "Architecture", "color": "indigo",
         "body": "تتابعية، متوازية، بواقي، تحسين مشترك، معلّم–طالب، متعددة المهام، تنافسية، حلقة مغلقة."},
        {"title": "حسب دور النظرية", "en": "Role of theory", "color": "violet",
         "body": "بلا قيد، عقوبة مرنة (Soft penalty)، قيد صلب (Hard constraint)، قبْلي تدريب مسبق (Pretraining prior)، إسقاط لاحق (Post-hoc projection)."},
        {"title": "حسب مصدر المعرفة", "en": "Knowledge source", "color": "teal",
         "body": "بيانات حقيقية، نظرية رياضية، محاكاة هيكلية، نماذج مدربة مسبقًا، وثائق، معرفة خبراء."},
        {"title": "حسب نوع البيانات", "en": "Data type", "color": "orange",
         "body": "مقطعية، زمنية، بانل، مكانية، شبكية، تواتر مختلط، نص، صورة، متعددة الوسائط."},
        {"title": "حسب غاية البحث", "en": "Research goal", "color": "rose",
         "body": "تنبؤ، استدلال سببي، قياس، تقدير هيكلي، تعلم سياسات، توليد بيانات."},
    ], cols=5)
    ui.section("مثال: تصنيف ثلاثة أبحاث على المحاور الستة", "Worked classification", "slate")
    ex = pd.DataFrame([
        ("Chernozhukov et al. (2018) DML", "L3", "تتابعية + توفيق متقاطع", "بلا قيد نظري", "بيانات حقيقية", "مقطعية", "استدلال سببي"),
        ("Azinovic et al. (2022) DEQN", "L6", "حلقة تدريب على بواقي التوازن", "قيد في الخسارة", "نظرية + محاكاة", "حالات نموذج", "حل هيكلي"),
        ("Casella et al. (2026) نص FOMC في DSGE", "L4 + L6", "تتابعية داخل فضاء الحالة", "قبْلي spike-and-slab", "نص + بيانات كلية", "زمنية + نص", "تقدير هيكلي"),
    ], columns=["البحث", "المستوى", "البنية", "دور النظرية", "مصدر المعرفة", "البيانات", "الغاية"])
    st.table(ex)

with tabs[6]:
    ui.section("عمق إدخال المعرفة الاقتصادية في الخوارزمية", "How deep does economic knowledge go?", "emerald")
    depth = ["AI بعد النموذج", "AI داخل التقدير", "النظرية داخل الخسارة", "النظرية داخل المعمارية"]
    exs = ["تصحيح البواقي، التوليف", "DML، Deep IV", "PINNs، SKINNs، بواقي أويلر", "SDF بلا مراجحة، DEQN، الكتل الجمعية"]
    fig = go.Figure(go.Bar(x=[1, 2, 3, 4], y=depth, orientation="h", text=exs, textposition="inside",
                           marker=dict(color=[ui.COLORS[c] for c in ["sky", "teal", "violet", "fuchsia"]])))
    fig.update_xaxes(title="عمق إدخال المعرفة الاقتصادية →", showticklabels=False)
    ui.style_fig(fig, "اتجاه توضيحي لعمق المعرفة — ليس ترتيبًا للجودة أو الدقة", height=380, legend_h=False)
    ui.show(fig)
