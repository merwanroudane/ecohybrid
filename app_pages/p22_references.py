import importlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import refs_data, ui

# reload so content pushed to the repo shows without restarting the app
R = importlib.reload(refs_data).R

ui.hero(
    "المراجع والقراءات",
    "Bibliography, Reference Books and Reading Paths",
    f"{len(R)} مرجعًا مصنّفة حسب المحاور، مع روابط DOI (تم التحقق من وجودها ومطابقة عناوينها آليًا عبر doi.org و Crossref) "
    "وحالة كل مرجع (محكّم، ورقة عمل، أولي، كتاب). مراجع 2026 غير الموثقة برابط مأخوذة من موسوعة د. مروان رودان كما صنّفها.",
    "amber", "sky",
)

df = pd.DataFrame(R, columns=["المحور", "المؤلفون", "السنة", "العنوان", "المصدر", "الرابط", "الحالة"])
df["الرابط"] = df["الرابط"].apply(lambda s: f"https://doi.org/{s}" if s.startswith("10.") else s)

tabs = st.tabs([":material/menu_book: كل المراجع", ":material/library_books: الكتب المرجعية", ":material/bar_chart: خريطة الأدبيات", ":material/route: مسارات القراءة"])

with tabs[0]:
    c1, c2, c3 = st.columns([2, 1, 1])
    q = c1.text_input("ابحث بالمؤلف أو العنوان", key="ref_q")
    cats = c2.multiselect("المحور", list(dict.fromkeys(df["المحور"])), key="ref_c")
    stt = c3.multiselect("الحالة", sorted(df["الحالة"].unique()), key="ref_s")
    sub = df.copy()
    if q:
        sub = sub[sub.apply(lambda r: q.lower() in (str(r["المؤلفون"]) + " " + str(r["العنوان"])).lower(), axis=1)]
    if cats:
        sub = sub[sub["المحور"].isin(cats)]
    if stt:
        sub = sub[sub["الحالة"].isin(stt)]
    st.caption(f"{len(sub)} مرجعًا")
    st.dataframe(sub.sort_values(["المحور", "السنة"]), hide_index=True, height=640,
                 column_config={"الرابط": st.column_config.LinkColumn("الرابط", display_text="فتح ↗"),
                                "السنة": st.column_config.NumberColumn(format="%d"),
                                "العنوان": st.column_config.TextColumn(width="large")})

with tabs[1]:
    ui.section("الكتب التي يستند إليها الدليل", "Reference books used in this guide", "amber")
    ui.cards([
        {"title": "Machine Learning for Econometrics", "en": "Gaillac & L'Hour (2025), Oxford University Press", "color": "violet",
         "body": "الاستدلال بعد الاختيار، الأبعاد العالية والداخلية، الأثر غير المتجانس، تعلم السياسات المثلى، التحكم التركيبي، التنبؤ عالي الأبعاد، النصوص والتضمينات ونماذج اللغة الحديثة، مع تمارين."},
        {"title": "Econometrics with Machine Learning", "en": "Mátyás & Chan, eds. (2022), Springer", "color": "teal",
         "body": "نماذج خطية ولاخطية منظّمة، آثار المعالجة، التنبؤ، TMLE، اقتصاد الشبكات، العدالة، النماذج الرسومية، الفقر والتنمية، تسعير الأصول، وملحق مصطلحات."},
        {"title": "Causal Inference and Machine Learning", "en": "Yuksel & Aydede, CRC Press", "color": "orange",
         "body": "مقرر كامل: التحيز والتباين، الإفراط في التوفيق، الضبط، الانحدار المنظّم، الأشجار والتعزيز، DML و AIPW و DML-IV، الغابات السببية، المتعلمات الوصفية، DML-DiD، SDID و RDD، التنبؤ بالغابات، والشبكات."},
        {"title": "Statistical Learning for Big Dependent Data", "en": "Peña & Tsay (2021), Wiley", "color": "rose",
         "body": "البيانات المعتمدة زمنيًا: VAR و VARMA والتكامل المشترك، العوامل الديناميكية، لاسو للبيانات المعتمدة، التنبؤ الآني، الغابات والشبكات والتعلم العميق والتعلم المعزز، والبيانات الزمانية-المكانية."},
        {"title": "Macroeconomic Forecasting in the Era of Big Data", "en": "Fuleky, ed. (2020), Springer", "color": "sky",
         "body": "مصادر البيانات الضخمة، DFM و FAVAR و BVAR الكبيرة، التقلب، الشبكات العصبية، الانحدار المنظّم، اختيار المتغيرات، متوسط النماذج، الغابات والتعزيز، التنبؤ الكثيفي وتقييمه، جذور الوحدة والتكامل المشترك، نقاط التحول، المجال الترددي."},
        {"title": "Econometrics and Data Science", "en": "Nokeri (2022), Apress", "color": "emerald",
         "body": "مدخل تطبيقي بـ Python: الانحدار بـ statsmodels و scikit-learn، التنبؤ بالنمو، التصنيف اللوجستي، HMM و GMM، العناقيد، الشبكات العصبية، المحاكاة، ونماذج المعادلات الهيكلية."},
        {"title": "الموسوعة الأكاديمية للدمج (2026)", "en": "Dr Merwan Roudane — 39 chapters, 75 references", "color": "fuchsia",
         "body": "موسوعة نظرية عربية من الصفر حتى أكتوبر 2026: مستويات الدمج الثمانية، الشبكات الموجّهة اقتصاديًا وماليًا، المتغيرات المولّدة وأخطاؤها، النماذج التأسيسية، الحالات، النقد، الأجندة، المعجم وقائمة التحقق."},
        {"title": "Applied Causal Inference Powered by ML and AI", "en": "Chernozhukov, Hansen, Kallus, Spindler & Syrgkanis (2024)", "color": "amber",
         "body": "كتاب مفتوح (causalml-book.org): SEMs و DAGs و SCMs، DML للاستدلال، عدم التجانس، IV و DiD و RDD مع ML."},
    ], cols=2)

with tabs[2]:
    by = df.groupby(["السنة", "المحور"]).size().reset_index(name="n")
    fig = go.Figure()
    for i, (cat, g) in enumerate(by.groupby("المحور")):
        fig.add_trace(go.Bar(x=g["السنة"], y=g["n"], name=cat, marker_color=ui.PLOT_COLORS[i % 12]))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="سنة النشر"); fig.update_yaxes(title="عدد المراجع في هذا الدليل")
    ui.style_fig(fig, "توزيع مراجع الدليل عبر الزمن والمحاور (يعكس اختيار الدليل لا حجم الأدبيات الكلي)", height=480)
    ui.show(fig)
    st.dataframe(df["الحالة"].value_counts().rename_axis("الحالة").reset_index(name="العدد"), hide_index=True)

with tabs[3]:
    ui.section("مسارات قراءة متدرجة", "Graded reading paths", "emerald")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.note("1. Varian (2014)\n2. Mullainathan & Spiess (2017)\n3. Athey & Imbens (2019)\n4. Breiman (2001)\n5. Hastie et al., ESL (الفصول 1–7)\n6. Yuksel & Aydede (الفصول 1–15)",
                "المسار التمهيدي", "emerald", "🌱")
    with c2:
        ui.note("1. Belloni et al. (2014)\n2. Chernozhukov et al. (2018)\n3. Wager & Athey (2018)\n4. Athey & Wager (2021)\n5. Gaillac & L'Hour (2025)\n6. Chernozhukov et al. (2024) CausalML book",
                "مسار الاستدلال السببي", "violet", "🎯")
    with c3:
        ui.note("1. Stock & Watson (2002)\n2. Goulet Coulombe et al. (2022)\n3. Babii et al. (2022)\n4. Smeekes & Wijler (2021)\n5. Peña & Tsay (2021)\n6. Fuleky (2020)",
                "مسار السلاسل الزمنية والتنبؤ", "sky", "📈")
    c4, c5, c6 = st.columns(3)
    with c4:
        ui.note("1. Maliar et al. (2021)\n2. Azinovic et al. (2022)\n3. Kaji et al. (2023)\n4. Igami (2020)\n5. Fernández-Villaverde et al. (2023)\n6. موسوعة رودان، الفصول 11–15",
                "مسار الهيكلي والموجّه بالنظرية", "fuchsia", "🏛️")
    with c5:
        ui.note("1. Gentzkow, Kelly & Taddy (2019)\n2. Korinek (2023)\n3. Angelopoulos et al. (2023)\n4. Battaglia et al. (2024)\n5. Ludwig, Mullainathan & Rambachan (2025)\n6. Dell (2025)",
                "مسار النص والذكاء التوليدي", "orange", "🤖")
    with c6:
        ui.note("1. Rust (1987)\n2. Calvano et al. (2020)\n3. Kasy & Sautmann (2021)\n4. Charpentier et al. (2023)\n5. Zheng et al. (2022)\n6. Peña & Tsay (2021) ف 8.6",
                "مسار التعلم المعزز", "lime", "🎮")
