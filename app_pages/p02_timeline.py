import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui
from core.timeline_data import EVENTS, LANE_AR, LANES

ui.hero(
    "الخط الزمني للدمج وموجاته الست",
    "Timeline of AI × Econometrics — From 1933 to 2026",
    "تتبّع تفاعلي لأكثر من سبعين محطة في ستة مسارات متوازية: القياس الاقتصادي، الذكاء الاصطناعي، البيانات، "
    "الدمج المنهجي، التعلم المعزز، والذكاء التوليدي. اضغط ▶ Play لترى كيف تتقارب المسارات حتى تلتقي.",
    "sky", "fuchsia",
)

tabs = st.tabs([
    ":material/play_circle: الخط الزمني المتحرك",
    ":material/waves: موجات الدمج الست",
    ":material/article: الأوراق التأسيسية",
    ":material/compare: مقارنة العصور",
    ":material/view_timeline: مخطط زمني",
])

df = pd.DataFrame(EVENTS, columns=["year", "lane", "label", "ar"])
df["y"] = df["lane"].map({l: i for i, l in enumerate(LANES)})
rng = np.random.default_rng(3)
df["yj"] = df["y"] + rng.uniform(-0.28, 0.28, len(df))
lane_color = dict(zip(LANES, [ui.COLORS[c] for c in ["amber", "violet", "teal", "rose", "emerald", "sky"]]))

# ------------------------------------------------------------ animated -------
with tabs[0]:
    ui.section("اضغط تشغيل: تقارب المسارات عبر الزمن", "Animated convergence of six tracks", "sky")
    steps = list(range(1935, 2031, 5))

    def lane_traces(upto: int) -> list:
        out = []
        for lane in LANES:
            s = df[(df.lane == lane) & (df.year <= upto)]
            out.append(go.Scatter(
                x=s.year, y=s.yj, mode="markers+text", name=LANE_AR[lane],
                text=s.label, textposition="top center", textfont=dict(size=9, color=lane_color[lane]),
                marker=dict(size=13, color=lane_color[lane], line=dict(width=1.5, color="white"), opacity=.9),
                customdata=s.ar, hovertemplate="<b>%{text}</b> (%{x})<br>%{customdata}<extra></extra>",
            ))
        return out

    fig = go.Figure(data=lane_traces(steps[0]),
                    frames=[go.Frame(data=lane_traces(y), name=str(y)) for y in steps])
    fig.update_yaxes(tickvals=list(range(len(LANES))), ticktext=[LANE_AR[l] for l in LANES], range=[-0.7, len(LANES) - 0.3])
    fig.update_xaxes(range=[1928, 2030], dtick=10, title="السنة")
    ui.style_fig(fig, "محطات الدمج بين الذكاء الاصطناعي والقياس الاقتصادي", height=640)
    fig.update_layout(margin=dict(l=190))
    ui.add_play(fig, steps, duration=700, prefix="حتى سنة: ", redraw=True)
    ui.show(fig)
    ui.note("لاحظ كيف يبقى مسار **«الدمج المنهجي»** شبه فارغ حتى نهاية الثمانينيات، ثم تظهر نقاط متفرقة (وايت، تشانغ، GETS)، "
            "ثم ينفجر بعد **2014** عندما تلتقي ثلاثة عوامل: البيانات الضخمة، والتعلم العميق، ونظرية الاستدلال بعد الاختيار (PDS/DML).",
            "قراءة الرسم", "sky", "👁️")
    with st.expander("جدول كل المحطات"):
        st.dataframe(df[["year", "lane", "label", "ar"]].rename(columns={"year": "السنة", "lane": "المسار", "label": "المحطة", "ar": "الشرح"}),
                     hide_index=True)

# ---------------------------------------------------------------- waves ------
with tabs[1]:
    ui.section("ست موجات للدمج", "Six waves of integration", "fuchsia")
    st.markdown("نقترح تحقيبًا (Periodization) للدمج في ست موجات متداخلة؛ لكل موجة **سؤال مهيمن** و**أداة مهيمنة** و**نوع بيانات** و**مشكلة استدلال**.")
    ui.ladder([
        ("الموجة 0 — ما قبل الدمج", "1930–1985",
         "قياس اقتصادي معلمي، نماذج آنية، ARIMA/VAR، معايير معلوماتية. الذكاء الاصطناعي رمزي ومنفصل تمامًا."),
        ("الموجة 1 — الشبكات العصبية واللامعلمي", "1986–2000",
         "وايت وتلامذته: الشبكات كنماذج انحدار لا خطية؛ اختبارات اللاخطية؛ تقدير الغربال والنواة؛ أعوان متعلمون في سانتا في؛ GETS."),
        ("الموجة 2 — البيانات الضخمة والتنظيم", "2000–2013",
         "نماذج العوامل، لاسو والشبكة المرنة، التنبؤ الآني، Google Trends، النص كبيانات المالية، الأقمار الصناعية، الهجين ARIMA–ANN."),
        ("الموجة 3 — تعلم الآلة السببي", "2014–2019",
         "فاريان (2014)، PDS (2014)، الأشجار والغابات السببية (2016–2019)، DML (2018)، تعلم السياسات (2018–2021)، Athey & Imbens (2019)."),
        ("الموجة 4 — التعلم العميق والبيانات غير المهيكلة", "2018–2022",
         "تسعير الأصول بالشبكات (GKX 2020)، الاستدلال بالشبكات العميقة (2021)، حل النماذج الهيكلية بالشبكات، BERT للنصوص الاقتصادية، الغابات الكلية."),
        ("الموجة 5 — الذكاء التوليدي والوكلاء", "2022–2026",
         "LLMs كأدوات قياس ومحاكاة (Homo silicus)، الاستدلال الصالح بالمتغيرات المولّدة (PPI, DSL)، تحيّز النظر للأمام، نماذج الأساس للسلاسل الزمنية، وكلاء بحث قابلون للتدقيق."),
    ], start=0)
    waves = pd.DataFrame({
        "الموجة": ["0", "1", "2", "3", "4", "5"],
        "بداية": [1930, 1986, 2000, 2014, 2018, 2022],
        "نهاية": [1986, 2000, 2014, 2019, 2022, 2026],
        "الاسم": ["ما قبل الدمج", "الشبكات واللامعلمي", "البيانات الضخمة والتنظيم", "تعلم الآلة السببي",
                  "التعلم العميق وغير المهيكل", "الذكاء التوليدي والوكلاء"],
    })
    figw = go.Figure()
    for i, r in waves.iterrows():
        figw.add_trace(go.Bar(x=[r["نهاية"] - r["بداية"]], y=[r["الاسم"]], base=[r["بداية"]], orientation="h",
                              marker_color=ui.PLOT_COLORS[i], name=r["الاسم"], showlegend=False,
                              hovertemplate=f"{r['الاسم']}: {r['بداية']}–{r['نهاية']}<extra></extra>"))
    figw.update_yaxes(autorange="reversed")
    ui.style_fig(figw, "المدى الزمني لكل موجة", height=360)
    ui.show(figw)

# --------------------------------------------------------- foundational ------
with tabs[2]:
    ui.section("الأوراق التأسيسية التي يجب أن يعرفها كل باحث", "Must-know foundational papers", "violet")
    papers = pd.DataFrame([
        ("Hornik, Stinchcombe & White", 1989, "Neural Networks", "التقريب الشامل بالشبكات", "تبرير المرونة الدالية"),
        ("Breiman", 2001, "Statistical Science", "الثقافتان في النمذجة الإحصائية", "التنبؤ الخوارزمي مقابل نمذجة البيانات"),
        ("Varian", 2014, "J. Econ. Perspectives", "حيل جديدة للاقتصاد القياسي", "بيان الدمج الأول للجمهور الاقتصادي"),
        ("Belloni, Chernozhukov & Hansen", 2014, "Rev. Econ. Studies", "الاستدلال بعد الاختيار المزدوج", "لاسو مع استدلال صالح"),
        ("Kleinberg, Ludwig, Mullainathan & Obermeyer", 2015, "AER P&P", "مسائل السياسات التنبؤية", "متى يكون التنبؤ هو السؤال"),
        ("Athey & Imbens", 2016, "PNAS", "التقسيم التكراري للأثر غير المتجانس", "الأشجار السببية الأمينة"),
        ("Mullainathan & Spiess", 2017, "J. Econ. Perspectives", "تعلم الآلة: مقاربة قياسية تطبيقية", "ŷ مقابل β̂"),
        ("Chernozhukov et al.", 2018, "Econometrics Journal", "Double/Debiased ML", "التعامد + التوفيق المتقاطع"),
        ("Wager & Athey", 2018, "JASA", "الغابات السببية والاستدلال", "توزيع طبيعي تقاربي للغابات"),
        ("Athey & Imbens", 2019, "Annual Review of Econ.", "طرق تعلم الآلة للاقتصاديين", "مراجعة مرجعية شاملة"),
        ("Athey, Tibshirani & Wager", 2019, "Annals of Statistics", "الغابات العشوائية المعممة", "إطار GRF"),
        ("Gu, Kelly & Xiu", 2020, "Rev. Financial Studies", "تسعير الأصول تجريبيًا بتعلم الآلة", "تفوق الشبكات والأشجار"),
        ("Farrell, Liang & Misra", 2021, "Econometrica", "الشبكات العميقة للتقدير والاستدلال", "معدلات تقارب للشبكات"),
        ("Athey & Wager", 2021, "Econometrica", "تعلم السياسات بالبيانات المشاهدة", "قواعد معالجة مثلى"),
        ("Goulet Coulombe et al.", 2022, "J. Applied Econometrics", "كيف يفيد تعلم الآلة في التنبؤ الكلي؟", "اللاخطية مفتاح الكسب"),
        ("Korinek", 2023, "J. Econ. Literature", "الذكاء التوليدي للبحث الاقتصادي", "ست فئات استخدام"),
        ("Dell", 2025, "J. Econ. Literature", "التعلم العميق للاقتصاديين", "دليل عملي شامل"),
        ("Ludwig, Mullainathan & Rambachan", 2025, "NBER WP 33344", "LLMs: إطار قياسي تطبيقي", "التنبؤ مقابل التقدير، التسرب، عينة التحقق"),
    ], columns=["المؤلفون", "السنة", "المجلة", "العنوان (معرّب)", "الإسهام"])
    st.dataframe(papers, hide_index=True, height=640)
    ui.refs("القائمة الكاملة بروابط DOI في صفحة «المراجع والقراءات».")

# -------------------------------------------------------------- compare ------
with tabs[3]:
    ui.section("ماذا تغيّر من عصر إلى عصر؟", "What changed across eras?", "emerald")
    comp = pd.DataFrame([
        ("السؤال المهيمن", "ما قيمة β؟", "هل العلاقة لا خطية؟", "كيف نتنبأ بمئات المتغيرات؟", "ما الأثر السببي ولمن؟", "كيف نقيس من النص والصورة؟", "كيف نستخدم LLMs بأمان؟"),
        ("البيانات", "جداول صغيرة", "سلاسل زمنية متوسطة", "لوحات كبيرة، إنترنت", "إدارية وتجريبية كبيرة", "نص، صور، أقمار", "كل شيء + مولَّد بالآلة"),
        ("الأداة", "OLS / IV / MLE", "شبكات ضحلة، نواة", "لاسو، عوامل", "DML، غابات سببية", "شبكات عميقة، BERT", "LLMs، وكلاء، نماذج أساس"),
        ("الاستدلال", "نظرية تقاربية معلمية", "تقارب لا معلمي", "نادرًا ما يُناقش", "دوال متعامدة", "متغيرات مولَّدة", "PPI / DSL / تصحيح التحيز"),
        ("الخطر الرئيسي", "سوء المواصفة", "الإفراط في التوفيق", "تسرب، عدم استقرار", "انتهاك التداخل", "خطأ القياس", "تسرب، هلوسة، درجات حرية"),
    ], columns=["البعد", "الموجة 0", "الموجة 1", "الموجة 2", "الموجة 3", "الموجة 4", "الموجة 5"])
    st.table(comp)

# -------------------------------------------------------------- mermaid ------
with tabs[4]:
    ui.section("مخطط زمني مكثف", "Condensed timeline diagram", "orange")
    st.mermaid_chart(
        """
timeline
    title AI x Econometrics
    1930-1985 : Econometric Society : Haavelmo : Cowles SEM : Box-Jenkins : Lucas & Sims critiques
    1986-2000 : Backprop : White neural nets : Universal approximation : NN nonlinearity test : GETS
    2000-2013 : LASSO & forests : Factor models : Nowcasting : Text & night lights : Hybrid ARIMA-ANN
    2014-2019 : Varian new tricks : Post-double-selection : Causal trees & forests : Double ML : Policy learning
    2018-2022 : Deep learning asset pricing : Deep nets inference : Solving models with DL : Macro random forest
    2022-2026 : LLMs & homo silicus : Valid inference with AI labels : TS foundation models : Auditable AI agents
"""
    )
