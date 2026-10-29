import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Lasso, Ridge

from core import ui

ui.hero(
    "ثورة البيانات الضخمة والبيانات غير المهيكلة",
    "The Big Data Revolution & Unstructured Data in Economics",
    "الدمج لم يبدأ بالخوارزميات بل بالبيانات: عندما أصبحت البيانات «أطول» (n ضخم) و«أعرض» (p ضخم) و«أغنى» "
    "(نص، صورة، صوت، موقع جغرافي)، عجزت الأدوات الكلاسيكية وظهرت الحاجة إلى الذكاء الاصطناعي.",
    "teal", "amber",
)

tabs = st.tabs([
    ":material/dataset: ما البيانات الضخمة؟",
    ":material/trending_up: ثورة البيانات في الاقتصاد",
    ":material/category: أنواع غير المهيكلة",
    ":material/description: جذور «النص كبيانات»",
    ":material/satellite_alt: جذور الصورة والأقمار",
    ":material/science: مختبر: لعنة الأبعاد",
    ":material/report_problem: التحديات والدروس",
])

with tabs[0]:
    ui.section("تعريف البيانات الضخمة بعين الاقتصادي", "Defining Big Data for economists", "teal")
    ui.cards([
        {"title": "الحجم", "en": "Volume", "icon": "📦", "body": "ملايين أو مليارات المشاهدات: معاملات بطاقات، سجلات ضريبية، بيانات ماسحات الأسعار."},
        {"title": "السرعة", "en": "Velocity", "icon": "⚡", "body": "تدفق آني (Real-time): أسعار يومية من الإنترنت، بحث Google، بيانات الهاتف المحمول."},
        {"title": "التنوع", "en": "Variety", "icon": "🎨", "body": "نصوص، صور، صوت، فيديو، شبكات، مواقع جغرافية — لا تأتي في جداول."},
        {"title": "الموثوقية", "en": "Veracity", "icon": "🔍", "body": "ضوضاء، انتقائية، عدم تمثيل، تغيّر الخوارزميات المولّدة للبيانات."},
        {"title": "القيمة", "en": "Value", "icon": "💎", "body": "القدرة على الإجابة عن أسئلة لم تكن ممكنة: قياس الفقر من الفضاء، التضخم يوميًا."},
    ], cols=5)
    ui.section("طويلة، عريضة، غنية", "Tall, wide (fat) and rich data", "violet")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.note("$n$ ضخم جدًا و $p$ صغير. **المشكلة**: الحساب، لا الإحصاء. أي فرق صغير يصبح «دالًا» ⇒ الدلالة الاقتصادية أهم من الإحصائية.",
                "بيانات طويلة (Tall)", "teal", "📏")
    with c2:
        ui.note("$p$ كبير مقارنة بـ $n$ أو أكبر منه $p \\gg n$. **المشكلة**: OLS غير معرّف/متفجر التباين ⇒ التنظيم، الاختيار، العوامل.",
                "بيانات عريضة (Wide / Fat)", "violet", "↔️")
    with c3:
        ui.note("المعلومة مخبأة في نص أو صورة. **المشكلة**: تحويل غير المهيكل إلى متغير $\\hat z = f(\\text{text})$ ثم استدلال صالح عليه.",
                "بيانات غنية (Rich / Unstructured)", "orange", "🖼️")
    st.markdown(
        """
يلخّص **هال فاريان (2014)** الفكرة: البيانات الضخمة تتطلب «**حيلًا جديدة**» — أدوات للتعامل مع الحجم (قواعد بيانات موزعة)،
ولاختيار المتغيرات (لاسو، الأشجار)، وللتعامل مع العلاقات غير الخطية — مع تذكير بأن **السببية تبقى السؤال الأهم**.
        """
    )
    ui.chips([("Volume", "الحجم"), ("Velocity", "السرعة"), ("Variety", "التنوع"), ("Veracity", "الموثوقية"),
              ("High-dimensional", "عالي الأبعاد"), ("Sparsity", "التناثر"), ("Real-time data", "بيانات آنية")])

with tabs[1]:
    ui.section("كيف تغيّرت البيانات المتاحة للاقتصاديين؟", "The data revolution in economics", "orange")
    st.markdown(
        "لاحظ **آينف وليفن (2014)** أنه قبل 15–20 سنة كانت مجموعات البيانات المثيرة غير المدروسة **موردًا نادرًا**، وأن نسبة الأوراق "
        "المنشورة التي تستخدم بيانات إدارية أو خاصة كانت صغيرة (نحو 5–10% في 2006–2008) ثم ارتفعت بقوة. وهذه البيانات ليست مجرد نسخ أكبر "
        "من البيانات التقليدية؛ فهي **آنية** و**دقيقة الحبيبات** (Granular) و**غير مصمَّمة للبحث** أصلًا."
    )
    ui.cards([
        {"title": "البيانات الإدارية", "en": "Administrative data", "icon": "🏢",
         "body": "سجلات ضريبية وتأمينات (تشيتي وآخرون عن الحراك الاجتماعي)، سجلات تعليمية وصحية بملايين الأفراد."},
        {"title": "بيانات ماسحات الأسعار", "en": "Scanner data", "icon": "🛒", "body": "مبيعات وأسعار المتاجر على مستوى المنتج والأسبوع — مرونات الطلب وقوة السوق."},
        {"title": "أسعار الإنترنت", "en": "Billion Prices Project", "icon": "🌐", "body": "كافالو وريغوبون (2016): كشط أسعار المتاجر الإلكترونية يوميًا لقياس التضخم."},
        {"title": "المعاملات المالية", "en": "Card & payment data", "icon": "💳", "body": "استهلاك آني على مستوى المدن والقطاعات — ازدهر خلال جائحة 2020."},
        {"title": "بحث الإنترنت", "en": "Google Trends", "icon": "🔎", "body": "تشوي وفاريان (2012): «التنبؤ بالحاضر» للبطالة ومبيعات السيارات والسفر."},
        {"title": "الهاتف والموقع", "en": "Mobile & geolocation", "icon": "📱", "body": "التنقل، التباعد الاجتماعي، الشبكات الاجتماعية، الأسواق المحلية."},
        {"title": "الأقمار الصناعية", "en": "Remote sensing", "icon": "🛰️", "body": "الأضواء الليلية، الغطاء الزراعي، التلوث، البناء — قياس حيث لا توجد إحصاءات."},
        {"title": "المنصات الرقمية", "en": "Platform data", "icon": "🧑‍💻", "body": "أوبر، أمازون، إير بي إن بي: تجارب A/B ضخمة وأسواق ثنائية الجانب."},
    ], cols=4)
    ui.note("البيانات الضخمة **لم تُصمَّم للبحث**: ⇒ الحاجة إلى أدوات تتعامل مع الضوضاء والأبعاد العالية والأشكال غير المهيكلة "
            "— وهذا بالضبط ما يقدّمه تعلم الآلة. ولذلك نقول إن **ثورة البيانات سبقت ثورة الخوارزميات في الاقتصاد**.",
            "الاستنتاج", "orange", "🧭")

with tabs[2]:
    ui.section("خريطة البيانات غير المهيكلة وأدواتها", "Unstructured data types and AI tools", "rose")
    types = pd.DataFrame([
        ("النص", "Text", "تقارير البنوك المركزية، الأخبار، العقود، براءات الاختراع، الخطب، وسائل التواصل",
         "قواميس، TF-IDF، LDA، Word2Vec، BERT، LLMs", "عدم اليقين، النبرة، الابتكار، الاستقطاب"),
        ("الصورة", "Images", "صور الأقمار، صور الشوارع، المنتجات، الوثائق الممسوحة",
         "CNN، التعلم بالنقل، ViT، OCR وتحليل التخطيط", "الفقر، الدخل، جودة السكن، الأسعار"),
        ("الصوت", "Audio", "مكالمات الأرباح، خطب السياسة النقدية", "تحويل الكلام لنص، تحليل النبرة الصوتية", "عدم اليقين الإداري، المصداقية"),
        ("الفيديو", "Video", "المؤتمرات الصحفية، الإعلانات", "شبكات زمانية-مكانية", "تعابير الوجه وردود فعل السوق"),
        ("الشبكات", "Networks / Graphs", "سلاسل الإمداد، الشبكات المالية، التجارة", "GNN، تضمين الرسوم البيانية", "العدوى، النظامية، الانتشار"),
        ("الجغرافيا", "Geospatial", "إحداثيات، خرائط، تنقل", "نماذج مكانية + شبكات رسومية", "التكتل، الأسواق المحلية"),
        ("السلاسل عالية التواتر", "High-frequency", "أوامر البورصة، أجهزة الاستشعار", "شبكات متكررة، محوّلات", "السيولة والتقلب"),
    ], columns=["النوع", "English", "مصادر اقتصادية", "أدوات الذكاء الاصطناعي", "مفاهيم تُقاس"])
    st.table(types)
    ui.section("الخطوة المفهومية الحاسمة", "From raw data to an economic variable", "violet")
    st.mermaid_chart(
        """
flowchart LR
    A[Raw unstructured data<br/>text, image, audio] --> B[Representation<br/>tokens, pixels, embeddings]
    B --> C[AI model f<br/>classifier, CNN, LLM]
    C --> D[Generated variable z-hat]
    D --> E[Econometric model<br/>y = a + b z-hat + e]
    E --> F{Valid inference?}
    F -->|naive| G[Bias: measurement error]
    F -->|corrected| H[PPI, DSL, bias correction<br/>validation sample]
"""
    )
    st.markdown("السؤال الذي يطرحه القياس الاقتصادي هنا: **$\\hat z$ متغير مولَّد (Generated regressor) يحمل خطأ قياس**؛ "
                "إدخاله كأنه بيانات حقيقية يحيّز $\\hat b$ ويُبطل فترات الثقة (Battaglia et al., 2024). التفاصيل في صفحتي «النص والصورة» و«الذكاء التوليدي».")

with tabs[3]:
    ui.section("كيف أصبح النص بيانات اقتصادية؟", "Origins of text-as-data in economics & finance", "indigo")
    ui.ladder([
        ("أنتويلر وفرانك", "Antweiler & Frank, 2004, J. Finance",
         "1.5 مليون رسالة في منتديات Yahoo! Finance: هل للحديث على الإنترنت أثر في الأسهم؟ بداية تيار «النص يتنبأ بالأسواق»."),
        ("تيتلوك", "Tetlock, 2007, J. Finance",
         "النبرة المتشائمة لعمود في وول ستريت جورنال تتنبأ بانخفاض قصير الأجل للأسعار يتبعه ارتداد نحو الأساسيات."),
        ("جنتزكو وشابيرو", "Gentzkow & Shapiro, 2010, Econometrica",
         "قياس الانحياز الإعلامي من تكرار العبارات الحزبية في الصحف؛ الانحياز يستجيب لتفضيلات القراء أكثر من الملكية."),
        ("لوغران وماكدونالد", "Loughran & McDonald, 2011, J. Finance",
         "قاموس مالي متخصص: نحو ثلاثة أرباع الكلمات «السلبية» في قاموس Harvard IV-4 ليست سلبية في السياق المالي — درس في خطر نقل الأدوات بين المجالات."),
        ("مؤشر عدم اليقين", "Baker, Bloom & Davis, 2016, QJE",
         "مؤشر عدم اليقين في السياسة الاقتصادية (EPU) من تغطية الصحف — أصبح متغيرًا قياسيًا في آلاف الدراسات."),
        ("شفافية الفدرالي", "Hansen, McMahon & Prat, 2018, QJE",
         "نماذج المواضيع (LDA) على محاضر FOMC لقياس أثر الشفافية على المداولات."),
        ("المسح المرجعي", "Gentzkow, Kelly & Taddy, 2019, JEL",
         "«النص كبيانات»: إطار موحد للتمثيل والنمذجة والاستدلال."),
        ("الأخبار ودورات الأعمال", "Bybee, Kelly, Manela & Xiu, 2024, J. Finance",
         "مواضيع أخبار الأعمال تتتبع النشاط الاقتصادي وتفسر العوائد."),
    ])
    ui.note("تطوّر التمثيل النصي: **حقيبة الكلمات (Bag-of-Words)** ← **TF-IDF** ← **نماذج المواضيع (LDA)** ← **التضمينات (Word2Vec, GloVe)** "
            "← **المحوّلات السياقية (BERT, FinBERT)** ← **نماذج اللغة الكبيرة (GPT, Llama, Claude)**.", "تطور الأدوات", "indigo", "🧠")

with tabs[4]:
    ui.section("من الفضاء إلى الانحدار", "Images and satellites as economic data", "sky")
    ui.cards([
        {"title": "الأضواء الليلية والناتج", "en": "Henderson, Storeygard & Weil (2012) AER", "icon": "🌃",
         "body": "الأضواء الليلية من أقمار DMSP كمقياس بديل للنمو حيث الحسابات القومية ضعيفة؛ دمج الإحصاءات الرسمية مع الضوء في مقدّر أمثل."},
        {"title": "الفقر بالتعلم العميق", "en": "Jean et al. (2016) Science", "icon": "🛰️",
         "body": "شبكة التفافية مدرَّبة بالنقل (Transfer learning): تتعلم من الأضواء ثم تتنبأ بالاستهلاك والثروة من صور النهار في إفريقيا."},
        {"title": "صور الشوارع", "en": "Street-level imagery", "icon": "🏘️",
         "body": "قياس جودة الأحياء والتحسين العمراني والدخل المحلي من صور الشوارع بالرؤية الحاسوبية."},
        {"title": "نقد ومراجعة", "en": "Gibson et al. (2020, 2025)", "icon": "⚠️",
         "body": "الأقمار لم تُصمَّم للاقتصاديين: تشبّع، تغيّر المستشعرات، ضعف في المناطق الريفية الصغيرة؛ VIIRS (منذ 2012) أدق من DMSP."},
    ], cols=2)
    st.latex(r"\text{Satellite image } I_i \xrightarrow{\ \text{CNN}\ } \hat{w}_i \;(\text{predicted wealth})"
             r"\xrightarrow{\ \text{econometrics}\ } \hat\beta \text{ in } y_i=\alpha+\beta \hat w_i+u_i")

with tabs[5]:
    ui.section("مختبر: لماذا ينهار OLS عندما تكبر p؟", "Lab — the curse of dimensionality", "pink")
    st.markdown(
        "عينة تدريب ثابتة $n=100$ مع 5 متغيرات مؤثرة فقط، ونضيف تدريجيًا متغيرات لا قيمة لها حتى يقترب $p$ من $n$. "
        "نقيس **خطأ التنبؤ خارج العينة**. شغّل الحركة: OLS ينفجر قرب $p\\to n$ بينما **لاسو (L1)** و**ريدج (L2)** يبقيان مستقرين."
    )

    @st.cache_data(show_spinner=False)
    def curse(reps: int = 25) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        n, nt = 100, 2000
        ps = list(range(5, 100, 5))
        rows = []
        for p in ps:
            e = {"OLS": [], "Lasso": [], "Ridge": []}
            for _ in range(reps):
                beta = np.zeros(p)
                beta[:5] = [1.5, -1, .8, .6, -.5]
                X = rng.standard_normal((n, p)); y = X @ beta + rng.standard_normal(n)
                Xt = rng.standard_normal((nt, p)); yt = Xt @ beta + rng.standard_normal(nt)
                b = np.linalg.lstsq(X, y, rcond=None)[0]
                e["OLS"].append(np.mean((yt - Xt @ b) ** 2))
                e["Lasso"].append(np.mean((yt - Lasso(alpha=0.1).fit(X, y).predict(Xt)) ** 2))
                e["Ridge"].append(np.mean((yt - Ridge(alpha=10).fit(X, y).predict(Xt)) ** 2))
            rows.append({"p": p, **{k: float(np.median(v)) for k, v in e.items()}})
        return pd.DataFrame(rows)

    d = curse()
    names = {"OLS": ("OLS (بدون تنظيم)", "rose"), "Lasso": ("Lasso (L1)", "teal"), "Ridge": ("Ridge (L2)", "violet")}
    fig = go.Figure(
        data=[go.Scatter(x=d.p[:1], y=d[k][:1], mode="lines+markers", name=v[0],
                         line=dict(width=4, color=ui.COLORS[v[1]])) for k, v in names.items()],
        frames=[go.Frame(data=[go.Scatter(x=d.p[:i], y=d[k][:i]) for k in names], name=str(int(d.p[i - 1])))
                for i in range(1, len(d) + 1)],
    )
    fig.update_xaxes(title="عدد المتغيرات p (مع n = 100)", range=[0, 100])
    fig.update_yaxes(title="متوسط مربع خطأ التنبؤ خارج العينة (لوغاريتمي)", type="log", range=[-0.1, np.log10(d.OLS.max() * 1.5)])
    ui.style_fig(fig, "لعنة الأبعاد: OLS مقابل التنظيم")
    ui.add_play(fig, [int(x) for x in d.p], duration=300, prefix="p = ")
    ui.show(fig)
    ui.note("الخطأ الأدنى الممكن هنا هو 1 (تباين الضجيج). عند $p=95$ يتضاعف خطأ OLS مرات عديدة لأن تباين المقدّر يتناسب مع $p/(n-p)$، "
            "بينما يستغل لاسو **التناثر (Sparsity)**. هذا هو الدافع الأول تاريخيًا لإدخال تعلم الآلة إلى القياس الاقتصادي.",
            "التفسير", "pink", "📌")

with tabs[6]:
    ui.section("دروس قاسية من البيانات الضخمة", "Hard lessons", "amber")
    ui.cards([
        {"title": "حكاية Google Flu Trends", "en": "Lazer et al. (2014) Science", "icon": "🤒",
         "body": "تنبأ بالإنفلونزا من البحث ثم أخفق بشدة (بالغ في التقدير): تغيّر خوارزمية البحث، الإفراط في التوفيق، «غطرسة البيانات الضخمة»."},
        {"title": "الانتقائية وعدم التمثيل", "en": "Selection & representativeness", "icon": "🎯",
         "body": "مستخدمو تويتر أو بطاقات الائتمان ليسوا المجتمع. n الضخم لا يصحح تحيّز العينة."},
        {"title": "الخصوصية والوصول", "en": "Privacy & access", "icon": "🔐", "body": "البيانات الإدارية والخاصة مقيدة؛ قابلية التكرار صعبة؛ الخصوصية التفاضلية (Differential privacy) تضيف ضجيجًا."},
        {"title": "الدلالة الإحصائية الزائفة", "en": "Everything is significant", "icon": "📉", "body": "مع n بالملايين كل معامل «دال»؛ ركّز على الحجم الاقتصادي وفترات الثقة."},
        {"title": "عدم الثبات", "en": "Algorithm-generated data drift", "icon": "🔄", "body": "البيانات نفسها ناتجة عن خوارزميات تتغير (التوصيات، الأسعار الديناميكية)."},
        {"title": "خطأ القياس", "en": "Measurement error", "icon": "📏", "body": "المؤشرات المستخلصة آليًا تحمل أخطاء منهجية تحيّز الانحدارات اللاحقة."},
    ])
    ui.refs("Varian (2014) JEP; Einav & Levin (2014) Science; Choi & Varian (2012) Econ. Record; Cavallo & Rigobon (2016) JEP; Lazer et al. (2014) Science; "
            "Henderson et al. (2012) AER; Jean et al. (2016) Science; Gentzkow, Kelly & Taddy (2019) JEL; Loughran & McDonald (2011) JF.")
