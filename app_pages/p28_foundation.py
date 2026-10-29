import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "النماذج التأسيسية والفروع الرمزية واكتشاف السببية",
    "Foundation Models, Symbolic Regression, KANs and Causal Discovery",
    "من ARIMA و VAR إلى نماذج تعلّمت على مجموعات واسعة من السلاسل وتتكيف مع مهام جديدة — دون مساواة «Zero-shot» بالموثوقية السببية. "
    "ومعها فروع «علمية» تبحث عن معادلات صريحة وعن رسوم سببية من البيانات.",
    "cyan", "fuchsia",
)

tabs = st.tabs([
    ":material/foundation: ما النموذج التأسيسي؟",
    ":material/timeline: نماذج السلاسل الزمنية",
    ":material/table: النماذج الجدولية",
    ":material/history: التلوث والمراجعات",
    ":material/functions: الانحدار الرمزي",
    ":material/science: مختبر رمزي",
    ":material/share: اكتشاف السببية",
])

with tabs[0]:
    ui.section("تعريف النموذج التأسيسي وعائلاته", "What is a foundation model?", "cyan")
    st.markdown("**النموذج التأسيسي (Foundation model)** يتعلم تمثيلات وأنماطًا من نطاق واسع في مرحلة **التدريب المسبق**، ثم يستخدمها في مهمة جديدة دون تدريب كامل "
                "(Zero/few-shot) أو بضبط محدود (Fine-tuning). في السلاسل الزمنية يتعلم الإيقاع والاعتماد والتغيرات عبر سلاسل عديدة — وهو مختلف عن ARIMA الموفَّق لسلسلة واحدة، "
                "ومختلف عن LLM لغوي يُطلب منه تخمين رقم.")
    fam = pd.DataFrame([
        ("نماذج أساس للسلاسل الزمنية", "Time-series FMs", "سلاسل من مجالات كثيرة", "توقعات نقطية/احتمالية", "قد لا تحفظ الهويات المحاسبية وقيود السببية"),
        ("نماذج أساس كلية", "Macroeconomic FMs", "بيانات بنوك مركزية ومؤشرات بلدان", "خط أساس + تنبؤ شرطي", "تحتاج التحقق من الإصدارات والسياسات"),
        ("نماذج أساس جدولية", "Tabular FMs", "مهام جدولية اصطناعية/متنوعة", "توزيعات تنبؤ لمهام جدولية", "مفيدة للمقاطع العرضية، لا تحل ديناميكية البانل"),
        ("نماذج لغة اقتصادية", "Economic LMs", "نصوص اقتصادية/مالية", "تصنيف واستخراج معلومات", "المقاييس المولّدة تحتاج تحققًا"),
        ("نماذج متعددة الوسائط", "Multimodal economic models", "صور + نص + جداول + زمن", "متغيرات مركبة وتوقع", "صعوبة ربط الإشارة بالمعنى الاقتصادي"),
    ], columns=["العائلة", "English", "مصدر التعلم المسبق", "المخرجات", "فارق اقتصادي"])
    st.table(fam)

with tabs[1]:
    ui.section("نماذج الأساس للسلاسل الزمنية والاقتصاد الكلي", "Time-series & macro foundation models", "teal")
    ui.cards([
        {"title": "TimeGPT", "en": "Nixtla (2023)", "color": "teal", "body": "من أوائل نماذج الأساس التجارية للتنبؤ بلا تدريب."},
        {"title": "TimesFM", "en": "Google (2024)", "color": "sky", "body": "محوّل مفكك (Decoder-only) يعمل على رقع (Patches) حقيقية القيم."},
        {"title": "Chronos / Chronos-2", "en": "Amazon (2024; 2025)", "color": "orange", "body": "يعيد استخدام جهاز نماذج اللغة: يحوّل القيم إلى رموز ويعامل التنبؤ كتصنيف على صناديق قيم؛ Chronos-2 للتنبؤ متعدد المتغيرات."},
        {"title": "Moirai", "en": "Salesforce (2024)", "color": "violet", "body": "أي عدد من المتغيرات وتواترات مختلطة؛ تفوقت متغيراته على Persistence و ARIMA في ناتج نيوزيلندا (Jetwiriyanon et al., 2025)."},
        {"title": "BISTRO", "en": "BIS (16 مارس 2026)", "color": "rose", "body": "BIS Time-series Regression Oracle: محوّل لتوقع السلاسل الكلية بلا إعداد خاص لكل مهمة، مع تنبؤ شرطي وغير شرطي؛ كود و Colab متاحان — أداة سيناريوهات لا نموذج سببي."},
        {"title": "MACROCAST", "en": "Carriero, Pettenuzzo & Shekhar (يونيو 2026)", "color": "emerald", "body": "يعالج التلوث الزمني وتحيز المراجعات بتدريب مسبق اصطناعي وضبط على إصدارات ALFRED مع مقارنة حقيقية زمنية."},
    ], cols=3)
    ui.section("كيف ندمج النماذج التأسيسية بالقياس الاقتصادي؟", "Six integration routes", "indigo")
    ui.ladder([
        ("معيار مستقل", "Benchmark", "مقارنة مع ARIMA/ARDL/VAR/BVAR/DFM بنفس البروتوكول الزمني."),
        ("مستخرج ميزات", "Feature extractor", "تمثيلات أو تنبؤات أولية داخل توليف النماذج (Model averaging)."),
        ("محرك سيناريوهات شرطية", "Conditional scenario engine", "تجارب «ماذا لو» — مع وصفها سيناريوهات تنبؤية لا آثارًا سببية."),
        ("عمود فقري مسبق التدريب", "Pretrained backbone", "ثم ضبط دقيق مضبوط على سلاسل البلد المستهدف."),
        ("معايرة قياسية", "Econometric calibration", "تصحيح التحيز، فترات تنبؤ مطابقة، إدراج القيود المحاسبية."),
        ("إدارة مخاطر النماذج", "Model risk management", "تتبع تغير الدقة مع الزمن وتغير النظام الاقتصادي."),
    ])
    ui.note("مراجعة Chen & Wang (يوليو 2026) «من VAR إلى التنبؤ بالذكاء الاصطناعي» تربط المحوّلات ونماذج الأساس والمولّدات الانتشارية بتقليد VAR عبر كائن مشترك: "
            "**التوزيع الشرطي للمستقبل بشرط الماضي** — وتلاحظ أنها كثيرًا ما **تفتقر** إلى أدوات الاستدلال والبنية التي تجعل النماذج الكلاسيكية مفيدة لاختبار الفرضيات وتحليل السياسات.",
            "خلاصة مراجعة حديثة", "indigo", "📚")

with tabs[2]:
    ui.section("النماذج التأسيسية للبيانات الجدولية", "Tabular foundation models", "orange")
    st.markdown(
        """
**TabPFN** (Hollmann et al., 2025, *Nature*): شبكة محوّلات مدربة مسبقًا على ملايين **مهام جدولية اصطناعية** مولّدة من نماذج سببية عشوائية،
تقوم بـ «الاستدلال البايزي التقريبي» في تمريرة أمامية واحدة (In-context learning) ⇒ أداء قوي جدًا في **البيانات الصغيرة والمتوسطة** دون ضبط.

**للاقتصادي**: منافس قوي للتعزيز في التنبؤ المقطعي ومتعلم محتمل للدوال المساعدة في DML — لكن **لا يحل** ديناميكية البانل أو الاعتماد الزمني، ويجب التحقق من شروط معدلات التقارب.
        """
    )

with tabs[3]:
    ui.section("لماذا لا يكفي تقسيم تدريب/اختبار عادي؟", "Pretraining contamination & data vintages", "rose")
    ui.cards([
        {"title": "التلوث الزمني", "en": "Temporal contamination", "color": "rose", "body": "النموذج المدرب مسبقًا على سلاسل منشورة لاحقًا قد «رأى المستقبل» حتى لو فصلت بياناتك محليًا."},
        {"title": "تحيز المراجعات", "en": "Revision bias", "color": "amber", "body": "التعلم من القيم النهائية المنقحة لا المتاحة وقت التنبؤ — ALFRED/Real-time datasets هي العلاج."},
        {"title": "Zero-shot ≠ لم يرَ", "en": "Zero-shot is not unseen", "color": "violet", "body": "قد يكون النموذج مدربًا على بيانات قريبة بل على السلسلة نفسها."},
        {"title": "الأعقد ليس الأفضل", "en": "IMF (Dec. 2025)", "color": "teal", "body": "نماذج Bridge والعوامل الديناميكية والطرق التقليدية تفوقت غالبًا على ML المعقد في التنبؤ الآني للبلدان والبيانات المفحوصة؛ Lasso/Elastic Net أكثر تنافسية من الشبكات العميقة في السلاسل القصيرة."},
    ], cols=2)
    ui.note("**البروتوكول الصحيح**: تحقق من تاريخ إصدار النموذج ونطاق تدريبه المسبق، واستخدم **تقييمًا متدحرجًا بالإصدارات الآنية** (Rolling-origin real-time vintage backtesting) "
            "مع اختبارات Diebold–Mariano ومجموعات ثقة النماذج.", "البروتوكول", "rose", "📋")

with tabs[4]:
    ui.section("الانحدار الرمزي والفروع العلمية", "Symbolic regression, program synthesis, KANs", "fuchsia")
    st.markdown("يبحث **الانحدار الرمزي (Symbolic regression)** عن **شكل دالة صريح** مثل $Y=a+b\\log X+cZ^2$ عبر بحث تطوري (Genetic programming) أو شبكات أو تركيب برامج — "
                "أدوات مثل **PySR** و **AI Feynman**. في الاقتصاد قد يكتشف مرشحات لدوال الإنتاج أو منحنيات الاستهلاك أو علاقات التسعير.")
    ui.cards([
        {"title": "متى يكون دمجًا حقيقيًا؟", "en": "Genuine integration", "color": "fuchsia",
         "body": "بحث محدود بوحدات متجانسة (Dimensional consistency) وإشارات مشتقات معروفة؛ فحص المرونات والحدود؛ مقارنة خارج العينة وعدم يقين اختيار النموذج بالـ bootstrap؛ ربط النتيجة بعلاقة بنيوية قابلة للدحض."},
        {"title": "شبكات كولموغوروف–أرنولد", "en": "KAN (Liu et al., 2024); TimeKAN (2026)", "color": "violet",
         "body": "دوال أحادية المتغير قابلة للتعلم على الحواف بدل أوزان ثابتة؛ منحنيات قابلة للرسم — لكن القابلية للتفسير الاقتصادي تتطلب اشتقاق المقاييس الحدية والتحقق من الاستقرار."},
        {"title": "ليس بديلًا عن القياس", "en": "Not a substitute", "color": "rose",
         "body": "الصيغة المكتشفة لا تعالج الداخلية والانتقاء والآنية. المسار الأعمق: اكتشاف مرشح رمزي ← تفسير بنيوي ← تعرّف قياسي ← تحقق مستقل."},
    ])

with tabs[5]:
    ui.section("مختبر: بحث رمزي مصغّر بمعيار BIC", "Mini symbolic search", "pink")
    st.markdown("نولّد بيانات من دالة «مخفية» ونبحث في مكتبة صغيرة من الأشكال الدالية $y=a+b\\,f(x)$ و $y=a+b\\,f(x)+c\\,g(x)$ ونرتبها بمعيار BIC — نسخة تعليمية مبسطة من فكرة الانحدار الرمزي.")
    true_form = st.selectbox("الدالة الحقيقية المخفية", ["2 + 3·log(x)", "1 + 0.5·x²", "4 − 2/x", "1 + 2·√x − 0.3·x"], key="sym_true")
    noise = st.slider("الضجيج", 0.05, 1.0, 0.3, 0.05, key="sym_noise")
    rng = np.random.default_rng(2)
    x = rng.uniform(0.5, 5, 150)
    truth = {"2 + 3·log(x)": 2 + 3 * np.log(x), "1 + 0.5·x²": 1 + .5 * x ** 2, "4 − 2/x": 4 - 2 / x, "1 + 2·√x − 0.3·x": 1 + 2 * np.sqrt(x) - .3 * x}[true_form]
    y = truth + rng.normal(0, noise, x.size)
    lib = {"x": x, "log(x)": np.log(x), "x²": x ** 2, "√x": np.sqrt(x), "1/x": 1 / x, "exp(−x)": np.exp(-x), "x³": x ** 3}
    rows = []
    names = list(lib)
    cands = [(a,) for a in names] + [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    for c in cands:
        X = np.column_stack([np.ones_like(x)] + [lib[t] for t in c])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = np.sum((y - X @ coef) ** 2); k = X.shape[1]
        bic = len(y) * np.log(rss / len(y)) + k * np.log(len(y))
        expr = f"{coef[0]:.2f}" + "".join(f" {'+' if b >= 0 else '−'} {abs(b):.2f}·{t}" for b, t in zip(coef[1:], c))
        rows.append({"الصيغة المرشحة": expr, "عدد الحدود": len(c), "BIC": bic, "RSS": rss})
    res = pd.DataFrame(rows).sort_values("BIC").head(8).reset_index(drop=True)
    st.dataframe(res.style.format({"BIC": "{:.1f}", "RSS": "{:.2f}"}), hide_index=True)
    best = res.iloc[0]["الصيغة المرشحة"]
    fig = go.Figure([go.Scatter(x=x, y=y, mode="markers", marker=dict(color="#cbd5e1"), name="البيانات"),
                     go.Scatter(x=np.sort(x), y=truth[np.argsort(x)], mode="lines", line=dict(color="#111827", dash="dot"), name=f"الحقيقية: {true_form}")])
    ui.style_fig(fig, f"أفضل صيغة بمعيار BIC: {best}", height=380)
    ui.show(fig)
    ui.note("مع ضجيج مرتفع قد يختار البحث صيغة «خاطئة» تتنبأ جيدًا (مثلًا √x بدل log x) — **الصيغة البسيطة ليست قانونًا اقتصاديًا بمجرد بساطتها**. "
            "وتعدد الصيغ المجرَّبة يعيدنا إلى مشكلة التنقيب في البيانات ⇒ تحقق خارج العينة.", "الدرس", "pink", "🔍")

with tabs[6]:
    ui.section("اكتشاف السببية من البيانات", "Causal discovery", "emerald")
    st.markdown("خوارزميات **PC و FCI** (Spirtes, Glymour & Scheines)، **GES**، **LiNGAM**، **NOTEARS** (Zheng et al., 2018)، و **PCMCI** للسلاسل الزمنية تستخدم أنماط "
                "**الاستقلال الشرطي** أو قيودًا شكلية لاقتراح رسم سببي مرشح. المثال الكلاسيكي: الأنماط الثلاثة تعطي بصمات استقلال مختلفة:")

    @st.cache_data(show_spinner=False)
    def patterns(n: int = 5000):
        rng = np.random.default_rng(1)
        def pc(a, b, c):
            ra = a - np.polyval(np.polyfit(c, a, 1), c); rb = b - np.polyval(np.polyfit(c, b, 1), c)
            return np.corrcoef(a, b)[0, 1], np.corrcoef(ra, rb)[0, 1]
        X = rng.normal(size=n); Y = .8 * X + rng.normal(size=n); Z = .8 * Y + rng.normal(size=n)
        chain = pc(X, Z, Y)
        Y2 = rng.normal(size=n); X2 = .8 * Y2 + rng.normal(size=n); Z2 = .8 * Y2 + rng.normal(size=n)
        fork = pc(X2, Z2, Y2)
        X3 = rng.normal(size=n); Z3 = rng.normal(size=n); Y3 = .8 * X3 + .8 * Z3 + rng.normal(size=n)
        coll = pc(X3, Z3, Y3)
        return chain, fork, coll

    ch, fk, cl = patterns()
    pat = pd.DataFrame([
        ("سلسلة", "X → Y → Z", f"{ch[0]:.2f}", f"{ch[1]:.2f}", "مستقلان بشرط Y"),
        ("شوكة", "X ← Y → Z", f"{fk[0]:.2f}", f"{fk[1]:.2f}", "مستقلان بشرط Y"),
        ("مصادم", "X → Y ← Z", f"{cl[0]:.2f}", f"{cl[1]:.2f}", "مستقلان دون شرط، مرتبطان بشرط Y!"),
    ], columns=["النمط", "الرسم", "corr(X,Z)", "corr(X,Z | Y)", "البصمة"])
    st.table(pat)
    ui.note("السلسلة والشوكة **لا يمكن تمييزهما** من الاستقلال وحده (فئة تكافؤ ماركوف)، أما المصادم فيُكتشف — ولهذا تعطي الخوارزميات **فئات تكافؤ** لا رسمًا وحيدًا. "
            "وكلها تحتاج افتراضات قوية (عدم وجود مرْبكات كامنة في PC، الإخلاص Faithfulness، عدم الدورية). **الاستخدام السليم**: اقتراح DAG مرشح ثم اختباره بالتصميم والنظرية — "
            "لا «استخراج السببية تلقائيًا». وضبط مصادم (Collider) خطأ شائع يولّد تحيزًا.", "التفسير", "emerald", "🕸️")
    ui.refs("Hollmann et al. (2025) Nature; Ansari et al. (2024) Chronos; Das et al. (2024) TimesFM; Woo et al. (2024) Moirai; BIS (2026) BISTRO; Carriero, Pettenuzzo & Shekhar (2026); "
            "Jetwiriyanon, Susnjak & Ranathunga (2025); Chen & Wang (2026) arXiv 2607.14279; Spirtes, Glymour & Scheines (2000); Zheng et al. (2018); Runge et al. (2019).")
