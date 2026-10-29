import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import ui

ui.hero(
    "الذكاء الاصطناعي التوليدي ونماذج اللغة الكبيرة في القياس الاقتصادي",
    "Generative AI & Large Language Models — Measurement, Simulation, Hypotheses and Agents",
    "الموجة الخامسة: نماذج اللغة كأدوات قياس وتصنيف، كأعوان اقتصاديين محاكَين، كمولّدات فرضيات، وكوكلاء بحث. "
    "ومعها مشكلات قياسية جديدة: التسرب وتحيز النظر للأمام، خطأ القياس غير الكلاسيكي، اختلاف النماذج والمطالبات، ودرجات حرية الباحث.",
    "violet", "pink",
)

tabs = st.tabs([
    ":material/memory: كيف تعمل LLMs؟",
    ":material/work: استخدامات البحث",
    ":material/rule: إطار قياسي للـ LLMs",
    ":material/science: مختبر الاستدلال المدعوم بالتنبؤ",
    ":material/groups: الإنسان السيليكوني",
    ":material/history_toggle_off: تحيز النظر للأمام",
    ":material/lightbulb: توليد الفرضيات",
    ":material/smart_toy: الوكلاء في البحث",
])

with tabs[0]:
    ui.section("من الرموز إلى التوليد", "How LLMs work", "violet")
    st.mermaid_chart(
        """
flowchart LR
    T[Text] --> K[Tokenizer<br/>sub-word tokens] --> E[Embeddings<br/>vectors in R^d]
    E --> TR[Transformer blocks<br/>self-attention + MLP] --> P[Next-token<br/>probabilities]
    P --> S[Sampling<br/>temperature, top-p] --> O[Generated text]
    PT[Pre-training on web-scale corpus<br/>self-supervised] -.-> TR
    FT[Fine-tuning + RLHF<br/>instructions, preferences] -.-> TR
"""
    )
    st.latex(r"P(w_{t+1}=j\mid w_{1:t})=\frac{\exp(z_j/T)}{\sum_k\exp(z_k/T)}\qquad\text{(softmax with temperature }T)")
    st.markdown("**الحرارة (Temperature)** تتحكم في عشوائية التوليد. شغّل الحركة لترى كيف يتغير توزيع الكلمة التالية بعد جملة «The central bank decided to …» (احتمالات توضيحية مصطنعة):")
    words = ["raise", "hold", "cut", "pause", "review", "signal", "ignore"]
    logits = np.array([2.2, 1.9, 1.1, 0.6, 0.1, -0.2, -1.5])
    temps = [0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0]

    def soft(t):
        p = np.exp(logits / t); return p / p.sum()

    fig = go.Figure(data=[go.Bar(x=words, y=soft(temps[0]), marker_color=ui.PLOT_COLORS[:7])],
                    frames=[go.Frame(data=[go.Bar(x=words, y=soft(t), marker_color=ui.PLOT_COLORS[:7])], name=str(t)) for t in temps])
    fig.update_yaxes(range=[0, 1], title="الاحتمال", tickformat=".0%")
    ui.style_fig(fig, "توزيع الكلمة التالية حسب الحرارة (أرقام توضيحية)", height=420, legend_h=False)
    ui.add_play(fig, temps, duration=800, prefix="T = ", redraw=True)
    ui.show(fig)
    ui.note("لأغراض **القياس** في البحث: استخدم حرارة 0 (أو قريبة)، ثبّت **إصدار النموذج** والمطالبة والبذرة، وسجّل كل ذلك. "
            "النماذج المغلقة تتغير خلف الواجهة (Model drift) فتنتج انقطاعات مصطنعة في السلاسل.", "للباحث", "violet", "📌")

with tabs[1]:
    ui.section("استخدامات الذكاء التوليدي في البحث الاقتصادي", "Use cases — Korinek (2023 JEL; 2025 agents update)", "pink")
    uc = pd.DataFrame([
        ("التفكير والتغذية الراجعة", "Ideation & feedback", "عصف ذهني، نقد المسودات، تحديد نقاط الضعف", "مرتفعة"),
        ("الكتابة", "Writing", "تحرير، تلخيص، ترجمة، صياغة", "مرتفعة"),
        ("البحث الخلفي", "Background research", "مراجعة الأدبيات، شرح المفاهيم (مع التحقق من المراجع!)", "متوسطة — خطر الهلوسة"),
        ("البرمجة", "Coding", "كتابة وتصحيح أكواد Python/R/Stata، الترجمة بين اللغات", "مرتفعة مع التنفيذ والتحقق"),
        ("تحليل البيانات", "Data analysis", "استخراج وتصنيف وتنظيف وتلخيص البيانات النصية", "مرتفعة مع عينة تحقق"),
        ("الاشتقاقات الرياضية", "Math derivations", "إعداد النماذج، الاشتقاق، التحقق", "متوسطة — تحقق صارم"),
        ("الوكلاء (2025)", "Agentic workflows", "تنفيذ متعدد الخطوات: بيانات ← كود ← تقدير ← تقرير", "واعدة — تحتاج تدقيقًا"),
    ], columns=["المجال", "English", "أمثلة", "النضج"])
    st.table(uc)
    ui.note("وجد معيار من 40 مهمة قياسية في Stata و R و Python (عمود VoxEU لسوسا وزملائه، 2026) أن السماح للنماذج بتشغيل كودها ومراجعته رفع النتائج الصحيحة من **79% إلى 97%** "
            "مقارنة بإجابات الدردشة فقط. وفي المقابل أشار تدقيق 2026 إلى أخطاء برمجية في نحو ربع 110 دراسة منشورة — فالتحقق الآلي مفيد للبشر أيضًا.",
            "أرقام حديثة", "pink", "📈")

with tabs[2]:
    ui.section("إطار قياسي تطبيقي لنماذج اللغة", "Ludwig, Mullainathan & Rambachan (2025, NBER WP 33344)", "teal")
    c1, c2 = st.columns(2)
    with c1:
        ui.note("نستخدم LLM للتنبؤ بنتيجة من النص (مثلًا: هل تتنبأ لغة التقرير بالعوائد؟).\n\n**الشرط الحاسم**: **لا تسرب** بين بيانات تدريب النموذج وعينة الباحث. "
                "إن كان النموذج «رأى» النتيجة في تدريبه فالدقة وهمية. **الحل**: نماذج مفتوحة موثقة البيانات، أو نماذج مؤرخة بتاريخ قطع سابق للعينة.",
                "1) LLM للتنبؤ (Prediction)", "teal", "🔮")
    with c2:
        ui.note("نستخدم LLM لقياس مفهوم اقتصادي يدخل لاحقًا في تحليل (معنويات، تصنيف، توقعات).\n\n**الشرط الحاسم**: **عينة تحقق بشرية صغيرة**. بدونها لا يمكن كشف أخطاء LLM، "
                "واختيارات اعتباطية (أي نموذج؟ أي مطالبة؟) تعطي تقديرات مختلفة جدًا. مع عينة التحقق يمكن تصحيح الاستدلال.",
                "2) LLM للتقدير/القياس (Estimation)", "violet", "📏")
    st.latex(r"\underbrace{\hat\theta_{\text{naive}}}_{\text{uses }\hat X=\text{LLM}(text)}\neq\theta_0\quad\text{in general};\qquad "
             r"\hat\theta_{\text{corrected}}=\hat\theta_{\text{naive}}-\widehat{\text{bias}}\big(\text{validation sample}\big)")
    ui.cards([
        {"title": "متغيرات مولّدة: التحيز", "en": "Battaglia, Christensen, Hansen & Sacher (2024)", "color": "rose",
         "body": "معاملة المتغيرات المولّدة بالذكاء الاصطناعي كبيانات تحيّز التقدير وتُبطل الاستدلال؛ يقترحون تصحيحًا صريحًا للتحيز وتقديرًا مشتركًا."},
        {"title": "التعلم الموجّه بالتصميم", "en": "Egami, Hinck, Stewart & Wei (2023) DSL", "color": "sky",
         "body": "يجمع الوسوم البديلة مع عينة وسوم ذهبية بإجراء مزدوج المتانة؛ يضمن الاستدلال الصالح حتى لو كانت الوسوم متحيزة اعتباطيًا بفضل التحكم في احتمال اختيار الوثائق للتوسيم."},
        {"title": "الاستدلال المدعوم بالتنبؤ", "en": "Angelopoulos et al. (2023, Science) PPI", "color": "emerald",
         "body": "تنبؤات وفيرة + عينة صغيرة موسومة ⇒ فترات ثقة صالحة وأضيق من استخدام العينة الموسومة وحدها."},
    ])

with tabs[3]:
    ui.section("مختبر متحرك: الساذج مقابل الذهبي فقط مقابل PPI", "Prediction-powered inference lab", "emerald")
    st.markdown("نريد تقدير متوسط مفهوم $\\theta=\\mathbb E[Y]$ (مثلًا نسبة التصريحات المتشائمة). لدينا 5000 وثيقة وسمها LLM بخطأ منهجي، و $n$ وثيقة فقط وسمها خبراء. "
                "نقارن ثلاثة مقدّرات عبر 400 محاكاة: **الساذج** (متوسط وسوم LLM)، **الذهبي فقط**، و **PPI**. شغّل الحركة مع زيادة حجم العينة الذهبية.")
    bias = st.slider("تحيز وسوم LLM", 0.0, 0.3, 0.12, 0.02, key="ppi_bias")

    @st.cache_data(show_spinner=False)
    def ppi_sim(b: float, reps: int = 400, N: int = 5000):
        rng = np.random.default_rng(3)
        out = {}
        for n in (50, 100, 200, 400, 800):
            cov = {"ساذج (LLM فقط)": [], "ذهبي فقط": [], "PPI": []}
            wid = {k: [] for k in cov}
            for _ in range(reps):
                Yall = rng.binomial(1, 0.35, N + n).astype(float)
                f = np.clip(Yall * 0.8 + (1 - Yall) * 0.15 + b + rng.normal(0, .15, N + n), 0, 1)
                Yl, fl, fu = Yall[:n], f[:n], f[n:]
                theta = 0.35
                m = fu.mean(); se = fu.std() / np.sqrt(N)
                cov["ساذج (LLM فقط)"].append(abs(m - theta) < 1.96 * se); wid["ساذج (LLM فقط)"].append(3.92 * se)
                m = Yl.mean(); se = Yl.std() / np.sqrt(n)
                cov["ذهبي فقط"].append(abs(m - theta) < 1.96 * se); wid["ذهبي فقط"].append(3.92 * se)
                rect = Yl - fl
                m = fu.mean() + rect.mean(); se = np.sqrt(fu.var() / N + rect.var() / n)
                cov["PPI"].append(abs(m - theta) < 1.96 * se); wid["PPI"].append(3.92 * se)
            out[n] = ({k: np.mean(v) for k, v in cov.items()}, {k: np.mean(v) for k, v in wid.items()})
        return out

    res = ppi_sim(bias)
    ns = list(res.keys())
    meth = ["ساذج (LLM فقط)", "ذهبي فقط", "PPI"]
    colors = [ui.COLORS["rose"], ui.COLORS["sky"], ui.COLORS["emerald"]]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("تغطية فترة الثقة 95%", "متوسط عرض فترة الثقة"))

    def ppi_tr(n):
        c, w = res[n]
        return [go.Bar(x=meth, y=[c[m] for m in meth], marker_color=colors, showlegend=False, text=[f"{c[m]:.0%}" for m in meth], textposition="outside"),
                go.Bar(x=meth, y=[w[m] for m in meth], marker_color=colors, showlegend=False, text=[f"{w[m]:.3f}" for m in meth], textposition="outside")]

    for j, t in enumerate(ppi_tr(ns[0])):
        fig.add_trace(t, row=1, col=j + 1)
    fig.frames = [go.Frame(data=ppi_tr(n), traces=[0, 1], name=str(n)) for n in ns]
    fig.add_hline(y=0.95, line_dash="dash", row=1, col=1)
    fig.update_yaxes(range=[0, 1.12], tickformat=".0%", row=1, col=1)
    fig.update_yaxes(range=[0, max(res[ns[0]][1].values()) * 1.25], row=1, col=2)
    ui.style_fig(fig, height=460, legend_h=False)
    ui.add_play(fig, ns, duration=1000, prefix="حجم العينة الذهبية n = ", redraw=True)
    ui.show(fig)
    ui.note("**الساذج**: فترة ضيقة جدًا لكنها **خاطئة** (تغطية قريبة من الصفر عند وجود تحيز) — أخطر نتيجة: ثقة زائفة. **الذهبي فقط**: صحيح لكن عريض. "
            "**PPI**: صحيح **وأضيق** من الذهبي لأنه يستغل التنبؤات الوفيرة ويصحح تحيزها بالعينة الصغيرة.", "القراءة", "emerald", "✅")

with tabs[4]:
    ui.section("نماذج اللغة كأعوان اقتصاديين محاكَين", "Homo silicus — Horton (2023) and beyond", "orange")
    st.markdown(
        """
يقترح **هورتون (2023)** استخدام LLMs كـ **«إنسان سيليكوني»** (Homo silicus) — نظير حاسوبي للإنسان يمكن منحه تفضيلات ومعلومات وأوقافًا ثم دراسة سلوكه في سيناريوهات،
كما يستخدم الاقتصاديون **الإنسان الاقتصادي** (Homo economicus). أعاد تجارب كلاسيكية (Charness & Rabin 2002؛ Kahneman, Knetsch & Thaler 1986؛ Samuelson & Zeckhauser 1988)
بنتائج **متشابهة نوعيًا**، مع سهولة بالغة في تجريب التنويعات. وطوّر **Manning, Zhu & Horton (2024)** «العلوم الاجتماعية المؤتمتة»: نماذج لغة كعلماء وكمبحوثين معًا.
        """
    )
    ui.cards([
        {"title": "الاستخدامات الواعدة", "en": "Promising uses", "color": "orange", "body": "تجريب أولي (Piloting) للاستبيانات والتجارب، توليد فرضيات، اختبار تصميم المعالجات قبل التكلفة البشرية."},
        {"title": "حدود قوية", "en": "Strong limits", "color": "rose", "body": "دراسة Nature Human Behaviour (أكتوبر 2026، وفق موسوعة رودان): استجابات LLM المحاكاة بدائل ضعيفة لتفضيلات البشر الاقتصادية عبر 74 بلدًا."},
        {"title": "تحيزات منهجية", "en": "Systematic distortions", "color": "violet", "body": "تجانس مفرط، تمثيل ناقص للفئات، حساسية للمطالبة، «تملق» النموذج، وانعكاس بيانات التدريب لا المجتمع."},
        {"title": "الإفصاح", "en": "Disclosure", "color": "slate", "body": "ورقة هورتون تفصح عن مصالح مالية للمؤلفين في منصة Expected Parrot وعمله مستشارًا اقتصاديًا لشركة ذكاء اصطناعي."},
    ], cols=2)
    ui.note("**قاعدة**: المحاكاة بالـ LLM أداة **لتوليد الفرضيات والتجريب الأولي**، وليست بديلًا عن البيانات البشرية في الاستدلال. "
            "و«20,000 إجابة مولّدة» ليست 20,000 مشاهدة مستقلة.", "قاعدة منهجية", "orange", "⚖️")

with tabs[5]:
    ui.section("تحيز النظر للأمام وتلوث التدريب المسبق", "Look-ahead bias & pretraining contamination", "rose")
    ui.ladder([
        ("المشكلة", "The problem", "LLM مدرَّب على نصوص حتى 2024 «يعرف» ما حدث في 2020–2023. إن طلبنا منه «التنبؤ» بعوائد 2021 من نص 2020 فقد يستدعي النتيجة لا أن يتنبأ بها."),
        ("اختبارات إحصائية", "Sarkar & Vafa (ICML 2025 workshop)", "اختبارات مباشرة لتحيز النظر للأمام؛ وجدوه في التنبؤ بعوامل المخاطر من مكالمات الأرباح وبالفائزين بالانتخابات من السير الذاتية."),
        ("نماذج مؤرخة", "ChronoBERT / ChronoGPT (He, Lv, Manela & Wu, 2025)", "نماذج تُدرَّب فقط على نصوص متاحة حتى كل تاريخ؛ في التنبؤ بالعوائد اليومية من الأخبار وجدوا التحيز «متواضعًا»."),
        ("نماذج بقطع سنوي", "DatedGPT (2026)", "نموذج 1.3 مليار معلمة مدرَّب على بيانات مقسمة بتواريخ قطع سنوية."),
        ("قياس متعارض", "LookaheadBench (2026)", "يجد تحيزًا «كبيرًا» في النماذج القياسية مقاسًا بتآكل العوائد غير الطبيعية — مقاييس ونماذج مختلفة عن ChronoBERT."),
        ("المعرفة الكلية", "Total Recall? (Federal Reserve, 2025); ChatMacro (FRBSF, 2026)", "النماذج تخلط بين البيانات الأولية والمنقحة؛ وتنبؤاتها «الورقية» قد تختلف عن التنبؤ الحقيقي في الوقت الفعلي."),
    ])
    ui.note("**العلاجات**: نماذج مفتوحة بتاريخ قطع معروف وسابق للعينة؛ تقييم على فترات **بعد** تاريخ القطع؛ اختبارات تلوث (هل يكمل النموذج نصوصًا من العينة حرفيًا؟)؛ "
            "إخفاء الهوية (أسماء الشركات والتواريخ)؛ التقييم بالبيانات الآنية لا المنقحة.", "العلاج", "rose", "💊")

with tabs[6]:
    ui.section("تعلم الآلة كأداة لتوليد الفرضيات", "Ludwig & Mullainathan (2024, QJE)", "amber")
    st.markdown(
        """
السؤال: من أين تأتي الفرضيات العلمية؟ تقليديًا من حدس الباحث. يقترح **لودفيغ ومولايناثان** إجراءً منهجيًا: (1) درّب خوارزمية تتنبأ بقرار بشري
(قرارات القضاة بالاحتجاز) من بيانات غنية (صور وجوه المتهمين)، (2) إن تنبأت الخوارزمية بما لا تفسره المتغيرات المعروفة فهناك **إشارة غير مفسرة**، (3) استخدم نماذج توليدية
لـ **تشويه الصور** في اتجاه التنبؤ وعرضها على بشر ليسمّوا الخاصية ⇒ فرضيات جديدة قابلة للاختبار («الوجه الأكثر تجهّمًا/الأقل عناية» يرتبط بالاحتجاز).
        """
    )
    st.mermaid_chart(
        """
flowchart LR
    A[Rich data: images, text] --> B[Train predictor of human decision]
    B --> C{Unexplained predictive signal?}
    C -->|yes| D[Generative model morphs inputs]
    D --> E[Humans name the feature]
    E --> F[New testable hypothesis]
    F --> G[Classic econometric test on new data]
"""
    )
    ui.note("الفصل بين **مرحلة الاكتشاف** (Discovery) و**مرحلة الاختبار** (Testing) على بيانات مختلفة هو ما يمنع هذا الإجراء من أن يصبح «تنقيبًا في البيانات».",
            "الضمانة المنهجية", "amber", "🔒")

with tabs[7]:
    ui.section("وكلاء الذكاء الاصطناعي في دورة البحث", "AI agents in the research cycle", "indigo")
    ag = pd.DataFrame([
        ("0 — صياغة", "تحويل سؤال عام إلى فرضيات", "نعم، مع تحقق الباحث"),
        ("1 — معرفة/بحث", "استرجاع أوراق ومقارنة طرق", "نعم، بعد التثبت من كل مرجع"),
        ("2 — بيانات", "استخراج/تنظيف/ترميز", "نعم، مع اختبارات صحة وتوثيق"),
        ("3 — تقدير", "كتابة وتنفيذ النماذج", "نعم، في بيئة قابلة لإعادة التنفيذ"),
        ("4 — تشخيص", "قراءة الأخطاء المعيارية والبواقي", "جزئيًا، مع التحقق من الشروط"),
        ("5 — تفسير", "القول إن β̂ سببي أو السياسة ناجحة", "لا، دون صلاحية منهج التعرّف"),
        ("6 — قرار", "توصيات سياسة أو تداول بأثر فعلي", "يحتاج إشرافًا وقيودًا وتحكيمًا مستقلًا"),
    ], columns=["مستوى الاستقلالية", "النشاط", "هل يُقبل بحثيًا؟"])
    st.table(ag)
    ui.cards([
        {"title": "حلقة وكيل قابلة للتدقيق", "en": "Shin (2026) — forecast combination", "color": "indigo",
         "body": "مقيّم ثابت + سجل تجارب كامل + عينة محجوزة بعد البحث: من ثلاث قواعد اكتشفها الوكيل، اثنتان صمدتا خارج العينة وواحدة لم تصمد. الوكلاء يوسّعون درجات حرية الباحث الخفية."},
        {"title": "الإنسان في الحلقة", "en": "HLER (2026)", "color": "violet",
         "body": "الأنظمة المستقلة تمامًا تتعثر في الاقتصاد لأن الأسئلة تُربط بالبيانات المتاحة، والتعرّف يحتاج تصميمًا، والدلالة الاقتصادية تحتاج حكمًا بشريًا."},
        {"title": "بناء البيانات بالوكلاء", "en": "Deep Research on a Loop (NBER WP 35188, 2026)", "color": "teal",
         "body": "وكلاء يبنون مجموعات بيانات من مصادر عامة (129 مصدرًا و136 سجل دليل في تجربتهم) بتكلفة ساعات قليلة من عمل مساعد بحث."},
        {"title": "البرمجة القياسية", "en": "Galiani, López & Sosa (2026, NBER)", "color": "orange",
         "body": "مقارنة الدردشة والوكلاء وهندسة المطالبات في مهام القياس الاقتصادي؛ تقييم موثوقية الأدوات لا استبدال التفكير النظري."},
    ], cols=2)
    ui.note("جملة واثقة من LLM عن سببية نتيجة أو عن شروط اختبار ARDL **ليست برهانًا**. على الباحث إثبات مصدر القاعدة، والتمييز بين تنبؤات الوكيل وتقدير نماذج قابلة للدحض.",
            "فخ «الخبير الاصطناعي»", "rose", "🎭")
    ui.refs("Korinek (2023) JEL; Horton (2023) NBER w31122; Angelopoulos et al. (2023) Science; Egami et al. (2023); Battaglia et al. (2024); Ludwig & Mullainathan (2024) QJE; "
            "Ludwig, Mullainathan & Rambachan (2025) NBER w33344; Sarkar & Vafa (2025); He, Lv, Manela & Wu (2025) arXiv 2502.21206; Shin (2026) arXiv 2603.17381.")
