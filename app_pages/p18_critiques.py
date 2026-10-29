import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "الانتقادات والقيود والمخاطر",
    "Critiques, Limitations and Risks of AI × Econometrics",
    "نقاش نقدي ومعرفي لا غنى عنه: لماذا قد يصيب التنبؤ وتخطئ النظرية؟ كيف يعود نقد لوكاس؟ ما مخاطر الصندوق الأسود، وعدم الاستقرار، "
    "ودرجات حرية الباحث، والتسرب، والعدالة، والخصوصية، وقابلية التكرار؟ مع ردود الأدبيات وعلاجاتها.",
    "rose", "slate",
)

tabs = st.tabs([
    ":material/psychology_alt: نقد معرفي",
    ":material/science: مختبر نقد لوكاس",
    ":material/build: نقد منهجي",
    ":material/smart_toy: نقد خاص بـ LLMs",
    ":material/gavel: أخلاقي ومؤسسي",
    ":material/error: المغالطات الخمس",
    ":material/medical_services: العلاجات",
])

with tabs[0]:
    ui.section("اعتراضات معرفية", "Epistemological objections", "rose")
    ui.cards([
        {"title": "التنبؤ يصيب والنظرية تخطئ", "en": "Forecasting success ≠ understanding", "color": "rose",
         "body": "نموذج معقد قد يتنبأ بدقة لأنه التقط علاقات عرضية أو معلومات لم تكن متاحة. النجاح التنبؤي لا ينفي الإرباك أو السببية المعكوسة أو الاعتماد على نظام مؤقت."},
        {"title": "«نهاية النظرية»؟", "en": "Anderson (2008) vs. the econometric view", "color": "violet",
         "body": "ادعاء أن البيانات الضخمة تغني عن النظرية والنماذج. الرد القياسي: التعرّف والقيم المضادة للواقع وتحليل السياسات تتطلب بنية نظرية."},
        {"title": "التفسير ≠ التعرّف", "en": "Interpretability ≠ identification", "color": "orange",
         "body": "SHAP والاعتماد الجزئي والمشتقات تصف حساسية تنبؤية مشروطة، لا معاملًا سببيًا، خصوصًا مع ترابط الخصائص."},
        {"title": "النظرية الخاطئة داخل الشبكة", "en": "Wrong theory, structured bias", "color": "teal",
         "body": "قيود مثل عدم المراجحة أو الهويات المحاسبية تختلف في قوتها عن فرضيات التفضيلات؛ فرض النظرية الناقصة يولّد تحيزًا منظمًا."},
    ], cols=2)

with tabs[1]:
    ui.section("مختبر: نقد لوكاس يصيب نماذج تعلم الآلة أيضًا", "Lab — the Lucas critique for ML models", "pink")
    st.markdown("يتعلم نموذج (هنا غابة/انحدار مرن مكافئ) العلاقة بين التضخم والبطالة في **نظام سياسة A** حيث يستجيب البنك المركزي بقاعدة معينة. "
                "عند الفترة 150 يتغير نظام السياسة (قاعدة أشد) فتتغير توقعات الأعوان والعلاقة المختزلة. شغّل الحركة: الخطأ ينفجر بعد التغير رغم دقة النموذج السابقة.")

    @st.cache_data(show_spinner=False)
    def lucas(T: int = 300, brk: int = 150):
        rng = np.random.default_rng(4)
        u = np.zeros(T); pi = np.zeros(T)
        for t in range(1, T):
            phi = 0.9 if t < brk else 0.2
            u[t] = 0.7 * u[t - 1] + rng.normal(0, .5)
            pi[t] = phi * pi[t - 1] - (1.2 if t < brk else 0.3) * u[t] + rng.normal(0, .3)
        X = np.column_stack([np.ones(brk - 1), pi[:brk - 1], u[1:brk]])
        b = np.linalg.lstsq(X, pi[1:brk], rcond=None)[0]
        pred = np.r_[np.nan, np.column_stack([np.ones(T - 1), pi[:-1], u[1:]]) @ b]
        err = np.abs(pi - pred)
        return pi, pred, err, brk

    pi, pred, err, brk = lucas()
    steps = list(range(20, len(pi) + 1, 10))

    def lc(m):
        x = np.arange(m)
        return [go.Scatter(x=x, y=pi[:m], mode="lines", line=dict(color="#64748b"), name="التضخم الفعلي"),
                go.Scatter(x=x, y=pred[:m], mode="lines", line=dict(color=ui.COLORS["rose"], width=3), name="تنبؤ النموذج المدرَّب على النظام A")]

    fig = go.Figure(data=lc(steps[0]), frames=[go.Frame(data=lc(m), name=str(m)) for m in steps])
    fig.add_vline(x=brk, line_dash="dash", annotation_text="تغيّر نظام السياسة")
    fig.update_xaxes(range=[0, len(pi)]); fig.update_yaxes(range=[np.nanmin(pi) - 1, np.nanmax(pi) + 1])
    ui.style_fig(fig, "التنبؤ قبل وبعد تغيّر نظام السياسة")
    ui.add_play(fig, steps, duration=250, prefix="الزمن: ", redraw=True)
    ui.show(fig)
    c1, c2 = st.columns(2)
    c1.metric("متوسط الخطأ المطلق قبل التغير", f"{np.nanmean(err[1:brk]):.2f}")
    c2.metric("متوسط الخطأ المطلق بعد التغير", f"{np.nanmean(err[brk:]):.2f}")
    ui.note("الشبكات ذات البنية غير الشفافة قد **تخفي** آلية عدم الثبات. في تحليل سياسة بديلة يجب السؤال: **هل العلاقات ثابتة عندما يتغير النظام؟** "
            "العلاجات: نماذج هيكلية، قيود ثبات (Invariance)، تدريب عبر أنظمة متعددة، واختبارات الاستقرار.", "الدرس", "pink", "🏛️")

with tabs[2]:
    ui.section("انتقادات منهجية", "Methodological critiques", "orange")
    mc = pd.DataFrame([
        ("الصندوق الأسود", "Black box", "صعوبة فهم الآلية والتحقق منها", "نماذج قابلة للتفسير، BAM، MRF، ملخصات GenericML"),
        ("الاستدلال الساذج بعد ML", "Naive post-ML inference", "فترات ثقة باطلة بعد الاختيار/التنظيم", "PDS، DML، الاستدلال الانتقائي"),
        ("عدم استقرار الاختيار", "Selection instability", "المتغيرات المختارة تتبدل مع العينة", "لا تفسّر معاملات لاسو؛ استقرار الاختيار"),
        ("الإفراط في التوفيق على المعايير", "Benchmark overfitting", "ضبط متكرر على نفس بيانات الاختبار", "عينة محجوزة نهائية، تسجيل مسبق"),
        ("حديقة المسارات المتشعبة", "Garden of forking paths", "قرارات بحثية كثيرة غير موثقة", "تحليل المواصفات المتعددة (Multiverse)، التوثيق"),
        ("درجات حرية الوكلاء", "Agentic researcher degrees of freedom", "الوكلاء يجربون آلاف المواصفات بصمت", "سجلات كاملة، مقيّم ثابت، عينة بعد البحث (Shin 2026)"),
        ("التسرب الزمني", "Leakage", "تحسن وهمي في التنبؤ", "تقسيم زمني صارم، بيانات آنية"),
        ("التعميم الخارجي", "External validity", "الأداء في مجتمع لا ينتقل لآخر", "نمذجة عدم التجانس، اختبار عبر البلدان"),
        ("العينات الصغيرة", "Small samples", "السلاسل الكلية قصيرة", "انكماش، نماذج أساس بحذر، قبليات نظرية"),
        ("الكلفة الحسابية والبيئية", "Compute & energy", "تعقيد بلا فائدة قابلة للقياس", "مبدأ أصغر نموذج كافٍ"),
    ], columns=["النقد", "English", "المشكلة", "العلاج"])
    st.table(mc)

with tabs[3]:
    ui.section("انتقادات خاصة بنماذج اللغة الكبيرة", "LLM-specific critiques", "violet")
    ui.cards([
        {"title": "الهلوسة", "en": "Hallucination", "color": "violet", "body": "مراجع وأرقام مختلقة بثقة عالية — كل مخرج يحتاج تحققًا."},
        {"title": "التسرب والنظر للأمام", "en": "Leakage & look-ahead", "color": "rose", "body": "النموذج «يتذكر» نتائج الفترة التي يُطلب منه التنبؤ بها."},
        {"title": "عدم القابلية للتكرار", "en": "Non-reproducibility", "color": "orange", "body": "نماذج مغلقة تتغير، عشوائية التوليد، اعتماد على المطالبة."},
        {"title": "خطأ قياس غير كلاسيكي", "en": "Nonclassical error", "color": "teal", "body": "يختلف حسب اللغة والقطاع والطول — يحيّز في أي اتجاه."},
        {"title": "محاكاة البشر", "en": "Synthetic respondents", "color": "sky", "body": "استجابات محاكاة بدائل ضعيفة لتفضيلات البشر عبر الثقافات."},
        {"title": "التحيزات المضمّنة", "en": "Embedded biases", "color": "pink", "body": "تحيزات بيانات التدريب الثقافية واللغوية — العربية أقل تمثيلًا."},
    ], cols=3)

with tabs[4]:
    ui.section("أبعاد أخلاقية ومؤسسية", "Ethical & institutional dimensions", "slate")
    ui.ladder([
        ("قانون غودهارت والتنبؤ الأدائي", "Goodhart & performativity", "إذا استُخدمت درجة AI في سياسة فعلية، يتغير سلوك المتعاملين لإرضائها، ويعاد تشكيل البيانات المستقبلية."),
        ("العدالة", "Fairness", "اختلاف أداء المقاييس عبر البلدان واللغات واللهجات والقطاعات الصغيرة."),
        ("الخصوصية والسرية", "Privacy & confidentiality", "الحذر من إدخال نصوص مالية سرية في خدمات LLM خارجية؛ البيانات الاصطناعية ليست ضمانًا آليًا للخصوصية."),
        ("الترخيص والتوثيق", "Licensing & documentation", "توثيق الترخيص ونسخة الأداة والمطالبة والبذرة وبيانات التدريب المتاحة."),
        ("الأثر البيئي", "Environmental footprint", "توضيح الكلفة الحسابية عندما لا يحقق التعقيد فائدة قابلة للقياس."),
        ("المساءلة", "Accountability", "من يتحمل مسؤولية قرار سياسة مبني على نموذج لا يفهمه صانع القرار؟"),
    ])

with tabs[5]:
    ui.section("خمس مغالطات تتكرر في المقالات", "Five recurring fallacies", "amber")
    fl = pd.DataFrame([
        ("R² = 0.98 يثبت صحة النموذج الاقتصادي", "قد يكون نتيجة تسرب أو اتجاه مشترك أو إفراط في التوفيق"),
        ("الشبكة تحقق اتجاه الطلب إذن تعرّفنا على المرونة السببية", "الرتابة لا تحل الداخلية"),
        ("لدينا 10,000 مشاهدة اصطناعية فحجم العينة 10,000", "الملاحظات الاصطناعية ليست مرصودة مستقلة"),
        ("المؤشر الذي ينتجه LLM متغير جاهز للانحدار", "خطأ القياس قد يفسد المعاملات وفترات الثقة"),
        ("وجود Transformer يعني تفوقًا على VAR", "لا تفوق عام دون معيار بالزمن الحقيقي"),
    ], columns=["الادعاء الخاطئ", "الرد العلمي"])
    st.table(fl)
    ui.note("حتى أكتوبر 2026 **لا توجد** طريقة عامة تجعل كل LLM مناسبًا لقياس كل مفهوم اقتصادي، أو تحوّل أي شبكة موجّهة اقتصاديًا إلى مقدّر سببي صالح، "
            "أو تزيل مخاطر تغيّر الاقتصاد الحقيقي. ومستوى الأدلة متفاوت: مقالات محكّمة، أوراق مؤسسات، وأبحاث أولية.", "أين بقيت الحدود مفتوحة؟", "amber", "🚧")

with tabs[6]:
    ui.section("ردود الأدبيات: كيف تعالَج الانتقادات؟", "How the literature responds", "emerald")
    ui.cards([
        {"title": "استدلال صالح", "en": "Valid inference", "color": "emerald", "body": "DML، PDS، GenericML، PPI، تصحيح المتغيرات المولّدة، الاستدلال المطابق."},
        {"title": "بنية نظرية", "en": "Theory structure", "color": "violet", "body": "شبكات موجّهة بالنظرية، نماذج هيكلية عصبية، قيود ثبات."},
        {"title": "صدق زمني", "en": "Temporal honesty", "color": "orange", "body": "بيانات آنية، نماذج مؤرخة، تقييم متدحرج، Diebold–Mariano."},
        {"title": "شفافية", "en": "Transparency", "color": "sky", "body": "تسجيل مسبق، أكواد ونسخ منشورة، سجلات الوكلاء، تحليل المواصفات المتعددة."},
        {"title": "تحقق بشري", "en": "Human validation", "color": "rose", "body": "عينات ذهبية، اتفاق المقيّمين، تدقيق طبقي."},
        {"title": "تواضع منهجي", "en": "Methodological humility", "color": "teal", "body": "مقارنة الأركان الثلاثة ونشر حالات الفشل."},
    ], cols=3)
    ui.refs("Lucas (1976); Breiman (2001) والرد عليه؛ Mullainathan & Spiess (2017); Athey & Imbens (2019); Lazer et al. (2014); Obermeyer et al. (2019); "
            "Ludwig, Mullainathan & Rambachan (2025); Shin (2026); موسوعة رودان (2026) الفصل 34.")
