import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from core import ui

ui.hero(
    "النص والصورة كبيانات: الأدوات والمنهج",
    "Text, Images, Audio & Documents as Economic Data — Methods",
    "كيف نحوّل النص والصورة والصوت إلى متغيرات اقتصادية؟ خطوط المعالجة، التمثيلات من حقيبة الكلمات إلى نماذج اللغة الكبيرة، "
    "الرؤية الحاسوبية والأقمار الصناعية — ثم المشكلة القياسية الجوهرية: المتغير المولَّد يحمل خطأ قياس.",
    "indigo", "amber",
)

tabs = st.tabs([
    ":material/account_tree: خط معالجة النص",
    ":material/text_fields: تمثيلات النص",
    ":material/science: مختبر النص",
    ":material/image: الصور والأقمار",
    ":material/graphic_eq: الصوت والوثائق والشبكات",
    ":material/warning: مختبر خطأ القياس",
])

with tabs[0]:
    ui.section("من الوثيقة الخام إلى المتغير", "The text-as-data pipeline", "indigo")
    st.mermaid_chart(
        """
flowchart LR
    A[Corpus<br/>reports, news, transcripts] --> B[Preprocess<br/>tokenize, clean, dedupe, date-stamp]
    B --> C[Represent<br/>BoW, TF-IDF, embeddings]
    C --> D[Model<br/>dictionary, supervised, topic, LLM]
    D --> E[Aggregate<br/>doc to firm-quarter or country-month]
    E --> F[Validate<br/>gold labels, construct validity]
    F --> G[Econometric use<br/>with measurement-error correction]
"""
    )
    st.markdown(
        """
يلخص **Gentzkow, Kelly & Taddy (2019)** المسألة في ثلاث خطوات: (1) تمثيل النص بمصفوفة رقمية $\\mathbf C$، (2) ربطها بمتغير مستهدف $\\hat{\\mathbf V}=f(\\mathbf C)$،
(3) استخدام $\\hat{\\mathbf V}$ في التحليل اللاحق (وصف، تنبؤ، استدلال سببي). كل خطوة تتضمن قرارات بحثية (درجات حرية) يجب توثيقها.
        """
    )
    ui.chips([("Tokenization", "التقطيع"), ("Stop words", "كلمات الإيقاف"), ("Stemming / Lemmatization", "التجذير"), ("n-grams", "المتتاليات"),
              ("Document-term matrix", "مصفوفة الوثيقة–المصطلح"), ("Corpus", "المدوّنة"), ("Aggregation", "التجميع")])

with tabs[1]:
    ui.section("تطور تمثيلات النص", "From bag-of-words to LLMs", "violet")
    rep = pd.DataFrame([
        ("القواميس", "Dictionaries", "عدّ كلمات من قوائم محددة", "Loughran & McDonald (2011)؛ EPU (Baker, Bloom & Davis, 2016)", "شفاف؛ لكن السياق والنفي غائبان"),
        ("حقيبة الكلمات / TF-IDF", "Bag-of-words / TF-IDF", "تكرار الكلمات بأوزان الندرة", "Gentzkow & Shapiro (2010)", "أبعاد ضخمة متناثرة ⇒ لاسو"),
        ("الانحدار متعدد الحدود العكسي", "Multinomial inverse regression", "إسقاط النص على المتغير المستهدف", "Taddy (2013)", "فعال إحصائيًا للأبعاد العالية"),
        ("نماذج المواضيع", "Topic models (LDA)", "كل وثيقة مزيج من مواضيع", "Hansen, McMahon & Prat (2018)؛ Bybee et al. (2024)", "تفسير مفيد؛ عدد المواضيع اعتباطي"),
        ("التضمينات", "Word embeddings (Word2Vec, GloVe)", "متجهات كثيفة تلتقط التشابه الدلالي", "Ash & Chen؛ قياس الاستقطاب", "تحمل تحيزات المدوّنة"),
        ("المحوّلات السياقية", "BERT, FinBERT, RoBERTa", "تمثيل يعتمد على السياق", "تصنيف النبرة والتوقعات", "تحتاج ضبطًا دقيقًا بوسوم"),
        ("نماذج اللغة الكبيرة", "LLMs (GPT, Llama, Claude)", "تصنيف/استخراج بلا أمثلة أو بأمثلة قليلة", "AI Economy Score (Jha et al., 2026)", "اختلاف النماذج والمطالبات، التسرب"),
    ], columns=["التمثيل", "English", "الفكرة", "أمثلة اقتصادية", "المزايا/المحاذير"])
    st.dataframe(rep, hide_index=True, height=330)
    st.latex(r"\text{tf-idf}(t,d)=\text{tf}(t,d)\times\log\frac{N}{\text{df}(t)}")

with tabs[2]:
    ui.section("مختبر تفاعلي: قياس النبرة وخريطة الوثائق", "Interactive text lab", "pink")
    st.markdown("**تنبيه:** القوائم أدناه قصيرة ومصطنعة للتعليم فقط (مستوحاة من منطق قواميس النبرة المالية)، ولا تمثل قاموس Loughran–McDonald الأصلي.")
    pos = {"growth", "strong", "improve", "improved", "robust", "expansion", "gain", "gains", "recovery", "stable", "increase", "optimistic", "نمو", "قوي", "تحسن", "انتعاش", "استقرار", "ارتفاع", "تفاؤل"}
    neg = {"decline", "weak", "risk", "risks", "uncertainty", "loss", "losses", "recession", "slowdown", "volatility", "default", "crisis", "inflationary",
           "تراجع", "ضعيف", "مخاطر", "عدم", "يقين", "خسارة", "ركود", "تباطؤ", "أزمة", "تقلب"}
    txt = st.text_area("اكتب أو الصق نصًا (عربي أو إنجليزي)",
                       "The economy shows robust growth and stable employment, but inflationary risks and uncertainty remain elevated.", key="txt_in")
    toks = re.findall(r"[\w']+", txt.lower())
    p_cnt = sum(t in pos for t in toks); n_cnt = sum(t in neg for t in toks)
    score = (p_cnt - n_cnt) / max(p_cnt + n_cnt, 1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("عدد الكلمات", len(toks)); c2.metric("كلمات إيجابية", p_cnt); c3.metric("كلمات سلبية", n_cnt)
    c4.metric("مؤشر النبرة (−1 إلى +1)", f"{score:+.2f}")
    st.caption("لاحظ القصور: جملة «not weak» ستُحسب سلبية — القواميس تتجاهل النفي والسياق، وهذا ما تعالجه النماذج السياقية.")
    corpus = [
        ("The committee raised the policy rate to contain inflationary pressures.", "تشديد"),
        ("Inflation expectations remain elevated; further tightening may be appropriate.", "تشديد"),
        ("Higher interest rates are needed to bring inflation back to target.", "تشديد"),
        ("The committee lowered rates to support growth amid a slowdown.", "تيسير"),
        ("Weak demand and rising unemployment call for accommodative policy.", "تيسير"),
        ("Asset purchases will continue to ease financial conditions.", "تيسير"),
        ("Bank lending standards tightened and credit risk increased.", "مالي"),
        ("Financial stability risks rose with volatility in bond markets.", "مالي"),
        ("Defaults and losses in the banking sector raised systemic concerns.", "مالي"),
        ("Strong productivity gains supported robust output growth.", "نمو"),
        ("Investment and exports expanded, signalling a solid recovery.", "نمو"),
        ("Consumer spending improved as employment gains continued.", "نمو"),
    ]
    vec = TfidfVectorizer(stop_words="english")
    Xc = vec.fit_transform([c[0] for c in corpus])
    xy = TruncatedSVD(2, random_state=0).fit_transform(Xc)
    dfc = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "text": [c[0] for c in corpus], "topic": [c[1] for c in corpus]})
    fig = go.Figure()
    for i, (tpc, g) in enumerate(dfc.groupby("topic")):
        fig.add_trace(go.Scatter(x=g.x, y=g.y, mode="markers", marker=dict(size=15, color=ui.PLOT_COLORS[i]), name=tpc, text=g.text,
                                 hovertemplate="%{text}<extra></extra>"))
    ui.style_fig(fig, "خريطة جمل مصطنعة لبنك مركزي: TF-IDF ثم تفكيك القيم المفردة (LSA) إلى بعدين", height=470)
    fig.update_xaxes(title="المكوّن 1"); fig.update_yaxes(title="المكوّن 2")
    ui.show(fig)

with tabs[3]:
    ui.section("الرؤية الحاسوبية للاقتصاد", "Computer vision & remote sensing", "sky")
    st.mermaid_chart(
        """
flowchart LR
    A[Satellite / street / product images] --> B[Pretrained CNN or ViT<br/>ImageNet, foundation vision model]
    B --> C[Transfer learning<br/>fine-tune on proxy labels e.g. night lights]
    C --> D[Image features / embeddings]
    D --> E[Predict survey outcome<br/>consumption, wealth, crop yield]
    E --> F[Generated variable w-hat]
    F --> G[Econometric analysis + validation in the field]
"""
    )
    ui.cards([
        {"title": "التعلم بالنقل", "en": "Transfer learning (Jean et al., 2016)", "color": "sky", "body": "تدريب الشبكة على مهمة وفيرة الوسوم (الأضواء الليلية) ثم استخدام ميزاتها للتنبؤ بالفقر حيث الوسوم نادرة."},
        {"title": "الأقمار والنمو", "en": "IMF (2026) nowcasting with satellites", "color": "teal", "body": "أضواء ليلية + غابة عشوائية + مؤشرات تقليدية لتحسين تقديرات النمو في بلدان الفجوات الإحصائية."},
        {"title": "تحليل الوثائق", "en": "LayoutParser (Shen, Dell et al., 2021)", "color": "violet", "body": "استخراج الجداول التاريخية من الأرشيفات الممسوحة بالتعلم العميق — ثورة في التاريخ الاقتصادي."},
        {"title": "المحاذير", "en": "Caveats", "color": "rose", "body": "التشبع، الغيوم، تغيّر المستشعرات، كهرباء غير مرتبطة بالإنتاج، ضعف في المناطق الريفية الصغيرة."},
    ], cols=2)

with tabs[4]:
    ui.section("الصوت والوثائق والرسوم البيانية المعرفية", "Audio, documents and knowledge graphs", "teal")
    ui.cards([
        {"title": "الصوت", "en": "ASR + vocal tone", "color": "teal",
         "body": "تحويل الكلام إلى نص لمكالمات الأرباح ومؤتمرات البنوك المركزية؛ تحليل «طريقة القول» لا «ما قيل» فقط. فرّق بين خطأ التعرف على الكلام وخطأ تصنيف المشاعر."},
        {"title": "حل الكيانات", "en": "Entity resolution", "color": "orange",
         "body": "توحيد أسماء الشركات والأفراد عبر السجلات المتفرقة — أساس ربط قواعد البيانات الإدارية."},
        {"title": "شبكات التوريد", "en": "Supply-chain graphs", "color": "violet",
         "body": "استخلاص الروابط من التقارير والنصوص؛ الروابط الزائفة أو المفقودة تغيّر مركزية العقد ونتائج انتقال الصدمات."},
        {"title": "متعددة الوسائط", "en": "Multimodal LLMs", "color": "pink",
         "body": "استخراج إشارات من صور الرسوم المالية والجداول (دراسات 2026) — توليد ميزات لا يثبت صلاحية التحليل الفني."},
    ], cols=2)

with tabs[5]:
    ui.section("مختبر متحرك: كيف يحيّز خطأ التصنيف معامل الانحدار؟", "Lab — generated regressors and attenuation", "rose")
    st.markdown("لدينا متغير ثنائي حقيقي $X^*$ (مثلًا: «هل الإعلان الوظيفي يتطلب مهارات AI؟») أثره الحقيقي $\\beta=1$ على النتيجة. "
                "لكننا نستخدم وسمًا من مصنّف آلي $\\hat X$ بدقة متغيرة. شغّل الحركة: مع انخفاض الدقة **ينكمش** المعامل المقدّر نحو الصفر وتفقد فترة الثقة التغطية.")
    diff = st.toggle("خطأ تفاضلي (يعتمد على النتيجة) بدل الخطأ العشوائي", value=False, key="me_diff")

    @st.cache_data(show_spinner=False)
    def attenuation(differential: bool, reps: int = 200, n: int = 2000):
        rng = np.random.default_rng(21)
        accs = np.round(np.arange(1.0, 0.54, -0.05), 2)
        rows = []
        for a in accs:
            b = []
            for _ in range(reps):
                xs = rng.binomial(1, .4, n); y = 1.0 * xs + rng.normal(0, 1, n)
                if differential:
                    flip_p = np.clip((1 - a) * (1 + 0.8 * np.tanh(y)), 0, 1)
                else:
                    flip_p = np.full(n, 1 - a)
                xh = np.where(rng.uniform(size=n) < flip_p, 1 - xs, xs)
                b.append(np.polyfit(xh, y, 1)[0])
            b = np.array(b)
            rows.append((a, b))
        return rows

    rows = attenuation(diff)
    names = [f"{a:.0%}" for a, _ in rows]

    def att_tr(k):
        a, b = rows[k]
        return [go.Histogram(x=b, xbins=dict(start=-0.6, end=1.4, size=0.02), marker_color=ui.COLORS["rose"], opacity=.75, name="β̂ بالوسم الآلي")]

    fig = go.Figure(data=att_tr(0), frames=[go.Frame(data=att_tr(k), name=names[k],
                                                      layout=go.Layout(title_text=f"دقة المصنّف {names[k]} — متوسط β̂ = {rows[k][1].mean():.3f}"))
                                             for k in range(len(rows))])
    fig.add_vline(x=1.0, line_dash="dash", annotation_text="β الحقيقي = 1")
    fig.update_xaxes(title="β̂", range=[-0.6, 1.4]); fig.update_yaxes(title="التكرار")
    ui.style_fig(fig, f"دقة المصنّف {names[0]} — متوسط β̂ = {rows[0][1].mean():.3f}", legend_h=False)
    ui.add_play(fig, names, duration=800, prefix="الدقة: ", redraw=True)
    ui.show(fig)
    ui.note("حتى دقة **85%** تحيّز المعامل بقوة. ومع **الخطأ التفاضلي** (أخطاء ترتبط بالنتيجة، شائعة في نماذج اللغة) قد يتجه التحيز في **أي اتجاه**. "
            "الحلول (عينة تحقق بشرية، PPI، AI-PI، تصحيح Battaglia وآخرين، bootstrap الوسوم المقترنة) مفصّلة في صفحة **«ثورة القياس»**.",
            "الدرس", "rose", "📏")
    ui.refs("Gentzkow, Kelly & Taddy (2019) JEL; Taddy (2013); Jean et al. (2016) Science; Battaglia, Christensen, Hansen & Sacher (2024); Egami et al. (2023) DSL; Dell (2025) JEL.")
