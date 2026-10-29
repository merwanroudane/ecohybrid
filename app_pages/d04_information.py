import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "الدمج في المعلومات",
    "Integration at the information level — selecting, compressing and creating the variables that enter the model",
    "كل نموذج قياسي يرى العالم من خلال <b>مجموعة معلومات</b> محدودة اختارها الباحث. الذكاء الاصطناعي يوسّع هذه المجموعة بثلاث "
    "عمليات: <b>الاختيار</b> من آلاف المتغيرات، <b>الضغط</b> إلى عوامل قليلة، و<b>الإنشاء</b> من نصوص وصور وأصوات. لكل عملية "
    "رياضياتها ومخاطرها الخاصة على التقدير والاستدلال.",
    "sky", "emerald",
)

tabs = st.tabs([
    ":material/info: ما معنى الدمج في المعلومات؟",
    ":material/filter_alt: الاختيار في الأبعاد العالية",
    ":material/compress: الضغط: العوامل والمرمِّزات",
    ":material/image: الإنشاء من بيانات غير مهيكلة",
    ":material/straighten: خطأ القياس في المتغيرات المولدة",
    ":material/schedule: التواتر المختلط والتنبؤ الآني",
    ":material/visibility_off: التسرب والنظر إلى المستقبل",
    ":material/science: مختبر: الاختيار المفرد مقابل المزدوج",
    ":material/checklist: قائمة التحقق",
])

# --------------------------------------------------------------------------- 1
with tabs[0]:
    ui.section("مجموعة المعلومات: المكوِّن الذي ننساه", "The information set", "sky")
    st.markdown("نكتب البيانات الخام كلها $W$ (كل ما يمكن رصده)، والنموذج لا يرى $W$ بل تحويلًا منه:")
    st.latex(r"X=\phi(W),\qquad y=m(X;\theta)+u")
    st.markdown(
        "في القياس التقليدي $\\phi$ **يدوي**: الباحث يختار 5–10 متغيرات بحسب النظرية ويحولها (لوغاريتم، فرق). في الدمج يصبح $\\phi$ "
        "**متعلَّمًا** جزئيًا أو كليًا. وهنا ثلاث عمليات مختلفة:"
    )
    ui.cards([
        {"title": "الاختيار", "en": "Selection", "color": "sky",
         "body": "$X=W_S$ حيث $S\\subset\\{1,\\dots,p\\}$ مجموعة جزئية يختارها Lasso أو غابة. المتغيرات تبقى بمعناها الأصلي."},
        {"title": "الضغط", "en": "Compression", "color": "emerald",
         "body": "$X=F=\\Lambda^+W$ عوامل قليلة تلخص مئات السلاسل (PCA، Autoencoder). المتغيرات الجديدة مجردة."},
        {"title": "الإنشاء", "en": "Creation", "color": "amber",
         "body": "$X=h(\\text{نص، صورة})$ متغير جديد تمامًا لم يكن موجودًا رقميًا: معنويات، شدة، تصنيف، دخل من الضوء الليلي."},
    ], cols=3)
    ui.note(
        "**لماذا هذا دمج وليس مجرد «تحضير بيانات»؟** لأن $\\phi$ يحدد **معنى** المعلمة المقدَّرة. إذا اختار Lasso ضوابط مختلفة تغيّر "
        "معنى أثر المعالجة؛ وإذا كانت المعنويات مقيسة بخطأ تغيّر حجم المعامل. لذلك يجب أن يُعامل $\\phi$ المتعلَّم كجزء من المقدّر "
        "بكل عدم يقينه، لا كخطوة تمهيدية بريئة.",
        "النقطة الجوهرية", "violet", "🎯",
    )
    st.markdown("#### متى نحتاج الدمج في المعلومات؟")
    st.markdown(
        "- **$p$ كبير مقارنة بـ $n$:** مئات الضوابط المحتملة، أو 128 سلسلة كلية شهرية (FRED-MD) لـ 300 شهر.\n"
        "- **المفهوم غير مقاس رقميًا:** حالة عدم اليقين، النبرة، الانتباه، جودة المؤسسات.\n"
        "- **البيانات بتواترات مختلفة:** يومية وشهرية وفصلية في نموذج واحد.\n"
        "- **البيانات التقليدية غائبة:** الدخل في مناطق بلا مسوح، النشاط في الوقت الحقيقي."
    )

# --------------------------------------------------------------------------- 2
with tabs[1]:
    ui.section("Lasso: الاختيار بالتنظيم", "Selection by L1 penalty", "sky")
    st.latex(r"\hat\beta^{\text{Lasso}}=\arg\min_\beta\ \frac1{2n}\sum_{i=1}^{n}(y_i-x_i'\beta)^2+\lambda\sum_{j=1}^{p}|\beta_j|")
    st.markdown(
        "العقوبة $|\\beta_j|$ تدفع كثيرًا من المعاملات إلى **الصفر تمامًا**، فيحدث اختيار تلقائي. شرط عمله الجيد: **الندرة التقريبية** "
        "(عدد قليل من المتغيرات المهمة فعلًا) وعدم ترابط شديد بين المتغيرات. $\\lambda$ يُختار بالتحقق المتقاطع (للتنبؤ) أو "
        "بقاعدة نظرية (للاستدلال، مثل قاعدة Belloni et al.)."
    )
    ui.section("الفخ: الاختيار المفرد يحيّز الأثر السببي", "Why naive post-Lasso inference fails", "rose")
    st.markdown("لنموذج المعالجة $Y=\\theta D+X'\\beta+U$ و$D=X'\\gamma+V$. الطريقة الساذجة: Lasso لـ $Y$ على $(D,X)$ ثم OLS على المختار.")
    st.markdown(
        "**المشكلة:** متغير $X_j$ يؤثر **قليلًا** على $Y$ لكنه يؤثر **كثيرًا** على $D$. Lasso لمعادلة $Y$ قد يستبعده لأن أثره المباشر "
        "صغير، لكن استبعاده يُحدث **تحيز متغير محذوف**:"
    )
    st.latex(r"\text{التحيز}\approx\frac{\operatorname{Cov}(D,X_j)}{\operatorname{Var}(D)}\cdot\beta_j\neq0")
    st.markdown("ولأن الاستبعاد يحدث بشكل عشوائي من عينة لأخرى، يصبح توزيع $\\hat\\theta$ **غير طبيعي** وفترات الثقة خاطئة.")
    ui.section("الحل: الاختيار المزدوج (Post-Double-Selection)", "Belloni, Chernozhukov & Hansen (2014)", "emerald")
    ui.ladder([
        ("Lasso لـ Y على X", "Outcome selection", "اختر $S_Y$: المتغيرات التي تتنبأ بالنتيجة."),
        ("Lasso لـ D على X", "Treatment selection", "اختر $S_D$: المتغيرات التي تتنبأ بالمعالجة. هذه الخطوة تلتقط المربكات التي فاتت الأولى."),
        ("OLS على الاتحاد", "Final regression", "انحدر $Y$ على $D$ و$X_{S_Y\\cup S_D}$ بالمربعات الصغرى العادية، واستخدم أخطاء معيارية قوية."),
    ])
    st.markdown(
        "**لماذا ينجح؟** المتغير المربك يجب أن يرتبط بـ $Y$ **و** بـ $D$. لكي يُفقد يجب أن يفوت **كلا** الاختيارين، أي أن يكون أثره "
        "صغيرًا في المعادلتين معًا، وعندها يكون التحيز الناتج حاصل ضرب صغيرين فيصبح مهملًا. هذه هي نفس فكرة التعامد في DML، "
        "وفعلًا الاختيار المزدوج حالة خاصة من DML بمتعلم Lasso."
    )
    ui.refs("Tibshirani (1996) JRSS-B 58(1)؛ Belloni, Chernozhukov & Hansen (2014) REStud 81(2) و J. Economic Perspectives 28(2)؛ Leeb & Pötscher (2005) Econometric Theory 21(1).")

# --------------------------------------------------------------------------- 3
with tabs[2]:
    ui.section("نموذج العوامل: مئات السلاسل في بضعة أرقام", "Factor models and diffusion indexes", "emerald")
    st.latex(r"x_{it}=\lambda_i'F_t+e_{it},\qquad i=1,\dots,N,\ \ F_t\in\mathbb R^{r},\ r\ll N")
    st.markdown(
        "الفكرة: معظم تقلبات 128 مؤشرًا كليًا تقودها **قوى مشتركة قليلة** (النشاط الحقيقي، الأسعار، المال). المكونات الرئيسية "
        "تقدّر $F_t$ كأول $r$ متجهات ذاتية لمصفوفة التغاير. ثم يُستخدم $\\hat F_t$ كمتغيرات في انحدار التنبؤ (مؤشر الانتشار):"
    )
    st.latex(r"y_{t+h}=\alpha+\beta'\hat F_t+\gamma(L)y_t+\varepsilon_{t+h}")
    st.markdown(
        "أثبت Stock & Watson (2002) أن $\\hat F_t$ يتقارب إلى $F_t$ (حتى دوران) عندما يكبر $N$ و$T$ معًا، وأن الخطأ الناتج عن "
        "تقدير العوامل يصبح مهملًا في انحدار التنبؤ عندما $\\sqrt T/N\\to0$. هذا مهم: **المتغير المولَّد هنا لا يحيّز** في الحالات "
        "الكبيرة، بخلاف الكثير من المتغيرات المولدة بالذكاء الاصطناعي."
    )
    ui.section("من الخطي إلى اللاخطي", "Beyond PCA", "violet")
    ui.cards([
        {"title": "المتنبئات الموجهة", "en": "Targeted predictors (Bai & Ng 2008)", "color": "violet",
         "body": "PCA يلخص التباين لا العلاقة بـ $y$. الأفضل: اختيار المتغيرات المرتبطة بالهدف أولًا (Lasso أو اختبارات) ثم PCA عليها."},
        {"title": "PCA المتناثر", "en": "Sparse PCA", "color": "sky",
         "body": "عوامل كل منها يعتمد على عدد قليل من السلاسل، فيصبح لها اسم اقتصادي (عامل أسعار، عامل سوق عمل)."},
        {"title": "المرمِّز الذاتي", "en": "Autoencoder", "color": "emerald",
         "body": "شبكة $W\\to F\\to\\hat W$ بعنق ضيق. تلتقط بنية لاخطية لا يراها PCA. PCA هو حالة خاصة (مرمز خطي)."},
        {"title": "عوامل بالحد الأدنى من الإشراف", "en": "Supervised factors", "color": "amber",
         "body": "تدريب الضغط ليتنبأ بـ $y$ مباشرة (PLS، شبكات ذات رأسين). أقوى تنبؤيًا وأخطر في الإفراط."},
    ], cols=2)
    st.latex(r"\min_{\text{enc},\text{dec}}\ \frac1{NT}\sum_{t}\big\|W_t-\text{dec}\big(\text{enc}(W_t)\big)\big\|^2,\qquad \hat F_t=\text{enc}(W_t)")
    ui.refs("Stock & Watson (2002) JASA 97(460) و JBES 20(2)؛ Bai & Ng (2008) J. Econometrics 146(2)؛ McCracken & Ng (2016) JBES 34(4) FRED-MD؛ Gu, Kelly & Xiu (2021) J. Econometrics 222(1) autoencoder asset pricing.")

# --------------------------------------------------------------------------- 4
with tabs[3]:
    ui.section("خط إنتاج المتغير من بيانات غير مهيكلة", "From raw text and images to a variable", "amber")
    st.mermaid_chart(
        """
flowchart LR
    R[Raw data<br/>text, images, audio] --> P[Preprocess<br/>clean, segment, date-stamp]
    P --> M[Model<br/>dictionary / embeddings / LLM / CNN]
    M --> S[Score per unit<br/>sentiment, label, intensity]
    S --> A[Aggregate<br/>firm-month, region-year]
    A --> V[Validate<br/>human-coded sample]
    V --> E[Econometric model<br/>with measurement-error correction]
"""
    )
    ui.section("أجيال قياس النص", "Generations of text measurement", "violet")
    ui.ladder([
        ("المعاجم", "Dictionaries", "عدّ الكلمات الإيجابية والسلبية بقائمة جاهزة (Loughran & McDonald 2011 للنصوص المالية). شفاف وسريع، لكنه "
         "يتجاهل السياق والنفي («ليس جيدًا»)."),
        ("حقيبة الكلمات + تعلم آلي", "Bag-of-words + ML", "تمثيل النص بتكرارات الكلمات ثم Lasso أو غابة لتصنيفه بحسب عينة مرمَّزة. "
         "يتعلم الكلمات المهمة من البيانات."),
        ("التضمينات", "Embeddings (word2vec, BERT)", "كل نص يصبح متجهًا كثيفًا يحفظ المعنى. BERT مضبوط على عينة مرمَّزة يفهم السياق والنفي."),
        ("النماذج اللغوية الكبيرة", "LLMs zero/few-shot", "مطالبة تصف المفهوم والمقياس، والنموذج يعطي درجة لكل نص دون تدريب. مرن جدًا، "
         "لكنه حساس للمطالبة وللإصدار ويحمل معرفة لاحقة."),
    ])
    ui.section("الصور والأقمار الصناعية", "Images and remote sensing", "teal")
    st.markdown(
        "الأضواء الليلية من الأقمار الصناعية ترتبط بالنشاط الاقتصادي، واستُخدمت لتقدير الدخل في مناطق بلا إحصاءات "
        "(Henderson, Storeygard & Weil 2012). Jean et al. (2016) استخدموا **التعلم بالنقل**: شبكة تلافيفية تتعلم التنبؤ بالأضواء الليلية "
        "من صور النهار، ثم تُستخدم ميزاتها للتنبؤ بالفقر من مسوح قليلة. النتيجة متغير «فقر مقدَّر» لكل قرية."
    )
    st.latex(r"\widehat{\text{poverty}}_v=g\big(\text{CNN}_{\text{features}}(\text{image}_v)\big),\qquad g\ \text{مدرَّبة على القرى المسوحة فقط}")
    ui.note(
        "**المشترك بين كل هذه الطرق:** المتغير الناتج **تنبؤ** لا قياس مباشر. وله خطأ. السؤال القياسي ليس «هل النموذج دقيق بنسبة 90%؟» "
        "بل «كيف يؤثر خطؤه البالغ 10% في المعامل الذي سأقدّره، وهل هذا الخطأ عشوائي أم منهجي؟». التبويب التالي يجيب.",
        "تنبيه", "amber", "⚠️",
    )
    ui.refs("Loughran & McDonald (2011) J. Finance 66(1)؛ Gentzkow, Kelly & Taddy (2019) J. Economic Literature 57(3)؛ Henderson, Storeygard & Weil (2012) AER 102(2)؛ "
            "Jean et al. (2016) Science 353(6301)؛ Ash & Hansen (2023) Annual Review of Economics 15.")

# --------------------------------------------------------------------------- 5
with tabs[4]:
    ui.section("ماذا يفعل خطأ المتغير المولَّد بالانحدار؟", "Measurement error in AI-generated variables", "rose")
    st.markdown("لنفترض الانحدار الحقيقي $y=\\beta x+u$ لكننا نرى $\\hat x=x+v$ من نموذج ذكاء اصطناعي.")
    st.markdown("#### الحالة 1: خطأ في المتغير المستقل، كلاسيكي")
    st.latex(r"\operatorname{plim}\hat\beta=\beta\cdot\underbrace{\frac{\sigma_x^2}{\sigma_x^2+\sigma_v^2}}_{\lambda\in(0,1)}")
    st.markdown("**التخفيف (Attenuation):** المعامل ينكمش نحو الصفر. أداة دقتها مرتفعة لكن تباين خطئها 25% من تباين المتغير تعطي $\\lambda=0.8$.")
    st.markdown("#### الحالة 2: خطأ في المتغير التابع")
    st.latex(r"\hat y=y+w,\quad \operatorname{Cov}(w,x)=0\ \Rightarrow\ \hat\beta\ \text{غير متحيز لكن أقل دقة}")
    st.markdown("**لكن** إذا كان الخطأ مرتبطًا بـ $x$ (النموذج يبالغ في تقدير الفقر في المناطق الريفية مثلًا) يصبح $\\hat\\beta$ متحيزًا مباشرة.")
    st.markdown("#### الحالة 3: خطأ غير كلاسيكي")
    st.latex(r"\operatorname{Cov}(v,x)\neq0\ \ \text{أو}\ \ \operatorname{Cov}(v,u)\neq0\ \Rightarrow\ \text{اتجاه التحيز غير معروف}")
    st.markdown(
        "نماذج التنبؤ **تنكمش نحو المتوسط** بطبعها (تتنبأ بقيم أقل تطرفًا)، فخطؤها مرتبط سالبًا بالقيمة الحقيقية: $\\operatorname{Cov}(v,x)<0$. "
        "هذا يجعل الخطأ غير كلاسيكي **حكمًا**، وقد يضخم المعامل بدل أن يخففه."
    )
    ui.section("الحلول", "Corrections", "emerald")
    sol = pd.DataFrame([
        ("عينة تحقق + PPI", "عينة صغيرة مقيسة بدقة بشريًا", "تصحيح الفرق بين التنبؤ والحقيقة على العينة المرمَّزة", "Angelopoulos et al. (2023)"),
        ("DSL", "عينة مرمَّزة باحتمالات معروفة", "مقدّر قوي مزدوج يجمع التنبؤ والترميز", "Egami et al. (2023)"),
        ("متغير آلي بقياس ثانٍ", "نموذجان مستقلان الخطأ", "قياس ثانٍ كأداة للأول يزيل التخفيف الكلاسيكي", "Durbin (1954)؛ Hausman (2001)"),
        ("تصحيح التباين للمتغير المولَّد", "متغير من مرحلة أولى مقدَّرة", "Murphy–Topel أو bootstrap لكامل السلسلة", "Pagan (1984)؛ Murphy & Topel (1985)"),
        ("القياس كأداة لا كهدف", "تقييم الأداة على المفهوم لا على الدقة", "إطار Ludwig, Mullainathan & Rambachan", "Ludwig et al. (2025)"),
    ], columns=["الحل", "ما يتطلبه", "الفكرة", "المرجع"])
    st.dataframe(sol, hide_index=True)
    ui.refs("Pagan (1984) International Economic Review 25(1)؛ Bound, Brown & Mathiowetz (2001) Handbook of Econometrics 5؛ Angelopoulos et al. (2023) Science 382؛ Egami et al. (2023) NeurIPS.")

# --------------------------------------------------------------------------- 6
with tabs[5]:
    ui.section("دمج تواترات مختلفة: MIDAS والتعلم الآلي", "Mixed-frequency data and nowcasting", "teal")
    st.markdown("نريد الناتج الفصلي $y_t$ لكن لدينا مؤشرات شهرية أو يومية $x^{(m)}$. إدخال كل الفجوات اليومية يعطي مئات المعاملات. MIDAS يقيدها بدالة أوزان:")
    st.latex(r"y_t=\alpha+\beta\sum_{k=0}^{K}w(k;\vartheta)\,x^{(m)}_{t-k/m}+\varepsilon_t,\qquad w(k;\vartheta)=\frac{e^{\vartheta_1k+\vartheta_2k^2}}{\sum_je^{\vartheta_1j+\vartheta_2j^2}}")
    st.markdown(
        "معلمتان $\\vartheta$ بدل $K$ معاملًا. **الدمج مع التعلم الآلي** يأتي بعدة صور: MIDAS مع Lasso متجمّع (Sparse-group Lasso) لاختيار "
        "المؤشرات من مئات المرشحين (Babii, Ghysels & Striaukas 2022)، أو غابات/شبكات تأخذ الفجوات عالية التواتر مباشرة مع تنظيم، أو "
        "ARDL-MIDAS-Transformer حيث يتعلم الانتباه الأوزان بدل دالة ثابتة."
    )
    ui.section("الحافة غير المتساوية", "The ragged edge", "amber")
    st.markdown(
        "في أي يوم من الربع، بعض المؤشرات صدرت وبعضها لم يصدر. نموذج التنبؤ الآني يجب أن يتعامل مع **أنماط فقد متغيرة** في نهاية العينة. "
        "الحلول: نماذج فضاء الحالة مع مرشح كالمان (Giannone, Reichlin & Small 2008)، أو تدريب نماذج تعلم آلي على كل نمط إصدار، "
        "أو تعويض آخر القيم بالتنبؤ. الشرط الحاسم: **محاكاة توفر البيانات كما كان في الوقت الحقيقي**."
    )
    ui.refs("Ghysels, Santa-Clara & Valkanov (2004)؛ Giannone, Reichlin & Small (2008) J. Monetary Economics 55(4)؛ Babii, Ghysels & Striaukas (2022) JBES 40(3).")

# --------------------------------------------------------------------------- 7
with tabs[6]:
    ui.section("أخطر ثغرة في الدمج المعلوماتي: معلومات من المستقبل", "Leakage and look-ahead bias", "rose")
    ui.cards([
        {"title": "إصدارات البيانات", "en": "Data vintages", "color": "rose",
         "body": "الناتج والتوظيف يُراجعان لسنوات. استخدام القيم النهائية للتنبؤ بالماضي يعطي دقة لم تكن ممكنة. "
                 "الحل: قواعد البيانات في الوقت الحقيقي (ALFRED، Philadelphia Fed RTDSM)."},
        {"title": "ملاءمة التحويل على كل العينة", "en": "Full-sample transforms", "color": "orange",
         "body": "PCA أو التعيير أو التعويض على كامل العينة يُدخل معلومات فترة الاختبار في التدريب. يجب إعادة الملاءمة في كل نافذة."},
        {"title": "النماذج اللغوية تعرف المستقبل", "en": "LLM look-ahead bias", "color": "amber",
         "body": "نموذج دُرّب حتى 2024 ويقيّم تقرير شركة من 2015 قد «يعرف» ما حدث لها. المعنويات المقاسة تحمل نتيجة المستقبل."},
        {"title": "الاختيار بالنتيجة", "en": "Selection on outcome", "color": "violet",
         "body": "اختيار المؤشرات لأنها تنبأت جيدًا بفترة الاختبار ثم الإبلاغ عن دقتها في الفترة نفسها."},
    ], cols=2)
    ui.section("كيف نختبر تسرب النماذج اللغوية؟", "Testing LLM look-ahead", "violet")
    st.markdown(
        "- **إخفاء الهوية:** استبدال أسماء الشركات والتواريخ برموز؛ إذا انخفضت القدرة التنبؤية بشدة فالنموذج كان يتذكر لا يفهم.\n"
        "- **فترة ما بعد التدريب:** تقييم على نصوص بعد تاريخ معرفة النموذج فقط.\n"
        "- **نماذج بتاريخ معرفة محدد:** نماذج مدرّبة على نصوص حتى تاريخ معين (Time-stamped LMs).\n"
        "- **اختبار الاستدعاء:** سؤال النموذج مباشرة عن الحدث اللاحق؛ إن عرفه فالتسرب ممكن."
    )
    ui.refs("Croushore & Stark (2001) J. Econometrics 105(1)؛ Glasserman & Lin (2023)؛ Sarkar & Vafa (2024) «Lookahead bias in pretrained language models»؛ Ludwig et al. (2025).")

# --------------------------------------------------------------------------- 8
with tabs[7]:
    ui.section("مختبر مونت كارلو: لماذا نختار مرتين؟", "Single vs double selection", "emerald")
    st.markdown(
        "نولد $Y=\\theta D+X'\\beta+U$ و$D=X'\\gamma+V$ مع $\\theta=1$. بعض الضوابط تؤثر **ضعيفًا** في $Y$ و**قويًا** في $D$ — هذه هي "
        "المربكات التي يفقدها الاختيار المفرد. نقارن ثلاثة مقدرات عبر عينات متكررة:"
    )
    c1, c2, c3 = st.columns(3)
    n = c1.select_slider("حجم العينة n", [100, 200, 400], value=200, key="pds_n")
    p = c2.select_slider("عدد الضوابط p", [50, 100, 200], value=100, key="pds_p")
    reps = c3.select_slider("عدد التكرارات", [50, 100, 200], value=100, key="pds_reps")

    @st.cache_data(show_spinner="تشغيل المحاكاة...")
    def mc(n, p, reps, seed=0):
        from sklearn.linear_model import Lasso
        rng = np.random.default_rng(seed)
        beta = np.zeros(p)
        gamma = np.zeros(p)
        beta[:5] = 0.15        # weak effect on Y
        gamma[:5] = 0.9        # strong effect on D -> confounders
        beta[5:10] = 1.0       # pure outcome predictors
        lam = lambda y: 1.1 * np.std(y) * np.sqrt(2 * np.log(2 * p) / n)  # noqa: E731  plug-in style penalty
        out = {"single": [], "double": [], "oracle": []}
        for _ in range(reps):
            X = rng.normal(size=(n, p))
            D = X @ gamma + rng.normal(size=n)
            Y = 1.0 * D + X @ beta + rng.normal(size=n)
            Z = np.column_stack([D, X])
            sy = Lasso(alpha=lam(Y)).fit(Z, Y)
            S1 = np.flatnonzero(sy.coef_[1:] != 0)
            sd = Lasso(alpha=lam(D)).fit(X, D)
            S2 = np.flatnonzero(sd.coef_ != 0)
            for name, S in [("single", S1), ("double", np.union1d(S1, S2)), ("oracle", np.arange(10))]:
                W = np.column_stack([np.ones(n), D, X[:, S]])
                out[name].append(np.linalg.lstsq(W, Y, rcond=None)[0][1])
        return {k: np.array(v) for k, v in out.items()}

    r = mc(n, p, reps)
    lbl = {"single": "اختيار مفرد (Lasso لـ Y فقط)", "double": "اختيار مزدوج (BCH 2014)", "oracle": "المقدّر المثالي (يعرف الضوابط الصحيحة)"}
    col = {"single": "#e11d48", "double": "#0d9488", "oracle": "#7c3aed"}
    fig = go.Figure()
    for k in ["single", "double", "oracle"]:
        fig.add_trace(go.Histogram(x=r[k], name=lbl[k], opacity=0.55, marker_color=col[k], nbinsx=30))
    fig.add_vline(x=1.0, line_dash="dash", line_color="#0f172a", annotation_text="θ الحقيقي = 1")
    fig.update_layout(barmode="overlay")
    ui.style_fig(fig, "توزيع θ̂ عبر العينات", height=440)
    ui.show(fig)
    st.dataframe(pd.DataFrame([(lbl[k], r[k].mean() - 1, r[k].std(), np.sqrt(np.mean((r[k] - 1) ** 2))) for k in r],
                              columns=["المقدّر", "التحيز", "الانحراف المعياري", "RMSE"]).round(3), hide_index=True)
    ui.note(
        "**القراءة:** الاختيار المفرد يفقد المربكات الضعيفة في معادلة Y، فيتحيز التوزيع **بعيدًا** عن 1. الاختيار المزدوج يلتقطها من "
        "معادلة D فيعود التوزيع متمركزًا قرب 1 وقريبًا من المقدّر المثالي. هذا هو الفرق بين «تعلم آلي للتنبؤ» و«تعلم آلي مصمم للاستدلال».",
        "ماذا يظهر المختبر؟", "emerald", "🧪",
    )

# --------------------------------------------------------------------------- 9
with tabs[8]:
    ui.section("قائمة تحقق للدمج في المعلومات", "Checklist", "slate")
    st.markdown(
        "- [ ] حددتُ هل العملية **اختيار** أم **ضغط** أم **إنشاء**، وعالجتُ مخاطر كل منها.\n"
        "- [ ] للاستدلال مع ضوابط كثيرة: اختيار مزدوج أو DML، لا Lasso مفرد.\n"
        "- [ ] كل تحويل (تعيير، PCA، تعويض) يُعاد ملاءمته داخل كل نافذة تدريب.\n"
        "- [ ] للمتغيرات المولدة: عينة تحقق بشرية عشوائية، وقياس نمط الخطأ، وتصحيح (PPI/DSL/IV).\n"
        "- [ ] للنصوص بنماذج لغوية: مطالبة وإصدار مثبتان مسبقًا، واختبار تسرب المستقبل.\n"
        "- [ ] للسلاسل الكلية: إصدارات البيانات في الوقت الحقيقي.\n"
        "- [ ] تقرير حساسية النتائج لاختيار الأداة والمطالبة وعدد العوامل."
    )
