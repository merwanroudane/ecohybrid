import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "الدمج في الشكل الدالي والمعلمات",
    "Integration in the functional form — semiparametric models, neural sieves, nonlinear dynamics and learned coefficients",
    "أكثر قرارات النموذج القياسي تعسفًا هو <b>الشكل الدالي</b>: نكتب $x'\\beta$ لأنها مريحة لا لأنها صحيحة. الدمج هنا يستبدل "
    "جزءًا من الدالة أو كلها بمكوِّن متعلَّم، أو يجعل المعاملات نفسها دوال متعلَّمة، مع الحفاظ على ما نحتاجه من تفسير واستدلال. "
    "سنرى أن بعض النماذج القياسية الشهيرة (STAR) هي في الحقيقة شبكات عصبية صغيرة.",
    "indigo", "teal",
)

tabs = st.tabs([
    ":material/function: لماذا الشكل الدالي موضع دمج؟",
    ":material/category: العائلة شبه المعلمية",
    ":material/neurology: الشبكة كمقدّر غربالي",
    ":material/ssid_chart: الديناميكيات اللاخطية",
    ":material/tune: المعاملات المتغيرة والمتعلَّمة",
    ":material/school: النظرية داخل الدالة",
    ":material/visibility: كيف نقرأ دالة متعلَّمة؟",
    ":material/science: مختبر: تقريب دالة مجهولة",
    ":material/checklist: قائمة التحقق",
])

# --------------------------------------------------------------------------- 1
with tabs[0]:
    ui.section("ثمن الشكل الدالي الخاطئ", "The cost of misspecification", "indigo")
    st.markdown("إذا كانت الحقيقة $y=f(x)+u$ وقدّرنا $x'\\beta$، فإن OLS يتقارب إلى **أفضل تقريب خطي**، لا إلى الأثر:")
    st.latex(r"\beta^*=\big(\mathbb E[xx']\big)^{-1}\mathbb E[x\,f(x)]")
    st.markdown(
        "- إذا كان أثر الحد الأدنى للأجور على التوظيف صغيرًا عند المستويات المنخفضة وكبيرًا عند المرتفعة، فمعامل خطي واحد **يخفي** "
        "الحقيقتين.\n"
        "- إذا كانت العلاقة على شكل U، قد يكون $\\beta^*\\approx0$ رغم أثر قوي.\n"
        "- الأثر الحدي $\\partial f/\\partial x$ قد يختلف إشارته بين مناطق مختلفة من البيانات."
    )
    ui.section("لكن البديل غير المعلمي الكامل له ثمن أيضًا: لعنة الأبعاد", "Curse of dimensionality", "rose")
    st.latex(r"\text{أفضل معدل تقارب لتقدير } f\in C^s(\mathbb R^d):\qquad n^{-\frac{s}{2s+d}}")
    st.markdown(
        "مع $s=2$ (دالة ملساء) و$d=1$ المعدل $n^{-0.4}$؛ مع $d=10$ يصبح $n^{-0.14}$؛ أي نحتاج عينات هائلة. لذلك الحل الوسط "
        "هو **شبه المعلمي**: جزء مرن حيث لا نعرف الشكل، وجزء معلمي حيث نريد التفسير. أو **بنية** تقلل الأبعاد الفعلية (جمعية، "
        "مؤشر واحد). والشبكات العميقة تنجح عمليًا لأنها تستغل بنية تركيبية في الدوال الحقيقية تقلل الأبعاد الفعلية."
    )
    d = np.array([1, 2, 5, 10, 20])
    nn = np.logspace(2, 6, 60)
    fig = go.Figure()
    for i, dd in enumerate(d):
        fig.add_trace(go.Scatter(x=nn, y=nn ** (-2 / (4 + dd)), name=f"d = {dd}", line=dict(color=ui.COLORS[ui.color_at(i)], width=2.5)))
    fig.update_xaxes(type="log", title="حجم العينة n")
    fig.update_yaxes(type="log", title="خطأ التقدير التقريبي")
    ui.style_fig(fig, "لعنة الأبعاد: الخطأ يتناقص ببطء أكبر كلما زادت الأبعاد (s = 2)", height=420)
    ui.show(fig)
    ui.refs("White (1980) International Economic Review 21(1)؛ Stone (1982) Annals of Statistics 10(4)؛ Bauer & Kohler (2019) Annals of Statistics 47(4).")

# --------------------------------------------------------------------------- 2
with tabs[1]:
    ui.section("خمسة نماذج تتوسط بين المعلمي واللامعلمي", "The semiparametric family", "teal")
    models = [
        ("النموذج جزئي الخطية", "Partially linear", r"y=\theta d+g(x)+u",
         "معلمة واحدة مفسَّرة $\\theta$ وجزء مرن $g$ يمتص الضوابط. **التعرّف:** $\\mathbb E[u\\mid d,x]=0$. **النسخة الذكية:** $g$ بغابة أو شبكة + DML."),
        ("نموذج المؤشر الواحد", "Single index", r"y=g(x'\beta)+u",
         "الدالة مجهولة لكنها تعتمد على تركيبة خطية واحدة. يقلل الأبعاد من $d$ إلى 1. $\\beta$ معرّفة حتى مقياس. **النسخة الذكية:** شبكة بخلية واحدة في الطبقة الأولى."),
        ("النموذج الجمعي", "Additive (GAM)", r"y=\alpha+\textstyle\sum_{j=1}^{d}g_j(x_j)+u",
         "كل متغير له دالة خاصة، بلا تفاعلات. معدل التقارب كأنه بعد واحد. قابل للرسم والتفسير. **النسخة الذكية:** الشبكات الجمعية العصبية (NAM)."),
        ("المعاملات المتغيرة", "Varying coefficients", r"y=x'\beta(z)+u",
         "البنية خطية في $x$ لكن الأثر يتغير بحسب $z$ (الزمن، الدورة، خصائص الفرد). **النسخة الذكية:** $\\beta(\\cdot)$ بشبكة أو غابة (MRF، tvwnn)."),
        ("الآثار غير المتجانسة", "Heterogeneous effects", r"y=\tau(x)\,d+\mu(x)+u",
         "أثر المعالجة نفسه دالة في الخصائص. **النسخة الذكية:** الغابات السببية، المتعلمات الوصفية (Meta-learners)."),
    ]
    for t, en, eq, body in models:
        with ui.tinted("slate"):
            st.markdown(f"**{t}** · <span class='ltr' style='color:#64748b'>{en}</span>", unsafe_allow_html=True)
            st.latex(eq)
            st.markdown(body)
    ui.note(
        "**كيف تختار؟** اسأل: أين أحتاج **التفسير**؟ ضع المعلمة هناك. وأين **لا أعرف** الشكل ولا أحتاج تفسيره؟ ضع المكوِّن المرن هناك. "
        "النموذج جزئي الخطية مثالي لسؤال سببي واحد؛ والمعاملات المتغيرة مثالية عندما يكون سؤالك «كيف يتغير الأثر؟».",
        "قاعدة الاختيار", "teal", "🧭",
    )
    ui.refs("Robinson (1988)؛ Ichimura (1993) J. Econometrics 58؛ Hastie & Tibshirani (1990) Generalized Additive Models؛ Hastie & Tibshirani (1993) JRSS-B 55(4)؛ Agarwal et al. (2021) NeurIPS NAM.")

# --------------------------------------------------------------------------- 3
with tabs[2]:
    ui.section("ما هو المقدّر الغربالي؟", "Sieve estimation", "violet")
    st.markdown("نقرّب الدالة المجهولة بتركيبة من دوال أساس، ونزيد عددها مع حجم العينة:")
    st.latex(r"g(x)\approx g_K(x)=\sum_{k=1}^{K}\theta_k\,\psi_k(x),\qquad K=K(n)\to\infty")
    st.markdown(
        "كثيرات الحدود والشرائح (Splines) أمثلة كلاسيكية. **الشبكة العصبية غربال أيضًا**، لكن دوال الأساس نفسها **متعلَّمة**:"
    )
    st.latex(r"g_K(x)=\beta_0+\sum_{k=1}^{K}\beta_k\,\sigma\big(\gamma_k'x+c_k\big)")
    st.markdown(
        "حيث $\\sigma$ دالة تفعيل (لوجستية، ReLU). هذه هي الشبكة ذات الطبقة المخفية الواحدة و$K$ خلية. أثبت Hornik, Stinchcombe & White "
        "(1989) أنها **مقرِّب شامل**: تقرّب أي دالة مستمرة بأي دقة إذا كانت $K$ كبيرة بما يكفي. ثم أثبت Chen & White (1999) معدلات "
        "تقارب الشبكات كمقدرات غربالية، وFarrell, Liang & Misra (2021) أن **الشبكات العميقة** بـ ReLU تحقق معدلات تكفي لشروط DML، "
        "أي يمكن استخدامها للأجزاء المزعجة مع استدلال صحيح على المعلمة الهدف."
    )
    ui.section("نموذج AR-NN: هجين مبني من الداخل", "Linear + neural, with a linearity test", "teal")
    st.latex(r"y_t=\phi'z_t+\sum_{j=1}^{h}\beta_j\,G\big(\gamma_j'z_t\big)+\varepsilon_t,\qquad z_t=(1,y_{t-1},\dots,y_{t-p})'")
    st.markdown(
        "Teräsvirta, Medeiros وآخرون جعلوا الشبكة **امتدادًا للنموذج الخطي**: الجزء $\\phi'z_t$ هو AR العادي، والخلايا تضيف اللاخطية. "
        "الميزة القياسية: يمكن **اختبار** الخطية ($H_0:\\beta_j=0$) قبل إضافة الخلايا، واختيار عدد الخلايا باختبارات متتابعة بدل التجربة "
        "العشوائية. هذا دمج في الشكل الدالي مع الحفاظ على أدوات الاستدلال."
    )
    ui.refs("Hornik, Stinchcombe & White (1989) Neural Networks 2(5)؛ Chen & White (1999) IEEE Trans. Information Theory 45(2)؛ Chen (2007) Handbook of Econometrics 6B؛ "
            "Farrell, Liang & Misra (2021) Econometrica 89(1)؛ Medeiros, Teräsvirta & Rech (2006) J. Forecasting 25(1).")

# --------------------------------------------------------------------------- 4
with tabs[3]:
    ui.section("من النماذج الانتقالية إلى الشبكات: الصلة المباشرة", "STAR models are tiny neural networks", "teal")
    st.markdown("نموذج الانحدار الذاتي ذو الانتقال الناعم اللوجستي (LSTAR) كما صاغه Teräsvirta (1994):")
    st.latex(r"y_t=\phi_1'z_t+\phi_2'z_t\cdot G(s_t;\gamma,c)+\varepsilon_t,\qquad G(s;\gamma,c)=\frac{1}{1+e^{-\gamma(s-c)}}")
    st.markdown(
        "دالة الانتقال $G$ **هي بالضبط** خلية عصبية بتفعيل لوجستي، مدخلها متغير الانتقال $s_t$. أي أن LSTAR شبكة بخلية مخفية واحدة "
        "تتحكم في الانتقال بين نظامين خطيين. والشبكة الأكبر ببساطة تسمح بأنظمة وانتقالات أكثر، وبمتغيرات انتقال يتعلمها النموذج بدل "
        "أن يحددها الباحث."
    )
    c1, c2 = st.columns(2)
    with c1:
        gamma = st.slider("حدة الانتقال γ", 0.5, 30.0, 4.0, 0.5, key="star_g")
    with c2:
        cc = st.slider("موقع الانتقال c", -2.0, 2.0, 0.0, 0.1, key="star_c")
    s = np.linspace(-3, 3, 300)
    G = 1 / (1 + np.exp(-gamma * (s - cc)))
    fig = go.Figure(go.Scatter(x=s, y=G, line=dict(color="#0d9488", width=3), name="G(s)"))
    fig.update_xaxes(title="متغير الانتقال s (مثلًا نمو الناتج المتأخر)")
    fig.update_yaxes(title="وزن النظام الثاني", range=[-0.05, 1.05])
    ui.style_fig(fig, "دالة الانتقال اللوجستية = خلية عصبية واحدة", height=360)
    ui.show(fig)
    st.caption("γ كبيرة ← انتقال حاد يقترب من نموذج العتبة (TAR). γ صغيرة ← انتقال تدريجي.")
    ui.section("الاستجابات الدفعية في النماذج اللاخطية", "Generalized impulse responses", "violet")
    st.markdown(
        "في VAR الخطي، الاستجابة لصدمة لا تعتمد على حجمها ولا إشارتها ولا حالة الاقتصاد. في النماذج اللاخطية (STAR، NN-VAR) كل ذلك "
        "يختلف، فنستخدم **الاستجابة الدفعية المعممة** (Koop, Pesaran & Potter 1996) بالمحاكاة:"
    )
    st.latex(r"\text{GIRF}(h,\delta,\omega_{t-1})=\mathbb E\big[y_{t+h}\mid\varepsilon_t=\delta,\omega_{t-1}\big]-\mathbb E\big[y_{t+h}\mid\omega_{t-1}\big]")
    st.markdown(
        "نحاكي مسارات المستقبل مرتين، مرة مع الصدمة ومرة بدونها، من الحالة نفسها، ونأخذ متوسط الفرق. بهذا يمكن أن نقول: «صدمة السياسة "
        "النقدية أقوى في الركود منها في التوسع» — وهو سؤال لا يستطيع VAR الخطي طرحه أصلًا. (صفحة هجين السلاسل الزمنية فيها مختبر GIRF)."
    )
    ui.refs("Teräsvirta (1994) JASA 89(425)؛ Tong (1990) Non-linear Time Series؛ Koop, Pesaran & Potter (1996) J. Econometrics 74(1)؛ Kuan & White (1994) Econometric Reviews 13(1).")

# --------------------------------------------------------------------------- 5
with tabs[4]:
    ui.section("عندما لا يكون الأثر ثابتًا", "Coefficients that move", "amber")
    st.markdown("النموذج القياسي يفترض $\\beta$ ثابتة. ثلاث طرق لجعلها تتحرك، من الأكثر بنية إلى الأكثر مرونة:")
    ui.ladder([
        ("معاملات متغيرة زمنيًا بفضاء الحالة", "TVP via state space",
         "$y_t=x_t'\\beta_t+\\varepsilon_t,\\ \\beta_t=\\beta_{t-1}+\\eta_t$ مع مرشح كالمان. المعاملات تتحرك ببطء كسير عشوائي. لا تعرف <b>لماذا</b> تتحرك."),
        ("معاملات دالة في متغيرات حالة", "Varying coefficients β(s_t)",
         "$\\beta_t=\\beta(s_t)$ حيث $s_t$ مشاهَدة (البطالة، التضخم). الشبكة أو الغابة تتعلم الدالة. الآن نعرف <b>ما الذي يحرك</b> الأثر."),
        ("الغابة الكلية العشوائية", "Macroeconomic Random Forest",
         "Goulet Coulombe (2024): كل ورقة في الشجرة تحتوي انحدارًا خطيًا صغيرًا، والتقسيم يحدث على متغيرات الحالة. النتيجة: "
         "معاملات «تشبه» المعاملات القياسية لكنها تختلف عبر الأنظمة، مع تنظيم يمنع الإفراط."),
        ("شبكات بأوزان متغيرة زمنيًا", "Time-varying weight networks",
         "أوزان الشبكة نفسها دوال في الزمن (Rudd, Bondell & Silver 2026؛ مكتبة tvwnn)، فتتكيف العلاقة اللاخطية مع تغير البيئة."),
    ])
    st.latex(r"\hat\beta(s)=\frac1B\sum_{b=1}^{B}\hat\beta_{\ell_b(s)}^{(b)},\qquad \hat\beta_{\ell}^{(b)}=\arg\min_\beta\sum_{t\in\ell}(y_t-x_t'\beta)^2+\lambda\|\beta-\beta_{\text{parent}}\|^2")
    st.caption("في MRF: معامل الحالة s هو متوسط معاملات الأوراق التي تقع فيها s عبر الأشجار، مع تقليص كل ورقة نحو أبيها.")
    ui.note(
        "**الفرق الجوهري عن الغابة العادية:** الغابة العادية تعطي تنبؤًا $\\hat y$. MRF يعطي **معاملات** $\\hat\\beta_t$ يمكن رسمها وتفسيرها: "
        "«ميل منحنى فيليبس انخفض بعد 1990 وارتفع في 2021». هذا دمج يحافظ على لغة القياس الاقتصادي مع مرونة التعلم الآلي.",
        "لماذا هذا دمج قياسي حقيقي؟", "amber", "🎯",
    )
    ui.refs("Hamilton (1994) Time Series Analysis ch. 13؛ Primiceri (2005) REStud 72(3)؛ Goulet Coulombe (2024) J. Applied Econometrics؛ Rudd, Bondell & Silver (2026).")

# --------------------------------------------------------------------------- 6
with tabs[5]:
    ui.section("إدخال الاقتصاد في الدالة نفسها", "Economic structure inside the function", "emerald")
    ui.cards([
        {"title": "قيود الشكل", "en": "Shape constraints", "color": "emerald",
         "body": "رتابة (الطلب ينخفض بالسعر)، تقعر (عوائد متناقصة)، تجانس (دالة إنتاج بعوائد ثابتة). تُفرض بالمعمارية: أوزان موجبة + تفعيل رتيب = شبكة رتيبة بالبناء."},
        {"title": "المعادلة كطبقة", "en": "Equation as a layer", "color": "teal",
         "body": "معادلة GARCH كبوابة إخراج في LSTM (Zhao et al. 2024)، أو معادلة الطلب كطبقة أخيرة. المعلمات القياسية تبقى قابلة للقراءة بعد التدريب."},
        {"title": "التفسير بالتصميم", "en": "Interpretable by design", "color": "sky",
         "body": "الشبكات الجمعية (NAM) تعطي دالة لكل متغير يمكن رسمها. KRLS يعطي أثرًا حديًا لكل مشاهدة بصيغة مغلقة."},
        {"title": "الثوابت الاقتصادية", "en": "Invariances", "color": "violet",
         "body": "عدم التأثر بالوحدات النقدية (تجانس من الدرجة صفر في الأسعار والدخل)، التماثل بين السلع. تقلل ما يجب تعلمه."},
    ], cols=2)
    st.latex(r"f(p,y)=\text{softplus}\big(w_2^{\top}\,\sigma(W_1[-p;\,y]+b_1)+b_2\big),\quad W_1,w_2\ge0\ \Rightarrow\ \partial f/\partial p\le0")
    st.caption("شبكة رتيبة بالبناء: الأوزان غير السالبة مع إدخال −p تضمن طلبًا غير متزايد في السعر لكل المدخلات، لا في المتوسط فقط.")
    ui.refs("Chetverikov, Santos & Shaikh (2018) Annual Review of Economics 10؛ Hainmueller & Hazlett (2014) Political Analysis 22(2)؛ Zhao et al. (2024) AAAI؛ You et al. (2017) NeurIPS «Deep lattice networks».")

# --------------------------------------------------------------------------- 7
with tabs[6]:
    ui.section("الدالة المتعلَّمة صندوق؛ كيف نفتحه؟", "Reading a learned function", "sky")
    tools = [
        ("الأثر الحدي المتوسط (AME)", r"\text{AME}_j=\frac1n\sum_{i}\frac{\partial\hat f(x_i)}{\partial x_{ij}}",
         "النظير المباشر لمعامل الانحدار. يمكن حسابه لشبكة تلقائيًا (Autodiff) أو بالفروق العددية. مع الغابات الدالة متقطعة فنستخدم فروقًا محدودة."),
        ("الاعتماد الجزئي (PDP)", r"\text{PD}_j(v)=\frac1n\sum_i\hat f\big(v,\ x_{i,-j}\big)",
         "متوسط التنبؤ عندما نثبت $x_j=v$ لكل المشاهدات. **مشكلة:** إذا كان $x_j$ مرتبطًا بغيره فإن التركيبات $(v,x_{i,-j})$ قد تكون مستحيلة واقعيًا."),
        ("الآثار المحلية المتراكمة (ALE)", r"\text{ALE}_j(v)=\int_{z_0}^{v}\mathbb E\Big[\frac{\partial\hat f}{\partial x_j}\,\Big|\,x_j=z\Big]dz",
         "يحسب الأثر المحلي **ضمن البيانات المشاهدة فقط** ثم يجمعه. أصح من PDP عند الترابط (Apley & Zhu 2020)."),
        ("قيم شابلي (SHAP)", r"\phi_j(x)=\sum_{S\subseteq F\setminus\{j\}}\frac{|S|!\,(|F|-|S|-1)!}{|F|!}\big[\hat f_{S\cup\{j\}}(x)-\hat f_S(x)\big]",
         "توزيع التنبؤ على المتغيرات بعدالة مستمدة من نظرية الألعاب. يجيب «لماذا تنبأ النموذج بهذا لهذه المشاهدة؟»."),
    ]
    for t, eq, body in tools:
        with ui.tinted("slate"):
            st.markdown(f"**{t}**")
            st.latex(eq)
            st.markdown(body)
    ui.note(
        "**تحذير حاسم:** هذه الأدوات تفسر **النموذج** لا **العالم**. قيمة SHAP كبيرة لمتغير تعني أن النموذج يعتمد عليه في التنبؤ، لا أنه "
        "يسبب النتيجة. لتحويل التفسير إلى سببية تحتاج استراتيجية تعرّف (L3)، لا أداة تفسير.",
        "التفسير ليس سببية", "rose", "⚠️",
    )
    ui.refs("Friedman (2001) PDP؛ Apley & Zhu (2020) JRSS-B 82(4)؛ Lundberg & Lee (2017) NeurIPS؛ Molnar (2022) Interpretable Machine Learning.")

# --------------------------------------------------------------------------- 8
with tabs[7]:
    ui.section("مختبر: من يقرّب الدالة الحقيقية أفضل؟", "Approximating an unknown function", "indigo")
    c1, c2, c3 = st.columns(3)
    shape = c1.selectbox("شكل الدالة الحقيقية", ["عتبة ناعمة (STAR)", "جيبية", "على شكل U", "خطية (للمقارنة)"], key="fn_shape")
    noise = c2.slider("الضوضاء", 0.1, 1.5, 0.5, 0.1, key="fn_noise")
    n = c3.select_slider("حجم العينة", [100, 300, 1000, 3000], value=300, key="fn_n")

    @st.cache_data(show_spinner="تقدير النماذج...")
    def fit_all(shape, noise, n, seed=1):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        f = {"عتبة ناعمة (STAR)": lambda x: 0.4 * x + 2.0 / (1 + np.exp(-4 * (x - 0.5))),
             "جيبية": lambda x: 1.5 * np.sin(2 * x),
             "على شكل U": lambda x: 0.6 * x ** 2 - 1,
             "خطية (للمقارنة)": lambda x: 0.8 * x}[shape]
        rng = np.random.default_rng(seed)
        x = rng.uniform(-3, 3, n)
        y = f(x) + rng.normal(0, noise, n)
        xt = rng.uniform(-3, 3, 4000)
        yt = f(xt) + rng.normal(0, noise, 4000)
        grid = np.linspace(-3, 3, 300)
        X, Xt, Xg = x[:, None], xt[:, None], grid[:, None]
        poly = PolynomialFeatures(3)
        models = {
            "خطي (OLS)": LinearRegression().fit(X, y),
            "كثيرة حدود من الدرجة 3": None,
            "غابة عشوائية": RandomForestRegressor(300, min_samples_leaf=5, random_state=0).fit(X, y),
            "شبكة عصبية (MLP)": MLPRegressor(hidden_layer_sizes=(32, 32), alpha=1e-3, max_iter=3000, random_state=0).fit(X, y),
        }
        lp = LinearRegression().fit(poly.fit_transform(X), y)
        preds, tests, ame = {}, {}, {}
        for k, m in models.items():
            if m is None:
                pg, pt = lp.predict(poly.transform(Xg)), lp.predict(poly.transform(Xt))
                d = (lp.predict(poly.transform(X + 0.05)) - lp.predict(poly.transform(X - 0.05))) / 0.1
            else:
                pg, pt = m.predict(Xg), m.predict(Xt)
                d = (m.predict(X + 0.05) - m.predict(X - 0.05)) / 0.1
            preds[k], tests[k], ame[k] = pg, float(np.sqrt(np.mean((yt - pt) ** 2))), float(d.mean())
        true_ame = float(np.mean((f(x + 0.05) - f(x - 0.05)) / 0.1))
        return x, y, grid, f(grid), preds, tests, ame, true_ame

    x, y, grid, truth, preds, tests, ame, true_ame = fit_all(shape, noise, n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="البيانات", marker=dict(size=4, color="#cbd5e1")))
    fig.add_trace(go.Scatter(x=grid, y=truth, name="الدالة الحقيقية", line=dict(color="#0f172a", width=3, dash="dash")))
    cols = {"خطي (OLS)": "#7c3aed", "كثيرة حدود من الدرجة 3": "#f59e0b", "غابة عشوائية": "#0d9488", "شبكة عصبية (MLP)": "#e11d48"}
    for k, v in preds.items():
        fig.add_trace(go.Scatter(x=grid, y=v, name=k, line=dict(color=cols[k], width=2.4)))
    ui.style_fig(fig, f"التقريب: {shape}", height=470)
    ui.show(fig)
    tab = pd.DataFrame([(k, tests[k], ame[k]) for k in preds], columns=["النموذج", "RMSE خارج العينة", "الأثر الحدي المتوسط"])
    tab.loc[len(tab)] = ["الحقيقة", noise, true_ame]
    st.dataframe(tab.round(3), hide_index=True)
    ui.note(
        "**ملاحظات:**\n"
        "- مع دالة **خطية** يفوز OLS أو يتعادل: المرونة بلا داعٍ تضيف تباينًا.\n"
        "- مع **العتبة** و**U** يفشل الخطي، والشبكة والغابة تقتربان من الحقيقة؛ والغابة تعطي دالة متدرجة (درجات سلم).\n"
        "- **AME** قد يكون صحيحًا تقريبًا حتى للنموذج الخطي (أفضل تقريب خطي) بينما الأثر الحدي المحلي خاطئ تمامًا — "
        "خصوصًا في شكل U حيث AME ≈ 0 لكن الأثر قوي في الطرفين.\n"
        "- زِد حجم العينة: يتحسن المرن أكثر من المعلمي (لعنة الأبعاد أخف في بعد واحد).",
        "اقرأ النتائج", "indigo", "🧪",
    )

# --------------------------------------------------------------------------- 9
with tabs[8]:
    ui.section("قائمة تحقق للدمج في الشكل الدالي", "Checklist", "slate")
    st.markdown(
        "- [ ] اختبرتُ الخطية أولًا (RESET، Teräsvirta) وبررتُ الحاجة إلى المرونة.\n"
        "- [ ] وضعتُ المعلمات المفسَّرة حيث أحتاج التفسير، والمكوِّن المرن حيث لا أعرف الشكل.\n"
        "- [ ] اخترتُ تعقيد المكوِّن المرن (عدد الخلايا، عمق الأشجار) خارج العينة.\n"
        "- [ ] للمعاملات المتغيرة: رسمتُ مساراتها مع فترات عدم يقين وفسرتُ ما يحركها.\n"
        "- [ ] للتفسير: استخدمتُ ALE عند ترابط المتغيرات، ولم أقدّم SHAP كدليل سببي.\n"
        "- [ ] للاستجابات الدفعية في النماذج اللاخطية: GIRF بالمحاكاة بحسب الحالة وحجم الصدمة.\n"
        "- [ ] إن فُرضت قيود شكلية: فحصتُ أن البيانات لا ترفضها."
    )
