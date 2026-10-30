import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "الدمج في الاستدلال",
    "Integration at the inference level — orthogonality, cross-fitting, rate conditions, heterogeneous effects, conformal prediction and forecast tests",
    "الاستدلال هو ما يميز القياس الاقتصادي: ليس «ما التقدير؟» فقط بل «ما مدى ثقتنا به؟». التعلم الآلي لا يعطي أخطاء معيارية، "
    "وإذا أدخلناه في مقدّر قياسي بسذاجة <b>يفسد</b> الأخطاء المعيارية. هذه الصفحة تشرح بالتفصيل لماذا يحدث ذلك، وكيف تصلحه "
    "النظرية الحديثة: التعامد، التقسيم المتقاطع، الأشجار الأمينة، التنبؤ المطابق، واختبارات مقارنة التنبؤات.",
    "indigo", "sky",
)

tabs = st.tabs([
    ":material/report: لماذا يكسر التعلم الآلي الاستدلال؟",
    ":material/architecture: التعامد: الاشتقاق الكامل",
    ":material/call_split: التقسيم المتقاطع",
    ":material/speed: شروط معدلات التقارب",
    ":material/diversity_3: الاستدلال على الآثار غير المتجانسة",
    ":material/filter_alt: الاستدلال بعد الاختيار",
    ":material/radar: التنبؤ المطابق",
    ":material/compare: مقارنة التنبؤات إحصائيًا",
    ":material/science: مختبر: التغطية الهامشية والشرطية",
    ":material/rule: الاختبارات المدعومة بالذكاء الاصطناعي: الحجم والقوة",
])

# --------------------------------------------------------------------------- 1
with tabs[0]:
    ui.section("ثلاثة مصادر لانهيار الاستدلال", "Three ways ML breaks inference", "rose")
    ui.ladder([
        ("تحيز التنظيم", "Regularization bias", "المتعلم يقلص التقديرات عمدًا ليقلل التباين. إذا أدخلنا $\\hat g$ المقلص في معادلة المعلمة، "
         "يدخل التقليص في $\\hat\\theta$ كتحيز يتناقص **أبطأ** من $n^{-1/2}$، فيسيطر على التوزيع."),
        ("تحيز الإفراط في التوفيق", "Overfitting / own-observation bias", "إذا قدّرنا $\\hat g$ على المشاهدة نفسها التي نحسب عليها المعادلة، "
         "فإن $\\hat g(X_i)$ يحتوي جزءًا من ضوضاء $U_i$، فتصبح البواقي مرتبطة بالضوضاء."),
        ("الاختيار المعتمد على البيانات", "Data-driven model selection", "إذا اختار Lasso المتغيرات ثم استخدمنا أخطاء OLS المعيارية كأن الاختيار "
         "كان معروفًا مسبقًا، نتجاهل عدم اليقين في الاختيار نفسه."),
    ])
    st.markdown("صيغة التحيز للمقدّر الساذج في النموذج جزئي الخطية (ندخل $\\hat g$ ونقدّر $\\theta$ من $Y-\\hat g(X)$ على $D$):")
    st.latex(r"\sqrt n(\hat\theta-\theta_0)=\underbrace{\frac{\frac{1}{\sqrt n}\sum D_iU_i}{\mathbb E D^2}}_{\to\ N(0,\Sigma)}+\underbrace{\frac{\frac{1}{\sqrt n}\sum m_0(X_i)\big(g_0(X_i)-\hat g(X_i)\big)}{\mathbb E D^2}}_{\text{يتباعد: معدل }\hat g\text{ أبطأ من }n^{-1/2}}")
    st.markdown(
        "الحد الثاني هو **حاصل** $m_0$ (ثابت لا يصغر) في خطأ $\\hat g$. إذا كان خطأ $\\hat g$ من رتبة $n^{-1/4}$ مثلًا، فالحد الثاني من رتبة "
        "$\\sqrt n\\cdot n^{-1/4}=n^{1/4}\\to\\infty$. أي أن التحيز **ينمو** مع حجم العينة بدل أن يختفي. لهذا يكون التقدير الساذج خاطئًا "
        "حتى مع بيانات ضخمة."
    )
    ui.refs("Chernozhukov et al. (2018) Econometrics Journal 21(1) §1؛ Leeb & Pötscher (2005) Econometric Theory 21(1).")

# --------------------------------------------------------------------------- 2
with tabs[1]:
    ui.section("التعامد من الصفر", "Neyman orthogonality, derived", "indigo")
    st.markdown("لدينا معلمة هدف $\\theta$ وأجزاء مزعجة $\\eta$ (دوال). المعلمة معرّفة بعزم:")
    st.latex(r"\mathbb E\big[\psi(W;\theta_0,\eta_0)\big]=0")
    st.markdown("نقدّر $\\eta$ بالتعلم الآلي فنحصل على $\\hat\\eta$ بخطأ. بتوسيع تايلور حول $\\eta_0$:")
    st.latex(r"\mathbb E\big[\psi(W;\theta_0,\hat\eta)\big]\approx\underbrace{\mathbb E[\psi(W;\theta_0,\eta_0)]}_{=0}+\underbrace{\partial_\eta\mathbb E[\psi(W;\theta_0,\eta)]\big|_{\eta_0}[\hat\eta-\eta_0]}_{\text{أثر من الدرجة الأولى}}+O\big(\|\hat\eta-\eta_0\|^2\big)")
    ui.note(
        "**التعامد** هو أن نختار $\\psi$ بحيث **ينعدم الحد الأوسط**:\n"
        "$$\\partial_\\eta\\,\\mathbb E[\\psi(W;\\theta_0,\\eta)]\\big|_{\\eta=\\eta_0}=0$$\n"
        "عندها يبقى أثر خطأ التعلم الآلي **من الدرجة الثانية فقط** $O(\\|\\hat\\eta-\\eta_0\\|^2)$. إذا كان $\\|\\hat\\eta-\\eta_0\\|=o(n^{-1/4})$، "
        "فالأثر $o(n^{-1/2})$، أي مهمل في التوزيع الحدي.",
        "التعريف", "indigo", "📐",
    )
    ui.section("مثال 1: النموذج جزئي الخطية", "Partially linear", "teal")
    st.markdown("العزم الساذج $\\psi^{\\text{naive}}=(Y-\\theta D-g(X))D$ **غير متعامد**: مشتقته في اتجاه $g$ هي $-\\mathbb E[D\\,\\delta(X)]\\neq0$. العزم المتعامد:")
    st.latex(r"\psi=\big(Y-\ell(X)-\theta(D-m(X))\big)\big(D-m(X)\big),\qquad \ell=\mathbb E[Y\mid X],\ m=\mathbb E[D\mid X]")
    st.markdown("**التحقق:** المشتقة في اتجاه $\\ell$ هي $-\\mathbb E[\\delta_\\ell(X)V]=0$ لأن $\\mathbb E[V\\mid X]=0$؛ والمشتقة في اتجاه $m$ تساوي صفرًا لأن "
                "$\\mathbb E[(\\tilde Y-\\theta V)\\mid X]=0$. ∎")
    ui.section("مثال 2: متوسط أثر المعالجة (AIPW)", "Doubly robust score for the ATE", "violet")
    st.latex(r"\psi=\mu_1(X)-\mu_0(X)+\frac{D\big(Y-\mu_1(X)\big)}{e(X)}-\frac{(1-D)\big(Y-\mu_0(X)\big)}{1-e(X)}-\theta")
    st.markdown(
        "حيث $\\mu_d(X)=\\mathbb E[Y\\mid D=d,X]$ و$e(X)=P(D=1\\mid X)$. هذا العزم **مضاعف المتانة**: يبقى $\\mathbb E\\psi=0$ إذا كان **أحد** "
        "النموذجين صحيحًا (نموذج النتيجة أو نموذج درجة الميل)، وخطأ التقدير يدخل كحاصل ضرب الخطأين:"
    )
    st.latex(r"\text{Bias}\approx\mathbb E\Big[\frac{\big(e(X)-\hat e(X)\big)\big(\mu_1(X)-\hat\mu_1(X)\big)}{\hat e(X)}\Big]+\dots")
    ui.section("الوصفة العامة لبناء عزم متعامد", "General recipe", "amber")
    st.markdown(
        "ابدأ بأي عزم معرِّف $\\phi$، ثم **أضف دالة تأثير الجزء المزعج** (تصحيح من الدرجة الأولى). Chernozhukov, Escanciano, Ichimura, "
        "Newey & Robins (2022) أعطوا طريقة عامة: $\\psi=\\phi+\\alpha(X)\\cdot(\\text{باقي الجزء المزعج})$ حيث $\\alpha$ «ممثل ريس» يُقدَّر "
        "أيضًا بالتعلم الآلي (مكتبة pragdml تعتمد انحدار ريس). بهذا يمكن بناء مقدرات DML لمعلمات كثيرة: آثار متوسطة، مرونات، فائض مستهلك."
    )
    ui.refs("Neyman (1959)؛ Robins & Rotnitzky (1995) JASA 90(429)؛ Chernozhukov et al. (2018)؛ Chernozhukov, Escanciano, Ichimura, Newey & Robins (2022) Econometrica 90(4)؛ Chernozhukov, Newey & Singh (2022) Econometrica 90(3) Riesz.")

# --------------------------------------------------------------------------- 3
with tabs[2]:
    ui.section("التقسيم المتقاطع: لماذا وكيف؟", "Cross-fitting", "teal")
    st.markdown("حتى مع عزم متعامد، يبقى حد ثالث في التوسع:")
    st.latex(r"\frac1{\sqrt n}\sum_i\Big(\psi(W_i;\theta_0,\hat\eta)-\psi(W_i;\theta_0,\eta_0)-\mathbb E\big[\psi(W;\theta_0,\hat\eta)-\psi(W;\theta_0,\eta_0)\big]\Big)")
    st.markdown(
        "هذا الحد صغير **إذا كان $\\hat\\eta$ مستقلًا عن $W_i$**. لكن إذا قُدّر $\\hat\\eta$ على المشاهدة $i$ نفسها، فهو مرتبط بها، والحد قد لا "
        "يصغر (خصوصًا مع متعلمات معقدة جدًا كالغابات العميقة). الحل التقليدي كان فرض شروط تعقيد (Donsker) لا تحققها معظم متعلمات التعلم الآلي. "
        "الحل الحديث: **التقسيم المتقاطع**."
    )
    ui.ladder([
        ("قسّم العينة إلى K طيات", "Split", "عادةً K = 5 أو 10."),
        ("لكل طية k: درّب على البقية", "Train off-fold", "$\\hat\\eta^{(-k)}$ مدرَّب على كل الطيات ما عدا k."),
        ("احسب العزم على الطية k", "Evaluate in-fold", "$\\psi(W_i;\\theta,\\hat\\eta^{(-k)})$ لكل $i\\in k$ — النموذج لم ير هذه المشاهدات."),
        ("حل المعادلة على كل العينة", "Pool", "$\\frac1n\\sum_k\\sum_{i\\in k}\\psi(W_i;\\hat\\theta,\\hat\\eta^{(-k)})=0$ — لا نخسر بيانات."),
        ("كرر بتقسيمات مختلفة", "Repeat", "خذ وسيط التقديرات، وأضف لتباين كل تقدير مربع ابتعاده عن الوسيط لتأخذ عدم اليقين من التقسيم."),
    ])
    st.latex(r"\hat\theta^{\text{med}}=\operatorname{median}_s\hat\theta_s,\qquad \hat\sigma^{2,\text{med}}=\operatorname{median}_s\Big(\hat\sigma_s^2+\big(\hat\theta_s-\hat\theta^{\text{med}}\big)^2\Big)")
    ui.section("التقسيم في البيانات غير المستقلة", "Dependent data", "rose")
    st.markdown(
        "- **بيانات عنقودية/بانل:** قسّم **بالعناقيد** (الشركة، الولاية) لا بالمشاهدات، وإلا تسربت معلومات الوحدة بين الطيات.\n"
        "- **سلاسل زمنية:** طيات متصلة زمنيًا مع **فجوة** بين التدريب والتقييم بطول الاعتماد، أو التقسيم المتقاطع العكسي (مكتبة tsdml).\n"
        "- **بيانات مكانية:** كتل جغرافية."
    )
    ui.refs("Chernozhukov et al. (2018) §3–4؛ Chiang, Kato, Ma & Sasaki (2022) JBES cluster DML؛ Semenova et al. (2023) panel DML.")

# --------------------------------------------------------------------------- 4
with tabs[3]:
    ui.section("متى يكون التعلم الآلي «جيدًا بما يكفي»؟", "Rate conditions", "amber")
    st.latex(r"\|\hat m-m_0\|_2\cdot\|\hat\ell-\ell_0\|_2=o_P\big(n^{-1/2}\big)")
    st.markdown(
        "**المعنى:** لا نحتاج أن يكون كل متعلم سريعًا جدًا؛ نحتاج أن يكون **حاصل** الخطأين صغيرًا. يكفي مثلًا أن يتقارب كل منهما بمعدل "
        "$o(n^{-1/4})$، وهو أبطأ بكثير من المعدل المعلمي $n^{-1/2}$. ويمكن أن يكون أحدهما أسرع ليعوض بطء الآخر."
    )
    rates = pd.DataFrame([
        ("Lasso", "ندرة: s متغيرًا مهمًا من p", "√(s log p / n)", "o(n^-1/4) إذا s² log² p / n → 0"),
        ("الغابات العشوائية", "دالة ملساء، أبعاد منخفضة", "n^(-1/(d+2)) تقريبًا (نسخ أمينة)", "مقبول في أبعاد صغيرة"),
        ("الشبكات العميقة (ReLU)", "بنية تركيبية/ملاسة", "معدلات Farrell, Liang & Misra (2021)", "تكفي لـ DML تحت شروطهم"),
        ("التعزيز", "تجميع ضعيف + تنظيم", "نتائج جزئية", "يُستخدم عمليًا مع تحقق محاكاة"),
        ("النوى (Kernels)", "ملاسة s في d بعد", "n^(-s/(2s+d))", "يتطلب s > d/2"),
    ], columns=["المتعلم", "الافتراض البنيوي", "معدل التقارب", "هل يحقق شرط DML؟"])
    st.dataframe(rates, hide_index=True)
    ui.note(
        "**نصيحة عملية:** لا نستطيع التحقق من شروط المعدلات مباشرة في البيانات. لذلك: (1) جرّب عدة متعلمات وأبلغ عن حساسية $\\hat\\theta$؛ "
        "(2) استخدم توليف متعلمات (Super learner) للأجزاء المزعجة؛ (3) شغّل محاكاة بخصائص تشبه بياناتك وتحقق من التغطية.",
        "في التطبيق", "amber", "🛠️",
    )
    ui.refs("Belloni & Chernozhukov (2013) Bernoulli 19(2)؛ Wager & Walther (2015)؛ Farrell, Liang & Misra (2021) Econometrica 89(1)؛ Bach et al. (2022) DoubleML JMLR.")

# --------------------------------------------------------------------------- 5
with tabs[4]:
    ui.section("من أثر متوسط إلى أثر لكل فرد", "Inference on heterogeneous effects", "teal")
    st.latex(r"\tau(x)=\mathbb E\big[Y(1)-Y(0)\mid X=x\big]")
    st.markdown("تقدير دالة كاملة أصعب بكثير من رقم واحد، والاستدلال عليها يتطلب أفكارًا خاصة:")
    ui.ladder([
        ("الأشجار الأمينة", "Honest trees (Athey & Imbens 2016)", "قسّم العينة: نصف لبناء الشجرة (أين نقسم)، ونصف آخر لتقدير الأثر في كل ورقة. "
         "بما أن تقديرات الورقة من بيانات لم تُستخدم في اختيار التقسيم، تصبح غير متحيزة وتصح فترات الثقة."),
        ("الغابات السببية", "Causal forests (Wager & Athey 2018)", "متوسط أشجار أمينة مع اختيار فرعي عشوائي. أثبتا أن $\\hat\\tau(x)$ "
         "**طبيعي التوزيع تقاربيًا** مع تباين يُقدَّر بـ jackknife-after-bagging."),
        ("الغابات العشوائية المعممة", "GRF (Athey, Tibshirani & Wager 2019)", "إطار عام لأي معلمة محلية معرفة بعزم: آثار المعالجة، الانحدار الكمي، المتغيرات الآلية المحلية."),
        ("الاستدلال على خصائص الأثر", "Generic ML (Chernozhukov, Demirer, Duflo & Fernández-Val)", "بدل الاستدلال على $\\tau(x)$ نقطة بنقطة، نسأل أسئلة قابلة للاختبار: "
         "هل يوجد تجانس؟ (BLP)، ما الأثر لأعلى 20% وأدنى 20%؟ (GATES)، ما خصائص الأكثر استفادة؟ (CLAN). يعمل مع **أي** متعلم."),
    ])
    st.latex(r"\text{BLP}:\ \ Y=\alpha'X_1+\beta_1(D-e(X))+\beta_2(D-e(X))\big(\hat\tau(X)-\overline{\hat\tau}\big)+\varepsilon,\qquad H_0:\beta_2=0")
    st.caption("إذا كان β₂ معنويًا وقريبًا من 1، فالمتعلم التقط تجانسًا حقيقيًا في الأثر؛ إذا كان صفرًا فالتجانس المقدَّر ضوضاء.")
    ui.refs("Athey & Imbens (2016) PNAS 113(27)؛ Wager & Athey (2018) JASA 113(523)؛ Athey, Tibshirani & Wager (2019) Annals of Statistics 47(2)؛ Chernozhukov, Demirer, Duflo & Fernández-Val (2025) Econometrica.")

# --------------------------------------------------------------------------- 6
with tabs[5]:
    ui.section("الاختيار ثم الاستدلال: مشكلة مستحيلة جزئيًا", "Post-selection inference", "rose")
    st.markdown(
        "Leeb & Pötscher (2005, 2006) أثبتا نتيجة صادمة: **لا يمكن** تقدير توزيع المقدّر بعد الاختيار بشكل منتظم وموثوق، لأن التوزيع يعتمد "
        "على معلمات مجهولة بطريقة غير مستمرة. أي أن «اختر بـ Lasso ثم استخدم أخطاء OLS» ليس خطأً تقنيًا يمكن ترقيعه، بل خطأ مفاهيمي."
    )
    ui.cards([
        {"title": "تجنب الاختيار على المعلمة الهدف", "color": "emerald",
         "body": "الاختيار المزدوج وDML لا يختاران أبدًا إن كانت المعالجة في النموذج؛ يختاران الضوابط فقط وبطريقة متعامدة. هذا يتجاوز المشكلة."},
        {"title": "تقسيم البيانات", "color": "sky",
         "body": "اختر النموذج على نصف واستدل على النصف الآخر. بسيط وصحيح، لكنه يخسر نصف الدقة."},
        {"title": "الاستدلال الانتقائي", "color": "violet",
         "body": "حساب التوزيع **بشرط** حدث الاختيار (Lee, Sun, Sun & Taylor 2016). دقيق لكنه معقد ويعطي فترات واسعة أحيانًا."},
        {"title": "PoSI", "color": "amber",
         "body": "فترات صحيحة لكل النماذج الممكنة في آن واحد (Berk et al. 2013). محافظة جدًا لكنها آمنة."},
    ], cols=2)
    ui.refs("Leeb & Pötscher (2005) Econometric Theory 21(1)؛ Berk, Brown, Buja, Zhang & Zhao (2013) Annals of Statistics 41(2)؛ Lee, Sun, Sun & Taylor (2016) Annals of Statistics 44(3).")

# --------------------------------------------------------------------------- 7
with tabs[6]:
    ui.section("فترات تنبؤ صادقة لأي نموذج", "Conformal prediction", "sky")
    st.markdown("الشبكات والغابات تعطي تنبؤًا نقطيًا بلا فترة. التنبؤ المطابق يبني فترة بضمان **محدود العينة** دون افتراض التوزيع:")
    ui.ladder([
        ("قسّم البيانات", "Split", "مجموعة تدريب ومجموعة معايرة (Calibration)."),
        ("درّب أي نموذج", "Fit", "$\\hat f$ على مجموعة التدريب فقط."),
        ("احسب درجات عدم المطابقة", "Scores", "$s_i=|y_i-\\hat f(x_i)|$ على مجموعة المعايرة."),
        ("خذ الكمّي المصحح", "Quantile", "$\\hat q=$ الكمّي رقم $\\lceil(n_c+1)(1-\\alpha)\\rceil$ من الدرجات المرتبة."),
        ("ابنِ الفترة", "Interval", "$C(x)=[\\hat f(x)-\\hat q,\\ \\hat f(x)+\\hat q]$."),
    ])
    st.latex(r"P\big(Y_{n+1}\in C(X_{n+1})\big)\ \ge\ 1-\alpha\qquad\text{(بشرط قابلية التبادل فقط)}")
    ui.note(
        "**الضمان هامشي لا شرطي:** التغطية 90% **في المتوسط على كل نقاط $x$**، لا لكل $x$ بعينه. مع تباين متغير، الفترة الثابتة العرض "
        "تغطي أكثر من اللازم حيث التباين صغير وأقل من اللازم حيث هو كبير. الحلول: درجات معيارية $s_i=|y_i-\\hat f(x_i)|/\\hat\\sigma(x_i)$، "
        "أو التنبؤ المطابق الكمي (CQR). المختبر يوضح الفرق.",
        "تمييز مهم", "amber", "⚠️",
    )
    st.markdown(
        "**في السلاسل الزمنية** يُكسر شرط التبادل. حلول حديثة: التنبؤ المطابق التكيفي (ACI، Gibbs & Candès 2021) الذي يعدّل $\\alpha$ "
        "عبر الزمن بحسب الأخطاء الأخيرة فيحافظ على التغطية طويلة الأجل حتى مع تغير التوزيع، وEnbPI للتجميعات."
    )
    ui.refs("Vovk, Gammerman & Shafer (2005) Algorithmic Learning in a Random World؛ Lei et al. (2018) JASA 113(523)؛ Romano, Patterson & Candès (2019) NeurIPS CQR؛ Gibbs & Candès (2021) NeurIPS ACI.")

# --------------------------------------------------------------------------- 8
with tabs[7]:
    ui.section("هل النموذج الهجين أفضل فعلًا؟ إحصائيًا", "Testing forecast accuracy", "violet")
    tests = [
        ("Diebold–Mariano (1995)", r"DM=\frac{\bar d}{\sqrt{\hat\Omega/T}},\ \ d_t=L(e_{1t})-L(e_{2t})",
         "هل متوسط فرق الخسارة صفر؟ $\\hat\\Omega$ تباين طويل الأجل (Newey–West). تصحيح العينة الصغيرة: Harvey, Leybourne & Newbold (1997)."),
        ("Clark–West (2007)", r"\hat f_t=e_{1,t}^2-\big[e_{2,t}^2-(\hat y_{1,t}-\hat y_{2,t})^2\big]",
         "للنماذج **المتداخلة** (الهجين يحتوي النموذج القياسي): DM منحاز ضد النموذج الأكبر تحت العدم، وكلارك–ويست يصحح ذلك."),
        ("Giacomini–White (2006)", r"H_0:\ \mathbb E\big[d_{t+1}\mid\mathcal F_t\big]=0",
         "قدرة تنبؤية **شرطية**: هل يتفوق الهجين في ظروف معينة (الأزمات)؟ ويأخذ في الحسبان عدم يقين التقدير في نافذة متدحرجة."),
        ("مجموعة الثقة للنماذج MCS", r"\hat{\mathcal M}^*_{1-\alpha}=\{\text{النماذج غير المستبعدة}\}",
         "Hansen, Lunde & Nason (2011): من بين 30 نموذجًا، أي مجموعة تحتوي الأفضل باحتمال 90%؟ يتعامل مع تعدد المقارنات."),
        ("فحص الواقع", r"\max_k\sqrt T\,\bar d_k",
         "White (2000) وHansen (2005): بعد تجربة نماذج كثيرة، هل أفضلها أفضل من المرجع فعلًا أم بالصدفة؟"),
    ]
    for t, eq, body in tests:
        with ui.tinted("slate"):
            st.markdown(f"**{t}**")
            st.latex(eq)
            st.markdown(body)
    ui.refs("Diebold & Mariano (1995) JBES 13(3)؛ Harvey, Leybourne & Newbold (1997) IJF 13(2)؛ Clark & West (2007) J. Econometrics 138(1)؛ Giacomini & White (2006) Econometrica 74(6)؛ "
            "White (2000) Econometrica 68(5)؛ Hansen, Lunde & Nason (2011) Econometrica 79(2).")

# --------------------------------------------------------------------------- 9
with tabs[8]:
    ui.section("مختبر: فترات تنبؤ بغابة عشوائية تحت تباين متغير", "Marginal vs conditional coverage", "sky")
    st.markdown("البيانات: $y=\\sin(2x)+\\varepsilon$ مع انحراف ضوضاء يزداد مع $|x|$. نبني فترات 90% بثلاث طرق ونقيس التغطية على عينة اختبار كبيرة.")
    c1, c2 = st.columns(2)
    het = c1.slider("قوة عدم ثبات التباين", 0.0, 2.0, 1.2, 0.1, key="cp_het")
    n = c2.select_slider("حجم عينة التدريب+المعايرة", [400, 1000, 3000], value=1000, key="cp_n")

    @st.cache_data(show_spinner="بناء الفترات...")
    def conformal(het, n, seed=2):
        from sklearn.ensemble import RandomForestRegressor
        rng = np.random.default_rng(seed)
        sd = lambda x: 0.2 + het * np.abs(x) / 3  # noqa: E731
        def draw(m):
            x = rng.uniform(-3, 3, m)
            return x, np.sin(2 * x) + rng.normal(0, sd(x))
        x, y = draw(n)
        xt, yt = draw(6000)
        tr, ca = slice(0, n // 2), slice(n // 2, n)
        f = RandomForestRegressor(300, min_samples_leaf=10, random_state=0).fit(x[tr, None], y[tr])
        res_tr = np.abs(y[tr] - f.predict(x[tr, None]))
        s = RandomForestRegressor(300, min_samples_leaf=20, random_state=0).fit(x[tr, None], res_tr)
        a = 0.10
        k = int(np.ceil((n // 2 + 1) * (1 - a)))
        pc, pt = f.predict(x[ca, None]), f.predict(xt[:, None])
        # 1) naive Gaussian with in-sample residual sd (overfit forest -> too narrow)
        sig = np.std(y[tr] - f.predict(x[tr, None]))
        naive = (pt - 1.645 * sig, pt + 1.645 * sig)
        # 2) split conformal, constant width
        q = np.sort(np.abs(y[ca] - pc))[k - 1]
        split = (pt - q, pt + q)
        # 3) locally weighted conformal
        sc, stt = np.maximum(s.predict(x[ca, None]), 1e-3), np.maximum(s.predict(xt[:, None]), 1e-3)
        qn = np.sort(np.abs(y[ca] - pc) / sc)[k - 1]
        local = (pt - qn * stt, pt + qn * stt)
        bins = np.linspace(-3, 3, 13)
        mids = (bins[:-1] + bins[1:]) / 2
        out = {}
        for name, (lo, hi) in {"غاوسي ساذج (بواقي التدريب)": naive, "مطابق مقسَّم (عرض ثابت)": split, "مطابق موزون محليًا": local}.items():
            cov = (yt >= lo) & (yt <= hi)
            byb = [cov[(xt >= bins[i]) & (xt < bins[i + 1])].mean() for i in range(len(mids))]
            out[name] = (cov.mean(), float(np.mean(hi - lo)), byb)
        return mids, out

    mids, out = conformal(het, n)
    tab = pd.DataFrame([(k, v[0], v[1]) for k, v in out.items()], columns=["الطريقة", "التغطية الهامشية", "متوسط العرض"])
    st.dataframe(tab.round(3), hide_index=True)
    fig = go.Figure()
    colors = ["#e11d48", "#f59e0b", "#0d9488"]
    for (k, v), c in zip(out.items(), colors):
        fig.add_trace(go.Scatter(x=mids, y=v[2], name=k, mode="lines+markers", line=dict(color=c, width=2.6)))
    fig.add_hline(y=0.9, line_dash="dash", line_color="#0f172a", annotation_text="الهدف 90%")
    fig.update_xaxes(title="x")
    fig.update_yaxes(title="التغطية داخل كل شريحة من x", range=[0, 1.02])
    ui.style_fig(fig, "التغطية الشرطية بحسب x", height=430)
    ui.show(fig)
    ui.note(
        "**القراءة:**\n"
        "- الطريقة الغاوسية الساذجة تستخدم بواقي **التدريب**، وهي أصغر من أخطاء الاختبار، فتأتي الفترة أضيق من اللازم والتغطية دون 90% "
        "بلا أي ضمان؛ وكلما كان النموذج أكثر مرونة ساءت أكثر. وعرضها الثابت يفشل في الأطراف كذلك.\n"
        "- المطابق المقسَّم يحقق 90% **هامشيًا** كما يضمن، لكنه يفرط في التغطية في الوسط ويقصّر في الأطراف حيث الضوضاء أكبر.\n"
        "- الموزون محليًا يقترب من 90% في كل شريحة تقريبًا، أي تغطية شرطية أفضل، بعرض يتكيف مع الضوضاء.",
        "ماذا نتعلم؟", "sky", "🧪",
    )

# --------------------------------------------------------------------------- 10
with tabs[9]:
    ui.section("عندما يصبح التعلم الآلي جزءًا من اختبار إحصائي", "ML-assisted tests: size and power", "rose")
    st.markdown(
        "شبكة أو غابة تُستخدم لاختبار سببية غرانجر اللاخطية، أو جذر وحدة، أو تغير هيكلي. يمر الدمج هنا بأربع مراحل، ولا يكفي أن «يرفض» النموذج الفرضية:"
    )
    ui.ladder([
        ("الفرضيتان اقتصاديًا", "Define H0 and H1", "مثلًا H0: ماضي x لا يحسّن التنبؤ بـ y بأي شكل، خطي أو لاخطي."),
        ("قاعدة الكشف", "Detection rule", "معمارية الشبكة أو المتعلم، والإحصاءة (فرق الخسارة، مقياس الأهمية، ...)."),
        ("توزيع الإحصاءة تحت H0", "Null distribution", "من نظرية تقاربية، أو bootstrap، أو تبديل، أو مونت كارلو. **بدونه لا يوجد اختبار.**"),
        ("الحجم والقوة", "Size and power", "محاكاة تحت H0 لقياس معدل الرفض الخاطئ (يجب أن يقارب α)، وتحت بدائل لقياس القوة، عبر أحجام عينات وأحجام شبكات."),
    ])
    st.latex(r"\widehat{\text{size}}=\frac1R\sum_{r=1}^{R}\mathbb 1\{\text{reject}_r\mid H_0\}\ \approx\ \alpha,\qquad \widehat{\text{power}}(\delta)=\frac1R\sum_{r=1}^{R}\mathbb 1\{\text{reject}_r\mid H_1(\delta)\}")
    ui.note(
        "**مصدر الخلل الأكثر شيوعًا:** النموذج غير المقيد (مع x) أكثر مرونة من المقيد، فيتلاءم **داخل العينة** أفضل دائمًا حتى لو لم يكن لـ x أي أثر. "
        "إحصاءة مبنية على أخطاء داخل العينة ترفض H0 تقريبًا في كل مرة: حجم متضخم يُقرأ خطأً «سببيةً لاخطية». الضبط (tuning) واختيار حجم الشبكة "
        "بعد رؤية النتيجة يزيدان التضخم.",
        "لماذا تفشل الاختبارات الساذجة؟", "rose", "⚠️",
    )
    ui.section("مختبر مونت كارلو: غرانجر بغابة عشوائية", "Random-forest Granger test, simulated", "indigo")
    st.markdown(
        "تحت H0 لا يؤثر $x$ في $y$. تحت H1 يضاف $\delta\,\sin(2x_{t-1})$ إلى معادلة $y$. نقارن إحصاءتين لفرق مربع الخطأ بين غابة مقيدة "
        "(ماضي y فقط) وغير مقيدة (ماضي y وx)، باختبار t أحادي الجانب عند α = 5%:\n"
        "- **داخل العينة:** الغابتان مدربتان ومقيَّمتان على العينة نفسها.\n"
        "- **خارج العينة:** تدريب على أول 60% وتقييم على آخر 40%."
    )
    c1, c2, c3, c4 = st.columns(4)
    n = c1.select_slider("حجم العينة", [150, 300, 600], value=300, key="mlt_n")
    delta = c2.slider("قوة الأثر تحت H1 (δ)", 0.1, 1.5, 1.0, 0.1, key="mlt_d")
    reps = c3.select_slider("عدد التكرارات", [60, 100, 200], value=100, key="mlt_r")
    cap = c4.selectbox("سعة الغابة", ["منظَّمة (أوراق ≥ 20، عمق 4)", "مفرطة المرونة (أوراق ≥ 3، عمق 6)"], key="mlt_cap")
    leaf, depth = (20, 4) if cap.startswith("منظَّمة") else (3, 6)

    @st.cache_data(show_spinner="تشغيل مونت كارلو (قد يستغرق دقيقة)...")
    def mc_test(n, delta, reps, leaf, depth, seed=0):
        from sklearn.ensemble import RandomForestRegressor
        rng = np.random.default_rng(seed)

        def onesided_t(d):
            return d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)

        out = {"H0": [0, 0], "H1": [0, 0]}
        for scen, eff in (("H0", 0.0), ("H1", delta)):
            for _ in range(reps):
                m = n + 50
                x, y = np.zeros(m), np.zeros(m)
                for t in range(1, m):
                    x[t] = 0.6 * x[t - 1] + rng.normal()
                    y[t] = 0.5 * y[t - 1] + eff * np.sin(2 * x[t - 1]) + rng.normal()
                x, y = x[50:], y[50:]
                tgt, Xr, Xu = y[1:], y[:-1, None], np.column_stack([y[:-1], x[:-1]])
                rf = lambda: RandomForestRegressor(30, max_depth=depth, min_samples_leaf=leaf, random_state=0)  # noqa: E731
                # in-sample: fit and evaluate on the same data
                er = tgt - rf().fit(Xr, tgt).predict(Xr)
                eu = tgt - rf().fit(Xu, tgt).predict(Xu)
                out[scen][0] += onesided_t(er ** 2 - eu ** 2) > 1.645
                # out-of-sample: train on first 60 %, evaluate on the rest
                c = int(0.6 * len(tgt))
                er = tgt[c:] - rf().fit(Xr[:c], tgt[:c]).predict(Xr[c:])
                eu = tgt[c:] - rf().fit(Xu[:c], tgt[:c]).predict(Xu[c:])
                out[scen][1] += onesided_t(er ** 2 - eu ** 2) > 1.645
        return {k: [v[0] / reps, v[1] / reps] for k, v in out.items()}

    if st.button("شغّل المحاكاة", type="primary", icon=":material/play_arrow:", key="mlt_run"):
        st.session_state["mlt_res"] = (n, delta, reps, cap, mc_test(n, delta, reps, leaf, depth))
    if "mlt_res" in st.session_state:
        n0, d0, r0, cap0, res = st.session_state["mlt_res"]
        st.dataframe(pd.DataFrame([
            ("داخل العينة (ساذج)", res["H0"][0], res["H1"][0]),
            ("خارج العينة", res["H0"][1], res["H1"][1]),
        ], columns=["الإحصاءة", "معدل الرفض تحت H0 (الحجم، الهدف 0.05)", f"معدل الرفض تحت H1 (القوة، δ={d0})"]).round(3), hide_index=True)
        st.caption(f"n = {n0}، {r0} تكرارًا لكل سيناريو، سعة الغابة: {cap0}. خطأ مونت كارلو في الحجم ≈ ±0.04 مع 100 تكرار.")
        ui.note(
            "**القراءة:** الإحصاءة الداخلية ترفض H0 في معظم التكرارات رغم أن x لا يؤثر إطلاقًا: حجمها بعيد جدًا عن 5%، فرفضها تحت H1 لا يعني شيئًا. "
            "مع غابة **منظَّمة** تحافظ الإحصاءة الخارجية على حجم قريب من 5%، وقوتها تزيد مع δ ومع n. لكن اختر **مفرطة المرونة**: يتضخم الحجم "
            "حتى خارج العينة، لأن الغابة المقيدة ذات المتغير الواحد تفرط في التوفيق، وإضافة متغير عديم الأثر تجعل الأشجار أنعم فتبدو «مفيدة». "
            "**الدرس:** الاختبار الصالح يُعرف حجمه أولًا، وسعة النموذجين المقارَنين يجب أن تُضبط بعناية لا أن تُترك افتراضية.",
            "ماذا يظهر المختبر؟", "indigo", "🧪",
        )
    else:
        st.caption("اضغط «شغّل المحاكاة» لتقدير الحجم والقوة.")
    ui.refs("Clark & West (2007) J. Econometrics 138(1)؛ Diebold & Mariano (1995)؛ Tank et al. (2022) IEEE TPAMI «Neural Granger causality»؛ موسوعة آلية الدمج — د. مروان رودان (2026).")
