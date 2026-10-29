import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "الدمج في التوليف والقرار",
    "Integration at the output level — forecast combination, stacking, model averaging, mixtures of experts and decision-making",
    "أبسط أشكال الدمج وأقدمها: لا ندمج النماذج من الداخل، بل <b>ندمج مخرجاتها</b>. يبدو هذا سطحيًا، لكنه من أقوى النتائج "
    "التطبيقية في التنبؤ منذ 1969، وله نظرية دقيقة. ثم نصعد إلى المستوى الأخير: تحويل التنبؤات والتقديرات إلى <b>قرارات</b> "
    "(سياسات علاج، تخصيص، تعلم معزز).",
    "orange", "teal",
)

tabs = st.tabs([
    ":material/calculate: لماذا يفيد التوليف؟ (Bates–Granger)",
    ":material/psychology_alt: لغز التوليف",
    ":material/layers: التكديس والمتعلم الفائق",
    ":material/functions: المتوسط البيزي والديناميكي",
    ":material/group_work: مزيج الخبراء والأوزان المتكيفة",
    ":material/handshake: توليف القياسي والذكي",
    ":material/gavel: من التنبؤ إلى القرار",
    ":material/science: مختبر: الأوزان المثلى مقابل المتساوية",
    ":material/checklist: قائمة التحقق",
])

# --------------------------------------------------------------------------- 1
with tabs[0]:
    ui.section("الاشتقاق: لماذا متوسط تنبؤين أفضل غالبًا من كل منهما؟", "Bates & Granger (1969)", "orange")
    st.markdown("تنبؤان غير متحيزين بخطأين $e_1,e_2$ بتباينين $\\sigma_1^2,\\sigma_2^2$ وتغاير $\\sigma_{12}$. التوليف:")
    st.latex(r"\hat y_c=w\,\hat y_1+(1-w)\,\hat y_2\quad\Rightarrow\quad e_c=w\,e_1+(1-w)\,e_2")
    st.latex(r"\operatorname{Var}(e_c)=w^2\sigma_1^2+(1-w)^2\sigma_2^2+2w(1-w)\sigma_{12}")
    st.markdown("باشتقاق التباين بالنسبة إلى $w$ ومساواته بالصفر:")
    st.latex(r"w^*=\frac{\sigma_2^2-\sigma_{12}}{\sigma_1^2+\sigma_2^2-2\sigma_{12}},\qquad \operatorname{Var}(e_c^*)=\frac{\sigma_1^2\sigma_2^2-\sigma_{12}^2}{\sigma_1^2+\sigma_2^2-2\sigma_{12}}\ \le\ \min(\sigma_1^2,\sigma_2^2)")
    ui.note(
        "**ثلاث نتائج مهمة:**\n"
        "1. التباين الأمثل **لا يتجاوز** أصغر التباينين أبدًا: التوليف الأمثل لا يمكن أن يكون أسوأ من أفضل نموذج.\n"
        "2. المكسب أكبر كلما كان **الارتباط بين الأخطاء أضعف**. نموذج قياسي وشبكة عصبية يخطئان لأسباب مختلفة، فارتباط أخطائهما منخفض "
        "عادةً، وهذا بالضبط ما يجعل توليفهما مفيدًا.\n"
        "3. حتى نموذج **أسوأ** يمكن أن يحسّن التوليف إذا كانت أخطاؤه مختلفة بما يكفي.",
        "ما يعنيه الاشتقاق", "orange", "💡",
    )
    st.markdown("**بالمصفوفات لـ $K$ نموذجًا:** مع قيد مجموع الأوزان واحد:")
    st.latex(r"w^*=\frac{\Sigma^{-1}\iota}{\iota'\Sigma^{-1}\iota},\qquad \Sigma=\operatorname{Var}(e_1,\dots,e_K)")
    ui.refs("Bates & Granger (1969) Operational Research Quarterly 20(4)؛ Granger & Ramanathan (1984) J. Forecasting 3(2)؛ Timmermann (2006) Handbook of Economic Forecasting 1 ch. 4.")

# --------------------------------------------------------------------------- 2
with tabs[1]:
    ui.section("اللغز: الأوزان المتساوية تهزم «المثلى»", "The forecast combination puzzle", "rose")
    st.markdown(
        "Stock & Watson (2004) وغيرهم وجدوا تجريبيًا أن **المتوسط البسيط** للتنبؤات يتفوق غالبًا على الأوزان المقدَّرة بصيغة Bates–Granger. "
        "كيف يكون الأمثل نظريًا أسوأ عمليًا؟"
    )
    st.markdown("**التفسير (Smith & Wallis 2009؛ Claeskens et al. 2016):** الأوزان المثلى **مجهولة** وتُقدَّر من عينة محدودة:")
    st.latex(r"\mathbb E\big[\operatorname{Var}(e_c(\hat w))\big]=\operatorname{Var}(e_c(w^*))+\underbrace{\mathbb E\big[(\hat w-w^*)'\Sigma(\hat w-w^*)\big]}_{\text{تكلفة تقدير الأوزان}}")
    st.markdown(
        "إذا كانت النماذج متقاربة الجودة، فالمكسب من الأوزان المثلى صغير، بينما تكلفة تقديرها (الحد الثاني) كبيرة، خصوصًا مع عينات قصيرة "
        "ونماذج كثيرة وأخطاء مترابطة ($\\Sigma^{-1}$ غير مستقرة). والأوزان المتساوية تكلفة تقديرها **صفر**."
    )
    ui.cards([
        {"title": "متى تفوز الأوزان المتساوية؟", "color": "emerald",
         "body": "عينة تقييم قصيرة، نماذج متقاربة الأداء، أخطاء مترابطة بقوة، أو عدم استقرار في الأداء النسبي عبر الزمن."},
        {"title": "متى تستحق الأوزان المقدَّرة؟", "color": "sky",
         "body": "فروق كبيرة ومستقرة في الجودة، عينة طويلة، نماذج قليلة. أو استخدام أوزان مقلصة نحو المتساوية."},
        {"title": "الحل الوسط", "color": "violet",
         "body": "$\\hat w^{\\text{shrink}}=\\kappa\\,\\hat w^*+(1-\\kappa)\\,\\iota/K$ — تقليص الأوزان نحو المتساوية، كالتنظيم تمامًا."},
    ], cols=3)
    ui.refs("Stock & Watson (2004) J. Forecasting 23(6)؛ Smith & Wallis (2009) Oxford Bulletin 71(3)؛ Claeskens, Magnus, Vasnev & Wang (2016) IJF 32(3).")

# --------------------------------------------------------------------------- 3
with tabs[2]:
    ui.section("التكديس: نموذج يتعلم كيف يجمع النماذج", "Stacking and the Super Learner", "violet")
    st.markdown("التكديس (Wolpert 1992؛ Breiman 1996) يتعلم الأوزان من **تنبؤات خارج الطية** لكل نموذج أساسي:")
    ui.ladder([
        ("اختر مكتبة نماذج", "Base learners", "OLS، Lasso، غابة، تعزيز، شبكة، ARIMA... كلما تنوعت أفضل."),
        ("تنبؤات خارج الطية", "Out-of-fold predictions", "لكل مشاهدة، تنبؤ كل نموذج من نسخة لم ترها: $Z_{ik}=\\hat f_k^{(-k(i))}(x_i)$."),
        ("المتعلم الفوقي", "Meta-learner", "انحدار $y$ على $Z$ بقيود (أوزان موجبة مجموعها 1)، أو نموذج أبسط."),
        ("التنبؤ النهائي", "Predict", "النماذج الأساسية مدربة على كل البيانات، مجمَّعة بالأوزان المتعلَّمة."),
    ])
    st.latex(r"\hat w=\arg\min_{w\ge0,\ \sum w_k=1}\ \sum_i\Big(y_i-\sum_kw_k\,Z_{ik}\Big)^2")
    ui.note(
        "**الضمان النظري (van der Laan, Polley & Hubbard 2007):** المتعلم الفائق يؤدي تقاربيًا **بقدر أفضل توليف ممكن** من المكتبة (متباينة "
        "الأوراكل). لذلك يوصى به بقوة للأجزاء المزعجة في DML: بدل الرهان على غابة أو Lasso، دع البيانات تختار.",
        "لماذا هو مهم للقياس الاقتصادي؟", "violet", "🏆",
    )
    st.markdown(
        "**في السلاسل الزمنية:** «خارج الطية» يجب أن يعني «من الماضي فقط». التنبؤات التي تدخل المتعلم الفوقي تُولَّد بنافذة متدحرجة، والمتعلم الفوقي "
        "نفسه يُقيَّم على فترة لاحقة. استخدام التحقق المتقاطع العشوائي هنا يحقن المستقبل في الأوزان."
    )
    ui.refs("Wolpert (1992) Neural Networks 5(2)؛ Breiman (1996) Machine Learning 24(1)؛ van der Laan, Polley & Hubbard (2007) Statistical Applications in Genetics and Molecular Biology 6(1).")

# --------------------------------------------------------------------------- 4
with tabs[3]:
    ui.section("المتوسط البيزي للنماذج", "Bayesian Model Averaging", "teal")
    st.latex(r"p(y_{T+1}\mid\mathcal D)=\sum_{k=1}^{K}p(y_{T+1}\mid M_k,\mathcal D)\,p(M_k\mid\mathcal D),\qquad p(M_k\mid\mathcal D)\propto p(\mathcal D\mid M_k)\,p(M_k)")
    st.markdown(
        "الأوزان هي **الاحتمالات اللاحقة للنماذج**. تقريب شائع: $p(M_k\\mid\\mathcal D)\\propto\\exp(-\\tfrac12\\text{BIC}_k)$. BMA يدمج **عدم اليقين في "
        "النموذج** في فترات التنبؤ، وهو ما يتجاهله اختيار نموذج واحد. عيبه: إذا لم يكن أي نموذج صحيحًا (الحالة المعتادة) تتركز الأوزان على "
        "نموذج واحد بثقة زائدة."
    )
    ui.section("المتوسط الديناميكي", "Dynamic Model Averaging", "amber")
    st.latex(r"\pi_{t\mid t-1,k}=\frac{\pi_{t-1\mid t-1,k}^{\alpha}}{\sum_j\pi_{t-1\mid t-1,j}^{\alpha}},\qquad \pi_{t\mid t,k}\propto\pi_{t\mid t-1,k}\,p_k(y_t\mid\mathcal D_{t-1})")
    st.markdown(
        "Raftery, Kárný & Ettler (2010) وKoop & Korobilis (2012): الأوزان **تتغير عبر الزمن**. معامل النسيان $\\alpha<1$ يجعل الأداء الحديث أهم، فيمكن "
        "لنموذج قياسي أن يهيمن في الأوقات الهادئة وللنموذج المرن أن يهيمن في الأزمات."
    )
    ui.refs("Hoeting, Madigan, Raftery & Volinsky (1999) Statistical Science 14(4)؛ Raftery, Kárný & Ettler (2010) Technometrics 52(1)؛ Koop & Korobilis (2012) International Economic Review 53(3).")

# --------------------------------------------------------------------------- 5
with tabs[4]:
    ui.section("مزيج الخبراء: أوزان تعتمد على الحالة", "Mixture of experts", "sky")
    st.latex(r"\hat y(x)=\sum_{k=1}^{K}g_k(x)\,\hat f_k(x),\qquad g(x)=\operatorname{softmax}\big(Vx\big)")
    st.markdown(
        "شبكة **بوابة** $g(x)$ تقرر وزن كل خبير **بحسب المدخلات**. الخبراء يمكن أن يكونوا نماذج قياسية (VAR لكل نظام) والبوابة تتعلم متى "
        "يكون كل نظام فاعلًا. هذا تعميم سلس لنماذج تبديل الأنظمة (Markov switching): بدل احتمالات انتقال ثابتة، بوابة تعتمد على متغيرات مشاهَدة."
    )
    ui.section("التوليف المتصل: أوزان تتعلم من كل خطأ", "Online learning and regret", "violet")
    st.latex(r"w_{t+1,k}=\frac{w_{t,k}\,e^{-\eta\,L(y_t,\hat y_{t,k})}}{\sum_jw_{t,j}\,e^{-\eta\,L(y_t,\hat y_{t,j})}}")
    st.markdown(
        "خوارزمية الأوزان الأسية (Hedge). الضمان: **الندم** (الفرق بين خسارة التوليف وخسارة أفضل نموذج منفرد بالنظر إلى الوراء) ينمو كـ $\\sqrt{T\\log K}$، "
        "أي يتناقص لكل فترة كـ $\\sqrt{\\log K/T}$، **دون أي افتراض إحصائي** على البيانات. مفيد عندما تتغير البيئة ولا نثق بالاستقرار."
    )
    ui.refs("Jacobs, Jordan, Nowlan & Hinton (1991) Neural Computation 3(1)؛ Hamilton (1989) Econometrica 57(2)؛ Cesa-Bianchi & Lugosi (2006) Prediction, Learning, and Games.")

# --------------------------------------------------------------------------- 6
with tabs[5]:
    ui.section("هل يضيف النموذج الذكي شيئًا للقياسي؟", "Encompassing and complementarity", "emerald")
    st.markdown("**اختبار الشمول (Encompassing):** هل يحتوي تنبؤ النموذج القياسي كل المعلومات المفيدة في تنبؤ التعلم الآلي؟")
    st.latex(r"e_{1,t}=\lambda\,(e_{1,t}-e_{2,t})+\varepsilon_t,\qquad H_0:\lambda=0")
    st.markdown(
        "إذا قُبل $H_0$، فالنموذج القياسي «يشمل» الذكي: لا فائدة من التوليف. إذا رُفض، فلتنبؤ التعلم الآلي معلومات إضافية، و$\\hat\\lambda$ "
        "هو تقريبًا وزنه الأمثل. (Chong & Hendry 1986؛ Harvey, Leybourne & Newbold 1998 بنسخة متينة)."
    )
    ui.section("ما الذي يجعل التوليف الهجين ناجحًا؟", "Why econometric + ML combinations work", "teal")
    ui.cards([
        {"title": "أخطاء مختلفة المصدر", "color": "teal",
         "body": "النموذج القياسي يخطئ عند اللاخطية والانكسارات؛ الشبكة تخطئ عند الاستقراء خارج المدى وفي العينات الصغيرة. ارتباط منخفض = مكسب كبير."},
        {"title": "استقرار مقابل مرونة", "color": "sky",
         "body": "القياسي مستقر بمعلمات قليلة؛ الذكي مرن بتباين مرتفع. التوليف يأخذ استقرار الأول ومرونة الثاني."},
        {"title": "حماية من الكوارث", "color": "rose",
         "body": "إذا انهار أحد النموذجين في فترة ما (تغير هيكلي)، يحد الآخر من الضرر. التوليف تأمين لا مجرد تحسين."},
    ], cols=3)
    ui.refs("Chong & Hendry (1986) REStud 53(4)؛ Harvey, Leybourne & Newbold (1998) JBES 16(2)؛ Goulet Coulombe et al. (2022) JAE.")

# --------------------------------------------------------------------------- 7
with tabs[6]:
    ui.section("المستوى الأخير: من التنبؤ إلى القرار", "From prediction to decision", "indigo")
    st.markdown(
        "التنبؤ والتقدير ليسا غاية؛ الغاية قرار: من يُعالَج؟ ما سعر الفائدة؟ ما حجم المحفظة؟ الدمج على مستوى القرار يربط مخرجات النماذج "
        "بدالة رفاه أو ربح، ويقيّم **القرار** لا التنبؤ."
    )
    ui.ladder([
        ("قواعد العلاج المثلى", "Empirical welfare maximization", "Kitagawa & Tetenov (2018): اختيار قاعدة $\\pi(x)\\in\\{0,1\\}$ من فئة بسيطة تعظم الرفاه المقدَّر. "
         "الضمان: الندم يتناقص كـ $\\sqrt{\\text{VC}/n}$."),
        ("تعلم السياسات بعزوم متعامدة", "Policy learning (Athey & Wager 2021)", "استخدام درجات AIPW المتعامدة لتقييم كل سياسة، فيصبح الندم مضبوطًا "
         "حتى مع بيانات مشاهدة وتقدير الأجزاء المزعجة بالتعلم الآلي."),
        ("قطاع الآلات المتعددة", "Bandits", "عندما نتعلم ونقرر في الوقت نفسه: موازنة الاستكشاف والاستغلال (Thompson sampling). "
         "تجارب تكيفية في السياسات العامة."),
        ("التعلم المعزز", "Reinforcement learning", "قرارات متتابعة تغير الحالة: سياسة نقدية، إدارة مخزون، تسعير ديناميكي. "
         "صفحة «التعلم المعزز» تشرحه بالتفصيل."),
    ])
    st.latex(r"\hat\pi=\arg\max_{\pi\in\Pi}\ \frac1n\sum_{i=1}^{n}\big(2\pi(X_i)-1\big)\,\hat\Gamma_i,\qquad \hat\Gamma_i=\hat\mu_1(X_i)-\hat\mu_0(X_i)+\frac{D_i-\hat e(X_i)}{\hat e(X_i)(1-\hat e(X_i))}\big(Y_i-\hat\mu_{D_i}(X_i)\big)")
    st.caption("تعلم السياسات: كل مشاهدة تسهم بدرجتها المتعامدة Γ̂، والسياسة تختار العلاج حيث Γ̂ موجبة في المتوسط ضمن فئة قواعد قابلة للتفسير.")
    ui.refs("Manski (2004) Econometrica 72(4)؛ Kitagawa & Tetenov (2018) Econometrica 86(2)؛ Athey & Wager (2021) Econometrica 89(1)؛ Kasy & Sautmann (2021) Econometrica 89(1).")

# --------------------------------------------------------------------------- 8
with tabs[7]:
    ui.section("مختبر: لماذا تفوز الأوزان المتساوية أحيانًا؟", "Combination puzzle, simulated", "orange")
    st.markdown(
        "تنبؤان (قياسي وتعلم آلي) بأخطاء طبيعية. نقدّر الأوزان المثلى من عينة تقييم قصيرة بطول $T$، ثم نقيس الأداء على فترة جديدة طويلة. "
        "نكرر 3000 مرة."
    )
    c1, c2, c3 = st.columns(3)
    ratio = c1.slider("نسبة انحراف خطأ ML إلى القياسي", 0.6, 1.6, 1.0, 0.05, key="cb_ratio")
    rho = c2.slider("الارتباط بين الخطأين", -0.5, 0.95, 0.5, 0.05, key="cb_rho")
    T = c3.select_slider("طول عينة تقدير الأوزان T", [10, 20, 40, 80, 160, 400], value=20, key="cb_T")
    s1, s2 = 1.0, ratio
    s12 = rho * s1 * s2
    wgrid = np.linspace(-0.2, 1.2, 141)
    var = wgrid ** 2 * s1 ** 2 + (1 - wgrid) ** 2 * s2 ** 2 + 2 * wgrid * (1 - wgrid) * s12
    den = s1 ** 2 + s2 ** 2 - 2 * s12
    wstar = (s2 ** 2 - s12) / den if den > 1e-9 else 0.5
    rng = np.random.default_rng(0)
    L = np.linalg.cholesky(np.array([[s1 ** 2, s12], [s12, s2 ** 2]]) + 1e-12 * np.eye(2))
    E = rng.normal(size=(3000, T, 2)) @ L.T
    v1, v2 = E[..., 0].var(1), E[..., 1].var(1)
    c = ((E[..., 0] - E[..., 0].mean(1, keepdims=True)) * (E[..., 1] - E[..., 1].mean(1, keepdims=True))).mean(1)
    d = v1 + v2 - 2 * c
    what = np.clip(np.where(d > 1e-9, (v2 - c) / np.where(d > 1e-9, d, 1), 0.5), -1, 2)
    true_var = lambda w: w ** 2 * s1 ** 2 + (1 - w) ** 2 * s2 ** 2 + 2 * w * (1 - w) * s12  # noqa: E731
    est_mse = float(np.mean(true_var(what)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wgrid, y=var, name="تباين خطأ التوليف", line=dict(color="#f97316", width=3)))
    fig.add_vline(x=wstar, line_dash="dash", line_color="#0d9488", annotation_text=f"w* = {wstar:.2f}")
    fig.add_vline(x=0.5, line_dash="dot", line_color="#7c3aed", annotation_text="متساوية")
    fig.add_trace(go.Histogram(x=what, name="توزيع ŵ المقدَّر", yaxis="y2", opacity=0.35, marker_color="#0ea5e9", nbinsx=40))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="تكرار ŵ"), bargap=0.05)
    fig.update_xaxes(title="وزن النموذج القياسي w")
    fig.update_yaxes(title="تباين خطأ التوليف")
    ui.style_fig(fig, "منحنى التباين وتوزيع الأوزان المقدَّرة من عينة قصيرة", height=450)
    ui.show(fig)
    res = pd.DataFrame([
        ("القياسي وحده", s1 ** 2), ("التعلم الآلي وحده", s2 ** 2), ("متساوية (0.5)", true_var(0.5)),
        ("المثلى المعروفة w*", true_var(wstar)), (f"المثلى المقدَّرة من T={T}", est_mse),
    ], columns=["الطريقة", "تباين الخطأ خارج العينة"])
    st.dataframe(res.round(4), hide_index=True)
    ui.note(
        "**جرّب:** نسبة = 1 وارتباط = 0.5 وT = 20 ← الأوزان المتساوية تساوي المثلى، والمقدَّرة **أسوأ** بسبب تشتت ŵ (اللغز). "
        "نسبة = 1.5 وT = 400 ← المقدَّرة تقترب من المثلى وتتفوق على المتساوية. ارتباط 0.95 ← المنحنى مسطح والمكسب صغير، والتقدير يصبح غير مستقر.",
        "اقرأ المختبر", "orange", "🧪",
    )

# --------------------------------------------------------------------------- 9
with tabs[8]:
    ui.section("قائمة تحقق للتوليف والقرار", "Checklist", "slate")
    st.markdown(
        "- [ ] قارنتُ التوليف بالمتوسط البسيط دائمًا، لا بالنماذج المنفردة فقط.\n"
        "- [ ] الأوزان مقدَّرة من تنبؤات خارج العينة **سابقة زمنيًا** لفترة التقييم.\n"
        "- [ ] مع عينة قصيرة أو نماذج كثيرة: قلّصتُ الأوزان نحو المتساوية.\n"
        "- [ ] اختبرتُ الشمول لأعرف هل يضيف النموذج الذكي معلومات فعلًا.\n"
        "- [ ] فحصتُ تغير الأوزان المثلى عبر الزمن (أنظمة، أزمات).\n"
        "- [ ] للقرارات: قيّمتُ السياسة بالرفاه أو الربح المقدَّر بدرجات متعامدة، لا بدقة التنبؤ.\n"
        "- [ ] السياسات من فئة قابلة للتفسير ومقيدة بما يمنع الإفراط في التوفيق."
    )
