import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "الدمج في دالة الهدف والتقدير",
    "Integration at the objective level — losses, priors, theory penalties, moments, likelihoods and simulation-based estimation",
    "كل مقدّر — من المربعات الصغرى إلى شبكة عميقة — هو <b>حل لمسألة تحسين</b>: نختار المعلمات التي تصغّر دالة هدف. إذن أقوى طريقة "
    "للدمج هي أن <b>نغيّر ما نصغّره</b>: نضيف قيدًا نظريًا، أو معرفة مسبقة، أو عزمًا اقتصاديًا، أو خسارة تعكس قرارًا حقيقيًا. "
    "هذه الصفحة تشرح هذا الباب من أساسه الرياضي إلى أحدث المقدرات التنافسية.",
    "rose", "violet",
)

tabs = st.tabs([
    ":material/target: التقدير كتحسين",
    ":material/balance: الخسارة تتبع القرار",
    ":material/psychology: التنظيم كمعرفة مسبقة",
    ":material/school: الخسائر الموجهة بالنظرية",
    ":material/functions: العزوم: من GMM إلى التنافسي",
    ":material/stacked_bar_chart: الأرجحية في الشبكات",
    ":material/casino: التقدير بالمحاكاة والتنافس",
    ":material/science: مختبر: متى تفيد النظرية؟",
    ":material/checklist: قائمة التحقق",
])

# --------------------------------------------------------------------------- 1
with tabs[0]:
    ui.section("إطار واحد لكل المقدرات", "Extremum estimation", "rose")
    st.latex(r"\hat\theta=\arg\min_{\theta\in\Theta}\ Q_n(\theta)")
    est = pd.DataFrame([
        ("المربعات الصغرى", r"(1/n)Σ(y−x'θ)²", "متوسط شرطي خطي"),
        ("الأرجحية العظمى", r"−(1/n)Σ log p(y|x;θ)", "توزيع كامل"),
        ("العزوم المعممة GMM", r"ḡ(θ)' W ḡ(θ)", "شروط عزوم من النظرية"),
        ("الانحدار الكمّي", r"(1/n)Σ ρ_τ(y−x'θ)", "كمّي شرطي"),
        ("Lasso", r"LS + λ Σ|θ_j|", "متوسط شرطي متناثر"),
        ("شبكة عصبية", r"(1/n)Σ L(y, f_θ(x)) + λ‖θ‖²", "أي دالة تقريبًا"),
    ], columns=["المقدّر", "دالة الهدف", "ما الذي يقدّره"])
    st.dataframe(est, hide_index=True)
    st.markdown(
        "**الاتساق** يتطلب شرطين عامين: (1) **التعرّف**: للدالة الحدية $Q_0(\\theta)$ حد أدنى وحيد عند $\\theta_0$؛ (2) **التقارب المنتظم**: "
        "$\\sup_\\theta|Q_n-Q_0|\\to0$. التعلم الآلي يكسر الشرط الأول غالبًا: شبكة كبيرة لها **عدد لا نهائي** من الأوزان المتكافئة، فالأوزان "
        "غير معرّفة رغم أن **الدالة** $f_\\theta$ قد تكون معرّفة. لذلك في الدمج نحرص أن تكون **المعلمة التي نفسرها معرّفة** حتى لو كانت "
        "بقية الأوزان غير معرّفة."
    )
    ui.note(
        "**المعنى العملي:** «الدمج في الهدف» = كتابة $Q_n$ جديدة تجمع جزءًا من القياس الاقتصادي وجزءًا من التعلم الآلي:\n"
        "$$Q_n(\\theta)=\\underbrace{\\text{ملاءمة البيانات}}_{\\text{LS, ML, GMM}}+\\lambda_1\\underbrace{\\text{التعقيد}}_{\\text{تنظيم}}+\\lambda_2\\underbrace{\\text{مخالفة النظرية}}_{\\text{قيود اقتصادية}}$$\n"
        "كل حد يأتي من تقليد مختلف، والأوزان $\\lambda$ تحدد التوازن بينها.",
        "الفكرة المركزية", "violet", "🎯",
    )
    ui.refs("Newey & McFadden (1994) Handbook of Econometrics 4 ch. 36؛ Amemiya (1985) Advanced Econometrics.")

# --------------------------------------------------------------------------- 2
with tabs[1]:
    ui.section("ما تصغّره يحدد ما تتنبأ به", "The loss determines the target", "amber")
    st.markdown("لكل دالة خسارة «تنبؤ أمثل» مختلف. هذا ليس تفصيلًا تقنيًا؛ إنه اختيار **الهدف الاقتصادي**:")
    losses = [
        ("مربع الخطأ", r"L(e)=e^2\ \Rightarrow\ \hat y^*=\mathbb E[y\mid x]", "المتوسط. متماثل، يعاقب الأخطاء الكبيرة بشدة."),
        ("القيمة المطلقة", r"L(e)=|e|\ \Rightarrow\ \hat y^*=\operatorname{Med}(y\mid x)", "الوسيط. أقل حساسية للقيم المتطرفة."),
        ("الخسارة الكمية", r"\rho_\tau(e)=e\big(\tau-\mathbb 1\{e<0\}\big)\ \Rightarrow\ \hat y^*=Q_\tau(y\mid x)", "الكمّي τ. أساس القيمة المعرضة للخطر وفترات التنبؤ."),
        ("LINEX", r"L(e)=b\big(e^{ae}-ae-1\big)", "غير متماثلة: $a>0$ يجعل التنبؤ الناقص أكثر كلفة، فينحاز التنبؤ الأمثل للأعلى (Varian 1975)."),
        ("قواعد التقييم السليمة", r"\text{CRPS}(F,y)=\int\big(F(z)-\mathbb 1\{y\le z\}\big)^2dz", "لتقييم تنبؤ **توزيعي** كامل لا رقم واحد."),
    ]
    for t, eq, body in losses:
        with ui.tinted("slate"):
            st.markdown(f"**{t}**")
            st.latex(eq)
            st.markdown(body)
    ui.note(
        "**مثال:** بنك مركزي يرى أن التقليل من التضخم أخطر من المبالغة فيه. تدريب شبكة بمربع الخطأ يعطيه المتوسط، وهذا ليس ما يريده. "
        "تدريبها بـ LINEX أو بخسارة كمية عند 0.7 يعطيه تنبؤًا يعكس تفضيلاته. Granger (1969) وChristoffersen & Diebold (1997) "
        "أوضحا أن التنبؤ الأمثل تحت خسارة غير متماثلة مع تباين متغير **ليس** المتوسط زائد ثابت؛ إنه يعتمد على التباين الشرطي.",
        "دمج القرار في التقدير", "amber", "🏦",
    )
    ui.refs("Granger (1969) Operational Research Quarterly 20؛ Varian (1975)؛ Christoffersen & Diebold (1997) Econometric Theory 13(6)؛ Gneiting & Raftery (2007) JASA 102(477)؛ Koenker & Bassett (1978) Econometrica 46(1).")

# --------------------------------------------------------------------------- 3
with tabs[2]:
    ui.section("التنظيم = معرفة مسبقة بلغة أخرى", "Regularization is a prior", "violet")
    st.markdown("حل Ridge هو بالضبط **الوضع اللاحق (MAP)** تحت توزيع مسبق طبيعي، وLasso تحت توزيع لابلاس:")
    st.latex(r"\arg\min_\beta\ \|y-X\beta\|^2+\lambda\|\beta\|_2^2\ \equiv\ \arg\max_\beta\ p(y\mid\beta)\,\underbrace{\mathcal N\big(\beta;0,\tfrac{\sigma^2}{\lambda}I\big)}_{\text{مسبق}}")
    st.markdown("وهذا يفتح بابًا للدمج: بدل التقليص نحو **الصفر**، نقلّص نحو **ما تقوله النظرية**:")
    st.latex(r"\hat\beta(\lambda)=\arg\min_\beta\ \|y-X\beta\|^2+\lambda\|\beta-\beta^{\text{theory}}\|^2=(X'X+\lambda I)^{-1}\big(X'y+\lambda\beta^{\text{theory}}\big)")
    ui.section("ثلاثة أمثلة تاريخية مهمة", "Three landmark examples", "teal")
    ui.cards([
        {"title": "توزيع مينيسوتا المسبق", "en": "Minnesota prior (Litterman 1986)", "color": "teal",
         "body": "VAR بمعاملات كثيرة يتقلص نحو «كل متغير سير عشوائي»، والفجوات البعيدة تتقلص أكثر. أول دمج منهجي للمعرفة الاقتصادية في تقدير عالي الأبعاد، وأساس BVAR الحديثة."},
        {"title": "DSGE-VAR", "en": "Del Negro & Schorfheide (2004)", "color": "violet",
         "body": "VAR يتقلص نحو القيود التي يفرضها نموذج DSGE، ووزن النظرية $\\lambda$ يُقدَّر من البيانات. إن كان $\\hat\\lambda$ كبيرًا فالبيانات تؤيد النظرية."},
        {"title": "VAR الكبير بالتقليص", "en": "Bańbura, Giannone & Reichlin (2010)", "color": "amber",
         "body": "أثبتوا أن VAR بمئة متغير يتفوق تنبؤيًا إذا زاد التقليص مع زيادة عدد المتغيرات. ربط مباشر بين التنظيم في التعلم الآلي والإحصاء البيزي الكلي."},
    ], cols=3)
    ui.refs("Hoerl & Kennard (1970) Technometrics 12؛ Litterman (1986) JBES 4(1)؛ Del Negro & Schorfheide (2004) International Economic Review 45(2)؛ Bańbura, Giannone & Reichlin (2010) J. Applied Econometrics 25(1).")

# --------------------------------------------------------------------------- 4
with tabs[3]:
    ui.section("النظرية كحد في دالة الخسارة", "Theory-informed losses", "emerald")
    st.markdown("الصيغة العامة:")
    st.latex(r"\hat f=\arg\min_f\ \underbrace{\frac1n\sum_i L\big(y_i,f(x_i)\big)}_{\text{ملاءمة البيانات}}+\lambda\,\underbrace{\frac1M\sum_{m}R\big(f;\tilde x_m\big)^2}_{\text{بواقي النظرية}}")
    st.markdown(
        "حيث $R(f;\\tilde x)$ «باقي النظرية»: مقدار مخالفة $f$ لشرط اقتصادي عند نقطة $\\tilde x$. ولاحظ أن نقاط النظرية $\\tilde x_m$ "
        "**لا تحتاج بيانات $y$**: يمكن أن تكون نقاطًا اصطناعية في مناطق لا توجد فيها مشاهدات، فتملأ النظرية الفراغ حيث تصمت البيانات."
    )
    ex = [
        ("معادلة أويلر للاستهلاك", r"R=\beta\,\mathbb E\big[u'(c_{t+1})(1+r_{t+1})\big]-u'(c_t)"),
        ("رتابة الطلب", r"R=\big[\partial f/\partial p\big]_+"),
        ("عدم المراجحة", r"R=\mathbb E\big[M_{t+1}R^e_{i,t+1}\big]"),
        ("تجانس دالة الطلب", r"R=f(\kappa p,\kappa y)-f(p,y)"),
        ("قيد ميزانية", r"R=\textstyle\sum_jp_jq_j(p,y)-y"),
    ]
    st.dataframe(pd.DataFrame([(a, b) for a, b in ex], columns=["الشرط الاقتصادي", "باقي النظرية R"]), hide_index=True)
    ui.section("ماذا يحدث عندما تكون النظرية خاطئة؟", "Bias from a wrong restriction", "rose")
    st.latex(r"\lambda\to\infty:\ \hat f\to f^{\text{theory}},\qquad \text{Bias}\to f^{\text{theory}}-f_0\neq0\ \ \text{لا يختفي بزيادة } n")
    st.markdown(
        "مع $\\lambda=0$: لا تحيز من النظرية لكن تباين كبير. مع $\\lambda$ كبير: تباين صغير لكن تحيز بقدر خطأ النظرية. الـ $\\lambda$ الأمثل "
        "يتناقص كلما كبرت العينة **إذا كانت النظرية خاطئة**، ويبقى كبيرًا **إذا كانت صحيحة**. لهذا يُختار $\\lambda$ خارج العينة، ويُبلَّغ "
        "عن النتائج لعدة قيم. المختبر في هذه الصفحة يوضح ذلك."
    )
    ui.refs("Raissi, Perdikaris & Karniadakis (2019) J. Computational Physics 378 (PINN)؛ Chen, Pelger & Zhu (2024) Management Science 70(2)؛ Maliar, Maliar & Winant (2021).")

# --------------------------------------------------------------------------- 5
with tabs[4]:
    ui.section("العزوم المعممة: النظرية تعطي شروطًا لا معادلات", "GMM", "indigo")
    st.markdown("النظرية الاقتصادية تقول غالبًا إن **توقعًا معينًا يساوي صفرًا**، لا ما هي المعادلة كاملة:")
    st.latex(r"\mathbb E\big[g(W;\theta_0)\big]=0,\qquad \hat\theta=\arg\min_\theta\ \bar g_n(\theta)'\,W_n\,\bar g_n(\theta)")
    st.markdown("مثال كلاسيكي (Hansen & Singleton 1982): شرط أويلر مضروبًا في أي متغير معروف وقت $t$:")
    st.latex(r"\mathbb E\Big[\Big(\beta\big(\tfrac{c_{t+1}}{c_t}\big)^{-\gamma}R_{t+1}-1\Big)z_t\Big]=0")
    ui.section("التعلم الآلي في GMM: ثلاث صور", "ML meets moments", "teal")
    ui.ladder([
        ("أدوات كثيرة مع Lasso", "Many instruments", "عندما يوجد مئات المتغيرات الآلية المحتملة، Lasso يختار الأداة المثلى للمرحلة الأولى "
         "(Belloni, Chen, Chernozhukov & Hansen 2012). يتجنب تحيز الأدوات الكثيرة الضعيفة."),
        ("المرحلة الأولى المرنة", "Deep IV", "Hartford et al. (2017): شبكة تقدّر توزيع المتغير الداخلي بشرط الأداة، ثم شبكة ثانية للنتيجة تُدرَّب "
         "على التوقع تحت ذلك التوزيع. يسمح بأثر لاخطي للمتغير الداخلي."),
        ("العزوم التنافسية", "Adversarial GMM", "بدل اختيار عدد محدود من العزوم، نبحث عن **أسوأ** دالة اختبار $f$ (تحددها شبكة خصم) ونصغّر "
         "مخالفة الشرط عندها. هذا يقدّر دوال كاملة لا معلمات فقط."),
    ])
    st.latex(r"\hat h=\arg\min_{h\in\mathcal H}\ \max_{f\in\mathcal F}\ \frac1n\sum_i\big(y_i-h(x_i)\big)f(z_i)-\frac{1}{2}\|f\|_n^2")
    st.caption("المتغيرات الآلية غير المعلمية بصيغة لعبة: h يحاول احترام شرط الأداة، وf يبحث عن أكبر مخالفة. (Dikkala, Lewis, Mackey & Syrgkanis 2020؛ Bennett, Kallus & Schnabel 2019)")
    ui.refs("Hansen (1982) Econometrica 50(4)؛ Hansen & Singleton (1982) Econometrica 50(5)؛ Belloni et al. (2012) Econometrica 80(6)؛ Hartford et al. (2017) ICML؛ Dikkala et al. (2020) NeurIPS.")

# --------------------------------------------------------------------------- 6
with tabs[5]:
    ui.section("تدريب الشبكات بالأرجحية بدل مربع الخطأ", "Likelihood as the training loss", "sky")
    st.markdown(
        "مربع الخطأ يفترض ضمنًا توزيعًا طبيعيًا بتباين ثابت. في الاقتصاد والمالية هذا خاطئ غالبًا (ذيول ثقيلة، تباين متغير). الحل: تدريب "
        "الشبكة على **الأرجحية الكاملة**. مثال: GARCH عصبي يتنبأ بالتباين $\\sigma_t^2$ ويُدرَّب على أرجحية ستيودنت:"
    )
    st.latex(r"\ell_t(\theta)=\log\Gamma\!\big(\tfrac{\nu+1}{2}\big)-\log\Gamma\!\big(\tfrac\nu2\big)-\tfrac12\log\big(\pi(\nu-2)\sigma_t^2\big)-\tfrac{\nu+1}{2}\log\Big(1+\frac{\epsilon_t^2}{(\nu-2)\sigma_t^2}\Big)")
    st.markdown(
        "- **الشبكة تتعلم** كيف يعتمد $\\sigma_t^2$ على الماضي (بدل معادلة GARCH ثابتة).\n"
        "- **الأرجحية القياسية** تضمن أن التنبؤ توزيع صحيح لا رقم واحد، فيمكن حساب VaR وES.\n"
        "- **القيود القياسية** (موجبية التباين، الاستقرارية) تُفرض بالمعمارية: $\\sigma^2=\\text{softplus}(\\cdot)$، و$\\alpha+\\beta<1$ بتحويل.\n\n"
        "هذا ما تفعله مكتبة garchai لعشرين نموذجًا: معادلات قياسية قابلة للتفاضل تُدرَّب بالانتشار العكسي على الأرجحية."
    )
    ui.section("شبكات كثافة المزيج", "Mixture density networks", "violet")
    st.latex(r"p(y\mid x)=\sum_{k=1}^{K}\pi_k(x)\,\mathcal N\big(y;\mu_k(x),\sigma_k^2(x)\big)")
    st.markdown("الشبكة تخرج أوزان المزيج والمتوسطات والتباينات، فتمثل توزيعات متعددة القمم (نظامان اقتصاديان محتملان). تُدرَّب بالأرجحية.")
    ui.refs("Bollerslev (1987) REStat 69(3)؛ Bishop (1994) Mixture Density Networks؛ Zhao et al. (2024) AAAI؛ Nguyen et al. (2025) neural GARCH.")

# --------------------------------------------------------------------------- 7
with tabs[6]:
    ui.section("عندما لا نستطيع كتابة الأرجحية", "Simulation-based estimation", "fuchsia")
    st.markdown("كثير من النماذج الهيكلية يمكن **محاكاتها** لكن لا يمكن كتابة أرجحيتها. ثلاثة أجيال من الحلول:")
    ui.ladder([
        ("طريقة العزوم المحاكاة", "SMM", "اختر عزومًا (متوسطات، تباينات، ارتباطات)، حاكِ النموذج، وطابق العزوم المحاكاة مع الحقيقية: "
         "$\\hat\\theta=\\arg\\min(\\hat m-\\tilde m(\\theta))'W(\\hat m-\\tilde m(\\theta))$. **المشكلة:** اختيار العزوم يدوي."),
        ("الاستدلال غير المباشر", "Indirect inference (Gouriéroux, Monfort & Renault 1993)", "قدّر نموذجًا مساعدًا (VAR مثلًا) على البيانات وعلى المحاكاة، "
         "وطابق معاملاته. **الدمج:** النموذج المساعد يمكن أن يكون شبكة عصبية تلتقط خصائص أغنى."),
        ("التقدير التنافسي", "Adversarial estimation (Kaji, Manresa & Pouliot 2023)", "مميِّز (شبكة) يحاول التفريق بين البيانات الحقيقية والمحاكاة؛ "
         "المعلمات تُختار لخداعه. المميِّز يختار «العزوم» تلقائيًا. أثبتوا اتساقًا ومعدلات وكفاءة في حالات معينة."),
        ("المقدرات العصبية المُطفأة", "Amortized neural estimators", "درّب شبكة مرة واحدة على ملايين المحاكيات لتأخذ بيانات وتعطي المعلمات مباشرة. "
         "بعد التدريب، التقدير لحظي لأي عينة جديدة."),
    ])
    st.latex(r"\hat\theta=\arg\min_\theta\ \max_{D}\ \frac1n\sum_{i}\log D(X_i)+\frac1m\sum_{j}\log\big(1-D(\tilde X_j(\theta))\big)")
    st.caption("صيغة التقدير التنافسي: D يحاول التمييز، θ يحاول جعل المحاكاة لا تُميَّز. عند التوازن تطابق المحاكاة البيانات في كل ما يستطيع D رؤيته.")
    ui.refs("McFadden (1989) Econometrica 57(5)؛ Gouriéroux, Monfort & Renault (1993) J. Applied Econometrics 8؛ Kaji, Manresa & Pouliot (2023) Econometrica 91(6)؛ Goodfellow et al. (2014) NeurIPS.")

# --------------------------------------------------------------------------- 8
with tabs[7]:
    ui.section("مختبر: التقليص نحو النظرية — متى يفيد ومتى يضر؟", "Shrinking toward theory", "emerald")
    st.markdown(
        "نقدّر معاملًا $\\beta$ (مثلًا مرونة) من عينة صغيرة بالتقليص نحو قيمة نظرية $\\beta^{\\text{theory}}$:"
        " $\\hat\\beta(\\lambda)=(X'X+\\lambda)^{-1}(X'y+\\lambda\\beta^{\\text{theory}})$. نحسب متوسط مربع الخطأ عبر آلاف العينات لكل $\\lambda$."
    )
    c1, c2, c3 = st.columns(3)
    beta_true = 1.0
    gap = c1.slider("خطأ النظرية: β_theory − β_true", 0.0, 1.0, 0.0, 0.05, key="sh_gap")
    n = c2.select_slider("حجم العينة n", [10, 25, 50, 100, 400], value=25, key="sh_n")
    sigma = c3.slider("الضوضاء σ", 0.5, 4.0, 2.0, 0.25, key="sh_sig")
    rng = np.random.default_rng(0)
    R = 4000
    X = rng.normal(size=(R, n))
    Y = beta_true * X + rng.normal(0, sigma, size=(R, n))
    sxx, sxy = (X * X).sum(1), (X * Y).sum(1)
    lams = np.r_[0, np.logspace(-1, 3, 60)]
    bt = beta_true + gap
    mse, bias2, var = [], [], []
    for lam in lams:
        b = (sxy + lam * bt) / (sxx + lam)
        mse.append(np.mean((b - beta_true) ** 2))
        bias2.append((b.mean() - beta_true) ** 2)
        var.append(b.var())
    mse, bias2, var = map(np.array, (mse, bias2, var))
    best = int(np.argmin(mse))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lams[1:], y=mse[1:], name="MSE", line=dict(color="#7c3aed", width=3)))
    fig.add_trace(go.Scatter(x=lams[1:], y=bias2[1:], name="تحيز²", line=dict(color="#e11d48", dash="dot")))
    fig.add_trace(go.Scatter(x=lams[1:], y=var[1:], name="تباين", line=dict(color="#0d9488", dash="dot")))
    fig.add_hline(y=mse[0], line_dash="dash", line_color="#64748b", annotation_text="OLS بلا نظرية (λ=0)")
    fig.update_xaxes(type="log", title="وزن النظرية λ")
    fig.update_yaxes(title="الخطأ")
    ui.style_fig(fig, "مقايضة التحيز والتباين في وزن النظرية", height=440)
    ui.show(fig)
    c1, c2, c3 = st.columns(3)
    c1.metric("MSE لـ OLS", f"{mse[0]:.4f}")
    c2.metric("أفضل MSE", f"{mse[best]:.4f}", f"{100 * (mse[best] / mse[0] - 1):.0f}%")
    c3.metric("λ الأمثل", "0" if best == 0 else f"{lams[best]:.1f}")
    ui.note(
        "**جرّب:**\n"
        "- خطأ النظرية = 0: كلما زاد λ تحسن الأداء؛ النظرية الصحيحة «مجانية».\n"
        "- خطأ النظرية = 0.5 مع n صغير: λ متوسط أفضل من OLS (التحيز صغير مقارنة بالتباين المُوفَّر).\n"
        "- خطأ النظرية = 0.5 مع n = 400: λ الأمثل يقترب من الصفر؛ البيانات كافية والنظرية الخاطئة تضر.\n\n"
        "**الدرس:** قيمة النظرية في الدمج أكبر ما تكون عندما تكون البيانات **قليلة أو مشوشة**، وتصبح عبئًا إذا كانت خاطئة والبيانات وفيرة.",
        "اقرأ المختبر", "emerald", "🧪",
    )

# --------------------------------------------------------------------------- 9
with tabs[8]:
    ui.section("قائمة تحقق للدمج في دالة الهدف", "Checklist", "slate")
    st.markdown(
        "- [ ] الخسارة تعكس القرار الاقتصادي الفعلي (متماثلة؟ كمّية؟ توزيعية؟).\n"
        "- [ ] المعلمة التي أفسرها معرّفة حتى لو لم تكن بقية الأوزان معرّفة.\n"
        "- [ ] كل قيد نظري مكتوب رياضيًا بباقٍ واضح، ومبرَّر اقتصاديًا.\n"
        "- [ ] وزن النظرية λ مختار خارج العينة، والنتائج مُبلَّغ عنها لعدة قيم.\n"
        "- [ ] فحصتُ هل البيانات ترفض القيد (اختبار J في GMM، أو تدهور الأداء عند فرضه).\n"
        "- [ ] في التقدير بالمحاكاة: فصلتُ خطأ المحاكاة عن الخطأ الإحصائي.\n"
        "- [ ] الأخطاء المعيارية تأخذ في الحسبان التنظيم أو استُبدلت بـ bootstrap مناسب."
    )
