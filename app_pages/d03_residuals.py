import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "الدمج عبر البواقي",
    "Residual-based integration — from Frisch–Waugh–Lovell to DML, Zhang's hybrid, boosting and residual diagnostics",
    "البواقي هي «ما لم يفسره النموذج». إذا كان النموذج صحيحًا فلا يمكن التنبؤ بها من المعلومات المتاحة؛ وإن أمكن، فهذا دليل "
    "على نقص في المواصفة <b>وفرصة للدمج</b>. هذه الفكرة البسيطة هي أقدم أشكال الدمج وأكثرها انتشارًا، وتظهر في صورتين مختلفتين تمامًا: "
    "<b>البواقي للاستدلال</b> (من فريش–وو–لوفيل إلى DML) و<b>البواقي للتنبؤ</b> (هجين Zhang والتعزيز).",
    "amber", "violet",
)

tabs = st.tabs([
    ":material/lightbulb: الفكرة من الصفر",
    ":material/history_edu: الجذر: مبرهنة FWL",
    ":material/hub: من FWL إلى روبنسون إلى DML",
    ":material/merge: هجين Zhang ومتغيراته",
    ":material/troubleshoot: اختبار البواقي قبل الدمج",
    ":material/stacked_line_chart: بواقي التباين والكمّيات",
    ":material/sync_alt: الاتجاه المعاكس والتعزيز",
    ":material/science: مختبر: داخل العينة مقابل خارجها",
    ":material/checklist: الأخطاء وقائمة التحقق",
])

# --------------------------------------------------------------------------- 1
with tabs[0]:
    ui.section("ما هي البواقي؟ ولماذا هي موضع دمج؟", "What residuals are and why they matter", "amber")
    st.markdown("لنموذج قياسي $y_t=m(x_t;\\theta)+u_t$، البواقي هي الفرق بين المشاهَد والمقدَّر:")
    st.latex(r"\hat u_t=y_t-m(x_t;\hat\theta)")
    st.markdown(
        "النموذج **الصحيح** يحقق شرطًا أساسيًا: الخطأ لا يمكن التنبؤ به من مجموعة المعلومات $\\mathcal F_{t-1}$ المتاحة عند التنبؤ:"
    )
    st.latex(r"\mathbb E\big[u_t\mid\mathcal F_{t-1}\big]=0")
    st.markdown(
        "إذا وجدنا دالة $h$ بحيث $\\mathbb E[u_t\\mid\\mathcal F_{t-1}]=h(z_{t-1})\\neq0$، فالنموذج **أهمل شيئًا**: لاخطية، تفاعلًا، "
        "انكسارًا، أو متغيرًا. عندها يمكن تحسين التنبؤ مباشرة:"
    )
    st.latex(r"\mathbb E[y_t\mid\mathcal F_{t-1}]=m(x_t;\theta)+h(z_{t-1})\quad\Longrightarrow\quad \hat y_t=m(x_t;\hat\theta)+\hat h(z_{t-1})")
    ui.note(
        "هذا هو الأساس النظري لكل هجين بواقي: **النموذج القياسي يلتقط الجزء الذي نفهمه، والتعلم الآلي يبحث عن النمط في ما تبقى.** "
        "لكن لاحظ الشرط الدقيق: يجب أن يكون $h$ دالة في معلومات **سابقة** $z_{t-1}$. استخدام بواقي الفترة نفسها أو أي معلومة "
        "لاحقة يحول الدمج إلى تسرب.",
        "الفكرة المركزية", "amber", "💡",
    )
    ui.section("صورتان مختلفتان تمامًا للدمج عبر البواقي", "Two very different uses", "violet")
    c1, c2 = st.columns(2)
    with c1:
        with ui.tinted("indigo"):
            st.markdown("#### البواقي للاستدلال")
            st.markdown(
                "**الهدف:** معلمة واحدة $\\theta$ (أثر سببي).\n\n"
                "**الفكرة:** نزيل من $Y$ ومن $D$ كل ما تفسره الضوابط $X$، ثم ننحدر **باقيًا على باقٍ**. التعلم الآلي يحسن «الإزالة».\n\n"
                "**الأمثلة:** FWL، Robinson (1988)، DML.\n\n"
                "**المعيار:** عدم التحيز وصحة فترة الثقة."
            )
    with c2:
        with ui.tinted("amber"):
            st.markdown("#### البواقي للتنبؤ")
            st.markdown(
                "**الهدف:** تنبؤ $\\hat y$ أدق.\n\n"
                "**الفكرة:** نقدّر نموذجًا قياسيًا ثم نتنبأ ببواقيه بنموذج مرن ونجمع التنبؤين.\n\n"
                "**الأمثلة:** هجين Zhang (ARIMA–ANN)، ARIMA–LSTM، ARDL–غابة، التعزيز التدرجي.\n\n"
                "**المعيار:** خطأ التنبؤ خارج العينة."
            )
    st.markdown("الخلط بين الصورتين مصدر أخطاء كثيرة: تحسين RMSE لا يعني معلمة سببية أفضل، ومعلمة DML صحيحة لا تعني تنبؤًا أفضل.")

# --------------------------------------------------------------------------- 2
with tabs[1]:
    ui.section("مبرهنة فريش–وو–لوفيل: أصل «الباقي على الباقي»", "Frisch–Waugh (1933), Lovell (1963)", "indigo")
    st.markdown("لنموذج خطي بمجموعتين من المتغيرات:")
    st.latex(r"y=X_1\beta_1+X_2\beta_2+u")
    st.markdown("تقول المبرهنة إن معامل $\\beta_1$ من الانحدار الكامل **يساوي تمامًا** المعامل الناتج عن ثلاث خطوات:")
    ui.ladder([
        ("أزل أثر $X_2$ من $y$", "Partial out y", "انحدر $y$ على $X_2$ وخذ البواقي: $\\tilde y=M_2y$ حيث $M_2=I-X_2(X_2'X_2)^{-1}X_2'$."),
        ("أزل أثر $X_2$ من $X_1$", "Partial out X1", "انحدر $X_1$ على $X_2$ وخذ البواقي: $\\tilde X_1=M_2X_1$."),
        ("انحدر الباقي على الباقي", "Residual on residual", "$\\hat\\beta_1=(\\tilde X_1'\\tilde X_1)^{-1}\\tilde X_1'\\tilde y$ — مطابق للانحدار الكامل."),
    ])
    st.latex(r"\hat\beta_1=\big(X_1'M_2X_1\big)^{-1}X_1'M_2\,y")
    with st.expander("البرهان المختصر"):
        st.markdown(
            "من المعادلات الطبيعية للانحدار الكامل: $X_2'(y-X_1\\hat\\beta_1-X_2\\hat\\beta_2)=0$، إذن "
            "$\\hat\\beta_2=(X_2'X_2)^{-1}X_2'(y-X_1\\hat\\beta_1)$. بالتعويض في المعادلة الأولى "
            "$X_1'(y-X_1\\hat\\beta_1-X_2\\hat\\beta_2)=0$ نحصل على $X_1'M_2(y-X_1\\hat\\beta_1)=0$، أي "
            "$\\hat\\beta_1=(X_1'M_2X_1)^{-1}X_1'M_2y$. ولأن $M_2$ متماثلة ومتساوية القوى ($M_2=M_2'=M_2^2$) فهذا هو انحدار "
            "$M_2y$ على $M_2X_1$. ∎"
        )
    ui.section("تحقق رقمي حي", "Numerical check", "teal")
    rng = np.random.default_rng(11)
    n = 300
    X2 = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    x1 = X2[:, 1:] @ np.array([0.8, -0.5, 0.3]) + rng.normal(size=n)
    y = 1.5 * x1 + X2 @ np.array([2.0, 1.0, 0.5, -1.0]) + rng.normal(size=n)
    full = np.linalg.lstsq(np.column_stack([x1, X2]), y, rcond=None)[0][0]
    M2 = np.eye(n) - X2 @ np.linalg.pinv(X2)
    fwl = float((M2 @ x1) @ (M2 @ y) / ((M2 @ x1) @ (M2 @ x1)))
    c1, c2, c3 = st.columns(3)
    c1.metric("β₁ الحقيقي", "1.5000")
    c2.metric("β₁ من الانحدار الكامل", f"{full:.6f}")
    c3.metric("β₁ من الباقي على الباقي", f"{fwl:.6f}")
    ui.note(
        "**لماذا هذا مهم للدمج؟** لأن المبرهنة تفصل المشكلة إلى جزأين: **جزء إزالة** (التنبؤ بـ $y$ وبـ $X_1$ من الضوابط) و**جزء "
        "تقدير** (باقٍ على باقٍ). جزء الإزالة هو مسألة **تنبؤ بحتة**، أي بالضبط ما يتقنه التعلم الآلي. إذا كانت الضوابط كثيرة "
        "أو أثرها لاخطي، نستبدل الانحدار الخطي في خطوتي الإزالة بغابة أو شبكة. هذا هو الجسر المفاهيمي من القياس الكلاسيكي إلى DML.",
        "الجسر إلى التعلم الآلي", "violet", "🌉",
    )
    ui.refs("Frisch & Waugh (1933) Econometrica 1(4)؛ Lovell (1963) JASA 58(304)؛ Davidson & MacKinnon (2004) Econometric Theory and Methods, ch. 2.")

# --------------------------------------------------------------------------- 3
with tabs[2]:
    ui.section("ثلاث محطات لفكرة واحدة", "One idea, three milestones", "indigo")
    ui.ladder([
        ("FWL (1933/1963)", "Linear partialling-out", "الإزالة بانحدار خطي. صحيح فقط إذا كان أثر الضوابط خطيًا."),
        ("Robinson (1988)", "Nonparametric partialling-out", "النموذج جزئي الخطية $Y=\\theta D+g(X)+U$ مع $g$ مجهولة. "
         "الإزالة بمقدرات نواة غير معلمية لـ $\\mathbb E[Y\\mid X]$ و$\\mathbb E[D\\mid X]$. أثبت روبنسون أن $\\hat\\theta$ يتقارب بسرعة "
         "$\\sqrt n$ رغم أن $g$ تُقدَّر أبطأ. لكن النوى تنهار مع أبعاد كثيرة."),
        ("DML (2018)", "ML partialling-out + cross-fitting", "الإزالة بأي متعلم آلي (Lasso، غابة، شبكة) مع تقسيم متقاطع. "
         "يعمل مع مئات الضوابط وبشروط معدلات أضعف."),
    ])
    st.markdown("**الخطوات الرياضية في صيغة روبنسون:** بأخذ التوقع الشرطي على $X$ للنموذج:")
    st.latex(r"\mathbb E[Y\mid X]=\theta\,\mathbb E[D\mid X]+g(X)")
    st.markdown("وبطرحها من النموذج الأصلي تختفي $g$ تمامًا:")
    st.latex(r"\underbrace{Y-\mathbb E[Y\mid X]}_{\tilde Y}=\theta\,\underbrace{\big(D-\mathbb E[D\mid X]\big)}_{\tilde D}+U\qquad\Rightarrow\qquad \hat\theta=\frac{\sum_i\tilde Y_i\tilde D_i}{\sum_i\tilde D_i^2}")
    ui.note(
        "هذه هي الخدعة: **لا نحتاج أن نعرف $g$** ولا أن نفسرها. نحتاج فقط تنبؤين جيدين: $\\ell(X)=\\mathbb E[Y\\mid X]$ و"
        "$m(X)=\\mathbb E[D\\mid X]$. والتنبؤ هو ما يتقنه التعلم الآلي. ثم يبقى التقدير النهائي انحدارًا بسيطًا بمعلمة واحدة "
        "قابلة للتفسير وفترة ثقة.",
        "لماذا تختفي g؟", "teal", "✨",
    )
    st.markdown("#### لماذا التقسيم المتقاطع ضروري هنا؟")
    st.markdown(
        "لو قدّرنا $\\hat\\ell$ و$\\hat m$ على البيانات نفسها التي نحسب عليها البواقي، فإن متعلمًا مرنًا (غابة عميقة) سيحفظ جزءًا من "
        "الضوضاء $U$ و$V$، فتصبح البواقي **أصغر من الحقيقة ومرتبطة بالضوضاء**، فيتحيز $\\hat\\theta$. التقسيم المتقاطع يحسب باقي كل "
        "مشاهدة من نموذج **لم يرها**:"
    )
    st.latex(r"\tilde Y_i=Y_i-\hat\ell^{(-k(i))}(X_i),\qquad \tilde D_i=D_i-\hat m^{(-k(i))}(X_i)")
    st.markdown("حيث $\\hat\\ell^{(-k)}$ مدرَّب على كل الطيات ما عدا الطية $k$ التي تنتمي إليها المشاهدة $i$.")
    st.markdown("#### لماذا يتحمل المقدّر أخطاء التعلم الآلي؟ (التعامد)")
    st.latex(r"\hat\theta-\theta\approx\frac{\frac1n\sum V_iU_i}{\mathbb E V^2}+\underbrace{\frac{\frac1n\sum\big(m-\hat m\big)\big(\ell-\hat\ell\big)}{\mathbb E V^2}}_{\text{حاصل ضرب خطأين}}+\dots")
    st.markdown(
        "الحد الأول هو مصدر التوزيع الطبيعي. الحد الثاني هو أثر أخطاء التعلم الآلي، وهو **حاصل ضرب** خطأ نموذج المعالجة في خطأ "
        "نموذج النتيجة. إذا تقارب كل منهما بمعدل $n^{-1/4}$ فالحاصل $n^{-1/2}$ ويصبح مهملًا مقارنة بالحد الأول. هذا ما يسمى "
        "**التعامد** أو **الحماية المزدوجة**. صفحة «الدمج في الاستدلال» تفصّل هذا الاشتقاق."
    )
    ui.refs("Robinson (1988) Econometrica 56(4)؛ Chernozhukov et al. (2018) Econometrics Journal 21(1)؛ Newey (1994) Econometrica 62(6).")

# --------------------------------------------------------------------------- 4
with tabs[3]:
    ui.section("هجين Zhang (2003): ARIMA ثم شبكة على البواقي", "The classic sequential hybrid", "amber")
    st.markdown("يفترض Zhang أن السلسلة مجموع جزء خطي ذاتي الارتباط وجزء لاخطي:")
    st.latex(r"y_t=L_t+N_t")
    ui.ladder([
        ("قدّر ARIMA", "Linear stage", "اختر الرتب بمعيار AIC/BIC وقدّر $\\hat L_t$. الجزء الخطي يُلتقط بالكامل إن كانت المواصفة جيدة."),
        ("احسب البواقي", "Residuals", "$e_t=y_t-\\hat L_t$. إن بقيت لاخطية فستكون فيها."),
        ("درّب شبكة على البواقي المتأخرة", "Nonlinear stage", "$e_t=f(e_{t-1},\\dots,e_{t-p})+\\varepsilon_t$، شبكة بطبقة مخفية واحدة في الورقة الأصلية."),
        ("اجمع التنبؤين", "Combine", "$\\hat y_{t}=\\hat L_t+\\hat N_t$ حيث $\\hat N_t=\\hat f(e_{t-1},\\dots,e_{t-p})$."),
    ])
    ui.section("لماذا نجح؟ ومتى يفشل؟", "Why it works, when it fails", "violet")
    c1, c2 = st.columns(2)
    with c1:
        with ui.tinted("emerald"):
            st.markdown("**لماذا ينجح:**\n"
                        "- ARIMA يتقن البنية الخطية بمعلمات قليلة ومستقرة.\n"
                        "- الشبكة لا تُهدر سعتها في تعلم الاتجاه والارتباط الذاتي الخطي؛ تتفرغ للنمط اللاخطي.\n"
                        "- الشبكة وحدها ضعيفة في الاستقراء خارج المدى (الاتجاه)، وARIMA يعالج ذلك.")
    with c2:
        with ui.tinted("rose"):
            st.markdown("**متى يفشل (Taskaya-Temizel & Casey 2005):**\n"
                        "- إذا لم تكن العلاقة **جمعية**؛ قد تكون ضربية أو متداخلة.\n"
                        "- إذا التقط ARIMA جزءًا من اللاخطية خطأً، فالبواقي لا تمثل $N_t$ بدقة.\n"
                        "- إذا كانت البواقي ضوضاء بيضاء فعلًا؛ الشبكة ستتعلم ضوضاء.\n"
                        "- وُجدت حالات يكون فيها الهجين **أسوأ** من كل مكوِّن وحده.")
    ui.section("المتغيرات اللاحقة", "Extensions", "teal")
    ui.cards([
        {"title": "التوليف اللاخطي (Khashei & Bijari 2011)", "color": "teal",
         "body": "بدل الجمع البسيط: $\\hat y_t=f(\\hat L_t,\\ e_{t-1..p},\\ y_{t-1..q})$. الشبكة تقرر بنفسها كيف تجمع، فتسقط فرضية الجمعية."},
        {"title": "ARIMA–LSTM و ARIMA–GRU", "color": "sky",
         "body": "شبكات تكرارية تتعامل مع تبعيات طويلة في البواقي. مفيدة عندما تكون اللاخطية ممتدة زمنيًا."},
        {"title": "ARDL/VAR + تعلم آلي", "color": "violet",
         "body": "الجزء الخطي متعدد المتغيرات (فجوات موزعة، VAR)، والمتعلم يأخذ بواقيه مع متغيرات خارجية."},
        {"title": "تفكيك ثم دمج", "color": "amber",
         "body": "تفكيك موجي أو EMD للسلسلة إلى ترددات، نموذج مناسب لكل تردد، ثم جمع. الدمج هنا عبر <i>مكونات</i> لا بواقي."},
        {"title": "GARCH ثم تعلم آلي", "color": "rose",
         "body": "النموذج القياسي يعطي التقلب الشرطي، والمتعلم يصحح بواقيه المعيارية أو يتنبأ بالتقلب المحقق المتبقي."},
        {"title": "الهجين ثنائي الاتجاه", "color": "emerald",
         "body": "تعلم آلي أولًا لالتقاط اللاخطية، ثم نموذج قياسي على البواقي لاستعادة البنية الخطية القابلة للتفسير."},
    ], cols=3)
    ui.link("app_pages/d11_components.py", "الدمج عبر المكونات بالتفصيل: التفكيك وإعادة التركيب وفخ المرشحات ثنائية الجانب", ":material/waterfall_chart:")
    ui.refs("Zhang (2003) Neurocomputing 50:159–175؛ Taskaya-Temizel & Casey (2005) Neural Networks 18(5–6)؛ Khashei & Bijari (2011) Applied Soft Computing 11(2).")

# --------------------------------------------------------------------------- 5
with tabs[4]:
    ui.section("لا تدمج قبل أن تختبر: هل في البواقي ما يُتعلم؟", "Test residuals before learning from them", "sky")
    st.markdown("كل اختبار يكشف نوعًا مختلفًا من البنية المتبقية. اختيار الاختبار الخطأ قد يخفي فرصة أو يختلق واحدة:")
    tests = [
        ("Ljung–Box", r"Q=T(T+2)\sum_{k=1}^{h}\frac{\hat\rho_k^2}{T-k}\sim\chi^2_h",
         "ارتباط ذاتي **خطي** متبقٍ. إن رُفض: النموذج الخطي نفسه ناقص (أضف فجوات) قبل أي تعلم آلي."),
        ("McLeod–Li", r"Q_{e^2}=T(T+2)\sum_{k=1}^{h}\frac{\hat\rho_k^2(e^2)}{T-k}",
         "ارتباط في **مربعات** البواقي: تجمّع التقلب (ARCH). يوجه إلى نمذجة التباين لا المتوسط."),
        ("BDS", r"W_{m,\varepsilon}=\sqrt T\,\frac{C_{m,\varepsilon}-C_{1,\varepsilon}^{\,m}}{\sigma_{m,\varepsilon}}\xrightarrow{d}N(0,1)",
         "يختبر أن البواقي **مستقلة ومتماثلة التوزيع** ضد أي بنية (خطية، لاخطية، فوضى). قوي لكنه لا يقول ما نوع البنية."),
        ("Teräsvirta (اختبار الشبكة العصبية)", r"e_t=\alpha'z_t+\sum_{i\le j}\delta_{ij}z_iz_j+\sum_{i\le j\le k}\delta_{ijk}z_iz_jz_k+\nu_t",
         "تقريب تايلور لشبكة عصبية؛ اختبار $\\delta=0$ بإحصاءة LM. إن رُفض: توجد لاخطية **في المتوسط** تستطيع شبكة التقاطها. "
         "أنسب اختبار قبل هجين Zhang."),
        ("RESET (Ramsey)", r"y=X\beta+\gamma_1\hat y^2+\gamma_2\hat y^3+u,\quad H_0:\gamma=0",
         "خطأ في الشكل الدالي في النماذج المقطعية والانحدارية."),
        ("Tsay", r"\text{انحدار البواقي على حاصل ضرب الفجوات}",
         "لاخطية من نوع التفاعلات بين الفجوات، مناسب للسلاسل الزمنية."),
    ]
    for name, eq, txt in tests:
        with ui.tinted("slate"):
            st.markdown(f"**{name}**")
            st.latex(eq)
            st.markdown(txt)
    ui.note(
        "**قاعدة القرار العملية:**\n"
        "1. Ljung–Box مرفوض ← أصلح الجزء الخطي أولًا.\n"
        "2. Ljung–Box مقبول، McLeod–Li مرفوض، Teräsvirta مقبول ← البنية في التباين؛ استخدم GARCH أو تعلمًا آليًا على التقلب.\n"
        "3. Teräsvirta أو BDS مرفوض ← لاخطية في المتوسط؛ الهجين المتسلسل مبرَّر.\n"
        "4. كل الاختبارات مقبولة ← البواقي أقرب لضوضاء بيضاء؛ هجين البواقي سيتعلم الضوضاء غالبًا.",
        "شجرة القرار", "sky", "🌳",
    )
    ui.refs("Ljung & Box (1978) Biometrika 65(2)؛ McLeod & Li (1983) JTSA 4(4)؛ Brock, Dechert, Scheinkman & LeBaron (1996) Econometric Reviews 15(3)؛ "
            "Teräsvirta, Lin & Granger (1993) JTSA 14(2)؛ Ramsey (1969) JRSS-B 31(2)؛ Tsay (1986) Biometrika 73(2).")

# --------------------------------------------------------------------------- 6
with tabs[5]:
    ui.section("ليست البواقي للمتوسط فقط", "Residuals carry variance and distribution information", "rose")
    st.markdown("النموذج قد يكون صحيحًا للمتوسط لكن ناقصًا لـ**التباين** أو لـ**شكل التوزيع**. هنا يدخل الدمج من باب آخر:")
    ui.cards([
        {"title": "تعلم آلي على مربعات البواقي", "color": "rose",
         "body": "$\\hat e_t^2=h(z_{t-1})+\\nu_t$ — غابة أو شبكة تتنبأ بالتقلب من متغيرات كثيرة. بديل مرن لـ GARCH أو مكمل له."},
        {"title": "GARCH ثم شبكة", "color": "orange",
         "body": "$\\sigma^2_t$ من GARCH، ثم شبكة تتنبأ بالبواقي المعيارية $z_t=e_t/\\sigma_t$ أو بنسبة التقلب المحقق إلى المتوقع."},
        {"title": "كمّيات البواقي", "color": "violet",
         "body": "انحدار كمّي أو غابة كمّية على البواقي يعطي فترات تنبؤ غير متماثلة دون افتراض الطبيعية."},
        {"title": "HAR ثم LSTM", "color": "teal",
         "body": "نموذج HAR للتقلب المحقق يلتقط الذاكرة الطويلة خطيًا، والشبكة تلتقط اللاخطية المتبقية. (مكتبة harlstmgarch)"},
    ], cols=2)
    st.latex(r"e_t=\sigma_t z_t,\quad \sigma_t^2=\omega+\alpha e_{t-1}^2+\beta\sigma_{t-1}^2,\quad \widehat{\sigma}^{2,\text{hyb}}_t=\hat\sigma_t^2\cdot\exp\big(\hat h(z_{t-1})\big)")
    st.caption("صيغة ضربية تحافظ على موجبية التباين: الشبكة تعدّل تقلب GARCH بنسبة لا بفرق.")

# --------------------------------------------------------------------------- 7
with tabs[6]:
    ui.section("الاتجاه المعاكس: التعلم الآلي أولًا ثم القياس", "ML first, econometrics on its residuals", "teal")
    st.markdown(
        "أحيانًا نبدأ بالتعلم الآلي لالتقاط ما لا نهتم بتفسيره (موسمية معقدة، اتجاه لاخطي، أثر الطقس)، ثم نطبق النموذج القياسي على "
        "الباقي لنفسّر الجزء الذي يهمنا:"
    )
    st.latex(r"\tilde y_t=y_t-\hat f_{\text{ML}}(w_t),\qquad \tilde y_t=x_t'\beta+u_t")
    st.markdown(
        "**مثال:** الطلب على الكهرباء يعتمد بقوة ولاخطيًا على الحرارة والساعة واليوم. غابة تزيل هذه الآثار، ثم ننحدر الباقي على "
        "السعر لتقدير مرونة الطلب. **تنبيه:** يجب أن تكون $w_t$ غير متأثرة بالسعر، وإلا أزلنا جزءًا من الأثر الذي نريد قياسه "
        "(وهذا ما يحميه DML بإزالة أثر $w$ من السعر أيضًا)."
    )
    ui.section("التعزيز التدرجي: تعلم البواقي مرارًا", "Boosting is iterated residual fitting", "violet")
    st.markdown("التعزيز (Friedman 2001) هو في جوهره **دمج متكرر عبر البواقي**:")
    ui.ladder([
        ("ابدأ بتنبؤ بسيط", "Initialize", "$F_0(x)=\\bar y$ أو نموذج قياسي خطي كنقطة بداية."),
        ("احسب البواقي", "Pseudo-residuals", "$r_i^{(m)}=y_i-F_{m-1}(x_i)$ (لخسارة المربعات)."),
        ("درّب متعلمًا ضعيفًا على البواقي", "Weak learner", "شجرة صغيرة $h_m$ تتنبأ بـ $r^{(m)}$."),
        ("حدّث بخطوة صغيرة", "Shrinkage", "$F_m=F_{m-1}+\\nu\\,h_m$ مع $\\nu\\in(0,1]$، وكرر."),
    ])
    st.latex(r"F_M(x)=F_0(x)+\nu\sum_{m=1}^{M}h_m(x)")
    ui.note(
        "إذا بدأنا التعزيز من **نموذج قياسي** $F_0(x)=x'\\hat\\beta$ بدل المتوسط، نحصل على هجين بواقي «متدرج»: الجزء الخطي يبقى "
        "قابلًا للتفسير، وكل شجرة لاحقة تصحح جزءًا صغيرًا من البواقي. وعدد الأشجار $M$ ومعدل التعلم $\\nu$ يتحكمان في مقدار "
        "اللاخطية المسموح بها، ويُختاران خارج العينة.",
        "التعزيز من نقطة بداية قياسية", "violet", "🔁",
    )
    ui.refs("Friedman (2001) Annals of Statistics 29(5)؛ Bühlmann & Hothorn (2007) Statistical Science 22(4)؛ Mincer & Zarnowitz (1969) لانحدارات أخطاء التنبؤ.")

# --------------------------------------------------------------------------- 8
with tabs[7]:
    ui.section("مختبر: هل تحسن الهجين حقيقي أم حفظ للضوضاء؟", "In-sample vs out-of-sample", "amber")
    st.markdown(
        "نولد سلسلة $y_t=0.6y_{t-1}+\\text{لاخطية}(y_{t-1})+\\varepsilon_t$. نقدّر AR(2) بالمربعات الصغرى على أول 70%، ثم نتعلم "
        "البواقي من فجوات $y$ والبواقي بمتعلم مرن، ونقارن داخل العينة وخارجها (تنبؤ خطوة واحدة)."
    )
    c1, c2, c3, c4 = st.columns(4)
    strength = c1.slider("قوة اللاخطية", 0.0, 2.0, 1.0, 0.1, key="res_str")
    noise = c2.slider("انحراف الضوضاء", 0.2, 1.5, 0.5, 0.1, key="res_noise")
    learner = c3.selectbox("المتعلم على البواقي", ["غابة عشوائية", "تعزيز تدرجي", "أقرب الجيران (k=3)", "أقرب الجيران (k=25)"], key="res_learner")
    T = c4.select_slider("طول السلسلة", [200, 400, 800, 1500], value=400, key="res_T")

    @st.cache_data(show_spinner=False)
    def run(strength, noise, learner, T, seed=5):
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        rng = np.random.default_rng(seed)
        y = np.zeros(T + 50)
        for t in range(2, T + 50):
            y[t] = 0.6 * y[t - 1] + strength * np.sin(2.2 * y[t - 1]) * np.exp(-0.15 * y[t - 2] ** 2) + rng.normal(0, noise)
        y = y[50:]
        Z = np.column_stack([np.ones(T - 2), y[1:-1], y[:-2]])
        tgt = y[2:]
        cut = int(0.7 * len(tgt))
        b = np.linalg.lstsq(Z[:cut], tgt[:cut], rcond=None)[0]
        L = Z @ b
        e = tgt - L
        F = np.column_stack([y[1:-1], y[:-2], np.r_[0, e[:-1]], np.r_[0, 0, e[:-2]]])
        mdl = {"غابة عشوائية": RandomForestRegressor(300, min_samples_leaf=1, random_state=0),
               "تعزيز تدرجي": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0),
               "أقرب الجيران (k=3)": KNeighborsRegressor(3),
               "أقرب الجيران (k=25)": KNeighborsRegressor(25)}[learner]
        mdl.fit(F[:cut], e[:cut])
        N = mdl.predict(F)
        rm = lambda a, s: float(np.sqrt(np.mean((tgt[s] - a[s]) ** 2)))  # noqa: E731
        ins, oos = slice(0, cut), slice(cut, None)
        return dict(ar_in=rm(L, ins), hy_in=rm(L + N, ins), ar_out=rm(L, oos), hy_out=rm(L + N, oos),
                    t=np.arange(len(tgt)), y=tgt, L=L, H=L + N, cut=cut, noise=noise)

    r = run(strength, noise, learner, T)
    tab = pd.DataFrame({
        "": ["داخل العينة (70%)", "خارج العينة (30%)"],
        "AR(2)": [r["ar_in"], r["ar_out"]],
        "الهجين": [r["hy_in"], r["hy_out"]],
    })
    tab["تحسن الهجين %"] = 100 * (1 - tab["الهجين"] / tab["AR(2)"])
    st.dataframe(tab.round(3), hide_index=True)
    st.caption(f"مرجع: لا يمكن لأي نموذج أن ينزل خارج العينة تحت انحراف الضوضاء = {r['noise']:.2f}")
    fig = go.Figure()
    s = slice(r["cut"] - 60, r["cut"] + 90)
    fig.add_trace(go.Scatter(x=r["t"][s], y=r["y"][s], name="y", mode="markers", marker=dict(size=5, color="#94a3b8")))
    fig.add_trace(go.Scatter(x=r["t"][s], y=r["L"][s], name="AR(2)", line=dict(color="#7c3aed", width=2)))
    fig.add_trace(go.Scatter(x=r["t"][s], y=r["H"][s], name="الهجين", line=dict(color="#e11d48", width=2)))
    fig.add_vline(x=r["cut"], line_dash="dash", line_color="#0d9488", annotation_text="بداية الاختبار")
    ui.style_fig(fig, "حول نقطة التقسيم: يسار = تدريب، يمين = اختبار", height=430)
    ui.show(fig)
    ui.note(
        "**جرّب هذه السيناريوهات:**\n"
        "1. اللاخطية = 0 ← أي «تحسن» داخل العينة وهمي، وخارجها يسوء الهجين غالبًا: المتعلم حفظ الضوضاء.\n"
        "2. أقرب الجيران k=3 ← تحسن داخلي ضخم وخارجي ضعيف أو سلبي (إفراط في التوفيق). k=25 أهدأ وأصدق.\n"
        "3. لاخطية قوية وسلسلة طويلة ← تحسن حقيقي خارج العينة يقترب من حد الضوضاء.\n\n"
        "**الخلاصة:** لا تُقيَّم هجن البواقي إلا خارج العينة وبترتيب زمني.",
        "ماذا تتعلم من المختبر؟", "amber", "🧪",
    )

# --------------------------------------------------------------------------- 9
with tabs[8]:
    ui.section("الأخطاء الشائعة في الدمج عبر البواقي", "Common mistakes", "rose")
    ui.cards([
        {"title": "تعلم بواقي الفترة نفسها", "color": "rose",
         "body": "استخدام $e_t$ كمدخل للتنبؤ بـ $e_t$، أو حساب البواقي بنموذج قُدّر على كامل العينة بما فيها فترة الاختبار."},
        {"title": "تقييم داخل العينة", "color": "orange",
         "body": "متعلم مرن سيخفض خطأ البواقي داخل العينة دائمًا. المعيار الوحيد: تنبؤ متدحرج خارج العينة."},
        {"title": "نسيان إعادة التقدير", "color": "amber",
         "body": "في التقييم المتدحرج يجب إعادة تقدير الجزء القياسي ثم إعادة حساب البواقي في كل نافذة."},
        {"title": "البواقي الداخلية لـ DML", "color": "violet",
         "body": "حساب بواقي DML من نماذج دُربت على المشاهدة نفسها يحيّز θ. التقسيم المتقاطع ليس اختياريًا."},
        {"title": "افتراض الجمعية دون اختبار", "color": "sky",
         "body": "إن كانت البنية ضربية أو تفاعلية فالتوليف اللاخطي (Khashei–Bijari) أو النموذج الموحد أنسب."},
        {"title": "التفسير الخاطئ", "color": "teal",
         "body": "معاملات الجزء الخطي في هجين التنبؤ ليست آثارًا سببية؛ والجزء المرن لا يُقرأ كعلاقة اقتصادية."},
    ], cols=3)
    ui.section("قائمة تحقق قبل النشر", "Checklist", "emerald")
    st.markdown(
        "- [ ] شخّصتُ البواقي (Ljung–Box، McLeod–Li، Teräsvirta أو BDS) وبررتُ الدمج.\n"
        "- [ ] الجزء القياسي ومتعلم البواقي قُدّرا على بيانات التدريب فقط في كل نافذة.\n"
        "- [ ] مدخلات متعلم البواقي كلها من الماضي ($t-1$ وما قبل).\n"
        "- [ ] التقييم خارج العينة بترتيب زمني، مع اختبار Diebold–Mariano.\n"
        "- [ ] قارنتُ بالمكوِّنين منفردين وبالتوليف البسيط.\n"
        "- [ ] في الاستدلال (DML): تقسيم متقاطع وتكرار التقسيم وتقرير الوسيط.\n"
        "- [ ] فحصتُ استقرار المكسب عبر فترات فرعية."
    )
