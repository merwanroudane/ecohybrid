import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "آلية الدمج الفعلية: كيف تنتقل المعلومات بين النموذجين؟",
    "The integration mechanism — roles, signals, interfaces, the fusion operator and the full training cycle",
    "معرفة «مستوى» الدمج لا تكفي. السؤال الحاسم: <b>ما الذي يخرج من النموذج القياسي؟ أين يدخل إلى نموذج التعلم الآلي؟ كيف يتعلم "
    "النموذجان؟ وما الذي يعود إلى الباحث في النهاية؟</b> هذه الصفحة تشرح الميكانيزم بوصفه سلسلة تحويلات واتصالات محددة يمكن كتابتها "
    "وتدقيقها وإعادة إنتاجها، كما بُني في موسوعة آلية الدمج لـ د. مروان رودان.",
    "violet", "amber",
)

tabs = st.tabs([
    ":material/account_tree: التعريف العملي والأدوار الأربعة",
    ":material/sync_alt: طبقة التوافق",
    ":material/sensors: إشارات النموذج القياسي الست",
    ":material/cable: واجهات الاتصال السبع",
    ":material/functions: عامل الدمج A(·)",
    ":material/alt_route: الميكانيزمات الثلاثة",
    ":material/hub: الميكانيزم داخل النماذج الشهيرة",
    ":material/science: مختبر: قيمة إشارة التوازن",
    ":material/autorenew: دورة التدريب الكاملة",
    ":material/fact_check: الشرح في دقيقة وقائمة الفحص",
])

# --------------------------------------------------------------------------- 1
with tabs[0]:
    ui.section("ميكانيزم الدمج: سلسلة تحويلات لا اسمان متجاوران", "A chain of transformations", "violet")
    st.markdown(
        "«ARDL–XGBoost» ليس وصفًا لنموذج؛ هو اسمان. **التعريف العملي للميكانيزم:** سلسلة تحويلات واتصالات تنقل معلومات اقتصادية من طبقة "
        "إلى طبقة، ويمكن وصف كل حلقة فيها بدقة:"
    )
    st.mermaid_chart(
        """
flowchart LR
    D[Data<br/>Y, X, text, images] --> R[Representation<br/>features, lags, transforms]
    R --> E[Econometric model<br/>ARDL, VAR, VECM, GMM]
    E --> S[Interface signal<br/>fitted, residual, ECT, sigma2, moments]
    S --> M[ML model<br/>learns nonlinearity, nuisance, parameters]
    M --> F[Fusion operator A<br/>sum, weights, gating, joint loss]
    F --> O[Output<br/>forecast, effect, test, simulation]
    O -.feedback / joint training / theory constraint.-> E
"""
    )
    ui.section("الأدوار الأربعة: حدّدها قبل أي سطر كود", "Four roles", "amber")
    roles = pd.DataFrame([
        ("النموذج القياسي", "ما العلاقة الاقتصادية؟ ما الأثر أو التوازن أو الديناميكية؟",
         "معاملات، قيم مقدَّرة، بواقي، حد تصحيح الخطأ، عزوم، تباين شرطي"),
        ("نموذج التعلم الآلي", "هل توجد لاخطية أو تفاعلات أو معلومات عالية الأبعاد؟",
         "متغيرات مولدة، تصحيح بواقي، تنبؤ، أوزان، دالة معلمات، تمثيل كامن"),
        ("طبقة الدمج", "كيف تتحول مخرجات الطرفين إلى نتيجة واحدة؟",
         "مجموع، متوسط موزون، تكديس، بوابة، خسارة مشتركة، عزم متعامد"),
        ("طبقة التحقق", "هل أضاف الدمج قيمة صحيحة؟",
         "خطأ خارج العينة، فترة ثقة، تغطية، متانة، تفسير اقتصادي مستقر"),
    ], columns=["المكوِّن", "السؤال الذي يجيب عنه", "ما يرسله أو ينتجه"])
    st.dataframe(roles, hide_index=True)
    ui.note(
        "**القاعدة:** لا تكتب «استخدمنا ARDL وXGBoost» قبل أن تحدد **ما الذي يقدمه ARDL إلى XGBoost** (أي إشارة؟) و**ما الذي يعيده XGBoost "
        "إلى التحليل** (تصحيح؟ وزن؟ معلمة؟). إذا لم تستطع كتابة هاتين الجملتين، فلا يوجد ميكانيزم بعد.",
        "قبل أن تبدأ", "violet", "🧭",
    )
    st.latex(r"\underbrace{\mathcal D}_{\text{بيانات}}\ \xrightarrow{\ \phi\ }\ \underbrace{X}_{\text{تمثيل}}\ \xrightarrow{\ \text{Econ}\ }\ \underbrace{s_t}_{\text{إشارة}}\ \xrightarrow{\ f_{\text{ML}}\ }\ \underbrace{\hat r_t}_{\text{مخرَج ML}}\ \xrightarrow{\ A\ }\ \underbrace{\hat y^{H}_{t+h},\ \hat\theta,\ \text{test}}_{\text{قرار الباحث}}")

# --------------------------------------------------------------------------- 2
with tabs[1]:
    ui.section("أول اتصال حقيقي: جعل المدخلات متوافقة", "The compatibility layer", "sky")
    st.markdown("قبل أن يتحدث النموذجان، يجب أن يعيشا في **الزمن نفسه** و**مجموعة المعلومات نفسها**. مجموعة المعلومات عند لحظة التنبؤ $t$:")
    st.latex(r"\mathcal I_t=\big\{\,Y_s,\ X_s\ :\ s\le t\,\big\}\qquad\Longrightarrow\qquad \text{كل مدخل لـ } f_{\text{ML}} \text{ يجب أن يكون دالة في } \mathcal I_t \text{ فقط}")
    st.markdown("إذا دخلت معلومة من $s>t$ إلى $\\mathcal I_t$ فقد صار الدمج **غير صالح** مهما كان النموذج متطورًا. متطلبات طبقة التوافق:")
    ui.ladder([
        ("فهرس موحد", "Index alignment", "توحيد الفهرس الزمني، أو معرف الوحدة والزمن في بيانات البانل، قبل أي دمج للجداول."),
        ("الأفق أولًا", "Horizon first", "تحديد أفق التنبؤ $h$ **قبل** بناء الإبطاءات؛ مدخلات تنبؤ $t+h$ تتوقف عند $t$."),
        ("التوفر في الوقت الحقيقي", "Real-time availability", "لا يُستخدم متغير لم يكن منشورًا في لحظة التنبؤ (تأخر النشر، المراجعات)."),
        ("التحويلات نفسها", "Same transforms", "تطبيق ما يفرضه النموذج الاقتصادي (لوغاريتم، فروق) بالطريقة نفسها على الطرفين."),
        ("ماذا يرى ML؟", "Declare what ML sees", "المتغيرات الأصلية؟ الفروق؟ الإبطاءات؟ مكونات ECM؟ يُصرَّح بذلك في الورقة."),
        ("خط معالجة مخزَّن", "Stored pipeline", "كل معالجة (تعيير، تعويض، تفكيك) داخل pipeline يُلائم على التدريب ويُطبَّق كما هو خارج العينة."),
    ])
    ui.refs("Croushore & Stark (2001) J. Econometrics 105(1)؛ Kaufman et al. (2012) ACM TKDD «Leakage in data mining».")

# --------------------------------------------------------------------------- 3
with tabs[2]:
    ui.section("ما الذي يمكن أن يرسله النموذج القياسي؟", "Six interface signals", "teal")
    st.markdown("كل ميكانيزم دمج يبدأ بإشارة **مصرَّح بها مسبقًا** يستخرجها النموذج القياسي. ست إشارات تغطي تقريبًا كل الأدبيات:")
    sig = [
        ("① إشارة القيمة المقدَّرة", "Fitted signal", r"\hat Y^{E}_{t}",
         "ما يفسره النموذج. تُستخدم في توليف التنبؤات والتكديس، أو كمدخل لنموذج ML يصحح التنبؤ الاقتصادي."),
        ("② إشارة البواقي", "Residual signal", r"e_t=Y_t-\hat Y^{E}_t",
         "الجزء غير المفسر. أساس تصحيح البواقي (Zhang 2003). يجب أن يتسق تعريف الباقي مع الأفق: باقي خطوة واحدة ليس باقي h خطوات."),
        ("③ إشارة التوازن", "Equilibrium signal", r"ect_{t-1}=\hat\beta'Y_{t-1}",
         "حد تصحيح الخطأ من VECM أو ARDL. يحمل معلومة «بُعد الاقتصاد عن توازنه طويل الأجل» التي لا يكتشفها ML من الفروق وحدها."),
        ("④ إشارة التقلب", "Volatility signal", r"\hat\sigma_t^2,\quad z_t=e_t/\hat\sigma_t",
         "التباين الشرطي أو البواقي المعيارية من GARCH. تنقل الدمج من المتوسط إلى التوزيع (GARCH–LSTM)."),
        ("⑤ إشارة العزوم", "Moment signal", r"\psi(W;\theta,\eta),\qquad g(W;\theta)",
         "بواقي شروط العزوم أو الدرجات في GMM وDML وDeep IV. المعلومة المنتقلة هي «مقدار مخالفة الشرط»."),
        ("⑥ الإشارة البنيوية", "Structural signal", r"\theta^{\text{econ}},\ \ \text{sign}(\partial f/\partial x),\ \ R(f)=0",
         "معلمات أو قيود يريد الباحث الحفاظ على معناها: إشارة أثر، رتابة، معادلة توازن، تُمرَّر كقيد أو كبيانات تدريب أولية."),
    ]
    c1, c2 = st.columns(2)
    for i, (t, en, eq, body) in enumerate(sig):
        with (c1 if i % 2 == 0 else c2):
            with ui.tinted(["teal", "amber", "violet", "rose", "indigo", "emerald"][i]):
                st.markdown(f"**{t}** · <span class='ltr' style='color:#64748b'>{en}</span>", unsafe_allow_html=True)
                st.latex(eq)
                st.markdown(body)
    ui.note(
        "**لماذا التصنيف بالإشارة مفيد؟** لأنه يحدد ما يجب حمايته: إشارة البواقي تتطلب منع تعلم ضوضاء الاختبار؛ إشارة التوازن تتطلب تقدير "
        "متجه التكامل على التدريب فقط؛ إشارة العزوم تتطلب التعامد والتقسيم المتقاطع؛ والإشارة البنيوية تتطلب أن يكون القيد صحيحًا.",
        "من الإشارة إلى الحماية", "teal", "🛡️",
    )

# --------------------------------------------------------------------------- 4
with tabs[3]:
    ui.section("الواجهة: نوع المعلومة واتجاهها ووقت استخدامها", "Seven communication interfaces", "indigo")
    iface = pd.DataFrame([
        ("الدمج بالأعمدة", "Concatenation", "متغيرات اقتصادية + متغيرات ML", "دمج في متجه مدخلات واحد", "إبطاءات + تضمينات نصية + متغيرات كلية"),
        ("التصحيح الجمعي", "Additive correction", "تنبؤ اقتصادي + تصحيح ML", "جمع المخرجات", "ARDL + XGBoost على البواقي"),
        ("التوليف الموزون", "Weighted combination", "تنبؤان أو أكثر", "أوزان ثابتة أو متغيرة", "توليف Bates–Granger"),
        ("البوابة", "Gating", "إشارة اختيار النموذج", "شبكة تتعلم متى تثق بكل نموذج", "تنبؤ يعتمد على النظام"),
        ("تحويل المعلمات", "Parameter mapping", "حالة اقتصادية ← معامل", "ML يولّد βₜ أو θₜ", "معاملات متغيرة زمنيًا"),
        ("اقتران الدرجة", "Score coupling", "بواقي أو عزوم", "تعلم مشروط بالتعامد أو العزم", "DML، DeepGMM"),
        ("اقتران الخسارة", "Loss coupling", "خطأ التنبؤ + مخالفة النظرية", "خسارة مشتركة واحدة", "شبكات موجهة بالنظرية، SKINN"),
    ], columns=["الواجهة", "English", "ما الذي ينتقل؟", "العملية", "مثال"])
    st.dataframe(iface, hide_index=True)
    st.markdown("#### الصيغة الرياضية لكل واجهة")
    eqs = [
        ("الدمج بالأعمدة", r"Z_t=\big[\,x_t,\ \ y_{t-1:t-p},\ \ h_{\text{NLP}}(D_t)\,\big],\qquad \hat y=f_{\text{ML}}(Z_t)"),
        ("التصحيح الجمعي", r"\hat y^{H}=\hat y^{E}+\hat e^{\text{ML}}"),
        ("التوليف الموزون", r"\hat y^{H}=w_t\,\hat y^{E}+(1-w_t)\,\hat y^{\text{ML}}"),
        ("البوابة", r"\hat y^{H}=g(s_t)\,\hat y^{E}+\big(1-g(s_t)\big)\hat y^{\text{ML}},\quad g=\sigma(v's_t)"),
        ("تحويل المعلمات", r"y_t=x_t'\beta_t+u_t,\qquad \beta_t=\text{NN}(s_t)"),
        ("اقتران الدرجة", r"\tfrac1n\textstyle\sum_i\psi\big(W_i;\hat\theta,\hat\eta^{(-k(i))}\big)=0"),
        ("اقتران الخسارة", r"\min_{\theta,W}\ L_{\text{pred}}(Y,f_{\theta,W}(X))+\lambda\,L_{\text{econ}}(\theta,W)"),
    ]
    for name, eq in eqs:
        c1, c2 = st.columns([1, 3])
        c1.markdown(f"**{name}**")
        with c2:
            st.latex(eq)

# --------------------------------------------------------------------------- 5
with tabs[4]:
    ui.section("عامل الدمج: الدالة التي تحول مخرجين إلى نتيجة", "The fusion operator", "orange")
    st.latex(r"\hat y^{H}_{t+h}=A\Big(\hat y^{E}_{t+h},\ f_{\text{ML}}(Z_t)\Big)")
    st.markdown("كل الهجن تختلف في شكل $A$. وهذا الاختيار ليس تقنيًا فقط؛ إنه يحدد ما يمكن تفسيره وما يجب تقديره وكيف يُقيَّم:")
    ops = pd.DataFrame([
        ("المجموع", "A(a,b)=a+b", "لا شيء", "الهجين المتسلسل (Zhang)", "يفترض الجمعية"),
        ("المتوسط الموزون", "A(a,b)=wa+(1−w)b", "w", "توليف Bates–Granger", "تكلفة تقدير الأوزان (لغز التوليف)"),
        ("التكديس", "A(a,b)=g(a,b)", "متعلم فوقي g", "Stacking / Super learner", "يحتاج تنبؤات خارج الطية سابقة زمنيًا"),
        ("البوابة", "A=g(s)a+(1−g(s))b", "شبكة البوابة", "مزيج الخبراء", "عدم استقرار مع عينات قصيرة"),
        ("الخسارة المشتركة", "A ضمني في التدريب", "θ و W معًا", "الهجن المندمجة", "اختيار λ وقابلية التعرّف"),
        ("الحل المتعامد", "حل ∑ψ=0", "θ فقط", "DML", "شروط المعدلات والتعرّف"),
        ("اللعبة min–max", "توازن g و f", "g و f", "Adversarial GMM", "عدم استقرار التدريب"),
    ], columns=["العامل", "الصيغة", "ما الذي يُقدَّر في طبقة الدمج", "يظهر في", "الخطر"])
    st.dataframe(ops, hide_index=True)
    ui.note(
        "**كلما زاد ما يُقدَّر داخل $A$ زادت المرونة وزاد خطر الإفراط في التوفيق.** المجموع لا يقدّر شيئًا؛ البوابة تقدّر شبكة كاملة. "
        "لذلك قارن دائمًا العامل المعقد بالعامل البسيط (المجموع أو المتوسط المتساوي) خارج العينة.",
        "قاعدة التعقيد", "orange", "⚖️",
    )

# --------------------------------------------------------------------------- 6
with tabs[5]:
    ui.section("ثلاثة ميكانيزمات أساسية", "Sequential, parallel, end-to-end", "teal")
    c1, c2, c3 = st.columns(3)
    with c1:
        with ui.tinted("amber"):
            st.markdown("#### ① التسلسلي")
            st.caption("Econometric first, then ML")
            st.markdown(
                "1. قدّر النموذج القياسي على **التدريب فقط**.\n"
                "2. استخرج الإشارة المصرَّح بها (قيمة مقدرة، باقٍ، ECT، تقلب).\n"
                "3. ابنِ مدخلات ML من $\\mathcal I_t$ فقط.\n"
                "4. درّب ML على الإشارة أو الهدف المحدد.\n"
                "5. طبّق $A$.\n"
                "6. قيّم خارج العينة وقارن بكل نموذج منفرد."
            )
    with c2:
        with ui.tinted("sky"):
            st.markdown("#### ② المتوازي")
            st.caption("Two independent models, then fusion")
            st.markdown(
                "النموذجان يريان البيانات نفسها ولا يتعلمان من بعضهما أثناء التقدير. قيمة الدمج تأتي من **تنويع الأخطاء** أو من "
                "اختلاف المعلومات التي يستغلها كل نموذج.\n\n"
                "**البنية المتوازية–المتسلسلة (Parallel–Series):** تفكيك أو معالجة أولًا، ثم عدة نماذج على المكونات، ثم دمج نهائي "
                "(Hajirahimi & Khashei 2019)."
            )
    with c3:
        with ui.tinted("violet"):
            st.markdown("#### ③ التكاملي")
            st.caption("Joint, end-to-end training")
            st.markdown(
                "مخرجات النموذج القياسي ليست ملفًا وسيطًا ثابتًا. خسارة واحدة تُحدِّث الجزأين معًا:\n"
                "1. احسب التنبؤ أو البواقي.\n2. احسب خطأ البيانات.\n3. احسب مخالفة القيد أو العزم.\n"
                "4. اجمع الخسارتين.\n5. احسب التدرج وحدّث $\\theta$ و$W$.\n6. كرر حتى التوقف والمتانة."
            )
    st.latex(r"L(\theta,W)=L_{\text{prediction}}\big(Y,f_{\theta,W}(X)\big)+\lambda\,L_{\text{economic}}(\theta,W)")
    st.markdown(
        "في التكاملي يصبح السؤال: **هل القيد الاقتصادي قابل للتعلم؟ وهل $\\lambda$ مناسب؟** قيمة كبيرة تمنع ملاءمة البيانات، وقيمة صغيرة "
        "تجعل القيد شكليًا. (صفحة «الدمج في دالة الهدف» فيها مختبر هذه المقايضة.)"
    )
    ui.refs("Zhang (2003)؛ Hajirahimi & Khashei (2019) Engineering Applications of Artificial Intelligence 86 «Hybrid structures in time series modeling and forecasting: a review»؛ Zhao et al. (2024).")

# --------------------------------------------------------------------------- 7
with tabs[6]:
    ui.section("الميكانيزم داخل خمس عائلات شهيرة", "Mechanism inside well-known hybrids", "indigo")
    st.markdown("#### ARDL–XGBoost: إشارة البواقي + تصحيح جمعي")
    st.latex(r"Y_t=\alpha+\textstyle\sum_i\phi_iY_{t-i}+\sum_j\beta_jX_{t-j}+\varepsilon_t,\qquad e_t=f_{\text{XGB}}(e_{t-1},X_t,\text{lags})+u_t,\qquad \hat Y^{H}_{t+h}=\hat Y^{\text{ARDL}}_{t+h}+\hat e^{\text{XGB}}_{t+h}")
    st.markdown("إذا لم يتحسن الأداء في تقييم متدحرج أو بإحصاءة Diebold–Mariano، فالبواقي لم تحمل معلومة قابلة للاستغلال أو أن التصحيح أفرط في التوفيق.")
    st.markdown("#### VECM ثم شبكة: إشارة التوازن تدخل المدخلات")
    st.latex(r"\Delta Y_t=\Pi Y_{t-1}+\textstyle\sum_i\Gamma_i\Delta Y_{t-i}+u_t,\qquad \Pi=\alpha\beta'")
    st.markdown(
        "يمكن تمرير **حد تصحيح الخطأ** $\\beta'Y_{t-1}$، و**سرعة التصحيح** $\\alpha$، والفروق المتأخرة، أو بواقي VECM وتنبؤاته، إلى LSTM أو XGBoost. "
        "بهذا **لا يُطلب من ML أن يكتشف التكامل المشترك بنفسه**؛ يستقبل إشارة التوازن جاهزة ويحاول تحسين الجزء اللاخطي، مثل تصحيح لا يحدث إلا "
        "خارج نطاق معين. ويمكن أيضًا جعل مخالفة العلاقة طويلة الأجل حدًا في الخسارة. المختبر في التبويب التالي يقيس قيمة هذه الإشارة."
    )
    st.markdown("#### DML: اقتران الدرجة")
    st.latex(r"X\xrightarrow{\text{ML}}\hat\ell(X)\Rightarrow\tilde Y=Y-\hat\ell,\qquad X\xrightarrow{\text{ML}}\hat m(X)\Rightarrow\tilde D=D-\hat m,\qquad \hat\theta=\frac{\sum\tilde D\tilde Y}{\sum\tilde D^2}")
    st.markdown("ML لا يدخل لأننا نريد قراءة معاملاته؛ يدخل لأنه **يزيل الجزء المعقد**، ثم يحتفظ القياس الاقتصادي بالمعلمة السببية والخطأ المعياري من الدرجة.")
    st.markdown("#### DeepGMM والعزوم التنافسية: من نظرية عزوم إلى لعبة")
    st.latex(r"\min_g\max_f\ \mathbb E\big[(Y-g(X))f(Z)\big]-\tfrac14\mathbb E\big[f(Z)^2\big]")
    st.markdown("النموذج $g$ يحاول تحقيق شرط العزم، والخصم $f$ يبحث عن أكبر انتهاك. يستمر التدريب حتى يقترب النظام من توازن اللعبة.")
    st.markdown("#### القيود النظرية: ثلاث طرق للإدخال")
    st.dataframe(pd.DataFrame([
        ("العقوبة", "Penalty", "مخالفة النظرية حد في الخسارة", "معاقبة ميل طلب موجب أو بواقي أويلر", "القيد أثناء التدريب، لين"),
        ("المعمارية", "Architecture", "طبقات لا تسمح بسلوك معين", "شبكة رتيبة أو جمعية بالكتل", "القيد في شكل النموذج، صلب"),
        ("التدريب الأولي", "Structural pretraining", "تدريب على بيانات من نموذج بنيوي ثم ضبط بالبيانات الحقيقية", "نموذج بنيوي ← بيانات اصطناعية ← بيانات فعلية", "النظرية كمعرفة سابقة قابلة للتصحيح"),
    ], columns=["الطريقة", "English", "كيف تعمل؟", "مثال", "طبيعة القيد"]), hide_index=True)
    ui.refs("Chernozhukov et al. (2018)؛ Dikkala, Lewis, Mackey & Syrgkanis (2020)؛ Bennett, Kallus & Schnabel (2019) NeurIPS DeepGMM؛ Buckmann & Potjagailo (2025)؛ Fernández-Villaverde (2023) Handbook chapter on deep learning for economic models.")

# --------------------------------------------------------------------------- 8
with tabs[7]:
    ui.section("مختبر: ماذا تضيف إشارة التوازن إلى التعلم الآلي؟", "The equilibrium signal, tested", "violet")
    st.markdown(
        "نحاكي زوجًا متكاملًا تكاملًا مشتركًا: $x_t$ سير عشوائي و$z_t=y_t-\\beta x_t$ مستقر. **التصحيح نحو التوازن لاخطي:** ضعيف داخل نطاق "
        "$|z|<c$ وقوي خارجه (تكاليف تعديل). نتنبأ بـ $\\Delta y_t$ خطوة واحدة بأربع طرق، مع تقدير $\\beta$ وكل النماذج على أول 70% فقط:"
    )
    c1, c2, c3, c4 = st.columns(4)
    band = c1.slider("عرض نطاق عدم التصحيح c", 0.0, 3.0, 2.0, 0.25, key="mx_c")
    strong = c2.slider("سرعة التصحيح خارج النطاق", 0.05, 0.8, 0.6, 0.05, key="mx_a")
    noise = c3.slider("الضوضاء", 0.2, 1.5, 0.3, 0.1, key="mx_n")
    T = c4.select_slider("طول السلسلة", [400, 800, 1600], value=800, key="mx_T")

    @st.cache_data(show_spinner="تقدير النماذج الأربعة...")
    def eq_lab(band, strong, noise, T, seed=4):
        from sklearn.ensemble import GradientBoostingRegressor
        rng = np.random.default_rng(seed)
        n = T + 100
        x, y = np.zeros(n), np.zeros(n)
        for t in range(1, n):
            x[t] = x[t - 1] + rng.normal(0, 1)
            z = y[t - 1] - 1.0 * x[t - 1]
            adj = -0.03 * z if abs(z) < band else -strong * z
            y[t] = y[t - 1] + adj + 0.3 * (x[t - 1] - x[t - 2] if t > 1 else 0) + rng.normal(0, noise)
        x, y = x[100:], y[100:]
        dy, dx = np.diff(y), np.diff(x)
        # rows t = 2..: target dy[t], info up to t-1
        tgt = dy[2:]
        dy1, dy2, dx1 = dy[1:-1], dy[:-2], dx[1:-1]
        ylev, xlev = y[2:-1], x[2:-1]          # levels at t-1 for the ECT
        m = len(tgt)
        cut = int(0.7 * m)
        # econometric stage on training only: Engle-Granger beta, then linear ECM
        b = np.polyfit(xlev[:cut], ylev[:cut], 1)
        ect = ylev - (b[0] * xlev + b[1])                       # equilibrium signal
        Zlin = np.column_stack([np.ones(m), ect, dy1, dx1])
        coef = np.linalg.lstsq(Zlin[:cut], tgt[:cut], rcond=None)[0]
        ecm = Zlin @ coef
        gbm = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)  # noqa: E731
        F0 = np.column_stack([dy1, dy2, dx1])                   # lags only: no equilibrium information
        F1 = np.column_stack([ect, dy1, dy2, dx1])              # + equilibrium signal from the econometric stage
        ml0 = gbm().fit(F0[:cut], tgt[:cut]).predict(F0)
        ml1 = gbm().fit(F1[:cut], tgt[:cut]).predict(F1)
        res = gbm().fit(F1[:cut], (tgt - ecm)[:cut]).predict(F1)
        hyb = ecm + res
        oos = slice(cut, None)
        rm = lambda p: float(np.sqrt(np.mean((tgt[oos] - p[oos]) ** 2)))  # noqa: E731
        grid = np.linspace(np.percentile(ect, 1), np.percentile(ect, 99), 120)
        med = np.median(F1[:cut], axis=0)
        G = np.tile(med, (len(grid), 1)); G[:, 0] = grid
        Gl = np.column_stack([np.ones(len(grid)), grid, np.full(len(grid), med[1]), np.full(len(grid), med[3])])
        model1 = gbm().fit(F1[:cut], tgt[:cut])
        return dict(rmse={"ECM خطي (القياسي وحده)": rm(ecm), "ML على الإبطاءات فقط (بلا إشارة)": rm(ml0),
                          "ML + إشارة التوازن ect": rm(ml1), "هجين: ECM + ML على البواقي": rm(hyb)},
                    grid=grid, lin=Gl @ coef, ml=model1.predict(G), ect=ect[:cut], tgt=tgt[:cut], noise=noise, beta=b[0])

    r = eq_lab(band, strong, noise, T)
    tab = pd.DataFrame(list(r["rmse"].items()), columns=["الطريقة", "RMSE خارج العينة"])
    base = r["rmse"]["ECM خطي (القياسي وحده)"]
    tab["مقارنة بـ ECM %"] = 100 * (tab["RMSE خارج العينة"] / base - 1)
    st.dataframe(tab.round(4), hide_index=True)
    st.caption(f"β المقدَّر على التدريب = {r['beta']:.3f} (الحقيقي 1.0). حد الضوضاء الأدنى ≈ {r['noise']:.2f}.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r["ect"], y=r["tgt"], mode="markers", name="Δy مقابل ect (تدريب)", marker=dict(size=3, color="#cbd5e1")))
    fig.add_trace(go.Scatter(x=r["grid"], y=r["lin"], name="دالة التصحيح في ECM الخطي", line=dict(color="#7c3aed", width=3)))
    fig.add_trace(go.Scatter(x=r["grid"], y=r["ml"], name="دالة التصحيح التي تعلمها ML من الإشارة", line=dict(color="#e11d48", width=3)))
    fig.update_xaxes(title="إشارة التوازن ectₜ₋₁ = yₜ₋₁ − β̂xₜ₋₁")
    fig.update_yaxes(title="Δyₜ")
    ui.style_fig(fig, "التصحيح نحو التوازن: خطي مفروض مقابل لاخطي متعلَّم", height=440)
    ui.show(fig)
    ui.note(
        "**القراءة:**\n"
        "- **ML بلا إشارة** يرى الفروق فقط، فلا يعرف كم ابتعد الاقتصاد عن توازنه؛ يخسر المعلومة الأهم ويأتي أسوأ حتى من ECM الخطي.\n"
        "- **ECM الخطي** يعرف التوازن لكنه يفرض تصحيحًا متناسبًا في كل مكان.\n"
        "- **ML + إشارة التوازن** يتعلم دالة التصحيح الحقيقية (مسطحة داخل النطاق، حادة خارجه) — هذا هو معنى «إشارة يرسلها النموذج القياسي».\n"
        "- اجعل عرض النطاق c = 0: يصبح التصحيح خطيًا فيتفوق ECM، والمرونة الزائدة تضيف تباينًا فقط. وارفع الضوضاء: يتقلص المكسب لأن "
        "كل النماذج تقترب من حد الضوضاء.",
        "ماذا يظهر المختبر؟", "violet", "🧪",
    )

# --------------------------------------------------------------------------- 9
with tabs[8]:
    ui.section("الدمج دورة تتكرر مع الزمن لا سطر كود", "The full training cycle", "teal")
    st.code(
        """for each forecast origin t:
    1. build the information set I_t only from observations available at t
    2. fit the econometric baseline on the training history
    3. extract the declared interface signal: fitted / residuals / ECT / volatility / moments
    4. create ML features without any future information
    5. fit or update the ML model with time-aware validation
    6. apply the fusion operator: sum / weighted average / gating / joint loss
    7. produce the point forecast and its uncertainty interval
    8. store errors, diagnostics and model state
    9. move the forecast origin forward
   10. compare the hybrid with each baseline out of sample""",
        language="text",
    )
    st.markdown("مخطط زمني لثلاث نقاط أصل متتالية في تقييم بنافذة متوسعة (اضغط ▶):")
    T0, H = 40, 8
    frames = []
    for k in range(10):
        t = T0 + 4 * k
        frames.append((t, [go.Bar(x=[t], y=["تدريب"], base=0, orientation="h", marker_color="#7c3aed", name="تدريب", width=0.5),
                           go.Bar(x=[1], y=["تنبؤ"], base=t, orientation="h", marker_color="#e11d48", name="أصل التنبؤ t+1", width=0.5)]))
    fig = go.Figure(data=frames[0][1], frames=[go.Frame(data=f, name=str(i), layout=go.Layout(title_text=f"أصل التنبؤ t = {t}"))
                                               for i, (t, f) in enumerate(frames)])
    fig.update_xaxes(range=[0, T0 + 4 * 10 + 4], title="الزمن")
    fig.update_layout(barmode="overlay", showlegend=True)
    ui.style_fig(fig, f"أصل التنبؤ t = {T0}", height=420)
    ui.add_play(fig, [str(i) for i in range(10)], duration=700, prefix="الأصل: ")
    ui.show(fig)
    st.caption("في كل خطوة: يُعاد تقدير النموذج القياسي، تُعاد استخراج الإشارة، يُعاد تدريب ML، ثم يُتنبأ بالفترة التالية فقط.")

# --------------------------------------------------------------------------- 10
with tabs[9]:
    ui.section("كيف تشرح الميكانيزم للباحثين في دقيقة واحدة؟", "The one-minute explanation", "violet")
    ui.note(
        "«نبدأ بنموذج قياسي يلتقط الجزء الاقتصادي الذي نريد تفسيره، مثل العلاقة طويلة الأجل أو الارتباط الذاتي أو شرط العزم. نستخرج منه "
        "**إشارة محددة**، مثل التنبؤ أو البواقي أو حد تصحيح الخطأ. يدخل التعلم الآلي على هذه الإشارة ليتعلم اللاخطية أو المعلومات العالية "
        "الأبعاد التي لم يلتقطها النموذج. ثم نعيد دمج المخرجات بالجمع أو بالأوزان أو بخسارة مشتركة، ونختبر هل أضاف الدمج قيمة خارج "
        "العينة أو حسّن الاستدلال مع بقاء القيود الاقتصادية صحيحة.»",
        "د. مروان رودان — موسوعة آلية الدمج", "violet", "🎙️",
    )
    ui.section("متى لا توجد آلية دمج مبررة؟", "When integration is not justified", "rose")
    st.markdown(
        "- لا يوجد سؤال اقتصادي واضح سوى الرغبة في استخدام الذكاء الاصطناعي.\n"
        "- العينة صغيرة لا تسمح بتقييم خارج العينة.\n"
        "- النموذج الثاني لا يضيف معلومة جديدة.\n"
        "- البواقي ضوضاء بيضاء والتنبؤات لا تتحسن.\n"
        "- الدمج يُفقد التعريف السببي أو علاقة التوازن طويل الأجل.\n"
        "- لا يمكن شرح المعلومة التي انتقلت بين النموذجين.\n"
        "- لا يمكن إعادة إنتاج خطوات المعالجة أو الضبط."
    )
    ui.section("قائمة فحص الميكانيزم: ثمانية أسئلة يجب أن تجيب عنها الورقة", "Mechanism checklist", "emerald")
    st.dataframe(pd.DataFrame([
        ("ما وظيفة النموذج القياسي؟", "تفسير، توازن، تقدير أثر، أو خط أساس"),
        ("ما إشارة الاتصال؟", "متغيرات، قيم مقدرة، بواقي، ECT، عزوم، أو معلمات"),
        ("ما اتجاه الاتصال؟", "باتجاه واحد، تسلسلي، متوازٍ، تغذية راجعة، أو مشترك"),
        ("ما عامل الدمج؟", "جمعي، موزون، بوابة، تكديس، min–max، أو خسارة مقيدة"),
        ("ماذا يتعلم ML؟", "تنبؤ، لاخطية، دالة مزعجة، تصحيح، أو دالة معلمات"),
        ("ما الهدف النهائي؟", "تنبؤ، أثر سببي، اختبار، محاكاة، أو حل نموذج"),
        ("كيف تم التحقق؟", "تقييم متدحرج، تقسيم متقاطع، bootstrap، تغطية، ومتانة"),
        ("ماذا يحدث إذا أزلنا ML؟", "عرض أداء خط الأساس وقيمة الإضافة بوضوح"),
    ], columns=["السؤال", "الإجابة المطلوبة في البحث"]), hide_index=True)
    ui.refs("موسوعة آلية دمج القياس الاقتصادي والذكاء الاصطناعي — د. مروان رودان (2026)؛ Hajirahimi & Khashei (2019)؛ Chernozhukov et al. (2018).")
