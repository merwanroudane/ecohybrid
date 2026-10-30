import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "فلسفة وآليات الدمج (4): DML والسببية والنظرية وشروط اللحظات",
    "Part 4: DML from scratch, cross-fitting step by step, causal integration, Deep IV / Deep GMM / causal forests, theory-guided learning, moment and accounting constraints (chapters 23–28)",
    "هنا يتغير دور الذكاء الاصطناعي: لا يتنبأ بالمتغير التابع لذاته، بل يقدّر مكونات مساعدة تسمح بتركيز الاستدلال على معلمة اقتصادية، أو يتعلم داخل "
    "حدود تفرضها النظرية وشروط اللحظات والهويات المحاسبية. محاضرة <b>د. مروان رودان</b>، الفصول 23–28.",
    "indigo", "emerald",
)

tabs = st.tabs([
    ":material/hub: 23 DML من الصفر",
    ":material/call_split: 24 Cross-Fitting خطوة بخطوة",
    ":material/gavel: 25 من التنبؤ إلى أثر التدخل",
    ":material/account_tree: 26 Deep IV · Deep GMM · Causal Forests",
    ":material/school: 27 الدمج الموجَّه بالنظرية",
    ":material/functions: 28 شروط اللحظات والقيود المحاسبية",
])

# --------------------------------------------------------------------------- 23
with tabs[0]:
    ui.section("23 — DML من الصفر: الذكاء الاصطناعي يخدم تقدير معلمة وليس التنبؤ فقط", "DML from scratch", "indigo")
    st.markdown("لنأخذ برنامج تدريب مهني. نريد تقدير أثره $\\theta$ على الأجر $Y$، مع وجود متغيرات ضابطة $X$ مثل التعليم والخبرة، ومتغير معالجة $D$. النموذج الجزئي الخطي:")
    st.latex(r"Y_i=\theta_0 D_i+g_0(X_i)+\varepsilon_i, \quad E(\varepsilon_i\mid D_i,X_i)=0.")
    st.markdown("نُعرّف دالتين مساعدتين:")
    st.latex(r"\ell_0(X)=E(Y\mid X),\qquad m_0(X)=E(D\mid X).")
    st.markdown("يمكن استخدام Random Forest أو Lasso أو شبكة عصبية لتقديرهما. ثم نحسب:")
    st.latex(r"\widetilde Y_i=Y_i-\widehat\ell(X_i),\qquad \widetilde D_i=D_i-\widehat m(X_i),\qquad \boxed{\widehat\theta=\frac{\sum_i\widetilde D_i\widetilde Y_i}{\sum_i\widetilde D_i^2}.}")
    st.markdown("#### لماذا تطرح المتوسطات الشرطية؟")
    st.markdown("بأخذ التوقع الشرطي للنموذج الأصلي نحصل على:")
    st.latex(r"\ell_0(X)=\theta_0m_0(X)+g_0(X).")
    st.markdown("نطرح هذه المعادلة من النموذج، فيختفي المكون المعقد $g_0(X)$:")
    st.latex(r"Y-\ell_0(X)=\theta_0[D-m_0(X)]+\varepsilon.")
    st.markdown("إذن وظيفة AI هنا هي **تقدير مكونات مساعدة** تسمح لنا بتركيز الاستدلال على المعلمة المستهدفة. هذا المنطق مرتبط بأعمال Robinson (1988) وبإطار DML الذي طوّره Chernozhukov وزملاؤه (2018) [م4، م5].")
    st.markdown("#### مثال عددي بأربع مشاهدات")
    st.markdown("للتوضيح فقط، نفترض $Y=1+4X+2D$:")
    d = pd.DataFrame({"الفرد": [1, 2, 3, 4], "X": [0, 0, 1, 1], "D": [0, 1, 1, 2]})
    d["Y"] = 1 + 4 * d["X"] + 2 * d["D"]
    d["D̃"] = d["D"] - d.groupby("X")["D"].transform("mean")
    d["Ỹ"] = d["Y"] - d.groupby("X")["Y"].transform("mean")
    st.dataframe(d, hide_index=True)
    theta = float((d["D̃"] * d["Ỹ"]).sum() / (d["D̃"] ** 2).sum())
    st.markdown("عند $X=0$ يكون متوسط $D=0.5$ ومتوسط $Y=2$، وعند $X=1$ يكون متوسط $D=1.5$ ومتوسط $Y=8$. من ثم:")
    st.latex(r"\widehat\theta=\frac{(-0.5)(-1)+(0.5)(1)+(-0.5)(-1)+(0.5)(1)}{4\times 0.5^2}=2.")
    st.success(f"الحساب الحي من الجدول: θ̂ = {theta:.2f} — استعدنا المعلمة الحقيقية في المثال.", icon=":material/check_circle:")
    st.caption("هذه ليست دراسة استدلالية بأربع مشاهدات؛ إنما مثال تعليمي لإزالة المكونات المتوقعة. وفي بيانات فعلية نستخدم Cross-Fitting وشروط انتظام وتحديد مناسبة.")
    ui.refs("[م4] Robinson (1988) Econometrica 56(4)؛ [م5] Chernozhukov et al. (2018) The Econometrics Journal 21(1).")

# --------------------------------------------------------------------------- 24
with tabs[1]:
    ui.section("24 — كيف يعمل Cross-Fitting خطوة بخطوة؟", "Cross-fitting", "teal")
    st.markdown(
        "نفترض أن لدينا بيانات مستقلة بصورة مناسبة، ونقسمها إلى $K$ مجموعات (Folds). في الطية الأولى نقدر $\\widehat m$ و$\\widehat\\ell$ باستخدام **جميع "
        "الطيات عدا الأولى**، ثم نحسب التنبؤات والبواقي للمشاهدات المحجوبة في الأولى. نكرر لكل طية، حتى يحصل كل فرد على تقدير للدوال المساعدة من نموذج "
        "لم يستخدم مشاهدته في تدريب تلك الدوال."
    )
    st.markdown("**الشكل (8) — Cross-Fitting بثلاث طيات (تبسيط)**")
    z = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    txt = [["تنبؤ خارج التدريب" if v else "تدريب" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=["الطية 1", "الطية 2", "الطية 3"], y=["الدورة 1", "الدورة 2", "الدورة 3"], text=txt, texttemplate="%{text}",
                               colorscale=[[0, "#ede9fe"], [1, "#e11d48"]], showscale=False, xgap=6, ygap=6))
    fig.update_yaxes(autorange="reversed")
    ui.style_fig(fig, "في كل دورة: طيتان للتدريب وطية تُتنبأ خارج التدريب", height=300)
    ui.show(fig)
    st.caption("تستخدم التنبؤات خارج الطيات لتكوين درجات التقدير المتعامدة. طريقة تكوين الطيات يجب أن تحترم تصميم البيانات والاعتماد بينها.")
    st.markdown("بعد جمع التنبؤات خارج الطيات نحسب:")
    st.latex(r"\psi_i(\theta,\eta)=\{D_i-m(X_i)\}\{Y_i-\ell(X_i)-\theta[D_i-m(X_i)]\}.")
    st.markdown("يتم اختيار $\\theta$ بحيث يكون متوسط $\\psi_i$ قريبًا من الصفر. وعندما تكون هذه الدرجة Neyman-Orthogonal وتتوفر شروط تقارب الدوال المساعدة، يمكن بناء استدلال صحيح في أطر DML المناسبة [م5].")
    ui.note("Cross-Fitting ليس مرادفًا لـ Rolling-Origin Validation. الأول يُستخدم هنا لحماية استدلال المعلمة عند تقدير الدوال المساعدة، والثاني لتقييم التنبؤ زمنيًا. "
            "عند وجود اعتماد زمني أو تجميعي، يجب تعديل تقسيم الطيات وطريقة الاستدلال بما يتناسب معه.", "تنبيه", "teal", "⚠️")
    st.markdown("#### ماذا لو توقع AI قيمة D بدقة شبه كاملة؟")
    st.markdown(
        "إذا كان $D=m(X)$ تقريبًا، فإن $\\widetilde D$ يكون قريبًا من الصفر، وقد يصبح المقام $\\sum\\widetilde D^2$ صغيرًا جدًا. عندئذ يصعب تعريف أو تقدير "
        "$\\theta$ بدقة. وهذا يثبت أن **تحسين التنبؤ بالدوال المساعدة لا يصنع من العدم الاختلاف الضروري لتحديد المعلمة**."
    )
    sd = st.slider("انحراف الجزء غير المتوقع من D (V = D − m(X))", 0.02, 1.5, 1.0, 0.02, key="f4_vsd")
    n = 1000
    se = 1.0 / (np.sqrt(n) * sd)
    c1, c2 = st.columns(2)
    c1.metric("Σ D̃² / n ≈ Var(V)", f"{sd ** 2:.4f}")
    c2.metric("الخطأ المعياري التقريبي لـ θ̂ (σ_ε = 1، n = 1000)", f"{se:.3f}")
    st.caption("SE(θ̂) ≈ σ_ε / (√n · sd(V)): كلما اقترب D من أن يكون دالة تامة في X، انهار الاختلاف المستقل في المعالجة وانفجر عدم اليقين.")

# --------------------------------------------------------------------------- 25
with tabs[2]:
    ui.section("25 — الدمج السببي: من دالة التنبؤ إلى أثر التدخل", "From prediction to intervention", "rose")
    st.markdown("في التنبؤ نسأل عن $E(Y\\mid X)$، بينما في السببية نهتم بما قد يحدث تحت تدخل. في لغة النتائج المحتملة لدينا:")
    st.latex(r"Y_i(1),\qquad Y_i(0),\qquad \tau_i=Y_i(1)-Y_i(0).")
    st.markdown("لا نلاحظ عادة النتيجتين معًا للفرد نفسه. يمكن أن يساعد AI في تقدير احتمال المعالجة ومتوسطي النتيجة المشروطين:")
    st.latex(r"p(X)=P(D=1\mid X),\qquad \mu_d(X)=E(Y\mid D=d,X),\quad d\in\{0,1\}.")
    st.markdown("ومن أمثلة الدمج استخدام درجة ذات تصحيح مزدوج (AIPW):")
    st.latex(r"\psi_i=\mu_1(X_i)-\mu_0(X_i)+\frac{D_i[Y_i-\mu_1(X_i)]}{p(X_i)}-\frac{(1-D_i)[Y_i-\mu_0(X_i)]}{1-p(X_i)}.")
    st.markdown("يمكن أن يقدَّر متوسط هذه الدرجة لتقدير أثر المعالجة المتوسط في إطار مناسب. لكنها تحتاج إلى افتراضات تعريف مثل **Consistency**، و**Conditional Exchangeability**، و**Positivity/Overlap**، وإلى تصميم استدلال يراعي تقدير الدوال المساعدة.")
    st.markdown("#### لماذا لا تكفي دقة AI لاستخراج السببية؟")
    st.markdown("لأن نموذجًا يستطيع توقع الأجور من المشاركة في برنامج تدريب لا يضمن أن تغيير المشاركة سيؤدي إلى التغير نفسه في الأجر. قد توجد عوامل غير مشاهدة تؤثر في الاثنين معًا. التعلم الآلي يتعامل مع الأنماط في البيانات، بينما التعريف السببي يحتاج افتراضات أو مصدر اختلاف مقنع.")
    ui.note("لا توجد خوارزمية تحوّل الارتباط إلى سببية بمجرد زيادة عدد طبقاتها.", "القاعدة", "rose", "⚖️")

# --------------------------------------------------------------------------- 26
with tabs[3]:
    ui.section("26 — Deep IV وDeep GMM وCausal Forests: توسع الدمج إلى التحديد والآثار غير المتجانسة", "Identification and heterogeneity", "violet")
    st.markdown("#### أولًا: Deep IV")
    st.markdown("عند داخلية متغير المعالجة $D$ قد نستخدم أداة $Z$ صالحة. في Deep IV، تبني المرحلة الأولى نموذجًا مرنًا لتوزيع المعالجة المشروط بالأداة والضوابط، ثم تدخل هذه المعلومات في خسارة مرحلة ثانية مرتبطة بتقدير نتيجة مضادة للواقع. هذه ليست مجرد «انحدار مرحلتين بشبكتين»؛ فالمرحلة الثانية تتضمن **تكاملًا بالنسبة إلى توزيع المعالجة المشروط** [م16].")
    st.latex(r"\min_h\ \sum_i\Big(Y_i-\int h(d,X_i)\,d\hat F(d\mid Z_i,X_i)\Big)^2")
    st.markdown("#### ثانيًا: Deep GMM")
    st.markdown("في القياس الاقتصادي نستخدم شروط لحظات من قبيل:")
    st.latex(r"E[Z_i u_i(\theta)]=0.")
    st.markdown("يمكن تطوير خوارزميات مرنة لتقدير دوال أكثر تعقيدًا بينما تقيس مدى انتهاك شروط لحظات محددة. من الاتجاهات المعروفة DeepGMM التي تناقش معالجة نماذج IV مرنة وكثيرة الأبعاد [م17]. **صحة الأداة وقوة التعريف لا تتولدان تلقائيًا من خوارزمية التعلم.**")
    st.markdown("#### ثالثًا: Causal Forests وGeneralized Random Forests")
    st.markdown("إذا كان أثر البرنامج يختلف بحسب خصائص المستفيدين، نبحث عن:")
    st.latex(r"\tau(X)=E[Y(1)-Y(0)\mid X].")
    st.markdown("تساعد Causal Forests في تقدير هذا التغاير عند تحقق افتراضات السببية والتصميم الملائمة، بينما توسع Generalized Random Forests منطق الغابات إلى كميات تُعرف بشروط لحظات محلية [م18، م19].")
    st.markdown("#### ما الفرق الجوهري بين هذه التطبيقات؟")
    st.markdown("في Residual Learning كان الهدف تقليل أخطاء التنبؤ. أما هنا فالتعلم الآلي يصبح عنصرًا داخل **إجراء تعريف أو تقدير** يهدف إلى معلمة أو أثر اقتصادي محدد. ويجب قراءة خصائص الاستدلال الخاصة بكل منهجية، بدل نقل استنتاجات منطق التنبؤ إليها.")
    ui.refs("[م16] Hartford, Lewis, Leyton-Brown & Taddy (2017) ICML؛ [م17] Bennett, Kallus & Schnabel (2019)؛ [م18] Wager & Athey (2018) JASA؛ [م19] Athey, Tibshirani & Wager (2019) Annals of Statistics.")

# --------------------------------------------------------------------------- 27
with tabs[4]:
    ui.section("27 — الدمج الموجَّه بالنظرية: عندما تتعلم الشبكة داخل حدود اقتصادية", "Theory-guided integration", "emerald")
    st.markdown("ليس مطلوبًا دائمًا أن يقدم النموذج القياسي توقعات أو بواقي للشبكة. أحيانًا تكون مساهمته **قيدًا اقتصاديًا** يوجه عملية التعلم. لنفرض أننا نقدر دالة طلب:")
    st.latex(r"Q=f_\phi(P,I,Z),")
    st.markdown("حيث $P$ السعر، و$I$ الدخل، و$Z$ الخصائص الأخرى. في سياق محدد تبرره النظرية قد نريد فرض أن الطلب لا يزيد بارتفاع السعر مع تثبيت العوامل الأخرى:")
    st.latex(r"\frac{\partial f_\phi}{\partial P}\le 0.")
    st.markdown("#### القيود الناعمة Soft Constraints")
    st.latex(r"L_{\rm data}=\frac1N\sum_i(Q_i-f_\phi(P_i,I_i,Z_i))^2,\qquad L_{\rm theory}=\frac1N\sum_i\left[\max\!\left(0,\frac{\partial f_\phi}{\partial P_i}\right)\right]^2,\qquad \boxed{L_{\rm total}=L_{\rm data}+\lambda L_{\rm theory}.}")
    st.markdown("كلما زادت المخالفة زادت العقوبة، لكن وجود $\\lambda$ محدودة يعني أن النظام قد يقبل مخالفات لتحسين ملاءمة البيانات. أيضًا لا يضمن فرض العقوبة عند نقاط تدريب محددة احترام القيد في كل مجال المتغيرات.")
    st.markdown("#### القيود الصلبة Hard Constraints")
    st.markdown("هنا نبني الدالة بطريقة لا تسمح أصلًا بمخالفة القيد على المجال المستهدف. مثال بنيوي توضيحي:")
    st.latex(r"f_\phi(P,Z)=a_\phi(Z)-\int_{P_0}^{P}\operatorname{softplus}(r_\phi(s,Z))\,ds.")
    st.markdown("وبما أن $\\operatorname{softplus}(u)=\\log(1+e^u)>0$، فإن:")
    st.latex(r"\frac{\partial f_\phi}{\partial P}=-\operatorname{softplus}(r_\phi(P,Z))<0.")
    st.markdown("إذن الرتابة **مضمونة بنيويًا** ضمن التصميم. لكن هذا المثال قد يكون أثقل حسابيًا من معماريات رتيبة أخرى، كما لا ينبغي فرض اتجاه مشتقة لا تدعمه النظرية والسياق.")
    st.markdown("هذه الأفكار تظهر في أبحاث دمج قيود الرتابة وبنية المجموعات الاقتصادية داخل ML، مثل ورقة بنك إنجلترا سنة 2025 حول Block-Additive Models [م20].")
    ui.link("app_pages/p26_theory_nn.py", "مختبر الرتابة التفاعلي في صفحة «الشبكات الموجّهة بالنظرية»", ":material/neurology:")
    ui.refs("[م20] Buckmann & Potjagailo (2025) Bank of England Staff Working Paper 1144.")

# --------------------------------------------------------------------------- 28
with tabs[5]:
    ui.section("28 — الدمج عبر شروط اللحظات GMM والقيود المحاسبية والمالية", "Moment conditions and accounting constraints", "amber")
    st.markdown("قد تعطينا النظرية شرط لحظة:")
    st.latex(r"E[Z_tu_t(\theta,\phi)]=0.")
    st.markdown("نستطيع إدخال شرط مماثل ضمن خسارة مرنة:")
    st.latex(r"L_{\rm moment}=\left\|\frac1N\sum_tZ_tu_t(\theta,\phi)\right\|^2,\qquad L=L_{\rm prediction}+\lambda L_{\rm moment}.")
    st.markdown("أو يمكن إنشاء مقدّر يقوم مباشرة على شروط اللحظات، مع أوزان ومصفوفات تباين مناسبة كما في GMM.")
    st.markdown("#### أمثلة لقيود أخرى")
    st.markdown(
        "- **الهوية المحاسبية:** الإيرادات − التكاليف = الأرباح في تعريف محاسبي مناسب؛ يمكن تصميم مخرجات النموذج بحيث تحقق الهوية.\n"
        "- **عدم السلبية:** بعض الكميات الاقتصادية مثل الاستهلاك في نموذج محدد يجب ألا تكون سالبة.\n"
        "- **قيود الميزانية:** يجب أن تتوافق خيارات المستهلك أو الدولة مع الموارد المتاحة في النموذج.\n"
        "- **شروط عدم المراجحة No-Arbitrage:** توجد في نماذج تمويلية مختلفة وتقيّد أسعار الأدوات المالية؛ يجب تحديد فرضيات السوق والنموذج.\n"
        "- **شروط التوازن:** قد نقلل بواقي معادلات شروط الدرجة الأولى أو معادلات السوق."
    )
    ui.note("فلسفة هذا الدمج: لا نكتفي بأن تتوافق مخرجات AI مع البيانات، بل نحاول أن تتوافق أيضًا مع **علاقات رياضية نعتقد أنها تصف الاقتصاد**. ومع ذلك، لا تكفي إضافة عقوبة "
            "إلى الخسارة لإثبات تعريف المعلمات أو صحة فترات الثقة.", "الفلسفة والحد", "amber", "📐")
    st.markdown("**تصميم المخرجات بحيث تحقق الهوية المحاسبية بالبناء:** بدل معاقبة المخالفة، نجعل الشبكة تتنبأ بالإيرادات والتكاليف فقط ونحسب الربح منهما، فتتحقق الهوية في كل نقطة:")
    st.latex(r"(\widehat{\text{Rev}},\widehat{\text{Cost}})=f_\phi(X),\qquad \widehat{\text{Profit}}\equiv\widehat{\text{Rev}}-\widehat{\text{Cost}},\qquad \widehat{\text{Cons}}=\operatorname{softplus}(\cdot)\ge0")
