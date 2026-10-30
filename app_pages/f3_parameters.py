import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "فلسفة وآليات الدمج (3): المتغيرات المولدة والمعلمات والتحديد والتدريب المشترك",
    "Part 3: AI-generated variables, parameter learning, identifiability and its four remedies, two orthogonalities, joint training (chapters 14–22)",
    "عندما ينتج AI متغيرًا يدخل الانحدار، أو يتعلم معاملًا اقتصاديًا، أو يُدرَّب مع النموذج القياسي في خسارة واحدة، تظهر أسئلة جديدة: خطأ القياس، "
    "عدم التحديد، الفرق بين التعامد الهندسي وتعامد نيمان، وكيف يعود التدرج إلى المعاملات القياسية. محاضرة <b>د. مروان رودان</b>، الفصول 14–22.",
    "indigo", "rose",
)

tabs = st.tabs([
    ":material/newspaper: 14 الاتجاه المعاكس",
    ":material/straighten: 15 أخطاء القياس",
    ":material/tune: 16 دمج المعلمات",
    ":material/help_center: 17 عدم التحديد",
    ":material/build: 18 أربعة حلول",
    ":material/architecture: 19 التعامد الهندسي",
    ":material/compare: 20 لا تخلط بين التعامدين",
    ":material/sync: 21 التدريب المشترك",
    ":material/view_list: 22 أربعة أنماط للتدريب",
])

# --------------------------------------------------------------------------- 14
with tabs[0]:
    ui.section("14 — الاتجاه المعاكس: AI يخلق متغيرات تدخل في النموذج القياسي", "AI → Econometrics", "indigo")
    st.markdown("قد نريد تقدير أثر عدم اليقين الاقتصادي على الاستثمار، بينما البيانات الأصلية عبارة عن مقالات صحفية ونصوص وتقارير لا تصلح مباشرة كمتغير عددي في OLS. نستخدم نموذج NLP أو LLM لتحويل النصوص إلى مؤشر:")
    st.latex(r"\widehat U_t=h_\phi(\mathrm{Text}_t).")
    st.markdown("ثم نقدّر:")
    st.latex(r"I_t=\alpha+\beta\widehat U_t+\gamma^\top X_t+\varepsilon_t.")
    st.markdown("هنا اتجاه انتقال المعلومات هو **AI → Econometrics**. لقد أنتج AI متغيرًا يدخل في المعادلة. ليس مطلوبًا أن يتدرب النموذجان معًا: قد يكون نموذج النصوص مدربًا مسبقًا ثم يُستخدم لإنتاج المؤشر فقط.")
    st.markdown("**الشكل (5) — من النصوص إلى الاستدلال الاقتصادي**")
    st.mermaid_chart("flowchart LR\n  N[News and reports<br/>unstructured text] --> L[AI / LLM<br/>index extraction] --> R[Econometric regression<br/>index to investment]")
    st.markdown("#### هل يستطيع الانحدار اكتشاف خطأ مؤشر AI وحده؟")
    st.markdown(
        "ليس تلقائيًا. فالمؤشر الذي يولده النموذج قد يكون متحيزًا أو يحمل خطأ قياس يختلف باختلاف نوع النصوص أو اللغة أو توقيت النشر. يجب بناء عينة تحقق "
        "ذات تسميات موثوقة، وتوضيح كيف يوافق المؤشر المفهوم الاقتصادي الذي ندعي قياسه.\n\n"
        "ظهرت أعمال حديثة لتأطير هذا الاستخدام قياسيًا، منها Ludwig وMullainathan وRambachan (2025)، ومراجعة Dell وRambachan (سبتمبر 2026) حول الثورة في "
        "القياس الاقتصادي بواسطة AI [م10، م11]، إضافة إلى العرض التعليمي للتعلم العميق لدى Dell (2025) [م6]."
    )
    ui.refs("[م6] Dell (2025) JEL 63(1)؛ [م10] Ludwig, Mullainathan & Rambachan (2025) NBER w33344؛ [م11] Dell & Rambachan (2026) NBER w35744.")

# --------------------------------------------------------------------------- 15
with tabs[1]:
    ui.section("15 — المتغيرات المولَّدة بالذكاء الاصطناعي وأخطاء القياس", "Generated variables and measurement error", "amber")
    st.markdown("لنفترض أن لدينا مفهومًا اقتصاديًا حقيقيًا $S_t$ لا نستطيع قياسه مباشرة، وأن AI ينتج $\\widehat S_t$. يمكن تمثيل الفرق، في نموذج مبسط، هكذا:")
    st.latex(r"\widehat S_t=S_t+\nu_t,")
    st.markdown("حيث $\\nu_t$ خطأ القياس. إذا استخدمنا $\\widehat S_t$ في الانحدار بدلًا من $S_t$، فقد ينتج تحيز أو أخطاء معيارية غير صالحة. في نموذج انحدار بسيط ومع خطأ قياس كلاسيكي مستقل:")
    st.latex(r"\operatorname{plim}\widehat\beta=\beta\cdot\frac{\operatorname{Var}(S)}{\operatorname{Var}(S)+\operatorname{Var}(\nu)}.")
    st.markdown("هذه صيغة **انحياز التوهين** تحت افتراضات محددة؛ لا تنطبق تلقائيًا على أخطاء LLMs، فقد تكون مرتبطة بالموضوع أو الزمن أو الصنف أو المتغيرات الاقتصادية نفسها، وقد يختلف اتجاه التحيز.")
    c1, c2 = st.columns(2)
    vs = c1.slider("Var(S) — تباين المفهوم الحقيقي", 0.2, 3.0, 1.0, 0.1, key="f3_vs")
    vn = c2.slider("Var(ν) — تباين خطأ القياس", 0.0, 3.0, 0.25, 0.05, key="f3_vn")
    st.metric("معامل التوهين Var(S)/(Var(S)+Var(ν))", f"{vs / (vs + vn):.3f}", f"معامل حقيقي 0.50 يُقدَّر بنحو {0.5 * vs / (vs + vn):.3f}")
    st.markdown("#### ما الإجراءات التي يجب أن يفكر فيها الباحث؟")
    st.markdown(
        "- تعريف المفهوم الاقتصادي المراد قياسه (Construct) قبل اختيار النموذج أو Prompt.\n"
        "- جمع عينة تحقق تمثل المجتمع الذي سيستخدم فيه المؤشر، ويفضل وجود تقييم بشري موثوق.\n"
        "- حساب أخطاء التصنيف أو القياس حسب الفئات والزمن ومصادر النصوص، لا الاكتفاء بدقة كلية.\n"
        "- دراسة حساسية النتائج لاختيار نموذج اللغة وطريقة التوجيه والعتبات.\n"
        "- عند الاستدلال، استخدام إجراء صُمم للتعامل مع المتغيرات المولَّدة، بدل اعتبارها بيانات خالية من الخطأ.\n"
        "- الانتباه إلى **تسرب البيانات داخل التدريب السابق للنموذج اللغوي**، فلا يكفي فصل العينة في الانحدار إذا كان LLM قد رأى لاحقًا معلومات عن النتائج التي نحاول توقعها."
    )
    st.markdown(
        "في الأدبيات الحديثة: Battaglia وزملاؤه (2024/2025) ناقشوا تحيز الانحدارات بمتغيرات مولدة، وDuan وPelger (يوليو 2026) قدما منهجًا يستعمل عينة "
        "معايرة وتسوية لشروط اللحظات، بينما ناقش Christensen وHansen (مايو 2026) طرقًا عملية للاستدلال باستخدام هذه المتغيرات [م12، م14، م15]."
    )
    ui.note("AI لا ينقل إلى القياس الاقتصادي متغيرًا فقط؛ قد ينقل معه **خطأ القياس وعدم اليقين واحتمال تغير معنى المؤشر عبر الزمن**.", "النتيجة", "amber", "📌")
    ui.refs("[م12] Battaglia, Christensen, Hansen & Sacher (2024/2025) Cowles DP 2421؛ [م14] Duan & Pelger (2026) NBER w35481؛ [م15] Christensen & Hansen (2026) AEA P&P 116.")

# --------------------------------------------------------------------------- 16
with tabs[2]:
    ui.section("16 — الدمج على مستوى المعلمات: عندما تتعلم الخوارزمية معاملًا اقتصاديًا", "Parameter-level integration", "teal")
    st.markdown("في انحدار تقليدي:")
    st.latex(r"Y_t=\alpha+\beta X_t+\varepsilon_t,")
    st.markdown("تكون $\\beta$ ثابتة في المواصفة المختارة. لكن ماذا لو تغيرت العلاقة مع حالة الاقتصاد؟ يمكننا كتابة:")
    st.latex(r"Y_t=\alpha+\beta_t X_t+\varepsilon_t, \qquad \beta_t=h_\phi(Z_t)\qquad\Longrightarrow\qquad Y_t=\alpha+h_\phi(Z_t)X_t+\varepsilon_t.")
    st.markdown("هنا تتعلم الشبكة دالة تتحكم في **المعامل نفسه**، وليس في البواقي النهائية. قد يمثل $Z_t$ التضخم المتوقع أو تقلب أسعار النفط أو طور الدورة الاقتصادية.")
    st.markdown("#### كيف يتدرب النموذج؟")
    st.markdown("ندخل $Z_t$ إلى الشبكة فتولد $\\widehat\\beta_t$. نضربها في $X_t$، ثم نحسب التنبؤ والخسارة. إذا كان البناء قابلًا للتفاضل، يمكن تحديث أوزان الشبكة من خطأ التنبؤ النهائي، مع إمكانية فرض قيود على المجال الذي تقع فيه المعلمة.")
    st.mermaid_chart("flowchart LR\n  Z[State Z_t] --> H[Network h_phi] --> B[beta_t]\n  X[X_t] --> M((x))\n  B --> M --> P[Prediction] --> L[Loss]\n  L -.gradient.-> H")
    st.markdown("#### أين الخطورة؟")
    st.markdown(
        "- هل توجد فروق كافية في البيانات لتحديد $\\beta_t$ أم أنها تتداخل مع مكونات مرنة أخرى؟\n"
        "- هل المعلمة وصفية تتغير مع الحالة، أم ندعي أنها تمثل أثرًا سببيًا؟\n"
        "- هل نسمح للشبكة برؤية متغيرات مستقبلية؟\n"
        "- هل نحتاج إلى تنعيم أو Regularization حتى لا تتحرك المعلمات بلا أساس اقتصادي؟"
    )
    st.markdown("في إطار البانل يمكن أيضًا أن نتعلم $\\beta_i=h_\\phi(W_i)$، أي اختلاف الأثر عبر الوحدات الاقتصادية، مع الحاجة إلى التحقق من إمكانية تحديد المعلمة الخاصة بكل وحدة.")

# --------------------------------------------------------------------------- 17
with tabs[3]:
    ui.section("17 — مشكلة عدم التحديد: قد يتعلم الجزءان العلاقة نفسها", "Identifiability", "rose")
    st.markdown("لنكتب نموذجًا هجينًا:")
    st.latex(r"Y=\beta X+g_\phi(X)+\varepsilon.")
    st.markdown("نفترض أن العلاقة الحقيقية هي $Y=3X+X^2$. نستطيع تمثيل العلاقة باستخدام $\\beta=3$ و$g(X)=X^2$. لكن نستطيع أيضًا اختيار $\\beta=1$ وجعل $g(X)=2X+X^2$، أو اختيار $\\beta=5$ وجعل $g(X)=-2X+X^2$. جميع الخيارات الثلاثة تعطي **التنبؤ نفسه**. بصورة عامة، لأي عدد $c$:")
    st.latex(r"\beta'=\beta+c,\qquad g'(X)=g(X)-cX\qquad\Longrightarrow\qquad \beta'X+g'(X)=\beta X+g(X).")
    st.markdown("إذا استطاعت الشبكة تمثيل المكون الخطي، فهذه مشكلة تداخل مكونات **Identifiability**، ولا تُحل تلقائيًا بإضافة بيانات تدريب أو زيادة عدد Epochs.")
    st.markdown("**مختبر 3: غيّر β ولاحظ ثبات المجموع** — في المثال $Y=3X+X^2$، عندما $X=2$ يكون $Y=10$.")
    b = st.slider("المعامل القياسي β", -2.0, 8.0, 3.0, 0.1, key="f3_beta")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("المعامل β", f"{b:.1f}")
    c2.metric("الجزء القياسي βX", f"{2 * b:.1f}")
    c3.metric("الجزء المرن g(X)", f"{10 - 2 * b:.1f}")
    c4.metric("المجموع", "10.0")
    st.caption(f"عند β = {b:.1f}، يمكن للمكوّن المرن أن يتعلم g(x) = x² + ({3 - b:.1f})x، ولذلك لا يتغير مجموع التنبؤ.")
    xs = np.linspace(-2, 3, 120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=b * xs, name="βX", line=dict(color="#7c3aed", width=2.6)))
    fig.add_trace(go.Scatter(x=xs, y=xs ** 2 + (3 - b) * xs, name="g(X)", line=dict(color="#0d9488", width=2.6)))
    fig.add_trace(go.Scatter(x=xs, y=3 * xs + xs ** 2, name="المجموع = 3X + X² (ثابت)", line=dict(color="#e11d48", width=3, dash="dash")))
    ui.style_fig(fig, "إعادة توزيع العلاقة بين المكوّنين دون أن يتغير التنبؤ", height=380)
    ui.show(fig)
    ui.note("عند قراءة أي ورقة اسأل: **هل جرى تحديد المكوّن الاقتصادي بصورة تميزه عن الشبكة؟** إذا كانت الإجابة لا، فإن وجود المعلمة داخل المعادلة لا يضمن قابلية تفسيرها.",
            "سؤال القارئ", "rose", "❓")

# --------------------------------------------------------------------------- 18
with tabs[4]:
    ui.section("18 — أربعة حلول لتنظيم التعلم بين المكوّنين", "Four remedies", "emerald")
    ui.ladder([
        ("الحل الأول: التقدير ثم التثبيت", "Freeze", "نقدّر المعلمة القياسية أولًا، ثم نثبتها ونُدرّب AI على البواقي. هذا يمنع الشبكة من تعديل $\\widehat\\beta$ أثناء التدريب الثاني، "
         "لكنه **لا يثبت** أن التقدير الأول صحيح أو غير متحيز."),
        ("الحل الثاني: فصل المدخلات", "Feature separation", "نحدد متغيرات للجزء القياسي وأخرى للشبكة. لكن إذا كانت المجموعتان مترابطتين بشدة، فقد تستطيع الشبكة إعادة بناء المعلومات "
         "التي خصصناها للجزء القياسي. الفصل الاسمي وحده ليس ضمانًا للتحديد."),
        ("الحل الثالث: التنظيم", "Regularization", "نضع عقوبة على تعقيد الشبكة أو على تقلب معاملات النموذج: $\\mathcal L=\\mathcal L_{\\mathrm{data}}+\\lambda\\Omega(\\phi)$. "
         "يساعد ذلك أحيانًا على استقرار التقدير والحد من Overfitting، لكنه لا يوفر دائمًا فصلًا فريدًا بين المكوّنين."),
        ("الحل الرابع: التعامد", "Orthogonalization", "نفرض أن مساهمة الشبكة لا تعيد إنتاج المكون الخطي الذي يمثله الجزء القياسي، باستخدام الإسقاط على الفضاء المتعامد. "
         "حل هندسي مباشر لبعض مشكلات التداخل، بشرط فهم حدود صلاحيته."),
    ])
    st.latex(r"\mathcal L=\mathcal L_{\mathrm{data}}+\lambda\Omega(\phi)")
    st.markdown("هذه الأفكار تناقشها أدبيات النماذج شبه المهيكلة، ومنها أعمال Rügamer وزملائه وورقة Post-Hoc Orthogonalization سنة 2023 [م8، م9].")
    ui.refs("[م8] Rügamer et al. (2023) The American Statistician؛ [م9] Rügamer (2023) ICML/PMLR «A new PHO-rmula for improved performance of semi-structured networks».")

# --------------------------------------------------------------------------- 19
with tabs[5]:
    ui.section("19 — التعامد الهندسي: لماذا نسقط مخرجات الشبكة على فضاء آخر؟", "Geometric (component) orthogonality", "violet")
    st.markdown("في عينة فيها $N$ مشاهدة، لنكتب:")
    st.latex(r"\widehat{\mathbf Y}=\mathbf X\beta+\mathbf g_\phi.")
    st.markdown("إذا كانت $\\mathbf X$ ذات رتبة أعمدة كاملة، فإن مصفوفة الإسقاط على فضائها هي:")
    st.latex(r"P_X=\mathbf X(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top.")
    st.markdown("نزيل من مخرج الشبكة الجزء القابل للإسقاط خطيًا على $\\mathbf X$:")
    st.latex(r"\boxed{\mathbf g_\phi^\perp=(I-P_X)\mathbf g_\phi}\qquad\text{ومن خصائص الإسقاط:}\qquad \mathbf X^\top\mathbf g_\phi^\perp=0.")
    st.markdown("#### المعنى بصورة مبسطة")
    st.markdown("تخيل أن كل مخرج للشبكة متجه في فضاء المشاهدات. يمكن تقسيمه إلى مكوّن يقع في الاتجاهات التي يستطيع الانحدار القياسي تمثيلها، ومكوّن عمودي على تلك الاتجاهات. نستخدم المكوّن الثاني لتقليل التداخل في المساهمات.")
    st.markdown("**الشكل (6) — تمثيل توضيحي للتعامد** (حرّك زاوية مخرج الشبكة)")
    ang = st.slider("زاوية مخرج الشبكة الأصلي g (بالدرجات)", 5, 175, 50, key="f3_ang")
    gx, gy = 3 * np.cos(np.radians(ang)), 3 * np.sin(np.radians(ang))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-3.5, 3.5], y=[0, 0], mode="lines", name="فضاء X (المكوّن الخطي)", line=dict(color="#7c3aed", width=4)))
    for (x0, y0, x1, y1, col, nm) in [(0, 0, gx, gy, "#475569", "مخرج الشبكة الأصلي g"), (0, 0, gx, 0, "#7c3aed", "P_X g (جزء يكرر الخطي)"),
                                       (gx, 0, gx, gy, "#e11d48", "g⊥ = (I − P_X) g")]:
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=3, arrowcolor=col)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=col, width=3), name=nm))
    fig.update_xaxes(range=[-3.6, 3.6], zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[-0.4, 3.4], scaleanchor="x", showticklabels=False)
    ui.style_fig(fig, "المكوّن العمودي g⊥ هو ما تبقيه الشبكة بعد إزالة ما يستطيع X تمثيله", height=400)
    ui.show(fig)
    st.caption("رسم هندسي مبسط لفهم الفكرة؛ الإسقاط الحقيقي يجري في فضاء مشاهدات قد يكون عالي الأبعاد.")
    st.markdown("#### ثلاثة تحذيرات أساسية")
    st.markdown(
        "- التعامد هنا قائم على **فضاء العينة ومصفوفة التصميم**، ولا يعني الاستقلال الإحصائي بين المتغيرات.\n"
        "- التعامد الجبري لا يحقق السببية أو خارجية المتغيرات تلقائيًا.\n"
        "- قد يؤثر إدخاله أثناء التدريب في سرعة التقارب والتنبؤ؛ لذلك دُرست أيضًا بدائل مثل Post-Hoc Orthogonalization [م9]."
    )
    st.markdown("إذا لم تكن $\\mathbf X$ ذات رتبة كاملة فيمكن استخدام إسقاط مناسب عبر شبه المعكوس، لكن تظل مشكلة الرتبة نفسها مهمة من ناحية التحديد.")
    with st.expander("تحقق رقمي: β تعود فريدة بعد التعامد"):
        rng = np.random.default_rng(3)
        n = 200
        x = rng.normal(size=n)
        Xm = np.column_stack([np.ones(n), x])
        y = 1 + 3 * x + x ** 2 + rng.normal(0, 0.3, n)
        g_raw = 2 * x + x ** 2                       # a network output that also carries a linear part
        P = Xm @ np.linalg.pinv(Xm)
        g_perp = g_raw - P @ g_raw
        b_raw = np.linalg.lstsq(Xm, y - g_raw, rcond=None)[0][1]
        b_perp = np.linalg.lstsq(Xm, y - g_perp, rcond=None)[0][1]
        st.markdown(f"مع مخرج شبكة غير متعامد (يحمل 2x) يُقدَّر β = **{b_raw:.3f}**؛ بعد الإسقاط $(I-P_X)$ يُقدَّر β = **{b_perp:.3f}**، أي كل الجزء الخطي يعود إلى المكوّن القياسي "
                    f"(القيمة الكاملة لميل X في البيانات، لأن X² نفسه يملك إسقاطًا خطيًا صغيرًا). و$\\max|X^\\top g^\\perp|$ = {np.abs(Xm.T @ g_perp).max():.2e}.")

# --------------------------------------------------------------------------- 20
with tabs[6]:
    ui.section("20 — لا تخلط بين التعامد الهندسي وNeyman Orthogonality", "Two different orthogonalities", "sky")
    st.markdown(
        "هما مفهومان يحملان لفظ «التعامد» لكن لهما وظيفتان مختلفتان.\n\n"
        "**Component Orthogonality:** فصل إسهامات مكونات النموذج الهجين، مثل اشتراط $\\mathbf X^\\top\\mathbf g^\\perp=0$.\n\n"
        "**Neyman Orthogonality:** بناء معادلة تقدير لا يتأثر متوسطها من الدرجة الأولى بأخطاء صغيرة في تقدير الدوال المساعدة عند القيم الحقيقية. في الحالة الثانية لدينا معادلة تقدير:"
    )
    st.latex(r"E[\psi(W;\theta_0,\eta_0)]=0.")
    st.markdown("حيث $\\theta_0$ المعلمة المستهدفة، و$\\eta_0$ دوال مساعدة قد تقدرها ML. نقول إن المعادلة متعامدة إذا كانت مشتقة المتوسط بالنسبة إلى اضطراب محلي في $\\eta$ تساوي صفرًا عند الحقيقة:")
    st.latex(r"\left.\frac{\partial}{\partial r}E[\psi(W;\theta_0,\eta_0+r h)]\right|_{r=0}=0.")
    st.markdown("هذا يقلل حساسية المقدّر للأخطاء من الدرجة الأولى، ضمن الشروط اللازمة. لا يعني أن كل خطأ في الدوال المساعدة يختفي، ولا يعني أن المعلمة أصبحت سببية بدون افتراضات تعريف مناسبة.")
    st.markdown(
        "| المصطلح | وظيفة التعامد | النتيجة التي لا يضمنها |\n|---|---|---|\n"
        "| Component Orthogonality | الحد من تداخل المساهمات داخل النظام الهجين | لا يضمن السببية |\n"
        "| Neyman Orthogonality | تخفيف حساسية معادلة التقدير لأخطاء الدوال المساعدة | لا يلغي الحاجة إلى شروط انتظام وتحديد |\n"
        "| Exogeneity / Identification | افتراضات أو شروط تعرف المعلمة المستهدفة | لا تنتج تلقائيًا من أي تعامد |"
    )
    ui.link("app_pages/d07_inference.py", "الاشتقاق الكامل لتعامد نيمان والتقسيم المتقاطع وشروط المعدلات", ":material/verified:")

# --------------------------------------------------------------------------- 21
with tabs[7]:
    ui.section("21 — التدريب المشترك Joint End-to-End: كيف يعود الخطأ إلى النموذج القياسي؟", "How the error flows back", "orange")
    st.markdown("نكتب:")
    st.latex(r"\widehat Y_i=f_\theta(X_i)+g_\phi(Z_i).")
    st.markdown("نعرّف خسارة مشتركة:")
    st.latex(r"\mathcal L(\theta,\phi)=\frac{1}{2N}\sum_i\left[Y_i-f_\theta(X_i)-g_\phi(Z_i)\right]^2.")
    st.markdown("إذا سمحنا بتحديث مجموعتي المعلمات، فإن التدرجات هي:")
    st.latex(r"\nabla_\theta\mathcal L=-\frac1N\sum_i(Y_i-\widehat Y_i)\nabla_\theta f_\theta(X_i),\qquad \nabla_\phi\mathcal L=-\frac1N\sum_i(Y_i-\widehat Y_i)\nabla_\phi g_\phi(Z_i).")
    st.markdown("لاحظ أن **خطأ التنبؤ النهائي نفسه** يدخل في التحديثين، لكن حساسية كل مكوّن للأخطاء تختلف بحسب مشتقاته.")
    st.markdown("#### إذا دخل توقع النموذج القياسي إلى الشبكة مباشرة")
    st.latex(r"u=f_\theta(X),\qquad \widehat Y=u+g_\phi(u,Z)\qquad\Longrightarrow\qquad \frac{\partial\widehat Y}{\partial\theta}=\left(1+\frac{\partial g_\phi}{\partial u}\right)\frac{\partial f_\theta}{\partial\theta}.")
    st.markdown("تغيير $\\theta$ يؤثر في التنبؤ بطريقتين: مباشرة عبر $u$، وبصورة غير مباشرة لأن $u$ دخل أيضًا إلى الشبكة. هذه هي **قاعدة السلسلة** التي تشرح رجوع التدرج.")
    st.markdown("**الشكل (7) — تدفق أمامي وخلفي**")
    st.mermaid_chart(
        """
flowchart TB
    T[Parameters theta<br/>econometric part] -->|Forward| P[Final prediction then loss]
    F[Weights phi<br/>neural part] -->|Forward| P
    P -.Backpropagation: gradient updates theta.-> T
    P -.Backpropagation: gradient updates phi.-> F
"""
    )
    st.caption("لا يعود التدرج إلى المكوّن القياسي إذا جُمّدت معلماته أو قطعت وصلة التفاضل عند مخرجه.")
    st.markdown("#### مثال عددي صغير")
    st.markdown("نفترض $Y=10$، و$X=2$، و$Z=1$، والشبكة في صورتها المبسطة $g_\\phi(Z)=\\phi Z$. نبدأ بـ $\\beta=2$ و$\\phi=3$، فيكون التنبؤ $7$. إذا كانت:")
    st.latex(r"L=\frac12(10-\beta X-\phi Z)^2,")
    st.markdown("فإن المشتقتين هما $-6$ و$-3$. باستخدام معدل تعلم $0.1$ يصبح $\\beta=2.6$ و$\\phi=3.3$، ويرتفع التنبؤ إلى $8.5$، وتنخفض الخسارة من $4.5$ إلى $1.125$.")
    steps = st.slider("عدد خطوات التدريب المشترك (η = 0.1)", 1, 15, 1, key="f3_steps")
    b, ph, rows = 2.0, 3.0, []
    for k in range(steps + 1):
        pred = b * 2 + ph * 1
        loss = 0.5 * (10 - pred) ** 2
        rows.append((k, b, ph, pred, loss))
        r = 10 - pred
        b, ph = b + 0.1 * r * 2, ph + 0.1 * r * 1
    st.dataframe(pd.DataFrame(rows, columns=["الخطوة", "β", "φ", "التنبؤ", "الخسارة"]).round(4), hide_index=True, height=min(400, 38 + 35 * len(rows)))
    ui.note("هذا **ليس دليلًا أن المعامل الاقتصادي الحقيقي تغير**؛ لقد تحرك لأنه يخفّض خسارة التنبؤ. يجب الفصل بين جودة التحسين وصلاحية تفسير المعلمات. "
            "لاحظ في الجدول أن β تستقر عند قيمة يحددها مسار التحسين لا الاقتصاد: β = 2 + 2c و φ = 3 + c لأي تقدم c، فكل توزيع يحقق 2β + φ = 10 يصفّر الخسارة.",
            "تحذير", "orange", "⚠️")

# --------------------------------------------------------------------------- 22
with tabs[8]:
    ui.section("22 — أربعة أنماط للتدريب: ما الذي يتغير فعلًا في كل واحد؟", "Four training regimes", "teal")
    st.markdown(
        "| طريقة التدريب | خطواتها | ما الذي يبقى ثابتًا؟ | متى نستخدمها مبدئيًا؟ |\n|---|---|---|---|\n"
        "| Sequential / Frozen | تقدير E ← تثبيت E ← تدريب AI | معاملات E أثناء تدريب AI | تصحيح البواقي أو استعمال توقعات نموذج سابق |\n"
        "| Alternating | تحديث E مع تثبيت AI ثم العكس | جزء مختلف في كل خطوة | عندما يكون التحسين المشترك صعبًا أو نريد تحكمًا في الأدوار |\n"
        "| Joint End-to-End | حساب خسارة مشتركة وتحديث المعلمات القابلة للتفاضل | فقط ما جرى تجميده صراحة | بناء مترابط يمكن التفاضل خلاله |\n"
        "| Pretraining → Fine-tuning | تدريب على بيانات/هدف أول ثم ضبط لاحق | قد نجمّد بعض الطبقات أو لا | نقل معلومات من نموذج هيكلي أو مجال سابق |"
    )
    st.markdown("#### ما المقصود بـ Stop Gradient؟")
    st.markdown("هو فصل مسار التفاضل: يمكن للشبكة أن **تقرأ** توقع ARDL أو مخرج مكوّن سابق، لكن التحديث الذي تحسبه لا يرجع إلى المعلمات التي أنتجت ذلك التوقع. إذن القراءة لا تعني إمكان التعديل.")
    st.markdown("#### ما المقصود بـ Alternating Optimization؟")
    st.latex(r"\theta^{k+1}=\arg\min_\theta L(\theta,\phi^k), \qquad \phi^{k+1}=\arg\min_\phi L(\theta^{k+1},\phi).")
    st.markdown("قد تكون خطوات التقليل تقريبية وليست حلولًا مغلقة. كما أن التقارب الحسابي لا يكفي لضمان التحديد الاقتصادي أو الوصول إلى الحد الأدنى العالمي.")
    st.markdown("#### ما الفرق بين Epoch وIteration وBatch؟")
    st.markdown("الـ**Epoch** مرور على أمثلة التدريب، والـ**Batch** مجموعة أمثلة تستخدم لحساب تحديث واحد، والـ**Iteration** قد تشير إلى خطوة تحسين واحدة بحسب الخوارزمية. وليس لأي منها معنى «شهر» أو «ربع سنة» تلقائيًا في السلسلة الزمنية.")
