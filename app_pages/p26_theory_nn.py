import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "الشبكات الموجّهة بالنظرية الاقتصادية",
    "Economics-Informed, Theory-Guided & Finance-Informed Neural Networks — SKINNs, DEQN, PINN, SDF-nets, BAM",
    "اتجاه حديث جوهري: الشبكة <b>لا تتعلم من البيانات وحدها</b>، بل يُوجَّه تعلّمها بمعرفة اقتصادية تُحوَّل إلى قيود، أو بيانات مولّدة من نموذج، "
    "أو معادلات توازن، أو إشارات للمشتقات. التحول من «تعلم الآلة لأجل الاقتصاد» إلى «تعلم الآلة ببنية اقتصادية».",
    "fuchsia", "emerald",
)

tabs = st.tabs([
    ":material/layers: طبقات المعرفة",
    ":material/settings_input_component: طرق الإدماج السبع",
    ":material/science: مختبر: قيد الرتابة",
    ":material/footprint: داخل الشبكة خطوة بخطوة",
    ":material/account_balance: الشبكات المالية",
    ":material/new_releases: جبهة 2025–2026",
    ":material/warning: المخاطر",
])

with tabs[0]:
    ui.section("خمس طبقات للمعرفة الاقتصادية", "Five layers of economic knowledge", "fuchsia")
    ui.ladder([
        ("قابلية التفسير", "Interpretable blocks", "تقسيم المتغيرات إلى كتل (طلب، عرض، سياسة نقدية) وإظهار مساهمة كل كتلة دون فرض الخطية داخلها — نماذج الكتل الجمعية BAM."),
        ("قيود الشكل", "Shape constraints", "إشارة أو حدود أو رتابة أو تحدب حين تدعمها النظرية (رتابة الطلب في ظروف محددة)."),
        ("معادلات التوازن", "Equilibrium conditions", "أويلر، قيود الميزانية، تصفية الأسواق، شروط HJB في الخسارة أو المعمارية."),
        ("المحاكاة المسبقة", "Simulation pretraining", "تدريب الشبكة على بيانات مولّدة من نموذج اقتصادي ثم تكييفها ببيانات العالم الحقيقي."),
        ("التقدير المشترك", "Joint estimation", "تعلّم دالة مرنة ومعلمات هيكلية قابلة للتفسير معًا — مسار SKINNs."),
    ])
    st.latex(r"\mathcal L(\omega,\theta)=\mathcal L_{data}(\omega,\theta)+\lambda_1\,\mathcal L_{economic}(\omega,\theta)+\lambda_2\,\mathcal L_{boundary}(\omega,\theta)+\lambda_3\,\Omega(\omega)")
    st.markdown("$\\omega$ أوزان الشبكة، $\\theta$ معلمات اقتصادية ذات معنى، $\\lambda$ أوزان القيود، $\\Omega$ عقوبة التعقيد — إطار عام لعائلة الخسائر الموجّهة بالنظرية لا وصف حرفي لكل ورقة.")
    ui.refs("مبني على الفصلين 11–15 من موسوعة د. مروان رودان (2026).")

with tabs[1]:
    ui.section("كيف تدخل المعرفة؟ سبع طرق تقنية", "Seven technical routes", "violet")
    rt = pd.DataFrame([
        ("عقوبات مرنة", "Soft penalties", "L = L_data + λ L_econ", "تفاوض بين الملاءمة والنظرية", "اختيار λ قد يغيّر الاستنتاج"),
        ("قيود صلبة / معلمة", "Hard constraints / parameterization", "بنية لا تسمح بالخرق أصلًا (مثل أوزان موجبة للرتابة)", "التزام مضمون حسابيًا", "تضييق المجال إن كان القيد خاطئًا"),
        ("تدريب مسبق ← ضبط", "Pretrain → fine-tune", "تعلّم على محاكاة اقتصادية ثم بيانات حقيقية", "مفيد مع الضجيج والعينات الصغيرة", "تحيز «المعلّم» الهيكلي"),
        ("لاغرانج / تحسين مقيّد", "Lagrangian / constrained optimization", "قيود بمضاعفات أو حل مقيد", "صياغة صريحة للإلزامات", "تعقّد الأمثلية"),
        ("إسقاط / إصلاح", "Projection / repair", "تصحيح المخرجات بعد التنبؤ لتلائم الشروط", "حفظ بعض القيود", "قد يضعف التنبؤ الأصلي"),
        ("لحظات تنافسية", "Adversarial moments", "خصم يبحث عن خروقات لحظات اقتصادية", "كشف نقاط ضعف غير متوقعة", "عدم استقرار الألعاب"),
        ("قبليات معمارية / كتل", "Architectural priors / blocks", "شبكة مجزأة وفق قنوات نظرية", "تفسير وربط بالآليات", "قد تفرض سردية غير صحيحة"),
    ], columns=["الطريقة", "English", "المبدأ", "المزايا", "الحدود"])
    st.table(rt)

with tabs[2]:
    ui.section("مختبر: قانون طلب مرن مع قيد رتابة", "Lab — a flexible demand curve with a monotonicity penalty", "pink")
    st.markdown("نوفّق منحنى طلب $\\hat Q(P)$ بكثير حدود مرن على بيانات قليلة ومشوّشة، مع **عقوبة** على أي مشتقة موجبة $\\partial\\hat Q/\\partial P>0$ عند نقاط تجميع "
                "(Collocation points) تغطي المجال. حرّك $\\lambda$: عند 0 نموذج بيانات صرف يتلوّى؛ مع زيادة $\\lambda$ يصبح المنحنى متناقصًا ومستقرًا خارج نطاق البيانات.")
    lam = st.select_slider("قوة القيد النظري λ", options=[0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0], value=0.001, key="mono_l")

    @st.cache_data(show_spinner=False)
    def mono_fit(lam: float):
        rng = np.random.default_rng(6)
        P = np.sort(rng.uniform(1, 6, 18)); Q = 20 * np.exp(-0.35 * P) + rng.normal(0, 1.2, P.size)
        deg = 7
        def basis(x): return np.vander((x - 3.5) / 2.5, deg + 1, increasing=True)
        def dbasis(x):
            z = (x - 3.5) / 2.5
            return np.column_stack([np.zeros_like(z)] + [k * z ** (k - 1) / 2.5 for k in range(1, deg + 1)])
        Bp, grid = basis(P), np.linspace(0.5, 7.5, 200)
        col = np.linspace(0.5, 7.5, 80); Dc = dbasis(col)
        # convex penalized least squares solved by an active-set iteration
        A0 = Bp.T @ Bp / len(P) + 1e-4 * np.eye(deg + 1)
        b0 = Bp.T @ Q / len(P)
        w = np.linalg.solve(A0, b0)
        for _ in range(100):
            act = (Dc @ w) > 0
            Da = Dc[act]
            w_new = np.linalg.solve(A0 + lam * Da.T @ Da / len(col), b0)
            if np.allclose(w_new, w, atol=1e-10):
                break
            w = w_new
        fit = basis(grid) @ w
        viol_share = np.mean(dbasis(grid) @ w > 1e-6)
        return P, Q, grid, fit, viol_share, np.mean((Bp @ w - Q) ** 2)

    P, Q, grid, fit, vs, mse = mono_fit(lam)
    fig = go.Figure([go.Scatter(x=P, y=Q, mode="markers", marker=dict(size=10, color=ui.COLORS["slate"]), name="بيانات السوق"),
                     go.Scatter(x=grid, y=20 * np.exp(-0.35 * grid), mode="lines", line=dict(color="#111827", dash="dot"), name="الطلب الحقيقي"),
                     go.Scatter(x=grid, y=np.clip(fit, -10, 40), mode="lines", line=dict(color=ui.COLORS["fuchsia"], width=4), name="الشبكة/النموذج المرن")])
    fig.add_vrect(x0=1, x1=6, fillcolor=ui.TINTS["emerald"], opacity=.5, line_width=0, annotation_text="نطاق البيانات", annotation_position="top left")
    fig.update_xaxes(title="السعر P"); fig.update_yaxes(title="الكمية Q", range=[-6, 30])
    ui.style_fig(fig, f"λ = {lam} — نسبة المجال الذي يخرق الرتابة: {vs:.0%}، MSE داخل العينة: {mse:.2f}", height=460)
    ui.show(fig)
    ui.note("**تحذير اقتصادي جوهري**: إن كانت البيانات **نقاط توازن** يتحرك فيها العرض والطلب معًا، فقد تكون العلاقة المشاهدة بين السعر والكمية غير سالبة دون أن يفشل قانون الطلب. "
            "فرض الرتابة على **دالة التوازن المشاهدة** بدل **الطلب الهيكلي** خطأ جوهري — القيد لا يحل مشكلة الداخلية؛ نحتاج تعرّفًا (أداة عرض مثلًا).",
            "قيد الرتابة ≠ تعرّف", "rose", "⚠️")

with tabs[3]:
    ui.section("كيف «تتعلم» الشبكة من الخصائص الاقتصادية فعلًا؟", "Inside the network, step by step", "teal")
    st.mermaid_chart(
        """
flowchart LR
    D[Real economic data<br/>X, Y, t, region] --> L[Training loss]
    K[Economic knowledge<br/>equilibrium, monotonicity, identities] --> R[Differentiable residuals<br/>on collocation points]
    R --> L
    L --> NN[Neural network weights w<br/>+ structural theta]
    NN --> O[Prediction consistent with theory]
    O --> V[Validation: data error, constraint violation,<br/>parameter stability, OOD]
"""
    )
    ui.ladder([
        ("ما تعرفه النظرية وما لا تعرفه", "Identity vs. hypothesis", "الهوية المحاسبية Y=C+I+G+NX قيد صلب؛ معادلة أويلر شرط احتمالي يُفرض تقريبيًا؛ «كل سعر يخفض كل طلب» ليس قانونًا لدالة توازن."),
        ("تحويل المعرفة إلى دالة قابلة للتفاضل", "Differentiable residual", "r_Euler(t) = u′(c_t) − βE_t[u′(c_{t+1})R_{t+1}]؛ L_Euler = mean(r²)؛ التفاضل الآلي يمرر أثر القيد إلى الأوزان."),
        ("نقاط التجميع", "Collocation points", "التحقق من القيد على حالات مولّدة من المجال ذي الصلة، لا عند المشاهدات فقط — وإلا خُرق القيد في الحالات النادرة."),
        ("ضبط الخسائر والتحقق", "Tune λ and validate", "مقارنة Data-only و Theory-only و Hybrid: خطأ البيانات، مخالفة القيود، استقرار المعلمات، التعميم خارج التوزيع."),
        ("القيد ليس إثباتًا للنظرية", "Constraint ≠ evidence", "تحقق الشرط في المخرجات نتيجة فرضه؛ اختبار النظرية يتطلب مقارنة غير دائرية ببيانات لم تُستخدم لفرضها."),
    ])

with tabs[4]:
    ui.section("الشبكات الموجّهة بالمنطق المالي", "Finance-informed networks", "orange")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**عدم المراجحة وعامل الخصم العشوائي**")
        st.latex(r"\mathbb E_t\big[M_{t+1}R^{e}_{i,t+1}\big]=0,\qquad M_{t+1}=1-\sum_i w_\omega(I_t,I_{i,t})\,R^{e}_{i,t+1}")
        st.markdown("Chen, Pelger & Zhu (2024): الشبكة تتعلم أوزان SDF وتختار أصعب اللحظات تنافسيًا — النظرية داخل هدف التدريب.")
        st.markdown("**قيود أسعار الخيارات**")
        st.latex(r"C(S,K,T)\ge\max(S-Ke^{-rT},0),\quad \partial C/\partial K\le0,\quad \partial^2C/\partial K^2\ge0")
    with c2:
        st.markdown("**PINN لمعادلة بلاك–شولز**")
        st.latex(r"\frac{\partial V}{\partial t}+\tfrac12\sigma^2S^2\frac{\partial^2V}{\partial S^2}+rS\frac{\partial V}{\partial S}-rV=0")
        st.markdown("الشبكة تتعلم $V(S,t)$ بحيث تقترب مشتقاتها الآلية من المعادلة مع شروط الحدود؛ وتمتد لـ PIDE القفزات وتبديل الأنظمة (AF-GPINN 2026). "
                    "**لكن** دقة حل المعادلة ليست صحة افتراضات السوق التي أنتجتها.")
    ui.cards([
        {"title": "التحوط العميق", "en": "Buehler et al. (2019)", "color": "orange", "body": "سياسات تحوط مع تكاليف وسيولة؛ المكافأة تعكس إدارة المخاطر لا الربح الاسمي."},
        {"title": "محوّلات داخل SDF", "en": "Kelly et al. AIPM (2025–2026)", "color": "violet", "body": "اتساق التوقع مع نظام تسعير الأصول — أعمق من توقع العوائد بـ LSTM."},
        {"title": "VAE مقيّد اقتصاديًا", "en": "Chen, Wang & Huang (2026) VAE-ER", "color": "teal", "body": "عوامل كامنة من خصائص الأسهم مرتبطة بحالات الاقتصاد الكلي عبر قيود منتقاة."},
        {"title": "Neural SDE / KAN", "en": "naming caution", "color": "rose", "body": "Neural SDE يتعلم ديناميكيات عشوائية وقد يحتاج ضبطًا مستقلًا لشروط التسعير؛ KAN وحده لا يضمن عدم المراجحة."},
    ], cols=4)

with tabs[5]:
    ui.section("ما الجديد في 2025–2026؟", "Frontier of theory-informed ML", "emerald")
    ui.cards([
        {"title": "SKINNs", "en": "Cao, Chen, Cong & Shi (2026)", "color": "emerald",
         "body": "<b>Structured-Knowledge-Informed Neural Networks</b>: معرفة منظمة (نظرية، محاكاة، معرفة مسبقة، خبرة من مجال آخر) كقيود قابلة للتفاضل مع تقدير مشترك لأوزان الشبكة والمعلمات الاقتصادية، ونقاط تجميع، وشروط للاستدلال والتعرّف."},
        {"title": "تعليم الاقتصاد للآلات", "en": "Chen et al. (2026) Teaching Economics to the Machines", "color": "violet",
         "body": "نموذج هيكلي ← بيانات اصطناعية ← تدريب مسبق ← ضبط على البيانات الحقيقية؛ يستفيد من شكل العلاقات الاقتصادي مع بيانات صغيرة."},
        {"title": "نماذج الكتل الجمعية", "en": "Buckmann & Potjagailo (2025, Bank of England)", "color": "orange",
         "body": "تنظيم المتنبئات في كتل ذات معنى (طلب، عرض) مع لاخطية داخل الكتلة وقيود رتابة لتحليل السياسات."},
        {"title": "عمليات غاوسية واعية اقتصاديًا", "en": "Shi (2026) EAGP thesis", "color": "sky",
         "body": "إدخال المعرفة في قبْليات العملية الغاوسية — الدمج النظري لا يقتصر على الشبكات العصبية."},
        {"title": "شبكات بتنظيم اليعقوبي", "en": "Jacobian-regularized asset pricing (2026)", "color": "rose",
         "body": "مشتقات العائد المتوقع بالنسبة لخصائص الشركات كأسعار مخاطر محلية مع عقوبة اليعقوبي للاستقرار — ليست أثر سياسة سببيًا."},
        {"title": "مراجعة ييل", "en": "Neural Networks in Economics: A Selective Review (2026)", "color": "amber",
         "body": "مراجعة من 129 صفحة: المعماريات، التقريب، القياس والاستخدامات (Cowles)."},
    ], cols=3)
    ui.note("تجمّع في 2026 ثلاثة خطوط: **تعلّم نظري داخل الخوارزمية**، و**استدلال على معلمات اقتصادية من شبكات معقدة**، و**قياس متغيرات جديدة** — مع اختلاف قوة الأدلة: "
            "كثير منها أوراق عمل وأبحاث أولية لا إجماع علمي.", "قراءة الجبهة", "emerald", "🔭")

with tabs[6]:
    ui.section("المخاطر الخاصة بالشبكات الموجّهة بالنظرية", "Specific risks", "rose")
    ui.ladder([
        ("النظرية الخاطئة", "Wrong theory", "فرض نسخة مبسطة خاطئة من النظرية في الخسارة يولّد تحيزًا منظمًا يصعب اكتشافه."),
        ("حساسية λ", "Penalty sensitivity", "الاستنتاج قد يتغير بقوة القيد ⇒ تحليل حساسية إلزامي ومقارنة Soft/Hard/Unconstrained."),
        ("الدائرية", "Circularity", "تحقق القيد ليس دليلًا تجريبيًا؛ الاختبار يتطلب بيانات مستقلة وحالات تنتهك الشرط."),
        ("نقاط تجميع خاطئة", "Bad collocation", "نقاط بعيدة عن مجال التحقق البنيوي تكافئ التزامًا بلا معنى."),
        ("التعرّف", "Identification", "الشبكة تحقق اتجاه الطلب ≠ تعرّفنا على مرونة سببية."),
    ])
    ui.refs("Raissi, Perdikaris & Karniadakis (2019) PINNs; Azinovic et al. (2022); Chen, Pelger & Zhu (2024); Buehler et al. (2019); Buckmann & Potjagailo (2025); "
            "Cao et al. (2026) SKINNs; Chen et al. (2026) Teaching Economics to the Machines — وفق موسوعة رودان.")
