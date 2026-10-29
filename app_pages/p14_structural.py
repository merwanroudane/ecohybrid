import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import ui

ui.hero(
    "النماذج الهيكلية والذكاء الاصطناعي",
    "Structural Econometrics Meets Deep Learning — Solving, Estimating and Interpreting",
    "هنا يدخل الذكاء الاصطناعي إلى قلب الاقتصاد النظري: شبكات تحل نماذج التوازن عالية الأبعاد (DEQN، PINN)، "
    "وشبكات تقدّر المعلمات الهيكلية (التقدير العصبي والتنافسي)، ونصوص تدخل معادلات القياس في DSGE — مع قراءة اقتصادية لـ AlphaGo نفسه.",
    "fuchsia", "teal",
)

tabs = st.tabs([
    ":material/architecture: الهيكلي مقابل المختزل",
    ":material/neurology: حل النماذج بالشبكات",
    ":material/science: مختبر تكرار دالة القيمة",
    ":material/model_training: التقدير العصبي",
    ":material/sports_kabaddi: التقدير التنافسي",
    ":material/article: النص داخل DSGE",
    ":material/sports_esports: الذكاء الاصطناعي كتقدير هيكلي",
])

with tabs[0]:
    ui.section("لماذا النماذج الهيكلية؟ وما مشكلتها الحسابية؟", "Structural vs. reduced form; the curse of dimensionality", "fuchsia")
    c1, c2 = st.columns(2)
    with c1:
        ui.note("تقدّر **معلمات عميقة** (تفضيلات، تقنية، تكاليف تعديل) ثابتة أمام تغيّر السياسات ⇒ تجيب عن أسئلة **مضادة للواقع** لسياسات لم تُطبَّق قط — الرد المباشر على نقد لوكاس.",
                "قوة النموذج الهيكلي", "fuchsia", "🏛️")
    with c2:
        ui.note("حل نموذج ديناميكي بتوقعات رشيدة وأسر غير متجانسة يتطلب تقريب دوال على فضاء حالات ضخم (توزيع الثروة كله!). الشبكات التقليدية (Grids) تنفجر أُسيًا: **لعنة الأبعاد** (Bellman).",
                "عنق الزجاجة", "rose", "🧱")
    st.latex(r"V(s)=\max_{a\in\Gamma(s)}\Big\{u(s,a)+\beta\,\mathbb E\big[V(s')\mid s,a\big]\Big\},\qquad \#\text{grid points}=m^{\dim(s)}")
    st.markdown("مع 10 نقاط لكل بُعد و 20 متغير حالة نحتاج $10^{20}$ نقطة. **الشبكات العصبية** تقرّب $V$ أو دالة السياسة $a(s)$ بعدد معلمات يتزايد ببطء مع الأبعاد، وتُدرَّب على حالات **محاكاة** تزورها الاقتصاد فعلًا.")

with tabs[1]:
    ui.section("حل نماذج التوازن بالتعلم العميق", "Deep learning for solving economic models", "violet")
    ui.ladder([
        ("بناء النموذج", "Model", "تفضيلات، تقنية، قيود، صدمات، وشروط توازن (أويلر، الميزانية، تصفية الأسواق)."),
        ("تعريف الحالة", "State", "رأس المال، الأجور، التوزيع (أو ملخصاته)، الصدمات."),
        ("تقريب السياسات", "Neural policy", "شبكة $a_\\omega(s)$ لدوال السياسة (الاستهلاك، الادخار، الأسعار)."),
        ("المحاكاة", "Simulate", "توليد مسارات حالات من النموذج بالسياسة الحالية (Ergodic set)."),
        ("حساب البواقي", "Residuals", "مخالفات أويلر والميزانية وشروط التكامل (Complementarity) على نقاط محاكاة."),
        ("التحديث", "Train", "تدني الخسارة $\\mathcal L(\\omega)=\\mathbb E_s\\|r(s;\\omega)\\|^2$ بالانحدار التدريجي العشوائي والتفاضل الآلي."),
        ("التحقق", "Verify", "دقة الحل في مناطق قيود الاقتراض والذيول وخارج نطاق التدريب؛ فصل خطأ الحل عن خطأ التقدير."),
    ])
    st.latex(r"r_{Euler}(s)=u'\big(c_\omega(s)\big)-\beta\,\mathbb E\big[u'\big(c_\omega(s')\big)R(s')\mid s\big],\qquad \mathcal L(\omega)=\mathbb E_{s\sim\text{ergodic}}\big[r_{Euler}(s)^2\big]")
    ui.cards([
        {"title": "طريقة «الكل في واحد»", "en": "Maliar, Maliar & Winant (2021, JME)", "color": "violet", "body": "تحويل الحل إلى مسألة تدنية واحدة بمؤثر توقع «الكل في واحد» وتدريب الشبكة بـ SGD."},
        {"title": "شبكات التوازن العميق", "en": "Azinovic, Gaegauf & Scheidegger (2022, IER)", "color": "teal", "body": "DEQN: نماذج أجيال متداخلة بعشرات الأجيال وصدمات كلية — بلا شبكات نقاط."},
        {"title": "الاحتكاكات وتوزيع الثروة", "en": "Fernández-Villaverde, Hurtado & Nuño (2023, Econometrica)", "color": "orange", "body": "حل نموذج زمني مستمر غير متجانس بشبكة تتعلم «قانون الحركة» الإدراكي للتوزيع."},
        {"title": "ترويض لعنة الأبعاد", "en": "Fernández-Villaverde, Nuño & Perla (2024)", "color": "rose", "body": "استغلال التماثل (Symmetry) والتجميع في نماذج بعدد كبير من الأعوان."},
        {"title": "PINN و HJB", "en": "Physics-informed nets in continuous time", "color": "sky", "body": "شبكات تقترب من حل معادلات هاميلتون–جاكوبي–بلمان وكولموغوروف الأمامية بتفاضل آلي ونقاط تجميع (Collocation)."},
        {"title": "مراجعة JEL", "en": "Fernández-Villaverde (2026) — deep learning for solving models", "color": "emerald", "body": "مراجعة شاملة منشورة في Journal of Economic Literature (سبتمبر 2026) وفق ما توثقه موسوعة رودان."},
    ], cols=3)
    ui.note("**DEQN** في الاقتصاد ليست «Deep Equilibrium Models» في أدبيات الرؤية الحاسوبية؛ الاسم متشابه والمفهوم مختلف. "
            "وحل التوازن بدقة لا يعني صحة النموذج الاقتصادي المفروض.", "تمييز مصطلحي", "amber", "⚠️")

with tabs[2]:
    ui.section("مختبر متحرك: تكرار دالة القيمة في نموذج النمو النيوكلاسيكي", "Value-function iteration (animated)", "pink")
    st.markdown("نموذج روبنسون كروزو: $\\max\\sum\\beta^t\\ln c_t$ بقيد $k_{t+1}=k_t^{\\alpha}-c_t$. نبدأ بـ $V_0=0$ ونكرر مؤثر بلمان. "
                "شغّل الحركة لترى تقارب دالة القيمة ودالة السياسة نحو الحل التحليلي $k'=\\alpha\\beta k^{\\alpha}$ — هذا ما تحاول الشبكات تقريبه في الأبعاد العالية.")
    beta_ = st.slider("معامل الخصم β", 0.85, 0.98, 0.95, 0.01, key="vfi_b")

    @st.cache_data(show_spinner=False)
    def vfi(beta: float, alpha: float = 0.3):
        kss = (alpha * beta) ** (1 / (1 - alpha))
        k = np.linspace(0.2 * kss, 2.0 * kss, 160)
        c = k[:, None] ** alpha - k[None, :]
        u = np.where(c > 0, np.log(np.maximum(c, 1e-12)), -1e10)
        V = np.zeros(len(k)); snaps = []
        keep = [0, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 160]
        for it in range(161):
            Q = u + beta * V[None, :]
            pol = k[np.argmax(Q, 1)]
            Vn = Q.max(1)
            if it in keep:
                snaps.append((it, Vn.copy(), pol.copy()))
            V = Vn
        exact = alpha * beta * k ** alpha
        return k, snaps, exact, kss

    k, snaps, exact, kss = vfi(beta_)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("دالة القيمة V(k)", "دالة السياسة k′(k)"))

    def vfi_tr(i):
        it, Vn, pol = snaps[i]
        return [go.Scatter(x=k, y=Vn, mode="lines", line=dict(color=ui.COLORS["violet"], width=4), name="V بعد التكرار"),
                go.Scatter(x=k, y=pol, mode="lines", line=dict(color=ui.COLORS["rose"], width=4), name="السياسة المقدّرة"),
                go.Scatter(x=k, y=exact, mode="lines", line=dict(color="#111827", dash="dot"), name="الحل التحليلي αβk^α"),
                go.Scatter(x=k, y=k, mode="lines", line=dict(color="#cbd5e1", dash="dash"), name="خط 45°")]

    for i, t in enumerate(vfi_tr(0)):
        fig.add_trace(t, row=1, col=1 if i == 0 else 2)
    fig.frames = [go.Frame(data=vfi_tr(i), traces=[0, 1, 2, 3], name=str(snaps[i][0])) for i in range(len(snaps))]
    vmin = min(s[1].min() for s in snaps[1:]); vmax = max(s[1].max() for s in snaps)
    fig.update_yaxes(range=[vmin - 1, vmax + 1], row=1, col=1)
    fig.update_xaxes(title="k", row=1, col=1); fig.update_xaxes(title="k", row=1, col=2)
    ui.style_fig(fig, height=500)
    ui.add_play(fig, [s[0] for s in snaps], duration=700, prefix="التكرار: ", redraw=True)
    ui.show(fig)
    st.caption(f"رأس المال في الحالة المستقرة k* = {kss:.3f}. الشبكة نقاط 160 لبُعد واحد؛ لنموذج بعشرين بُعدًا سنحتاج 160²⁰ نقطة — ومن هنا الحاجة إلى الشبكات العصبية.")

with tabs[3]:
    ui.section("التقدير العصبي للمعلمات الهيكلية", "Neural estimation of structural parameters", "teal")
    st.mermaid_chart(
        """
flowchart LR
    P[Draw theta from prior range] --> S[Simulate dataset from structural model]
    S --> M[Summaries / moments or raw data]
    M --> N[Train neural net: data -> theta]
    N --> E[Apply trained net to REAL data]
    E --> T[theta-hat + uncertainty]
"""
    )
    ui.cards([
        {"title": "التقدير العصبي المباشر", "en": "Wei & Jiang (2025, Marketing Science)", "color": "teal",
         "body": "تُدرَّب شبكة على مجموعات بيانات مولّدة من النموذج الهيكلي بقيم معلمات مختلفة لتتعلم الدالة العكسية «البيانات ← المعلمات»؛ تطبيق على نموذج بحث المستهلك بدقة أعلى وتكلفة أقل."},
        {"title": "المحاكيات العميقة", "en": "Deep surrogates / emulators", "color": "violet",
         "body": "شبكة تتعلم العلاقة θ ← لحظات النموذج، فتصبح محاكيًا سريعًا قابلًا للتفاضل داخل SMM أو بايز؛ يجب نقل عدم يقين التقريب إلى النتائج."},
        {"title": "الإمكان العصبي لنماذج HANK", "en": "Kase, Melosi & Rottner (2025, BIS WP)", "color": "orange",
         "body": "شبكة تحل وتقدّر نموذج HANK لا خطي مع الحد الأدنى الصفري للفائدة؛ تقدير بالإمكان للبيانات الأمريكية."},
        {"title": "الذكاء الاصطناعي للتقدير الهيكلي", "en": "Duarte & Fonseca (2026, NBER 35283)", "color": "rose",
         "body": "تقريب علاقة المعلمات باللحظات بشبكات مدربة على المحاكاة، ومعاملة أسعار التوازن كمعلمات مساعدة مع شروط تصفية السوق."},
    ], cols=2)
    ui.note("**الاستدلال القائم على المحاكاة (Simulation-based inference)** في الإحصاء الحديث (Neural posterior estimation) هو الإطار العام لهذه الأفكار؛ "
            "التحدي القياسي: التعرّف، وعدم اليقين، والأداء عند سوء المواصفة.", "الإطار العام", "teal", "🧭")

with tabs[4]:
    ui.section("مختبر: التقدير التنافسي على طريقة GAN", "Adversarial estimation — Kaji, Manresa & Pouliot (2023, Econometrica)", "orange")
    st.markdown("**الفكرة**: مولّد = النموذج الهيكلي يحاكي بيانات بمعلمة $\\theta$؛ مميّز = مصنّف يحاول التفريق بين البيانات الحقيقية والمحاكاة. "
                "نختار $\\theta$ التي **تخدع المميّز أكثر**. في المختبر: البيانات الحقيقية من توزيع بمعلمة شكل $\\theta_0=1.5$. "
                "شغّل الحركة لترى تداخل التوزيعين ودقة المميّز (انحدار لوجستي على $x, x^2, \\log x$) — أدناها عند القيمة الحقيقية.")

    @st.cache_data(show_spinner=False)
    def adversarial(n: int = 3000):
        rng = np.random.default_rng(5)
        real = rng.gamma(1.5, 1.0, n)
        thetas = np.round(np.arange(0.6, 2.61, 0.1), 2)
        acc, hists = [], []
        from sklearn.linear_model import LogisticRegression
        for th in thetas:
            sim = rng.gamma(th, 1.0, n)
            X = np.r_[real, sim]; F = np.column_stack([X, X ** 2, np.log(X + 1e-6)])
            yb = np.r_[np.ones(n), np.zeros(n)]
            clf = LogisticRegression(max_iter=500).fit(F, yb)
            acc.append(clf.score(F, yb))
            hists.append(sim)
        return real, thetas, np.array(acc), hists

    real, thetas, acc, hists = adversarial()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("التوزيع الحقيقي مقابل المحاكى", "دقة المميّز عبر θ"))

    def adv_tr(i):
        return [go.Histogram(x=real, histnorm="probability density", xbins=dict(start=0, end=8, size=0.2), marker_color=ui.COLORS["slate"], opacity=.5, name="حقيقي θ₀=1.5"),
                go.Histogram(x=hists[i], histnorm="probability density", xbins=dict(start=0, end=8, size=0.2), marker_color=ui.COLORS["orange"], opacity=.55, name="محاكى"),
                go.Scatter(x=thetas[:i + 1], y=acc[:i + 1], mode="lines+markers", line=dict(color=ui.COLORS["violet"], width=3), name="دقة المميّز")]

    for j, t in enumerate(adv_tr(0)):
        fig.add_trace(t, row=1, col=1 if j < 2 else 2)
    fig.frames = [go.Frame(data=adv_tr(i), traces=[0, 1, 2], name=f"{thetas[i]:.1f}") for i in range(len(thetas))]
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(range=[0, 8], row=1, col=1); fig.update_xaxes(range=[0.5, 2.7], title="θ", row=1, col=2)
    fig.update_yaxes(range=[0.45, max(acc) + 0.03], title="الدقة", row=1, col=2)
    ui.style_fig(fig, height=480)
    ui.add_play(fig, [f"{t:.1f}" for t in thetas], duration=450, prefix="θ = ", redraw=True)
    ui.show(fig)
    st.caption(f"θ̂ (أدنى دقة للمميّز) = {thetas[np.argmin(acc)]:.1f}؛ الدقة 50% تعني أن المميّز لا يستطيع التفريق إطلاقًا.")
    st.latex(r"\hat\theta=\arg\min_{\theta}\ \max_{D\in\mathcal D}\ \frac1n\sum_i\log D(X_i)+\frac1m\sum_j\log\big(1-D(\tilde X_j^{\theta})\big)")

with tabs[5]:
    ui.section("البيانات غير المهيكلة داخل التقدير الهيكلي", "Structural estimation with unstructured data", "indigo")
    st.mermaid_chart(
        """
flowchart LR
    A[Macro data<br/>GDP, inflation, policy rate] --> C[State-space DSGE]
    B[FOMC transcripts] --> L[LLM-derived text series]
    L --> M[Non-core measurement block<br/>spike-and-slab prior]
    M --> C
    C --> R[Structural inference<br/>parameters, shocks, IRFs]
"""
    )
    st.markdown(
        """
**Casella, Fernández-Villaverde, Hansen, Oishi & Shin (2026)** يضيفون سلاسل نصية مستخرجة من محاضر لجنة السوق المفتوحة الفدرالية كـ **كتلة قياس إضافية**
في نموذج DSGE بصيغة فضاء الحالة، مع قبْلي **spike-and-slab** لاختيار السلاسل النصية المفيدة. النتيجة وفق مواصفات الدراسة: تغيّر الاستدلال الهيكلي حول استجابة
الفائدة للتضخم ولزوجة الأسعار والفهرسة. **الدرس المنهجي**: أعمق أشكال الدمج ليس «DSGE + تنبؤ عصبي»، بل جعل عملية القياس بالذكاء الاصطناعي **جزءًا من معادلات المشاهدة**
مع انتشار عدم يقينها إلى النتائج النهائية.
        """
    )
    st.latex(r"\underbrace{\begin{bmatrix}y_t^{macro}\\ z_t^{text}\end{bmatrix}}_{\text{observables}}=\begin{bmatrix}H_{core}\\ \Lambda\odot\gamma\end{bmatrix}s_t+e_t,\qquad s_t=F(\theta)s_{t-1}+G(\theta)\varepsilon_t,\qquad \gamma_j\sim\text{spike-and-slab}")

with tabs[6]:
    ui.section("قراءة اقتصادية للذكاء الاصطناعي نفسه", "AI as structural estimation — Igami (2020, Econometrics Journal)", "emerald")
    ig = pd.DataFrame([
        ("Deep Blue (الشطرنج)", "دالة قيمة مُعايرة (Calibrated)", "معايرة يدوية لمعلمات دالة التقييم"),
        ("Bonanza (الشوغي)", "دالة قيمة مُقدَّرة", "طريقة النقطة الثابتة المتداخلة NFXP (Rust, 1987)"),
        ("AlphaGo — شبكة السياسة", "احتمالات الاختيار الشرطية", "مقدّر CCP لهوتز وميلر (1993)"),
        ("AlphaGo — شبكة القيمة", "محاكاة الاختيار الشرطي", "Hotz, Miller, Sanders & Smith (1994)"),
    ], columns=["نظام الذكاء الاصطناعي", "المقابل الاقتصادي", "الطريقة القياسية"])
    st.table(ig)
    ui.note("يرى إيغامي أن أنظمة الألعاب تحمل **افتراضات قياسية ضمنية**، وأن إدراج التفاعلات الاستراتيجية وعدم التجانس غير المشاهد صراحة سيزيد قابليتها للتفسير. "
            "وعلّق نيوي بأن تعلم الآلة يتفوق في التنبؤ لكن ليس بالضرورة في تقدير الكميات محل الاهتمام، واقترح **اللحظات المصححة للتحيز** — وهي فكرة DML بعينها.",
            "الخلاصة", "emerald", "♟️")
    ui.refs("Rust (1987) Econometrica; Hotz & Miller (1993) REStud; Maliar, Maliar & Winant (2021) JME; Azinovic, Gaegauf & Scheidegger (2022) IER; "
            "Fernández-Villaverde, Hurtado & Nuño (2023) Econometrica; Kaji, Manresa & Pouliot (2023) Econometrica; Igami (2020) Econometrics J.; Wei & Jiang (2025) Marketing Science.")
