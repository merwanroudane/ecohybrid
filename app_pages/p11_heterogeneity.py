import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from core import ui

ui.hero(
    "الأثر غير المتجانس وتعلم السياسات",
    "Heterogeneous Treatment Effects & Policy Learning",
    "الأثر المتوسط يخفي من يستفيد ومن يتضرر. هنا يأتي أقوى إسهامات تعلم الآلة للاقتصاد: الأشجار والغابات السببية، المتعلمات الوصفية "
    "(S/T/X/R/DR)، الاستدلال العام على عدم التجانس (GenericML)، وتعلم قواعد الاستهداف المثلى.",
    "emerald", "fuchsia",
)

tabs = st.tabs([
    ":material/diversity_3: مفهوم CATE",
    ":material/park: الأشجار والغابات السببية",
    ":material/school: المتعلمات الوصفية",
    ":material/science: مختبر CATE المتحرك",
    ":material/bar_chart: GATES و BLP",
    ":material/policy: تعلم السياسات",
    ":material/casino: المقاربات البايزية",
])


@st.cache_data(show_spinner="توليد البيانات وتدريب المتعلمات...")
def cate_lab():
    rng = np.random.default_rng(11)
    tau = lambda x: 1 + 2 / (1 + np.exp(-4 * x[:, 0])) - 0.8 * (x[:, 1] > 0.5)
    out = {}
    grid = np.column_stack([np.linspace(-2, 2, 120), np.zeros(120), np.zeros((120, 3))])
    for n in (500, 1000, 2000, 4000):
        X = rng.normal(0, 1, (n, 5))
        e = 1 / (1 + np.exp(-0.8 * X[:, 0]))
        D = rng.binomial(1, e)
        mu0 = X[:, 0] + np.sin(2 * X[:, 2])
        Y = mu0 + D * tau(X) + rng.normal(0, 1, n)
        rf = lambda: RandomForestRegressor(n_estimators=150, min_samples_leaf=10, random_state=0, n_jobs=-1)
        s = rf().fit(np.column_stack([X, D]), Y)
        s_cate = s.predict(np.column_stack([grid, np.ones(120)])) - s.predict(np.column_stack([grid, np.zeros(120)]))
        m1, m0 = rf().fit(X[D == 1], Y[D == 1]), rf().fit(X[D == 0], Y[D == 0])
        t_cate = m1.predict(grid) - m0.predict(grid)
        # DR-learner with 2-fold cross-fitting
        idx = rng.permutation(n); f = np.array_split(idx, 2); pseudo = np.empty(n)
        for k in range(2):
            te, tr = f[k], f[1 - k]
            ps = RandomForestClassifier(n_estimators=150, min_samples_leaf=20, random_state=0, n_jobs=-1).fit(X[tr], D[tr]).predict_proba(X[te])[:, 1]
            ps = np.clip(ps, 0.05, 0.95)
            a1 = rf().fit(X[tr][D[tr] == 1], Y[tr][D[tr] == 1]).predict(X[te])
            a0 = rf().fit(X[tr][D[tr] == 0], Y[tr][D[tr] == 0]).predict(X[te])
            pseudo[te] = a1 - a0 + D[te] * (Y[te] - a1) / ps - (1 - D[te]) * (Y[te] - a0) / (1 - ps)
        dr = RandomForestRegressor(n_estimators=200, min_samples_leaf=25, random_state=0, n_jobs=-1).fit(X, pseudo)
        out[n] = dict(S=s_cate, T=t_cate, DR=dr.predict(grid), X=X, D=D, Y=Y, tau=tau(X), dr_hat=dr.predict(X))
    return grid[:, 0], tau(grid), out


with tabs[0]:
    ui.section("من ATE إلى CATE", "From average to conditional effects", "emerald")
    st.latex(r"\tau(x)=\mathbb E\big[Y(1)-Y(0)\mid X=x\big],\qquad \text{ATE}=\mathbb E[\tau(X)],\qquad \text{GATE}_g=\mathbb E[\tau(X)\mid G=g]")
    ui.cards([
        {"title": "لماذا عدم التجانس؟", "en": "Why heterogeneity?", "color": "emerald",
         "body": "الاستهداف (من نعالج؟)، فهم الآليات، الصدق الخارجي (نقل النتائج لمجتمع آخر)، العدالة التوزيعية."},
        {"title": "المشكلة الكلاسيكية", "en": "Classical approach", "color": "amber",
         "body": "تفاعلات D×X يختارها الباحث يدويًا ⇒ خطر التنقيب (Fishing) وتعدد الاختبارات وفقدان التفاعلات غير المتوقعة."},
        {"title": "حل تعلم الآلة", "en": "ML solution", "color": "violet",
         "body": "اكتشاف عدم التجانس آليًا مع <b>أمانة</b> (Honesty) واستدلال صالح وتقسيم للعينة يمنع التنقيب."},
    ])
    ui.note("**المشكلة الأساسية للاستدلال السببي**: $\\tau_i=Y_i(1)-Y_i(0)$ غير مشاهد أبدًا لفرد واحد، فلا يمكن «تدريب» نموذج على الأثر مباشرة "
            "كما في التعلم الموجَّه. كل الطرق التالية حيل ذكية لتجاوز ذلك: نماذج منفصلة، دوال تحويل (Pseudo-outcomes)، أو معايير تقسيم معدّلة.",
            "لماذا هذا صعب؟", "rose", "🧩")

with tabs[1]:
    ui.section("الأشجار السببية الأمينة والغابات السببية", "Honest causal trees & causal forests", "teal")
    c1, c2 = st.columns(2)
    with c1:
        ui.note("نقسم العينة إلى جزأين: جزء **لبناء** الشجرة (اختيار التقسيمات)، وجزء مستقل **لتقدير** الأثر داخل كل ورقة. "
                "معيار التقسيم يعظّم **تباين الأثر** بين الأوراق مع عقوبة على عدم الدقة، لا دقة التنبؤ بـ Y.",
                "الشجرة السببية (Athey & Imbens, 2016)", "teal", "🌳")
    with c2:
        ui.note("متوسط آلاف الأشجار السببية الأمينة على عينات فرعية. أثبت **فاغر وآثي (2018)** أن تقديراتها متسقة وطبيعية تقاربيًا ⇒ فترات ثقة للأثر "
                "عند كل نقطة $x$. ثم عمّمها **GRF** لأي معلمة معرّفة بشرط لحظي محلي.", "الغابة السببية (Wager & Athey, 2018)", "violet", "🌲")
    st.latex(r"\hat\tau(x)=\frac{\sum_i \alpha_i(x)\,(D_i-\hat e(X_i))(Y_i-\hat m(X_i))}{\sum_i \alpha_i(x)\,(D_i-\hat e(X_i))^2},\qquad "
             r"\alpha_i(x)=\frac1B\sum_b\frac{\mathbf 1\{X_i\in L_b(x)\}}{|L_b(x)|}")
    st.markdown("الأوزان $\\alpha_i(x)$ = مدى تكرار وقوع المشاهدة $i$ في الورقة نفسها مع $x$ عبر الأشجار ⇒ **الغابة مقدّر محلي تكيّفي**. "
                "والتوجيه المسبق (Orthogonalization) بـ $\\hat e$ و $\\hat m$ يجعلها متينة للإرباك عبر $X$.")
    ui.cards([
        {"title": "Quantile forests", "en": "Meinshausen (2006)", "color": "sky", "body": "كمّيات شرطية — التأثير على التوزيع."},
        {"title": "Instrumental forests", "en": "GRF", "color": "orange", "body": "أثر محلي غير متجانس بمتغير أداتي (LATE(x))."},
        {"title": "Local linear forests", "en": "Friedberg et al. (2021)", "color": "rose", "body": "تصحيح انحياز الحواف والاستقراء الملساء."},
        {"title": "Causal survival forests", "en": "Cui et al. (2023)", "color": "teal", "body": "آثار على مدد البقاء مع الرقابة (Censoring)."},
    ], cols=4)

with tabs[2]:
    ui.section("المتعلمات الوصفية: أي خوارزمية تصبح مقدّرًا للأثر", "Meta-learners", "violet")
    ml = pd.DataFrame([
        ("S-learner", "نموذج واحد μ(x,d) مع D كمتغير", "τ̂ = μ̂(x,1) − μ̂(x,0)", "بسيط؛ قد يتجاهل D مع التنظيم (تحيز نحو الصفر)"),
        ("T-learner", "نموذجان μ₁ و μ₀ منفصلان", "τ̂ = μ̂₁(x) − μ̂₀(x)", "ضعيف عند عدم توازن المجموعات؛ ضجيج مزدوج"),
        ("X-learner", "يتعلم الآثار المنسوبة لكل مجموعة ثم يمزج بدرجة الميل", "τ̂ = e(x)τ̂₀(x)+(1−e(x))τ̂₁(x)", "ممتاز عندما تكون مجموعة صغيرة (Künzel et al., 2019)"),
        ("R-learner", "يدني خسارة روبنسون بعد إزالة الأثر", "min Σ[(Y−m̂)−τ(X)(D−ê)]²", "متعامد؛ أساس الغابة السببية (Nie & Wager, 2021)"),
        ("DR-learner", "انحدار للمتغير الزائف AIPW على X", "φ = μ̂₁−μ̂₀ + D(Y−μ̂₁)/ê − (1−D)(Y−μ̂₀)/(1−ê)", "مزدوج المتانة مع توفيق متقاطع (Kennedy, 2023)"),
    ], columns=["المتعلم", "الفكرة", "الصيغة", "الخصائص"])
    st.table(ml)
    st.mermaid_chart(
        """
flowchart LR
    A[Data Y, D, X] --> B{Meta-learner}
    B --> S[S: one model mu x,d]
    B --> T[T: mu1 and mu0]
    B --> X[X: imputed effects + propensity mix]
    B --> R[R: residual-on-residual loss]
    B --> DR[DR: AIPW pseudo-outcome regression]
    S & T & X & R & DR --> C[tau-hat x]
"""
    )

with tabs[3]:
    ui.section("مختبر متحرك: S و T و DR مقابل الأثر الحقيقي", "Animated CATE lab", "pink")
    st.markdown("الأثر الحقيقي $\\tau(x)$ دالة لا خطية في $x_1$ (منحنى لوجستي)، والمعالجة غير عشوائية (تعتمد على $x_1$). "
                "شغّل الحركة لترى كيف تتحسن تقديرات المتعلمات مع زيادة حجم العينة، ولاحظ انكماش **S-learner** نحو الصفر.")
    xg, tg, lab = cate_lab()
    ns = list(lab.keys())

    def cate_tr(n):
        return [go.Scatter(x=xg, y=tg, mode="lines", line=dict(color="#111827", width=4, dash="dot"), name="الأثر الحقيقي τ(x)"),
                go.Scatter(x=xg, y=lab[n]["S"], mode="lines", line=dict(color=ui.COLORS["amber"], width=3), name="S-learner"),
                go.Scatter(x=xg, y=lab[n]["T"], mode="lines", line=dict(color=ui.COLORS["sky"], width=3), name="T-learner"),
                go.Scatter(x=xg, y=lab[n]["DR"], mode="lines", line=dict(color=ui.COLORS["fuchsia"], width=4), name="DR-learner")]

    fig = go.Figure(data=cate_tr(ns[0]), frames=[go.Frame(data=cate_tr(n), name=str(n)) for n in ns])
    fig.update_xaxes(title="x₁ (مع تثبيت بقية المتغيرات عند 0)"); fig.update_yaxes(title="τ̂(x)", range=[-0.5, 3.8])
    ui.style_fig(fig, "تقدير الأثر الشرطي بثلاثة متعلمات وصفية (غابات عشوائية)")
    ui.add_play(fig, ns, duration=1100, prefix="n = ")
    ui.show(fig)

with tabs[4]:
    ui.section("الاستدلال العام على عدم التجانس", "GenericML: BLP, GATES, CLAN — Chernozhukov, Demirer, Duflo & Fernández-Val (2025)", "orange")
    st.markdown(
        """
بدل الادعاء بأن $\\hat\\tau(x)$ دقيق لكل فرد (وهو صعب)، نسأل أسئلة **قابلة للاستدلال الصالح** عن **ملامح** عدم التجانس باستخدام أي متعلم:
- **BLP** (أفضل إسقاط خطي): $\\mathbb E[\\tau(X)\\mid S(X)]\\approx\\beta_1+\\beta_2\\,(S(X)-\\bar S)$ — اختبار $\\beta_2\\neq0$ = هل يلتقط المتعلم عدم تجانس حقيقيًا؟
- **GATES** (آثار المجموعات المرتبة): نقسم العينة إلى خمس مجموعات حسب $S(X)$ ونقدّر الأثر في كل منها.
- **CLAN** (التحليل الخطي الشرطي): ما خصائص الأكثر والأقل تأثرًا؟
مع **تقسيم متكرر** للعينة وأخذ **وسيط** القيم الاحتمالية وفترات الثقة.
        """
    )
    n = 4000
    d = lab[n]
    q = pd.qcut(d["dr_hat"], 5, labels=False)
    est, true = [], []
    for g in range(5):
        m = q == g
        Y, D = d["Y"][m], d["D"][m]
        est.append(Y[D == 1].mean() - Y[D == 0].mean())
        true.append(d["tau"][m].mean())
    fig = go.Figure([go.Bar(x=[f"G{g + 1}" for g in range(5)], y=true, name="الأثر الحقيقي في المجموعة", marker_color=ui.COLORS["violet"]),
                     go.Bar(x=[f"G{g + 1}" for g in range(5)], y=est, name="فرق المتوسطات الخام في المجموعة", marker_color=ui.COLORS["orange"])])
    fig.update_xaxes(title="المجموعات مرتبة حسب τ̂ من DR-learner (G1 أدنى، G5 أعلى)"); fig.update_yaxes(title="الأثر")
    ui.style_fig(fig, "توضيح GATES من المختبر (n = 4000) — الخام متحيز لأن المعالجة غير عشوائية", height=430)
    ui.show(fig)
    ui.note("في التجارب العشوائية يكون فرق المتوسطات داخل المجموعة غير متحيز؛ هنا المعالجة تعتمد على $x_1$ فيظهر تحيز في العمود البرتقالي — "
            "لذلك يُطبّق GenericML في الأصل على **التجارب العشوائية** أو مع أوزان درجة الميل. التدرج الصاعد في البنفسجي يؤكد أن المتعلم رتّب الأفراد جيدًا.",
            "ملاحظة منهجية", "orange", "📌")

with tabs[5]:
    ui.section("من التقدير إلى القرار: تعلم قواعد المعالجة", "Policy learning", "rose")
    st.latex(r"\hat\pi=\arg\max_{\pi\in\Pi}\ \frac1n\sum_i\big(2\pi(X_i)-1\big)\hat\Gamma_i,\qquad \hat\Gamma_i=\text{AIPW score}")
    ui.cards([
        {"title": "تعظيم الرفاهية التجريبية", "en": "EWM — Kitagawa & Tetenov (2018, Econometrica)", "color": "rose",
         "body": "اختيار قاعدة من فئة بسيطة (عتبات، أشجار ضحلة) تعظّم الرفاهية المقدّرة مع ضمانات أسوأ حالة على الندم (Regret)."},
        {"title": "تعلم السياسات بالمشاهدات", "en": "Athey & Wager (2021, Econometrica)", "color": "violet",
         "body": "استخدام دوال AIPW المزدوجة المتانة مع شروط ميزانية؛ معدلات ندم مثلى بمعدل جذر n."},
        {"title": "أشجار السياسات", "en": "policytree (R)", "color": "teal",
         "body": "قواعد قابلة للتفسير: «عالج إن كان العمر < 30 والدخل < X» — مهمة للمؤسسات."},
        {"title": "القيود والعدالة", "en": "Budget & fairness constraints", "color": "amber",
         "body": "قيود ميزانية، حدود على التفاوت بين المجموعات، شفافية القاعدة."},
    ], cols=2)
    ui.section("محاكاة استهداف: عالج إذا تجاوز الأثر المتوقع التكلفة", "Interactive targeting", "pink")
    cost = st.slider("تكلفة المعالجة للفرد (بوحدات النتيجة)", 0.0, 3.0, 1.6, 0.1, key="pol_cost")
    tau_hat, tau_true = d["dr_hat"], d["tau"]
    treat = tau_hat > cost
    w_target = np.mean(np.where(treat, tau_true - cost, 0))
    w_all = np.mean(tau_true - cost)
    c1, c2, c3 = st.columns(3)
    c1.metric("نسبة المعالَجين", f"{treat.mean():.0%}")
    c2.metric("الرفاهية الصافية: استهداف", f"{w_target:.3f}")
    c3.metric("الرفاهية الصافية: معالجة الجميع", f"{w_all:.3f}", f"{w_target - w_all:+.3f} لصالح الاستهداف", delta_color="normal")
    order = np.argsort(-tau_hat)
    frac = np.arange(1, len(order) + 1) / len(order)
    gain = np.cumsum(tau_true[order] - cost) / len(order)
    oracle = np.cumsum(np.sort(tau_true - cost)[::-1]) / len(order)
    fig = go.Figure([go.Scatter(x=frac, y=gain, name="حسب ترتيب τ̂ (DR-learner)", line=dict(color=ui.COLORS["fuchsia"], width=4)),
                     go.Scatter(x=frac, y=oracle, name="الترتيب الأمثل (العرّاف)", line=dict(color=ui.COLORS["slate"], dash="dot")),
                     go.Scatter(x=frac, y=frac * np.mean(tau_true - cost), name="عشوائي", line=dict(color=ui.COLORS["amber"], dash="dash"))])
    fig.update_xaxes(title="نسبة السكان المعالَجين (الأعلى أثرًا متوقعًا أولًا)", tickformat=".0%"); fig.update_yaxes(title="الرفاهية الصافية للفرد")
    ui.style_fig(fig, "منحنى الكسب من الاستهداف (Uplift / Qini-type curve)", height=430)
    ui.show(fig)

with tabs[6]:
    ui.section("المقاربات البايزية لعدم التجانس", "BART & Bayesian causal forests", "indigo")
    ui.cards([
        {"title": "BART", "en": "Chipman, George & McCulloch (2010)", "color": "indigo",
         "body": "مجموع أشجار صغيرة بقبْليات تمنع الإفراط؛ استدلال بايزي كامل عبر MCMC. أداؤه السببي قوي في مسابقات ACIC."},
        {"title": "BCF", "en": "Hahn, Murray & Carvalho (2020)", "color": "violet",
         "body": "يفصل μ(x) عن τ(x) بقبْليات مختلفة ويضمّن درجة الميل كمتغير ⇒ يعالج «الإرباك المنظَّم بالتنظيم» (RIC)."},
        {"title": "BART في السلاسل الزمنية", "en": "Huber & Rossini; Clark et al. (2023)", "color": "teal",
         "body": "BAVART و BART-VAR متعدد المتغيرات للتنبؤ بالذيول والتنبؤ الآني (انظر صفحة هجين السلاسل الزمنية)."},
    ])
    ui.refs("Athey & Imbens (2016) PNAS; Wager & Athey (2018) JASA; Athey, Tibshirani & Wager (2019) AoS; Künzel et al. (2019) PNAS; Nie & Wager (2021) Biometrika; "
            "Kennedy (2023) EJS; Kitagawa & Tetenov (2018) Econometrica; Athey & Wager (2021) Econometrica; Hahn, Murray & Carvalho (2020) Bayesian Analysis.")
