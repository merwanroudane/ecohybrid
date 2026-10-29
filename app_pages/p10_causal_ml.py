import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from core import ui

ui.hero(
    "الاستدلال السببي بتعلم الآلة",
    "Causal Machine Learning — PDS-Lasso, Double/Debiased ML, AIPW, Auto-DML",
    "قلب الموجة الثالثة: كيف نستخدم متعلمات مرنة لتقدير الأجزاء المساعدة مع الحفاظ على تقدير غير متحيز وفترات ثقة صالحة "
    "للمعلمة السببية — مع مختبر محاكاة متحرك يبيّن لماذا يفشل الإدخال الساذج وكيف ينجح DML.",
    "indigo", "rose",
)

tabs = st.tabs([
    ":material/account_tree: الإطار السببي",
    ":material/report: لماذا يفشل الساذج؟",
    ":material/filter_alt: الاختيار المزدوج",
    ":material/functions: نظرية DML",
    ":material/science: مختبر DML المتحرك",
    ":material/sync_problem: مختبر التوفيق المتقاطع",
    ":material/shield: المقدّرات المزدوجة المتانة",
    ":material/new_releases: الامتدادات الحديثة",
])

with tabs[0]:
    ui.section("النتائج المحتملة والرسوم السببية", "Potential outcomes & DAGs", "indigo")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            r"""
**إطار نيمان–روبن (Potential outcomes):** لكل فرد نتيجتان محتملتان $Y_i(1), Y_i(0)$ ونرى واحدة فقط.

$$\text{ATE}=\mathbb E[Y(1)-Y(0)],\quad \text{ATT}=\mathbb E[Y(1)-Y(0)\mid D=1],\quad \tau(x)=\mathbb E[Y(1)-Y(0)\mid X=x]$$

**افتراضات التعرّف (الاختيار على المشاهدات):**
- عدم الإرباك: $\{Y(1),Y(0)\}\perp D\mid X$
- التداخل/الإيجابية: $0<\Pr(D=1\mid X)<1$
- SUTVA: لا تداخل بين الوحدات.
"""
        )
    with c2:
        st.mermaid_chart(
            """
flowchart TB
    X[X: high-dimensional confounders] --> D[D: treatment]
    X --> Y[Y: outcome]
    D -->|theta: causal effect| Y
    U[U: unobserved] -.-> D
    U -.-> Y
    Z[Z: instrument] --> D
"""
        )
    ui.note("**دور تعلم الآلة هنا محدد بدقة**: عندما يكون $X$ عالي الأبعاد أو تكون العلاقات لا خطية، نحتاج إلى تقدير $\\mathbb E[Y\\mid X]$ و $\\mathbb E[D\\mid X]$ بمرونة. "
            "أما عدم الإرباك نفسه فهو **افتراض تصميم** لا تثبته أي خوارزمية؛ وإن وُجد $U$ غير مشاهد نحتاج أداة $Z$ أو تصميمًا آخر.", "تقسيم الأدوار", "indigo", "🧭")

with tabs[1]:
    ui.section("مصدرا التحيز عند الإدخال الساذج لتعلم الآلة", "Regularization bias & overfitting bias", "rose")
    st.markdown("في النموذج شبه الخطي (Robinson, 1988):")
    st.latex(r"Y=\theta_0 D+g_0(X)+U,\qquad D=m_0(X)+V,\qquad \mathbb E[U\mid X,D]=0,\ \mathbb E[V\mid X]=0")
    st.markdown("لو قدّرنا $\\hat g$ بتعلم الآلة ثم حللنا $\\hat\\theta=\\big(\\tfrac1n\\sum D_i^2\\big)^{-1}\\tfrac1n\\sum D_i\\,(Y_i-\\hat g(X_i))$ مباشرة، فإن:")
    st.latex(r"\sqrt n(\hat\theta-\theta_0)=\underbrace{a^*}_{\to N(0,\Sigma)}+\underbrace{\big(\mathbb E D^2\big)^{-1}\frac{1}{\sqrt n}\sum_i m_0(X_i)\big(g_0(X_i)-\hat g(X_i)\big)}_{b^*:\ \text{regularization bias}}+o_P(1)")
    st.markdown(
        """
الحد $b^*$ **ينفجر**: لأن $\\hat g$ يتقارب أبطأ من $n^{-1/2}$ (ثمن المرونة والتنظيم)، فإن $\\sqrt n\\,\\mathbb E|g_0-\\hat g|\\to\\infty$.

**العلاج المزدوج (Chernozhukov et al., 2018):**
1. **التعامد (Orthogonalization)**: نُزيل أثر $X$ من $D$ أيضًا (باقي $\\hat V=D-\\hat m(X)$)، فيصبح الحد المتحيز **حاصل ضرب** خطأين: $\\sqrt n\\,\\|\\hat m-m_0\\|\\,\\|\\hat g-g_0\\|\\to 0$ إذا كان كل منهما $o(n^{-1/4})$.
2. **التوفيق المتقاطع (Cross-fitting)**: يزيل **تحيز الإفراط في التوفيق** الناتج عن استخدام المشاهدات نفسها في تعلم $\\hat\\eta$ وتقييمها.
        """
    )
    ui.chips([("Regularization bias", "تحيز التنظيم"), ("Overfitting bias", "تحيز الإفراط في التوفيق"), ("Partialling out", "إزالة الأثر"),
              ("Frisch–Waugh–Lovell", "مبرهنة فريش–وو–لوفيل"), ("Product of rates", "حاصل ضرب المعدلات"), ("√n-consistency", "اتساق بمعدل جذر n")])

with tabs[2]:
    ui.section("الاستدلال بعد الاختيار المزدوج", "Post-double-selection — Belloni, Chernozhukov & Hansen (2014)", "teal")
    ui.ladder([
        ("الاختيار الأول", "Lasso of Y on X", "اختر المتغيرات $\\hat I_1$ التي تتنبأ بالنتيجة $Y$."),
        ("الاختيار الثاني", "Lasso of D on X", "اختر المتغيرات $\\hat I_2$ التي تتنبأ بالمعالجة $D$ — هذه الخطوة هي مفتاح الحل!"),
        ("الانحدار النهائي", "OLS of Y on D and X[Î₁ ∪ Î₂]", "قدّر $\\theta$ بـ OLS على اتحاد المجموعتين، مع أخطاء معيارية متينة اعتيادية."),
    ])
    ui.note("**لماذا يفشل الاختيار المفرد؟** متغير مربك يؤثر بقوة في $D$ وبضعف في $Y$ سيُستبعد من معادلة $Y$ وحدها ⇒ **تحيز متغير محذوف** "
            "لا يزول تقاربيًا ويجعل فترات الثقة باطلة. الاختيار الثاني «ينقذ» هذه المتغيرات. وهو حالة خاصة من مقدّرات **التعامد** في DML.",
            "الفكرة المركزية", "teal", "💡")
    st.markdown("**التطبيقات الشهيرة في الورقة الأصلية:** أثر الإجهاض على الجريمة (إعادة تحليل Donohue & Levitt)، والتقارب في النمو الاقتصادي بين الدول (Barro–Lee) مع عشرات الضوابط.")

with tabs[3]:
    ui.section("نظرية Double/Debiased Machine Learning", "Neyman orthogonality, scores and the algorithm", "violet")
    st.markdown("**المبدأ العام**: نعرّف $\\theta_0$ كحل لشرط لحظي $\\mathbb E[\\psi(W;\\theta_0,\\eta_0)]=0$ يحقق **تعامد نيمان**:")
    st.latex(r"\partial_r\,\mathbb E\big[\psi\big(W;\theta_0,\eta_0+r(\eta-\eta_0)\big)\big]\Big|_{r=0}=0")
    st.markdown("أي أن أخطاء صغيرة في الدوال المساعدة $\\eta$ لا تؤثر في المعادلة من الدرجة الأولى. أهم الدوال:")
    scores = pd.DataFrame([
        ("PLR — النموذج شبه الخطي", r"(Y-\ell(X)-\theta(D-m(X)))(D-m(X))", "ℓ=E[Y|X], m=E[D|X]", "أثر معالجة مستمرة أو ثنائية بافتراض الخطية في D"),
        ("IRM — النموذج التفاعلي (AIPW)", r"g(1,X)-g(0,X)+\frac{D(Y-g(1,X))}{m(X)}-\frac{(1-D)(Y-g(0,X))}{1-m(X)}-\theta", "g(d,X)=E[Y|D=d,X], m=P(D=1|X)", "ATE لمعالجة ثنائية بأثر غير متجانس"),
        ("PLIV — شبه الخطي بأداة", r"(Y-\ell(X)-\theta(D-r(X)))(Z-m(X))", "ℓ, r=E[D|X], m=E[Z|X]", "معالجة داخلية مع أداة وضوابط كثيرة"),
        ("IIVM — LATE التفاعلي", "نسبة فرقي AIPW للنتيجة والمعالجة على الأداة", "g(z,X), r(z,X), m(X)", "أثر محلي للممتثلين"),
    ], columns=["النموذج", "الدالة ψ", "الدوال المساعدة η", "الاستخدام"])
    st.table(scores)
    st.markdown("**الخوارزمية (DML2):**")
    st.code("""1. Split sample into K folds  I_1, ..., I_K
2. For each k: fit nuisance  eta_hat_{-k} = (l_hat, m_hat)  with ANY ML learner on data NOT in I_k
3. Compute residuals on I_k:  V_hat = D - m_hat(X),  W_hat = Y - l_hat(X)
4. Solve pooled moment:  theta_hat = sum(V_hat * W_hat) / sum(V_hat * D)        (PLR)
5. Variance from the score:  sigma^2 = E[psi^2] / J^2  ->  CI = theta_hat +/- 1.96 * sigma / sqrt(n)
6. Repeat splits S times, report the median estimate (and median-adjusted SE)""", language="text")
    st.latex(r"\sqrt n(\hat\theta-\theta_0)\xrightarrow{d}N\!\Big(0,\ \frac{\mathbb E[\psi^2]}{J_0^2}\Big),\qquad J_0=\partial_\theta\mathbb E[\psi]")
    ui.note("**شروط النظرية**: (1) التعامد، (2) $\\|\\hat m-m_0\\|_2\\,\\|\\hat\\ell-\\ell_0\\|_2=o_P(n^{-1/2})$، (3) التداخل (Overlap) في IRM، (4) التوفيق المتقاطع. "
            "ليس مطلوبًا أن يكون المتعلم «صحيحًا» أو قابلًا للتفسير — فقط أن يتنبأ جيدًا بمعدل كافٍ.", "الشروط", "violet", "📐")

with tabs[4]:
    ui.section("مختبر متحرك: الساذج مقابل DML عبر أحجام العينة", "Monte-Carlo lab — naive vs. DML", "pink")
    st.markdown("النموذج: $Y=0.5\\,D+X\\beta+u$ و $D=X\\gamma+v$ مع $p=60$ ضابطًا. المرْبكات تؤثر **بقوة في $D$ وبضعف في $Y$**. "
                "نقارن: **OLS القصير** (بدون ضوابط)، **الاختيار المفرد بلاسو** (الساذج)، و **DML** (إزالة الأثر بلاسو + توفيق متقاطع). "
                "شغّل الحركة لترى توزيعات المقدّرات مع زيادة $n$: الساذج يبقى متحيزًا، و DML يتمركز حول القيمة الحقيقية 0.5.")
    conf = st.slider("قوة الإرباك عبر D", 0.5, 3.0, 1.5, 0.25, key="dml_conf")

    @st.cache_data(show_spinner="تشغيل محاكاة مونت كارلو...")
    def dml_sim(conf: float, reps: int = 80):
        rng = np.random.default_rng(2024)
        p, theta = 60, 0.5
        j = np.arange(1, p + 1)
        gamma = conf / j ** 1.2
        cov = 0.5 ** np.abs(np.subtract.outer(j, j)); Lc = np.linalg.cholesky(cov)
        out = {}
        for n in (200, 400, 800, 1600):
            res = {"OLS قصير": [], "اختيار مفرد (ساذج)": [], "DML": []}
            alpha = 0.6 * np.sqrt(np.log(p) / n)
            # direct effects of the confounders on Y shrink like 1/sqrt(n): always "hard to detect"
            beta = (3.5 / np.sqrt(n)) / j ** 1.2
            for _ in range(reps):
                X = rng.standard_normal((n, p)) @ Lc.T
                D = X @ gamma + rng.standard_normal(n)
                Y = theta * D + X @ beta + rng.standard_normal(n)
                res["OLS قصير"].append(np.polyfit(D, Y, 1)[0])
                # naive: one Lasso of Y on (D, X); keep only the X's it selects
                sel = np.flatnonzero(Lasso(alpha=alpha, max_iter=3000).fit(np.column_stack([D, X]), Y).coef_[1:])
                Z = np.column_stack([D, X[:, sel]]) if sel.size else D[:, None]
                res["اختيار مفرد (ساذج)"].append(LinearRegression().fit(Z, Y).coef_[0])
                idx = rng.permutation(n); folds = np.array_split(idx, 2)
                Vr, Wr = np.empty(n), np.empty(n)
                for k in range(2):
                    te, tr = folds[k], folds[1 - k]
                    Vr[te] = D[te] - Lasso(alpha=alpha, max_iter=3000).fit(X[tr], D[tr]).predict(X[te])
                    Wr[te] = Y[te] - Lasso(alpha=alpha, max_iter=3000).fit(X[tr], Y[tr]).predict(X[te])
                res["DML"].append((Vr @ Wr) / (Vr @ Vr))
            out[n] = {k: np.array(v) for k, v in res.items()}
        return out

    sim = dml_sim(conf)
    ns = list(sim.keys())
    cols = {"OLS قصير": ui.COLORS["amber"], "اختيار مفرد (ساذج)": ui.COLORS["rose"], "DML": ui.COLORS["teal"]}
    bins = dict(start=0.0, end=1.6, size=0.02)

    def hist_tr(n):
        return [go.Histogram(x=sim[n][k], name=k, marker_color=c, opacity=.65, xbins=bins) for k, c in cols.items()]

    fig = go.Figure(data=hist_tr(ns[0]), frames=[go.Frame(data=hist_tr(n), name=str(n)) for n in ns])
    fig.add_vline(x=0.5, line_dash="dash", line_color="#111827", annotation_text="θ₀ = 0.5")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="θ̂", range=[0.1, 1.6]); fig.update_yaxes(title="التكرار")
    ui.style_fig(fig, "توزيعات المقدّرات عبر محاكاة مونت كارلو")
    ui.add_play(fig, ns, duration=1100, prefix="n = ", redraw=True)
    ui.show(fig)
    rows = []
    for n in ns:
        for k in cols:
            v = sim[n][k]
            rows.append({"n": n, "المقدّر": k, "التحيز": v.mean() - 0.5, "الانحراف المعياري": v.std(),
                         "التحيز ÷ الانحراف": (v.mean() - 0.5) / v.std(), "RMSE": np.sqrt(((v - 0.5) ** 2).mean())})
    st.dataframe(pd.DataFrame(rows).style.format({"التحيز": "{:+.3f}", "الانحراف المعياري": "{:.3f}", "التحيز ÷ الانحراف": "{:+.2f}", "RMSE": "{:.3f}"}),
                 hide_index=True)
    ui.note("صُمّمت المحاكاة بحيث تكون الآثار المباشرة للمرْبكات على $Y$ «صعبة الالتقاط» (تتناقص بمعدل $1/\\sqrt n$) كما في الحجة النظرية لبيلوني وزملائه. "
            "لاحظ عمود **التحيز ÷ الانحراف المعياري**: يبقى كبيرًا للاختيار المفرد مهما كبرت $n$ ⇒ فترات ثقة لا تغطي القيمة الحقيقية؛ بينما يبقى قريبًا من الصفر لـ **DML**. "
            "ارفع «قوة الإرباك» لترى الفرق يتسع.", "القراءة", "pink", "📊")

with tabs[5]:
    ui.section("لماذا التوفيق المتقاطع ضروري؟", "Cross-fitting lab with an overfitting learner", "orange")
    st.markdown("نستخدم لمعادلة النتيجة متعلمًا **مفرطًا في التوفيق** عمدًا (k-NN بثلاثة جيران) ولمعادلة المعالجة متعلمًا أنعم (30 جارًا). "
                "بدون توفيق متقاطع، يحفظ المتعلم الأول جزءًا من ضجيج $Y_i$ نفسه — بما فيه $\\theta V_i$ — فتنكمش البواقي $\\hat W$ ويتحيز المقدّر نحو الصفر. "
                "مع التوفيق المتقاطع تُحسب البواقي على مشاهدات لم يرها المتعلم ⇒ يزول التحيز.")

    @st.cache_data(show_spinner="تشغيل المحاكاة...")
    def cf_sim(reps: int = 150, n: int = 600):
        rng = np.random.default_rng(7)
        res = {"بدون توفيق متقاطع": [], "مع توفيق متقاطع": []}
        for _ in range(reps):
            X = rng.uniform(-2, 2, (n, 2))
            m = np.sin(X[:, 0]) + X[:, 1] ** 2 / 2
            D = m + rng.standard_normal(n)
            Y = 0.5 * D + np.cos(X[:, 0]) + X[:, 1] + rng.standard_normal(n)
            ml_l = lambda: KNeighborsRegressor(n_neighbors=3)
            ml_m = lambda: KNeighborsRegressor(n_neighbors=30)
            V = D - ml_m().fit(X, D).predict(X); W = Y - ml_l().fit(X, Y).predict(X)
            res["بدون توفيق متقاطع"].append((V @ W) / (V @ V))
            idx = rng.permutation(n); f = np.array_split(idx, 2); V2, W2 = np.empty(n), np.empty(n)
            for k in range(2):
                te, tr = f[k], f[1 - k]
                V2[te] = D[te] - ml_m().fit(X[tr], D[tr]).predict(X[te]); W2[te] = Y[te] - ml_l().fit(X[tr], Y[tr]).predict(X[te])
            res["مع توفيق متقاطع"].append((V2 @ W2) / (V2 @ V2))
        return {k: np.array(v) for k, v in res.items()}

    cfs = cf_sim()
    fig = go.Figure([go.Violin(y=v, name=k, box_visible=True, meanline_visible=True, line_color=c, fillcolor=c, opacity=.55)
                     for (k, v), c in zip(cfs.items(), [ui.COLORS["rose"], ui.COLORS["teal"]])])
    fig.add_hline(y=0.5, line_dash="dash", annotation_text="θ₀ = 0.5")
    fig.update_yaxes(title="θ̂")
    ui.style_fig(fig, "متعلم مفرط في التوفيق: الفرق بين وجود التوفيق المتقاطع وغيابه", height=480)
    ui.show(fig)
    c1, c2 = st.columns(2)
    for col, (k, v) in zip((c1, c2), cfs.items()):
        col.metric(k, f"{v.mean():.3f}", f"تحيز {v.mean() - 0.5:+.3f}", delta_color="off")

with tabs[6]:
    ui.section("المقدّرات المزدوجة المتانة: AIPW و TMLE", "Doubly robust estimation", "emerald")
    st.latex(r"\hat\tau_{AIPW}=\frac1n\sum_i\Big[\hat g(1,X_i)-\hat g(0,X_i)+\frac{D_i\,(Y_i-\hat g(1,X_i))}{\hat e(X_i)}-\frac{(1-D_i)(Y_i-\hat g(0,X_i))}{1-\hat e(X_i)}\Big]")
    ui.cards([
        {"title": "المتانة المزدوجة", "en": "Double robustness (Robins, Rotnitzky & Zhao, 1994)", "color": "emerald",
         "body": "يكفي أن يكون <b>أحد</b> النموذجين صحيحًا (النتيجة أو درجة الميل) ليبقى المقدّر متسقًا."},
        {"title": "متانة المعدلات", "en": "Rate double robustness", "color": "teal",
         "body": "النسخة الحديثة مع ML: يكفي أن يكون حاصل ضرب أخطاء النموذجين صغيرًا — وهي نفسها دالة IRM في DML."},
        {"title": "TMLE", "en": "Targeted MLE (van der Laan & Rubin, 2006)", "color": "violet",
         "body": "يحدّث تقدير النتيجة في خطوة «استهداف» ليحل معادلة الدالة المؤثرة؛ يبقى داخل فضاء المعلمات. شائع في الاقتصاد الصحي."},
        {"title": "مشكلة التداخل", "en": "Limited overlap", "color": "rose",
         "body": "أوزان 1/ê تنفجر قرب الصفر أو الواحد ⇒ قصّ (Trimming) أو تغيير الهدف (ATO) وتشخيص توزيع درجات الميل."},
    ], cols=2)

with tabs[7]:
    ui.section("الامتدادات الحديثة لـ DML", "Frontier of debiased ML", "fuchsia")
    ui.cards([
        {"title": "Auto-DML وممثّل ريس", "en": "Chernozhukov, Newey & Singh (2022, Econometrica)", "color": "fuchsia",
         "body": "لأي دالة خطية للمعلمة (متوسط مشتقات، آثار سياسات)، يُتعلَّم «ممثّل ريس» α(X) آليًا دون اشتقاق صيغته يدويًا."},
        {"title": "GenericML", "en": "Chernozhukov, Demirer, Duflo & Fernández-Val (2025, Econometrica)", "color": "violet",
         "body": "استدلال صالح على ملامح عدم التجانس (BLP، GATES، CLAN) في التجارب العشوائية مع أي متعلم وتقسيم متكرر وتجميع بالوسيط."},
        {"title": "تحليل الحساسية", "en": "Omitted-variable bounds for causal ML", "color": "rose",
         "body": "Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022): حدود للتحيز إن وُجد مُربك غير مشاهد — بديل صادق لافتراض عدم الإرباك."},
        {"title": "DML بتمثيلات الذكاء الاصطناعي", "en": "Pragmatic DML with AI-learned representations (2026)", "color": "teal",
         "body": "Aradillas Fernandez, Chernozhukov, Cinelli, Newey وآخرون: التحيز ≈ خطأ التمثيل في معادلة النتيجة × خطأه في ممثّل ريس؛ تعلم التمثيل داخل الطيات، والتجميع المحدب."},
        {"title": "DML للمعالجات المستمرة والديناميكية", "en": "Colangelo & Lee; Lewis & Syrgkanis (2021)", "color": "orange",
         "body": "منحنيات الجرعة–الاستجابة، ومعالجات متتابعة عبر الزمن بتقدير g-estimation متعامد."},
        {"title": "DML للوساطة والانتقاء", "en": "Farbmacher et al. (2022); Bia, Huber & Lafférs (2023)", "color": "sky",
         "body": "الآثار المباشرة وغير المباشرة عبر وسيط، وتصحيح الانتقاء/التسرب في العينة — حزمة causalweight."},
    ], cols=2)
    ui.note("التطبيقات على **IV و DiD و RDD والبانل والسلاسل الزمنية** مشروحة بالتفصيل في صفحة «هجين البانل والسببية».", "انتقل إلى", "sky", "➡️")
    ui.refs("Robinson (1988); Belloni, Chernozhukov & Hansen (2014) REStud; Chernozhukov et al. (2018) Econometrics J.; Chernozhukov, Newey & Singh (2022) Econometrica; "
            "Chernozhukov et al. (2025) Econometrica; Chernozhukov, Hansen, Kallus, Spindler & Syrgkanis (2024) Applied Causal Inference Powered by ML and AI.")
