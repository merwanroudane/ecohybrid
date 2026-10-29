import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import lasso_path
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from core import ui

ui.hero(
    "الخوارزميات بعين قياسية",
    "Machine-Learning Algorithms Read Through an Econometric Lens",
    "لكل خوارزمية: الفكرة، الصيغة، النظير القياسي، متى نستخدمها في الاقتصاد، وما محاذيرها — مع رسوم متحركة تُظهر كيف «تتعلم» الخوارزمية خطوة بخطوة.",
    "violet", "orange",
)

tabs = st.tabs([
    ":material/linear_scale: الانحدار المنظَّم",
    ":material/account_tree: الأشجار",
    ":material/forest: الغابات والتجميع",
    ":material/rocket: التعزيز",
    ":material/blur_on: النواة و SVM و GP",
    ":material/hub: الشبكات العصبية",
    ":material/scatter_plot: غير الموجَّه",
    ":material/psychology_alt: المحوّلات والانتباه",
    ":material/table_chart: جدول المقارنة",
])

rng0 = np.random.default_rng(42)
X1 = np.sort(rng0.uniform(-3, 3, 220))
f1 = lambda x: np.sin(1.5 * x) + 0.3 * x
Y1 = f1(X1) + rng0.normal(0, .35, X1.size)
G1 = np.linspace(-3, 3, 300)

# ------------------------------------------------------------ lasso path -----
with tabs[0]:
    ui.section("مسار لاسو: كيف تدخل المتغيرات مع تخفيف العقوبة", "The LASSO path (animated)", "teal")

    @st.cache_data(show_spinner=False)
    def lpath():
        rng = np.random.default_rng(1)
        n, p = 200, 12
        X = rng.standard_normal((n, p)); X[:, 1] = .7 * X[:, 0] + .7 * X[:, 1]
        beta = np.array([2, 0, -1.5, 0, 1, 0, 0, .5, 0, 0, -.8, 0])
        y = X @ beta + rng.standard_normal(n)
        X = (X - X.mean(0)) / X.std(0)
        alphas, coefs, _ = lasso_path(X, y - y.mean(), n_alphas=40)
        return alphas, coefs, beta

    alphas, coefs, beta = lpath()
    labels = [f"x{i + 1}" for i in range(coefs.shape[0])]
    col = [ui.COLORS["teal"] if b != 0 else ui.COLORS["slate"] for b in beta]
    fig = go.Figure(
        data=[go.Bar(x=labels, y=coefs[:, 0], marker_color=col)],
        frames=[go.Frame(data=[go.Bar(x=labels, y=coefs[:, k], marker_color=col)], name=f"{alphas[k]:.3f}") for k in range(len(alphas))],
    )
    fig.update_yaxes(range=[-2.2, 2.4], title="المعامل"); fig.update_xaxes(title="المتغيرات (الأخضر = مؤثر فعلًا)")
    ui.style_fig(fig, "λ تتناقص ⇒ تدخل المتغيرات تباعًا", legend_h=False)
    ui.add_play(fig, [f"{a:.3f}" for a in alphas], duration=250, prefix="λ = ", redraw=True)
    ui.show(fig)
    c1, c2 = st.columns(2)
    with c1:
        st.latex(r"\hat\beta^{lasso}=\arg\min_\beta \tfrac{1}{2n}\|y-X\beta\|_2^2+\lambda\|\beta\|_1")
        st.markdown("**الهندسة**: قيد L1 على شكل معيّن (Diamond) له زوايا على المحاور ⇒ يلامس الحل المحور فيصبح المعامل صفرًا. أما L2 فدائري ⇒ انكماش بلا أصفار.")
    with c2:
        ui.note("- اختيار ضوابط عالية الأبعاد → **PDS / DML**\n- التنبؤ الكلي بمئات المؤشرات\n- لاسو-VAR ونماذج التكامل المشترك المنظّمة (SPECS)\n"
                "- **المحذور**: معاملات متحيزة نحو الصفر؛ الاختيار غير مستقر مع الترابط؛ الاستدلال الساذج بعد الاختيار باطل.",
                "الاستخدام الاقتصادي", "teal", "💼")

# --------------------------------------------------------------- trees -------
with tabs[1]:
    ui.section("شجرة الانحدار: تقسيم فضاء المتغيرات إلى مستطيلات", "CART — recursive binary partitioning", "orange")

    @st.cache_data(show_spinner=False)
    def tree_frames():
        rng = np.random.default_rng(3)
        X = rng.uniform(0, 1, (600, 2))
        y = np.where((X[:, 0] > .5) & (X[:, 1] > .4), 3, 0) + 2 * np.sin(4 * X[:, 0]) * (X[:, 1] < .4) + rng.normal(0, .4, 600)
        g = np.linspace(0, 1, 70); gx, gy = np.meshgrid(g, g); G = np.c_[gx.ravel(), gy.ravel()]
        out = []
        for d in range(1, 9):
            m = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X, y)
            out.append(m.predict(G).reshape(70, 70))
        return g, out, X, y

    g, tf, Xt_, yt_ = tree_frames()
    fig = go.Figure(
        data=[go.Heatmap(x=g, y=g, z=tf[0], colorscale="Sunset", zmin=-2, zmax=4, showscale=True)],
        frames=[go.Frame(data=[go.Heatmap(x=g, y=g, z=tf[d], colorscale="Sunset", zmin=-2, zmax=4)], name=str(d + 1)) for d in range(8)],
    )
    fig.update_xaxes(title="x₁"); fig.update_yaxes(title="x₂", scaleanchor="x")
    ui.style_fig(fig, "تنبؤات الشجرة حسب العمق: تتكاثر المستطيلات", height=560, legend_h=False)
    ui.add_play(fig, list(range(1, 9)), duration=800, prefix="العمق = ", redraw=True)
    ui.show(fig)
    st.markdown(
        """
**الخوارزمية (Breiman et al., 1984):** في كل عقدة نبحث عن المتغير $j$ ونقطة القطع $s$ اللذين يدنّيان مجموع مربعات البواقي:
$$\\min_{j,s}\\Big[\\sum_{x_i\\in R_1(j,s)}(y_i-\\bar y_{R_1})^2+\\sum_{x_i\\in R_2(j,s)}(y_i-\\bar y_{R_2})^2\\Big]$$
**النظير القياسي**: انحدار على متغيرات وهمية لتفاعلات معقدة تُكتشف آليًا؛ نماذج العتبة (Threshold models) متعددة الأنظمة.
**في الاقتصاد**: اكتشاف التفاعلات وعدم التجانس — وهي أساس **الشجرة السببية** لآثي وإمبنز (2016) حيث نقسّم لتعظيم **تباين الأثر** لا دقة التنبؤ.
        """
    )

# --------------------------------------------------------------- forests -----
with tabs[2]:
    ui.section("الغابة العشوائية: متوسط أشجار كثيرة «غير مترابطة»", "Bagging & random forests", "emerald")

    @st.cache_data(show_spinner=False)
    def rf_frames():
        rf = RandomForestRegressor(n_estimators=200, max_features=1, min_samples_leaf=3, random_state=0).fit(X1[:, None], Y1)
        preds = np.array([t.predict(G1[:, None]) for t in rf.estimators_])
        ks = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200]
        return ks, [preds[:k].mean(0) for k in ks], preds[:6]

    ks, avg, single = rf_frames()

    def rf_tr(i):
        return [go.Scatter(x=X1, y=Y1, mode="markers", marker=dict(color="#cbd5e1", size=5), name="البيانات"),
                go.Scatter(x=G1, y=f1(G1), mode="lines", line=dict(color=ui.COLORS["slate"], dash="dot"), name="الحقيقية"),
                go.Scatter(x=G1, y=avg[i], mode="lines", line=dict(color=ui.COLORS["emerald"], width=4), name="متوسط الغابة")]

    fig = go.Figure(data=rf_tr(0), frames=[go.Frame(data=rf_tr(i), name=str(k)) for i, k in enumerate(ks)])
    ui.style_fig(fig, "كلما زاد عدد الأشجار نعُم التقدير وانخفض التباين")
    ui.add_play(fig, ks, duration=700, prefix="عدد الأشجار = ")
    ui.show(fig)
    st.latex(r"\hat f_{RF}(x)=\frac1B\sum_{b=1}^{B}T_b(x;\,\text{bootstrap}_b,\ \text{random subset of } m \text{ features})")
    st.latex(r"\operatorname{Var}\big(\hat f_{RF}\big)=\rho\,\sigma^2+\frac{1-\rho}{B}\sigma^2")
    st.markdown("اختيار $m$ متغيرات عشوائية عند كل تقسيم يخفّض الارتباط $\\rho$ بين الأشجار، فيصبح المتوسط أقل تباينًا. "
                "**اقتصاديًا**: الغابة = **مقدّر محلي تكيّفي** يعطي أوزانًا للمشاهدات القريبة من $x$ — وهذه الرؤية هي أساس **الغابات العشوائية المعممة (GRF)** "
                "التي تقدّر أي معلمة محلية معرّفة بشرط لحظي $\\mathbb E[\\psi(\\theta(x);O)\\mid X=x]=0$ (Athey, Tibshirani & Wager, 2019).")

# -------------------------------------------------------------- boosting -----
with tabs[3]:
    ui.section("التعزيز: أشجار ضعيفة تتعلم من أخطاء ما قبلها", "Gradient boosting (Friedman, 2001)", "rose")

    @st.cache_data(show_spinner=False)
    def boost_frames():
        gb = GradientBoostingRegressor(n_estimators=300, max_depth=1, learning_rate=0.1, random_state=0).fit(X1[:, None], Y1)
        stages = list(gb.staged_predict(G1[:, None]))
        ks = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 180, 300]
        return ks, [stages[k - 1] for k in ks]

    ks, st_pred = boost_frames()

    def gb_tr(i):
        return [go.Scatter(x=X1, y=Y1, mode="markers", marker=dict(color="#fecdd3", size=5), name="البيانات"),
                go.Scatter(x=G1, y=f1(G1), mode="lines", line=dict(color=ui.COLORS["slate"], dash="dot"), name="الحقيقية"),
                go.Scatter(x=G1, y=st_pred[i], mode="lines", line=dict(color=ui.COLORS["rose"], width=4, shape="hv"), name="نموذج التعزيز")]

    fig = go.Figure(data=gb_tr(0), frames=[go.Frame(data=gb_tr(i), name=str(k)) for i, k in enumerate(ks)])
    ui.style_fig(fig, "كل مرحلة تضيف شجرة بعقدة واحدة (Stump) توفَّق على البواقي")
    ui.add_play(fig, ks, duration=700, prefix="المراحل = ")
    ui.show(fig)
    st.latex(r"F_m(x)=F_{m-1}(x)+\nu\,h_m(x),\qquad h_m=\arg\min_h\sum_i\big(r_{im}-h(x_i)\big)^2,\quad r_{im}=-\frac{\partial L(y_i,F)}{\partial F}\Big|_{F_{m-1}}")
    ui.cards([
        {"title": "XGBoost", "en": "Chen & Guestrin (2016)", "color": "rose", "body": "تعزيز منظَّم بسرعة عالية؛ مهيمن في المسابقات والبيانات الجدولية."},
        {"title": "LightGBM / CatBoost", "en": "Histogram & ordered boosting", "color": "orange", "body": "سرعة على البيانات الكبيرة، ومعالجة المتغيرات الفئوية."},
        {"title": "L2Boosting", "en": "Bühlmann & Yu (2003)", "color": "violet", "body": "تعزيز مكوّني خطي = طريقة اختيار متغيرات قريبة من لاسو؛ مستخدم في التنبؤ الكلي."},
        {"title": "GPBoost", "en": "Sigrist (2022, JMLR)", "color": "teal", "body": "تعزيز + آثار عشوائية/عمليات غاوسية — جسر مباشر مع نماذج البانل المختلطة."},
    ], cols=4)

# --------------------------------------------------------- kernels/SVM/GP ----
with tabs[4]:
    ui.section("طرق النواة، آلات المتجهات الداعمة، والعمليات الغاوسية", "Kernel methods, SVM/SVR, Gaussian processes", "sky")
    ui.cards([
        {"title": "حيلة النواة", "en": "Kernel trick", "color": "sky",
         "body": "نعمل في فضاء ميزات لا نهائي عبر دالة نواة K(x,x′) دون حسابه صراحة. صلة مباشرة بالانحدار اللامعلمي بالنواة لنادارايا–واتسون."},
        {"title": "SVM / SVR", "en": "Vapnik (1995)", "color": "violet",
         "body": "هامش أقصى للتصنيف؛ وانحدار بخسارة ε-غير حساسة. استُخدم في التنبؤ بالتعثر المالي وأسعار الصرف."},
        {"title": "انحدار ريدج بالنواة", "en": "Kernel ridge / KRLS", "color": "teal",
         "body": "Hainmueller & Hazlett (2014): KRLS للعلوم الاجتماعية — مرونة مع آثار حدية قابلة للتفسير (Pointwise marginal effects)."},
        {"title": "العمليات الغاوسية", "en": "Gaussian processes", "color": "orange",
         "body": "قبْلي بايزي على الدوال يعطي تنبؤًا مع عدم يقين. يُستخدم كمحاكٍ (Surrogate) للنماذج الهيكلية المكلفة، وفي EAGP الموجّهة اقتصاديًا."},
    ], cols=2)
    st.latex(r"\hat f(x)=\sum_{i=1}^n \alpha_i K(x,x_i),\qquad \alpha=(K+\lambda I)^{-1}y,\qquad K(x,x')=\exp\!\Big(-\frac{\|x-x'\|^2}{2\ell^2}\Big)")

# ------------------------------------------------------------ neural nets ----
with tabs[5]:
    ui.section("الشبكة العصبية تتعلم عبر الحقب (Epochs)", "A neural network learning, epoch by epoch", "indigo")

    @st.cache_data(show_spinner=False)
    def nn_frames():
        mlp = MLPRegressor(hidden_layer_sizes=(32, 32), activation="tanh", learning_rate_init=0.01, random_state=1, max_iter=1, warm_start=True)
        snaps, losses, keep = [], [], [1, 3, 5, 10, 20, 35, 60, 100, 160, 250, 400]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for ep in range(1, 401):
                mlp.partial_fit(X1[:, None], Y1)
                if ep in keep:
                    snaps.append(mlp.predict(G1[:, None])); losses.append(mlp.loss_)
        return keep, snaps, losses

    keep, snaps, nl = nn_frames()

    def nn_tr(i):
        return [go.Scatter(x=X1, y=Y1, mode="markers", marker=dict(color="#c7d2fe", size=5), name="البيانات"),
                go.Scatter(x=G1, y=f1(G1), mode="lines", line=dict(color=ui.COLORS["slate"], dash="dot"), name="الحقيقية"),
                go.Scatter(x=G1, y=snaps[i], mode="lines", line=dict(color=ui.COLORS["indigo"], width=4), name="الشبكة")]

    fig = go.Figure(data=nn_tr(0),
                    frames=[go.Frame(data=nn_tr(i), name=str(k), layout=go.Layout(title_text=f"الحقبة {k} — خسارة التدريب = {nl[i]:.3f}"))
                            for i, k in enumerate(keep)])
    ui.style_fig(fig, f"الحقبة 1 — خسارة التدريب = {nl[0]:.3f}")
    ui.add_play(fig, keep, duration=700, prefix="الحقبة = ", redraw=True)
    ui.show(fig)
    st.mermaid_chart(
        """
flowchart LR
    x1((x1)) --> h1((h1)) & h2((h2)) & h3((h3))
    x2((x2)) --> h1 & h2 & h3
    x3((x3)) --> h1 & h2 & h3
    h1 --> g1((g1)) & g2((g2))
    h2 --> g1 & g2
    h3 --> g1 & g2
    g1 --> y((y-hat))
    g2 --> y
"""
    )
    st.latex(r"\hat y=\sigma_2\!\big(W_2\,\sigma_1(W_1x+b_1)+b_2\big)\quad\text{— a nested nonlinear regression estimated by backprop + SGD}")
    ui.cards([
        {"title": "MLP", "en": "Feed-forward / multilayer perceptron", "color": "indigo", "body": "مقرّب شامل (Hornik et al., 1989). استدلال بالشبكات العميقة: Farrell, Liang & Misra (2021, Econometrica)."},
        {"title": "CNN", "en": "Convolutional nets", "color": "sky", "body": "للصور والأقمار الصناعية والوثائق الممسوحة (LayoutParser لديل وآخرين)."},
        {"title": "RNN / LSTM / GRU", "en": "Recurrent nets", "color": "teal", "body": "للتسلسلات الزمنية والنصوص؛ LSTM (Hochreiter & Schmidhuber, 1997) يحفظ الذاكرة الطويلة."},
        {"title": "Autoencoder / VAE", "en": "Representation learning", "color": "fuchsia", "body": "ضغط غير خطي = تحليل عاملي لا خطي (Gu, Kelly & Xiu, 2021: نماذج تسعير الأصول بالمشفّر الذاتي)."},
        {"title": "GAN", "en": "Generative adversarial nets", "color": "orange", "body": "مولّد ضد مميّز؛ ألهم التقدير الهيكلي التنافسي (Kaji, Manresa & Pouliot, 2023)."},
        {"title": "KAN", "en": "Kolmogorov–Arnold networks (2024)", "color": "lime", "body": "دوال أحادية المتغير قابلة للتعلم على الحواف؛ وعود بالتفسير، و TimeKAN للسلاسل (2026)."},
    ])

# ----------------------------------------------------------- unsupervised ----
with tabs[6]:
    ui.section("التعلم غير الموجّه: العناقيد والعوامل", "Clustering (k-means animated) & factor extraction", "fuchsia")

    @st.cache_data(show_spinner=False)
    def kmeans_frames():
        rng = np.random.default_rng(4)
        centers = np.array([[0, 0], [4, 1], [1.5, 4], [5, 5]])
        X = np.vstack([c + rng.normal(0, .8, (70, 2)) for c in centers])
        C = X[rng.choice(len(X), 4, replace=False)].copy()
        frames = []
        for _ in range(9):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
            frames.append((lab.copy(), C.copy()))
            C = np.array([X[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(4)])
        return X, frames

    Xk, kf = kmeans_frames()
    pal = [ui.COLORS[c] for c in ["violet", "teal", "orange", "rose"]]

    def km_tr(i):
        lab, C = kf[i]
        return [go.Scatter(x=Xk[:, 0], y=Xk[:, 1], mode="markers", marker=dict(color=[pal[l] for l in lab], size=7, opacity=.75), name="المشاهدات"),
                go.Scatter(x=C[:, 0], y=C[:, 1], mode="markers", marker=dict(symbol="x", size=18, color="black", line=dict(width=2)), name="المراكز")]

    fig = go.Figure(data=km_tr(0), frames=[go.Frame(data=km_tr(i), name=str(i + 1)) for i in range(len(kf))])
    ui.style_fig(fig, "خوارزمية k-means: إسناد ← تحديث المراكز ← تكرار", height=520)
    ui.add_play(fig, list(range(1, len(kf) + 1)), duration=900, prefix="التكرار = ", redraw=True)
    ui.show(fig)
    c1, c2 = st.columns(2)
    with c1:
        ui.note("PCA تستخرج اتجاهات أقصى تباين — وهي أساس **مؤشرات الانتشار** (Stock & Watson, 2002) ونماذج FAVAR وعوامل بانل (Bai & Ng, 2002). "
                "**الاقتصاد سبق تعلم الآلة هنا** بعقود (التحليل العاملي في الاقتصاد الكلي).", "العوامل و PCA", "fuchsia", "🧩")
    with c2:
        ui.note("العناقيد تحدد **أنواعًا** من الأسر أو الدول أو الأنظمة. نماذج المزيج و **سلاسل ماركوف المخفية (HMM)** = نماذج تبديل الأنظمة لهاميلتون (1989).",
                "العناقيد والأنظمة", "violet", "🗂️")

# ------------------------------------------------------------ transformers ---
with tabs[7]:
    ui.section("المحوّلات وآلية الانتباه", "Transformers & self-attention (Vaswani et al., 2017)", "cyan")
    st.latex(r"\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V")
    st.markdown("كل كلمة (أو نقطة زمنية) تنظر إلى كل الكلمات الأخرى وتحدد **أوزان انتباه**. تمثيل توضيحي (أرقام مصطنعة للتعليم) لجملة اقتصادية:")
    words = ["The", "Fed", "raised", "rates", "to", "fight", "inflation"]
    rng = np.random.default_rng(8)
    A = rng.uniform(0, 1, (7, 7)); A[2, 1] += 2.5; A[3, 2] += 2; A[6, 5] += 2.5; A[5, 6] += 1.5; A[3, 6] += 1.5; A[1, 3] += 1.2
    A = np.exp(A) / np.exp(A).sum(1, keepdims=True)
    fig = go.Figure(go.Heatmap(z=A, x=words, y=words, colorscale="Purples", text=np.round(A, 2), texttemplate="%{text}"))
    fig.update_yaxes(autorange="reversed", title="الكلمة التي «تنتبه»"); fig.update_xaxes(title="الكلمة المنتبَه إليها")
    ui.style_fig(fig, "مصفوفة انتباه توضيحية (أرقام مصطنعة)", height=470, legend_h=False)
    ui.show(fig)
    ui.cards([
        {"title": "BERT / FinBERT", "en": "Encoder models", "color": "cyan", "body": "تمثيلات سياقية لتصنيف النصوص الاقتصادية والمالية (النبرة، التوقعات)."},
        {"title": "GPT / Llama / Claude", "en": "Decoder LLMs", "color": "violet", "body": "توليد وتصنيف بلا أمثلة (Zero-shot)، أدوات قياس، وكلاء بحث."},
        {"title": "TFT / PatchTST / iTransformer", "en": "Time-series transformers", "color": "orange", "body": "تنبؤ متعدد السلاسل بآفاق متعددة مع متغيرات خارجية."},
        {"title": "TimesFM / Chronos / Moirai / BISTRO", "en": "Foundation models", "color": "teal", "body": "نماذج مسبقة التدريب للتنبؤ بلا تدريب (Zero-shot) — انظر صفحة النماذج التأسيسية."},
    ], cols=4)

# --------------------------------------------------------------- summary -----
with tabs[8]:
    ui.section("جدول مرجعي: الخوارزمية ودورها في الدمج", "Algorithm × role in integration", "slate")
    summ = pd.DataFrame([
        ("Lasso / Elastic Net", "خطي منظَّم", "PDS، DML، تنبؤ عالي الأبعاد، SPECS", "قابل للتفسير جزئيًا، سريع", "تحيز الانكماش، عدم استقرار الاختيار"),
        ("Random forest", "تجميع أشجار", "دوال مساعدة DML، GRF، تنبؤ كلي، MRF", "لا خطية وتفاعلات تلقائية", "ضعيف في الاستقراء خارج المدى"),
        ("Gradient boosting", "تجميع متسلسل", "Propensity score، تنبؤ، تصنيف", "دقة عالية جدًا على الجداول", "حساس للضبط؛ يفرط في التوفيق"),
        ("BART", "أشجار بايزية", "BCF للأثر غير المتجانس، BART-VAR", "عدم يقين بايزي طبيعي", "حسابيًا أثقل"),
        ("Neural nets (MLP)", "شبكات أمامية", "DML، Deep IV، حل النماذج الهيكلية", "مرونة قصوى، تفاضل آلي", "بيانات كثيرة، ضبط صعب، صندوق أسود"),
        ("LSTM / Transformers", "تسلسلية", "تنبؤ زمني، نصوص", "ذاكرة طويلة وسياق", "تسرب زمني، عينات قصيرة"),
        ("SVM / Kernels / GP", "طرق النواة", "تصنيف، محاكيات هيكلية، KRLS", "أداء جيد في العينات المتوسطة", "لا تتوسع مع n الضخم"),
        ("PCA / Autoencoders", "تقليل الأبعاد", "عوامل، FAVAR، تسعير أصول", "ضغط المعلومات", "العوامل لا تحمل معنى اقتصاديًا تلقائيًا"),
        ("Clustering / HMM", "غير موجّه", "أنظمة، تصنيف وحدات", "اكتشاف بنية", "اختيار عدد العناقيد"),
        ("LLMs", "نماذج لغة", "قياس، ترميز، محاكاة، وكلاء", "معالجة النص على نطاق واسع", "تسرب، هلوسة، خطأ قياس غير كلاسيكي"),
        ("RL", "تعلم معزز", "تجارب تكيفية، سياسات، حل DP", "قرارات تتابعية", "يحتاج نموذج بيئة/بيانات دعم"),
    ], columns=["الخوارزمية", "العائلة", "دورها في الدمج", "نقطة القوة", "المحذور"])
    st.dataframe(summ, hide_index=True, height=440)
    ui.refs("Breiman et al. (1984); Breiman (2001); Friedman (2001); Chen & Guestrin (2016); Hastie, Tibshirani & Friedman (2009) ESL; "
            "Chipman, George & McCulloch (2010); Hainmueller & Hazlett (2014); Goodfellow et al. (2014); Vaswani et al. (2017); Dell (2025) JEL.")
