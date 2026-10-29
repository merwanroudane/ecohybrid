import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import ui

ui.hero(
    "مفاهيم تعلم الآلة التي يحتاجها الاقتصادي",
    "Machine-Learning Foundations for Econometricians",
    "قبل الدمج يجب إتقان «لغة» تعلم الآلة: أنواع التعلم، مفاضلة التحيز والتباين، الإفراط في التوفيق، التنظيم، التحقق المتقاطع، "
    "الانحدار التدريجي، الانحدار المزدوج، ودوال الخسارة — مع ربط كل مفهوم بنظيره في القياس الاقتصادي.",
    "emerald", "violet",
)

tabs = st.tabs([
    ":material/category: أنواع التعلم",
    ":material/translate: قاموس الترجمة",
    ":material/tune: التحيز والتباين",
    ":material/view_week: التحقق المتقاطع",
    ":material/compress: التنظيم",
    ":material/south: الانحدار التدريجي",
    ":material/stacked_line_chart: الانحدار المزدوج",
    ":material/rule: الخسارة والتقييم",
])

# ------------------------------------------------------------ types ----------
with tabs[0]:
    ui.section("خريطة أنواع التعلم", "Supervised, unsupervised, self-supervised, reinforcement", "emerald")
    st.mermaid_chart(
        """
flowchart TB
    ML[Machine Learning] --> S[Supervised<br/>labelled y]
    ML --> U[Unsupervised<br/>no y]
    ML --> SS[Self-supervised<br/>labels from data itself]
    ML --> R[Reinforcement<br/>actions and rewards]
    S --> S1[Regression: Lasso, RF, GBM, NN]
    S --> S2[Classification: logit, SVM, trees]
    U --> U1[Dimension reduction: PCA, factors, autoencoders]
    U --> U2[Clustering: k-means, mixtures, HMM]
    SS --> SS1[Language models, embeddings, foundation models]
    R --> R1[Bandits, Q-learning, policy gradients]
"""
    )
    tbl = pd.DataFrame([
        ("التعلم الموجَّه", "Supervised", "تعلم $f$ من أزواج $(x,y)$", "الانحدار، نماذج الاختيار المتقطع (Logit/Probit)", "التنبؤ بالتعثر، الدخل من الصور"),
        ("التعلم غير الموجَّه", "Unsupervised", "اكتشاف بنية في $x$ وحده", "التحليل العاملي، نماذج المزيج، تحليل العناقيد", "عوامل كلية، أنواع الأسر، أنظمة الأسواق"),
        ("التعلم الذاتي الإشراف", "Self-supervised", "توليد مهمة من البيانات (توقع الكلمة التالية)", "لا نظير مباشر — قريب من نماذج AR", "LLMs، تضمينات النصوص الاقتصادية"),
        ("التعلم شبه الموجَّه", "Semi-supervised", "قليل من الوسوم + كثير بلا وسوم", "تصحيح خطأ القياس بعينة تحقق", "PPI و DSL بوسوم بشرية قليلة"),
        ("التعلم المعزَّز", "Reinforcement", "تعلم سياسة تعظّم مكافأة تراكمية", "البرمجة الديناميكية، نماذج الاختيار الديناميكي", "تجارب تكيفية، تسعير، سياسة نقدية"),
    ], columns=["النوع", "English", "الفكرة", "النظير القياسي", "مثال اقتصادي"])
    st.table(tbl)

# ---------------------------------------------------------- dictionary -------
with tabs[1]:
    ui.section("قاموس ترجمة بين الثقافتين", "ML ↔ Econometrics vocabulary", "violet")
    st.markdown("كثير من سوء الفهم بين الاقتصاديين ومهندسي تعلم الآلة سببه **اختلاف المصطلحات لمفاهيم واحدة** — أو **تشابه المصطلحات لمفاهيم مختلفة**.")
    voc = pd.DataFrame([
        ("Features / inputs", "المتغيرات المستقلة / المفسرة (Regressors)", "x"),
        ("Target / label", "المتغير التابع (Dependent variable)", "y"),
        ("Training", "التقدير (Estimation)", "fit"),
        ("Weights", "المعاملات (Coefficients)", "β"),
        ("Bias (unit)", "الحد الثابت (Intercept)", "α"),
        ("Loss function", "دالة الهدف / المعيار (Criterion)", "Σ(y−ŷ)²"),
        ("Learning rate", "خطوة الخوارزمية العددية", "η"),
        ("Hyperparameter", "معلمة ضبط (Tuning parameter)", "λ"),
        ("Generalization", "الأداء خارج العينة", "OOS"),
        ("Inference (في ML)", "التنبؤ بمشاهدة جديدة! — ليس الاستدلال الإحصائي", "ŷ"),
        ("Inference (في القياس)", "اختبارات الفرضيات وفترات الثقة", "CI, p-value"),
        ("Regularization", "الانكماش / العقوبة (Shrinkage / Penalization)", "λ‖β‖"),
        ("Data augmentation", "توليد مشاهدات مصطنعة — حذارِ: ليست مشاهدات مستقلة", "—"),
        ("Embedding", "تمثيل رقمي كثيف لنص/صورة (متغير كامن)", "z ∈ ℝ^d"),
        ("Data leakage", "استخدام معلومات غير متاحة وقت القرار (تحيز النظر للأمام)", "look-ahead"),
        ("Distribution shift", "عدم الثبات البنيوي / تغير النظام", "regime change"),
        ("Bias (statistical)", "التحيز = E[θ̂] − θ — بالمعنى نفسه في الثقافتين", "bias"),
    ], columns=["مصطلح تعلم الآلة", "المقابل القياسي", "رمز"])
    st.dataframe(voc, hide_index=True, height=600)
    ui.note("انتبه للفخ الأكبر: كلمة **Inference** في هندسة تعلم الآلة تعني «تشغيل النموذج المدرَّب على بيانات جديدة»، أما في القياس "
            "فتعني «الاستدلال الإحصائي». جملة مثل «النموذج سريع في الـ inference» لا علاقة لها بفترات الثقة!", "تحذير مصطلحي", "rose", "⚠️")

# ------------------------------------------------------------ bias-variance --
with tabs[2]:
    ui.section("مختبر متحرك: الإفراط في التوفيق ومفاضلة التحيز–التباين", "Overfitting & bias–variance (animated)", "orange")
    st.markdown(
        "نوفّق كثيرات حدود من الدرجة 1 إلى 15 على 30 مشاهدة من دالة لا خطية $f(x)=\\sin(2\\pi x)$. شغّل الحركة: "
        "يسارًا المنحنى الموفَّق، ويمينًا خطأ **التدريب** (ينخفض دائمًا) وخطأ **الاختبار** (ينخفض ثم يرتفع: منحنى U)."
    )
    noise = st.slider("مستوى الضجيج σ", 0.1, 0.6, 0.3, 0.05, key="bv_noise")

    @st.cache_data(show_spinner=False)
    def poly_frames(sigma: float):
        rng = np.random.default_rng(5)
        x = np.sort(rng.uniform(0, 1, 30)); y = np.sin(2 * np.pi * x) + rng.normal(0, sigma, 30)
        xt = rng.uniform(0, 1, 2000); yt = np.sin(2 * np.pi * xt) + rng.normal(0, sigma, 2000)
        grid = np.linspace(0, 1, 300)
        out, tr, te = [], [], []
        for d in range(1, 16):
            c = np.polyfit(x, y, d)
            out.append(np.clip(np.polyval(c, grid), -3, 3))
            tr.append(np.mean((y - np.polyval(c, x)) ** 2)); te.append(np.mean((yt - np.polyval(c, xt)) ** 2))
        return x, y, grid, out, np.array(tr), np.array(te)

    x, y, grid, fits, tr, te = poly_frames(noise)
    degs = list(range(1, 16))

    def frame_traces(k):
        return [
            go.Scatter(x=x, y=y, mode="markers", marker=dict(color=ui.COLORS["slate"], size=8), name="بيانات التدريب"),
            go.Scatter(x=grid, y=np.sin(2 * np.pi * grid), mode="lines", line=dict(color=ui.COLORS["emerald"], dash="dot"), name="الدالة الحقيقية"),
            go.Scatter(x=grid, y=fits[k], mode="lines", line=dict(color=ui.COLORS["orange"], width=4), name="النموذج الموفَّق"),
            go.Scatter(x=degs[:k + 1], y=tr[:k + 1], mode="lines+markers", line=dict(color=ui.COLORS["sky"], width=3), name="خطأ التدريب"),
            go.Scatter(x=degs[:k + 1], y=te[:k + 1], mode="lines+markers", line=dict(color=ui.COLORS["rose"], width=3), name="خطأ الاختبار"),
        ]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("المنحنى الموفَّق", "الخطأ مقابل التعقيد"))
    for i, t in enumerate(frame_traces(0)):
        fig.add_trace(t, row=1, col=1 if i < 3 else 2)
    fig.frames = [go.Frame(data=frame_traces(k), traces=[0, 1, 2, 3, 4], name=str(degs[k])) for k in range(15)]
    fig.update_yaxes(range=[-2.2, 2.2], row=1, col=1)
    fig.update_yaxes(type="log", title="MSE", row=1, col=2, range=[np.log10(max(tr.min(), 1e-3)) - 0.2, np.log10(min(te.max(), 50)) + 0.2])
    fig.update_xaxes(range=[0.5, 15.5], title="درجة كثير الحدود", row=1, col=2)
    ui.style_fig(fig, height=500)
    ui.add_play(fig, degs, duration=550, prefix="الدرجة = ", redraw=True)
    ui.show(fig)
    st.latex(r"\mathbb{E}\big[(y_0-\hat f(x_0))^2\big]=\sigma^2+\text{Bias}^2\big(\hat f(x_0)\big)+\operatorname{Var}\big(\hat f(x_0)\big)")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.note("نموذج بسيط جدًا (درجة 1): تحيز كبير، لا يلتقط البنية. **في القياس**: سوء المواصفة.", "نقص التوفيق (Underfitting)", "sky", "📉")
    with c2:
        ui.note("التعقيد الذي يُدنّي خطأ الاختبار — يُختار **تجريبيًا** بالتحقق المتقاطع لا بالنظرية.", "النقطة المثلى (Sweet spot)", "emerald", "🎯")
    with c3:
        ui.note("نموذج معقد جدًا: يحفظ الضجيج، خطأ تدريب ≈ 0 وخطأ اختبار ضخم. **في القياس**: «التنقيب في البيانات».", "الإفراط في التوفيق (Overfitting)", "rose", "📈")

# ------------------------------------------------------------- CV -----------
with tabs[3]:
    ui.section("التحقق المتقاطع: الضبط التجريبي للتعقيد", "k-fold and time-series cross-validation", "sky")
    mode = st.segmented_control("نوع التقسيم", ["K-fold عشوائي", "Rolling-origin زمني", "Blocked + Embargo"], default="K-fold عشوائي", key="cv_mode")
    n, K = 60, 5

    def cv_masks(kind: str):
        frames = []
        if kind == "K-fold عشوائي":
            rng = np.random.default_rng(0); perm = rng.permutation(n); folds = np.array_split(perm, K)
            for k in range(K):
                m = np.zeros(n); m[folds[k]] = 2; m[np.setdiff1d(np.arange(n), folds[k])] = 1
                frames.append(m)
        elif kind == "Rolling-origin زمني":
            for k in range(K):
                end = 25 + 7 * k; m = np.zeros(n); m[:end] = 1; m[end:end + 7] = 2
                frames.append(m)
        else:
            b = np.array_split(np.arange(n), K)
            for k in range(K):
                m = np.ones(n); m[b[k]] = 2
                lo, hi = b[k][0], b[k][-1]
                m[max(lo - 3, 0):lo] = 0; m[hi + 1:hi + 4] = 0
                frames.append(m)
        return frames

    masks = cv_masks(mode)
    cmap = {0: "#e5e7eb", 1: ui.COLORS["sky"], 2: ui.COLORS["orange"]}

    def bar(m):
        return go.Bar(x=np.arange(1, n + 1), y=np.ones(n), marker_color=[cmap[int(v)] for v in m],
                      hovertext=[{0: "مستبعد (Embargo/غير مستخدم)", 1: "تدريب", 2: "تحقق/اختبار"}[int(v)] for v in m], hoverinfo="text")

    fig = go.Figure(data=[bar(masks[0])], frames=[go.Frame(data=[bar(m)], name=str(k + 1)) for k, m in enumerate(masks)])
    fig.update_yaxes(visible=False); fig.update_xaxes(title="ترتيب المشاهدات (الزمن في التقسيم الزمني)")
    ui.style_fig(fig, f"{mode}: الأزرق = تدريب، البرتقالي = تحقق، الرمادي = مستبعد", height=330, legend_h=False)
    fig.update_layout(bargap=0.05)
    ui.add_play(fig, list(range(1, K + 1)), duration=900, prefix="الطيّة: ", redraw=True)
    ui.show(fig)
    ui.cards([
        {"title": "K-fold العشوائي", "en": "i.i.d. cross-sections", "color": "sky",
         "body": "مناسب للبيانات المقطعية المستقلة. يُختار λ أو العمق الذي يدني متوسط خطأ الطيات."},
        {"title": "الأصل المتدحرج", "en": "Rolling / expanding origin", "color": "orange",
         "body": "إلزامي في السلاسل الزمنية: لا نستخدم المستقبل للتنبؤ بالماضي. يحاكي التنبؤ الحقيقي."},
        {"title": "الكتل مع الحجز", "en": "Blocked CV with embargo / gap", "color": "rose",
         "body": "لبيانات مترابطة زمنيًا أو مكانيًا: نستبعد مشاهدات مجاورة لمنع التسرب عبر الارتباط الذاتي."},
        {"title": "التقسيم حسب الوحدة", "en": "Grouped CV (panel)", "color": "violet",
         "body": "في البانل: كل مشاهدات الفرد في الطيّة نفسها، وإلا تسرّب الأثر الفردي بين التدريب والاختبار."},
    ], cols=4)
    ui.note("وجد غوليه كولومب وآخرون (2022) أن **K-fold** يبقى أفضل ممارسة للضبط حتى في التنبؤ الكلي (مع نماذج AR الملائمة للبواقي)، "
            "لكن **التقييم النهائي** يجب أن يكون خارج العينة زمنيًا وبالبيانات المتاحة آنيًا (Real-time vintages).", "ملاحظة من الأدبيات", "teal", "📚")

# ----------------------------------------------------------- regularization --
with tabs[4]:
    ui.section("التنظيم: مقايضة قليل من التحيز بكثير من التباين", "Regularization", "fuchsia")
    st.latex(r"\hat\beta(\lambda)=\arg\min_\beta \;\frac1n\sum_{i=1}^n (y_i-x_i^\top\beta)^2+\lambda\,\mathcal{P}(\beta)")
    ui.cards([
        {"title": "ريدج", "en": "Ridge — L2 (Hoerl & Kennard, 1970)", "color": "violet",
         "body": "العقوبة ‖β‖₂². ينكمش بكل المعاملات نحو الصفر دون إلغائها؛ يعالج التعدد الخطي. مكافئ لـ <b>قبْلي غاوسي</b> بايزيًا."},
        {"title": "لاسو", "en": "LASSO — L1 (Tibshirani, 1996)", "color": "teal",
         "body": "العقوبة ‖β‖₁. يضع معاملات عند الصفر بالضبط ⇒ <b>اختيار متغيرات</b>. مكافئ لقبْلي لابلاس."},
        {"title": "الشبكة المرنة", "en": "Elastic Net (Zou & Hastie, 2005)", "color": "orange",
         "body": "مزيج L1+L2: يختار متغيرات ويحتفظ بالمجموعات المترابطة معًا."},
        {"title": "لاسو المجموعات", "en": "Group / Sparse-group LASSO", "color": "sky",
         "body": "يختار مجموعات كاملة (فجوات متغير، تواترات MIDAS). أساس sg-LASSO-MIDAS للتنبؤ الآني."},
        {"title": "لاسو التكيفي", "en": "Adaptive LASSO (Zou, 2006)", "color": "rose",
         "body": "أوزان عقوبة مختلفة ⇒ خاصية «العرّاف» (Oracle property). استُخدم لاختيار رتبة التكامل المشترك (Liao & Phillips)."},
        {"title": "التنظيم الضمني", "en": "Early stopping, dropout, pruning", "color": "emerald",
         "body": "في الأشجار والشبكات: العمق الأقصى، الحد الأدنى للورقة، الإيقاف المبكر، الإسقاط العشوائي للعصبونات."},
    ])
    st.markdown(
        """
**الصلة بالقياس الاقتصادي:**
- معايير **AIC/BIC** هي في جوهرها تنظيم بعقوبة $L_0$ (عدد المعلمات).
- **نماذج VAR البايزية (BVAR)** مع قبْلي مينيسوتا (Litterman, 1986) = ريدج منظّم بنيويًا — أقدم «تعلم آلة» في الاقتصاد الكلي!
- **الثمن**: التنظيم **يحيّز** المعاملات نحو الصفر (Shrinkage bias) ⇒ لا تصلح معاملات لاسو مباشرة للاستدلال — وهذا بالضبط ما عالجه **PDS** و**DML**.
        """
    )

# ------------------------------------------------------------ gradient -------
with tabs[5]:
    ui.section("الانحدار التدريجي: كيف «تتعلم» الشبكات", "Gradient descent on an OLS loss surface", "amber")
    lr = st.select_slider("معدل التعلم η", options=[0.02, 0.05, 0.1, 0.2, 0.35, 0.5], value=0.1, key="gd_lr")

    @st.cache_data(show_spinner=False)
    def gd_path(eta: float):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((200, 2)) @ np.array([[1, .6], [0, .8]]); b0 = np.array([1.5, -1.0])
        y = X @ b0 + rng.normal(0, .5, 200)
        H = X.T @ X / 200
        def loss(b): return np.mean((y - X @ b) ** 2)
        b = np.array([-2.0, 2.5]); path = [b.copy()]
        for _ in range(40):
            g = -2 * X.T @ (y - X @ b) / 200; b = b - eta * g; path.append(b.copy())
        g1 = np.linspace(-3, 3, 80); g2 = np.linspace(-3, 3, 80)
        Z = np.array([[loss(np.array([a, c])) for a in g1] for c in g2])
        return np.array(path), g1, g2, Z, np.linalg.lstsq(X, y, rcond=None)[0]

    path, g1, g2, Z, bols = gd_path(lr)
    base = [go.Contour(x=g1, y=g2, z=np.log(Z), colorscale="Viridis", showscale=False, opacity=.75, contours=dict(coloring="heatmap")),
            go.Scatter(x=[bols[0]], y=[bols[1]], mode="markers", marker=dict(symbol="star", size=18, color="white", line=dict(color="black", width=1)), name="حل OLS")]
    fig = go.Figure(
        data=base + [go.Scatter(x=path[:1, 0], y=path[:1, 1], mode="lines+markers", line=dict(color=ui.COLORS["rose"], width=3), marker=dict(size=7), name="مسار الخوارزمية")],
        frames=[go.Frame(data=[go.Scatter(x=path[:k + 1, 0], y=path[:k + 1, 1])], traces=[2], name=str(k)) for k in range(len(path))],
    )
    fig.update_xaxes(title="β₁", range=[-3, 3]); fig.update_yaxes(title="β₂", range=[-3, 3])
    ui.style_fig(fig, "سطح دالة الخسارة MSE ومسار الانحدار التدريجي", height=560)
    ui.add_play(fig, list(range(len(path))), duration=180, prefix="التكرار: ")
    ui.show(fig)
    st.latex(r"\beta^{(t+1)}=\beta^{(t)}-\eta\,\nabla_\beta \mathcal{L}\big(\beta^{(t)}\big),\qquad \nabla_\beta\mathcal L=-\tfrac{2}{n}X^\top(y-X\beta)")
    st.markdown("جرّب $\\eta=0.5$: تتذبذب الخوارزمية أو تتباعد. **الانحدار التدريجي العشوائي (SGD)** يستخدم عينة جزئية في كل خطوة، و**Adam** يكيّف الخطوة لكل معلمة. "
                "في القياس نستخدم منذ عقود خوارزميات **نيوتن–رافسون** و**BHHH** لتعظيم الإمكان — الفكرة نفسها مع معلومات الانحناء (Hessian).")

# ---------------------------------------------------------- double descent ---
with tabs[6]:
    ui.section("الانحدار المزدوج: لماذا تنجح النماذج «المفرطة في المعلمات»؟", "Double descent (Belkin et al., 2019)", "indigo")
    st.markdown(
        "الحكمة الكلاسيكية تقول إن الخطأ يرتفع بعد نقطة معينة من التعقيد. لكن عندما يتجاوز عدد المعلمات $p$ عدد المشاهدات $n$ "
        "ونأخذ **الحل ذا المعيار الأدنى (Minimum-norm)**، **ينخفض الخطأ مرة ثانية**. هذه الظاهرة تفسّر جزئيًا نجاح الشبكات العميقة، "
        "وهي أساس ورقة **Kelly, Malamud & Zhou (2024)** «فضيلة التعقيد» في التنبؤ بالعوائد."
    )

    @st.cache_data(show_spinner=False)
    def double_descent(reps: int = 30):
        rng = np.random.default_rng(9)
        n, d, nt = 40, 5, 1500
        ps = list(range(2, 121, 2))
        err = np.zeros(len(ps))
        for _ in range(reps):
            X = rng.standard_normal((n, d)); Xt = rng.standard_normal((nt, d))
            w = rng.standard_normal(d)
            f = lambda Z: np.tanh(Z @ w) + 0.3 * (Z[:, 0] ** 2)
            y = f(X) + rng.normal(0, .3, n); yt = f(Xt) + rng.normal(0, .3, nt)
            W = rng.standard_normal((d, max(ps))) / np.sqrt(d)
            F, Ft = np.maximum(X @ W, 0), np.maximum(Xt @ W, 0)
            for j, p in enumerate(ps):
                b = np.linalg.pinv(F[:, :p]) @ y
                err[j] += np.mean((yt - Ft[:, :p] @ b) ** 2)
        return np.array(ps), np.minimum(err / reps, 8)

    ps, err = double_descent()
    fig = go.Figure(
        data=[go.Scatter(x=ps[:1], y=err[:1], mode="lines", line=dict(color=ui.COLORS["indigo"], width=4), name="خطأ الاختبار")],
        frames=[go.Frame(data=[go.Scatter(x=ps[:k], y=err[:k])], name=str(int(ps[k - 1]))) for k in range(1, len(ps) + 1)],
    )
    fig.add_vline(x=40, line_dash="dash", line_color=ui.COLORS["rose"], annotation_text="عتبة الاستيفاء p = n", annotation_position="top")
    fig.update_xaxes(title="عدد المعلمات p (ميزات ReLU عشوائية)، n = 40", range=[0, 122])
    fig.update_yaxes(title="متوسط مربع خطأ الاختبار", range=[0, 8.3])
    ui.style_fig(fig, "منحنى الانحدار المزدوج")
    ui.add_play(fig, [int(p) for p in ps], duration=120, prefix="p = ")
    ui.show(fig)
    ui.note("**النظام الكلاسيكي** ($p<n$): منحنى U المعتاد. **عتبة الاستيفاء** ($p\\approx n$): الخطأ ينفجر. **النظام المفرط في المعلمات** ($p\\gg n$): "
            "يعود الخطأ للانخفاض لأن الحل ذا المعيار الأدنى يعمل كتنظيم ضمني. **الدرس للاقتصادي**: عدد المعلمات وحده ليس مقياسًا كافيًا للتعقيد.",
            "قراءة المنحنى", "indigo", "📈")

# ------------------------------------------------------------- losses --------
with tabs[7]:
    ui.section("دوال الخسارة: كل خسارة تقابل مقدّرًا قياسيًا", "Losses ↔ estimators", "rose")
    losses = pd.DataFrame([
        ("مربع الخطأ", "Squared (L2)", "(y−ŷ)²", "المتوسط الشرطي — OLS"),
        ("الخطأ المطلق", "Absolute (L1)", "|y−ŷ|", "الوسيط الشرطي — LAD"),
        ("خسارة الكمّيات", "Pinball / check loss", "ρ_τ(y−ŷ)", "الكمّي الشرطي — الانحدار الكمي (Koenker & Bassett, 1978)"),
        ("الإنتروبيا المتقاطعة", "Cross-entropy / log-loss", "−[y log p + (1−y) log(1−p)]", "الإمكان الأعظم لـ Logit"),
        ("خسارة هوبر", "Huber", "تربيعية قرب الصفر، خطية بعيدًا", "التقدير المتين (Robust M-estimation)"),
        ("خسارة المفصل", "Hinge", "max(0, 1−y·f)", "آلات المتجهات الداعمة"),
        ("ε-غير الحساسة", "ε-insensitive", "max(0, |y−ŷ|−ε)", "SVR — وجد غوليه كولومب أنها أسوأ من L2 للتنبؤ الكلي"),
        ("خسارة اللحظات", "Moment / GMM loss", "ḡ(θ)ᵀ W ḡ(θ)", "GMM — تستخدم في Adversarial GMM و DeepGMM"),
        ("بواقي التوازن", "Euler-equation residual", "‖u′(c) − βE[u′(c′)R]‖²", "الشبكات الموجّهة بالنظرية (DEQN)"),
    ], columns=["الخسارة", "English", "الصيغة", "المقدّر المقابل"])
    st.table(losses)
    ui.section("مقاييس التقييم", "Evaluation metrics", "teal")
    ui.cards([
        {"title": "للانحدار", "en": "Regression", "color": "teal", "body": "RMSE، MAE، MAPE، MASE، R² خارج العينة = 1 − Σ(y−ŷ)² / Σ(y−ȳ<sub>train</sub>)²."},
        {"title": "للتصنيف", "en": "Classification", "color": "violet", "body": "الدقة، الحساسية (Recall)، الدقة الموجبة (Precision)، F1، منحنى ROC و AUC، مصفوفة الالتباس."},
        {"title": "للتنبؤ الاحتمالي", "en": "Probabilistic", "color": "orange", "body": "Log score، CRPS، تغطية فترات التنبؤ، اختبار PIT."},
        {"title": "للمقارنة", "en": "Forecast comparison", "color": "rose", "body": "Diebold–Mariano (1995)، Clark–West للنماذج المتداخلة، مجموعة ثقة النماذج (MCS, Hansen et al., 2011)."},
    ], cols=4)
