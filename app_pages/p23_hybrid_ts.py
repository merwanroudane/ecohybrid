import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from core import ui

ui.hero(
    "الخوارزميات الهجينة للسلاسل الزمنية",
    "Hybrid Time-Series Algorithms — ARIMA/ARDL-NN, VAR-NN, VECM-ML, Neural Granger, GARCH-NN, Wavelet-ML, MIDAS-ML",
    "خوارزميات متخصصة تجمع مكوّنًا قياسيًا (ARIMA، ARDL، VAR، VECM، GARCH، اختبارات غرانجر والتكامل المشترك) مع مكوّن ذكي "
    "(شبكات، غابات، تنظيم، BART، تفكيك موجي). لكل هجين: الفكرة، المعادلة، المرجع، الأداة البرمجية، والمحاذير.",
    "sky", "rose",
)

tabs = st.tabs([
    ":material/map: الخريطة",
    ":material/merge_type: ARIMA/ARDL + NN",
    ":material/hub: VAR + ML",
    ":material/link: التكامل المشترك و ECM",
    ":material/compare_arrows: غرانجر + ML",
    ":material/show_chart: التقلب + ML",
    ":material/waves: الموجات والتردد",
    ":material/event: التواتر والانكسارات والأنظمة",
    ":material/table_view: جدول شامل",
])

with tabs[0]:
    ui.section("خريطة الهجين في السلاسل الزمنية", "Map of time-series hybrids", "sky")
    st.mermaid_chart(
        """
mindmap
  root((Time-series hybrids))
    Univariate
      ARIMA-ANN residual hybrid
      ARIMA-LSTM
      SARIMAX-LSTM
      ARDL-LSTM / GRU
    Multivariate
      Neural VAR / VANAR
      Deep VAR
      LSTM vs VAR
      BART-VAR / BAVART
      Lasso-VAR / BigVAR
      Macro random forest
    Long run
      SPECS penalized ECM
      Lasso VECM rank
      Neural ECM
      Error-correction NN
    Causality
      NN nonlinearity test
      Neural Granger cMLP cLSTM
      HD Granger post-double-selection
      PCMCI
    Volatility
      ANN-GARCH
      HAR-RNN
      ML realized volatility
    Time-frequency
      Wavelet-ANN
      WT-ARIMA-LSTM
      EMD / VMD hybrids
      TimeKAN
      QQ-KRLS
    Mixed and breaks
      sg-LASSO MIDAS
      NN change-point
      ML unit-root aggregation
"""
    )
    ui.note("كل هجين يجب أن يجيب عن ثلاثة أسئلة: **(1)** ما المكوّن الذي يبقى قياسيًا ولماذا (المعنى طويل الأجل، قابلية الاختبار، الاستقرار)؟ "
            "**(2)** ما المهمة الصغيرة التي تُسند للذكاء الاصطناعي (اللاخطية، الأبعاد، البواقي)؟ **(3)** كيف نقيّم بصدق (تقسيم زمني، Diebold–Mariano، بيانات آنية)؟",
            "الأسئلة الثلاثة", "sky", "❓")
    ui.note("تطبيقات Python جاهزة لكثير من هذه الهجائن متاحة في مكتبات د. مروان رودان على PyPI: **tsdml**، **ardldml**، **oracle-varx**، **drgct**، **shapneurgc**، "
            "**castor-causal**، **samovar-var**، **gpvar**، **tvwnn**، **hybridmetrics**، **bartardl**، **garchai**، **hybridecon**، **harlstmgarch**، **peltwtcn**، **wkrls**، **qqkrls**.",
            "مكتبات جاهزة", "fuchsia", "📦")
    ui.link("app_pages/p30_author_libs.py", "عرض صفحة المكتبات الهجينة بالتفصيل", ":material/deployed_code:")

# --------------------------------------------------------- ARIMA/ARDL + NN ---
with tabs[1]:
    ui.section("الهجين الخطي–اللاخطي: ARIMA–ANN و ARDL–LSTM", "Linear–nonlinear residual hybrids", "orange")
    st.latex(r"y_t=L_t+N_t,\qquad \hat L_t=\text{ARIMA/ARDL},\qquad e_t=y_t-\hat L_t,\qquad \hat N_t=f_{NN}(e_{t-1},\dots,e_{t-q}),\qquad \hat y_t=\hat L_t+\hat N_t")
    ui.cards([
        {"title": "ARIMA–ANN", "en": "Zhang (2003, Neurocomputing)", "color": "orange",
         "body": "الورقة المؤسسة (+3000 استشهاد): ARIMA للجزء الخطي، وشبكة عصبية لبواقيه اللاخطية. يقلل عدم اليقين في اختيار النموذج ويخفف الإفراط في توفيق الشبكة."},
        {"title": "تحسينات لاحقة", "en": "Khashei & Bijari (2011) and many others", "color": "amber",
         "body": "إدخال المتغيرات الأصلية والبواقي معًا للشبكة بدل افتراض الجمع البسيط L+N، وهجين ARIMA–SVR و ARIMA–LSTM."},
        {"title": "ARDL–LSTM / GRU", "en": "applied energy & finance hybrids", "color": "teal",
         "body": "ARDL للعلاقة الخطية قصيرة وطويلة الأجل، وLSTM/GRU للمكوّن اللاخطي. تُقيَّم غالبًا بـ RMSE/MAPE لا بالاستدلال على التكامل المشترك."},
        {"title": "SARIMAX–LSTM / VAR–LSTM–GRU", "en": "electricity demand & prices", "color": "violet",
         "body": "مكوّن موسمي/متعدد المتغيرات قياسي + شبكة متكررة؛ النتائج متباينة وقد يتفوق نموذج تسلسلي عميق صرف في آفاق طويلة."},
    ], cols=2)
    ui.section("مختبر متحرك: AR مقابل شبكة مقابل الهجين", "Animated residual-hybrid lab", "pink")

    @st.cache_data(show_spinner="تدريب النماذج...")
    def zhang_lab():
        rng = np.random.default_rng(12)
        T = 520
        y = np.zeros(T); e = rng.normal(0, .5, T)
        for t in range(2, T):
            y[t] = 0.6 * y[t - 1] - 0.2 * y[t - 2] + 0.9 * np.sin(1.6 * y[t - 1]) * np.exp(-0.3 * y[t - 2] ** 2) + e[t]
        tr = 400
        Z = np.column_stack([np.ones(T - 2), y[1:-1], y[:-2]]); tgt = y[2:]
        b = np.linalg.lstsq(Z[:tr - 2], tgt[:tr - 2], rcond=None)[0]
        lin = Z @ b
        res = tgt - lin
        # residual lags + level lags (Khashei-Bijari style); res[i] is dated t = i + 2
        R = np.column_stack([res[1:-1], res[:-2], y[3:-1], y[2:-2]])
        rt = res[2:]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nn_res = MLPRegressor(hidden_layer_sizes=(24,), activation="tanh", alpha=1e-3, max_iter=3000, random_state=0).fit(R[:tr - 4], rt[:tr - 4])
            nn_y = MLPRegressor(hidden_layer_sizes=(24,), activation="tanh", alpha=1e-3, max_iter=3000, random_state=0).fit(Z[:tr - 2, 1:], tgt[:tr - 2])
        hyb = lin[2:] + nn_res.predict(R)
        nn_only = nn_y.predict(Z[:, 1:])[2:]
        actual = tgt[2:]; ar = lin[2:]
        s = tr - 4
        return actual[s:], ar[s:], nn_only[s:], hyb[s:]

    act, ar, nn_o, hyb = zhang_lab()
    steps = list(range(10, len(act) + 1, 10))
    rm = lambda a, b: np.sqrt(np.mean((a - b) ** 2))

    def z_tr(m):
        x = np.arange(m)
        return [go.Scatter(x=x, y=act[:m], mode="lines", line=dict(color="#94a3b8"), name="الفعلي"),
                go.Scatter(x=x, y=ar[:m], mode="lines", line=dict(color=ui.COLORS["sky"]), name=f"AR(2): {rm(act[:m], ar[:m]):.3f}"),
                go.Scatter(x=x, y=nn_o[:m], mode="lines", line=dict(color=ui.COLORS["amber"]), name=f"شبكة فقط: {rm(act[:m], nn_o[:m]):.3f}"),
                go.Scatter(x=x, y=hyb[:m], mode="lines", line=dict(color=ui.COLORS["rose"], width=3), name=f"هجين AR+NN: {rm(act[:m], hyb[:m]):.3f}")]

    fig = go.Figure(data=z_tr(steps[0]), frames=[go.Frame(data=z_tr(m), name=str(m)) for m in steps])
    fig.update_xaxes(range=[0, len(act)], title="الفترة خارج العينة"); fig.update_yaxes(range=[act.min() - .5, act.max() + .5])
    ui.style_fig(fig, "تنبؤات خطوة واحدة خارج العينة — الأرقام في المفتاح = RMSE التراكمي")
    ui.add_play(fig, steps, duration=400, prefix="عدد التنبؤات: ", redraw=True)
    ui.show(fig)
    ui.note("في هذا المختبر تُدرَّب الشبكة على بواقي **عينة التدريب** فقط ثم تُقيَّم خارجها؛ النشر الصادق يتطلب ذلك. "
            "أشهر خطأ في أوراق الهجين: حساب البواقي وتدريب الشبكة على العينة كاملة ثم «التنبؤ» داخلها.", "تحذير تطبيقي", "amber", "⚠️")

# ---------------------------------------------------------------- VAR + ML ---
with tabs[2]:
    ui.section("نماذج VAR الهجينة", "VAR meets machine learning", "violet")
    st.latex(r"\mathbf y_t=\mathbf F_\omega(\mathbf y_{t-1},\dots,\mathbf y_{t-p})+\mathbf u_t\quad\text{vs.}\quad \mathbf y_t=\sum_{i=1}^{p}A_i\mathbf y_{t-i}+\mathbf u_t")
    vr = pd.DataFrame([
        ("Neural VAR (VANAR)", "شبكة عصبية بدل المعادلات الخطية مع استخراج ميزات آلي", "arXiv 1903.09395 (بيانات الفلبين)", "تنبؤ وسببية أفضل؛ صعوبة في المسارات المصدومة"),
        ("Deep VAR", "كل معادلة VAR تُستبدل بشبكة عميقة", "NeurIPS 2021 workshop (BSE)", "ملاءمة أفضل في الركود؛ دوال الاستجابة وفترات الثقة لم تُطوَّر بعد"),
        ("LSTM مقابل VAR", "شبكات متكررة متعددة المتغيرات للاقتصاد الكلي الأمريكي", "Verstyuk", "تنبؤ أفضل حتى بعينات صغيرة؛ استجابات أقرب للنظرية لصدمة الفائدة"),
        ("SVAR لا خطي", "SVAR لا خطي بالكامل يُعرَّف بتحليل الابتكارات المستقلة", "preprint 2026 (النفط)", "واعد؛ غير محكّم بعد"),
        ("BAVART / BART-VAR", "أشجار انحدار بايزية جمعية داخل VAR", "Huber & Rossini; Clark et al. (2023, IER)", "التنبؤ بالذيول ولا خطية التوزيع التنبؤي"),
        ("BART بتواتر مختلط", "VAR لا معلمي للتنبؤ الآني", "Huber et al. (2023, J. Econometrics)", "أداء جيد خلال الجائحة"),
        ("Lasso-VAR / BigVAR", "تنظيم هرمي للفجوات والمتغيرات في VAR كبير", "Nicholson, Wilms, Bien & Matteson (2020, JMLR)", "VAR بمئات المتغيرات مع تفسير"),
        ("BVAR الكبير", "قبْلي مينيسوتا كتنظيم ريدج منظّم", "Bańbura, Giannone & Reichlin (2010, JAE)", "مرجعية قوية يصعب التفوق عليها"),
        ("FAVAR + Autoencoder", "عوامل لا خطية بمشفّر ذاتي داخل FAVAR", "امتداد لـ Bernanke, Boivin & Eliasz (2005)", "العوامل الكامنة بلا معنى اقتصادي مباشر"),
        ("الغابة الكلية MRF", "انحدار خطي بمعاملات متغيرة زمنيًا معممة تُقدَّر بغابة", "Goulet Coulombe (2024, JAE)", "تفسير عبر GTVPs"),
    ], columns=["الهجين", "الفكرة", "المرجع", "النتيجة/المحذور"])
    st.dataframe(vr, hide_index=True, height=420)
    ui.section("مختبر: الاستجابة النبضية اللاخطية تعتمد على الحالة والإشارة والحجم", "Nonlinear impulse responses", "pink")
    st.markdown("في VAR الخطي الاستجابة متماثلة ومتناسبة مع حجم الصدمة. في النماذج اللاخطية (ومنها الشبكية) **ليست كذلك**؛ لذلك نستخدم **دوال الاستجابة المعممة (GIRF)** "
                "لكوب وبيساران وبوتر (1996) بالمحاكاة. النموذج التوضيحي: $y_t=0.5y_{t-1}+0.4\\max(y_{t-1},0)+\\varepsilon_t$.")
    size = st.select_slider("حجم الصدمة", [0.5, 1.0, 2.0, 3.0], value=1.0, key="girf_s")

    @st.cache_data(show_spinner=False)
    def girf(size: float, H: int = 15, R: int = 3000):
        rng = np.random.default_rng(1)
        f = lambda y: 0.5 * y + 0.4 * np.maximum(y, 0)
        out = {}
        for sign in (1, -1):
            base = np.zeros((R, H)); shock = np.zeros((R, H))
            e = rng.normal(0, 1, (R, H))
            yb = np.zeros(R); ys = np.zeros(R)
            for h in range(H):
                yb = f(yb) + e[:, h]
                ys = f(ys) + e[:, h] + (sign * size if h == 0 else 0)
                base[:, h] = yb; shock[:, h] = ys
            out[sign] = (shock - base).mean(0) / size
        return out

    gi = girf(size)
    fig = go.Figure([go.Scatter(y=gi[1], name="صدمة موجبة (مقسومة على حجمها)", line=dict(color=ui.COLORS["emerald"], width=4)),
                     go.Scatter(y=-gi[-1], name="صدمة سالبة (معكوسة ومقسومة على حجمها)", line=dict(color=ui.COLORS["rose"], width=4, dash="dash"))])
    fig.update_xaxes(title="الأفق h"); fig.update_yaxes(title="الاستجابة لكل وحدة صدمة")
    ui.style_fig(fig, "عدم التماثل: لو كان النموذج خطيًا لتطابق المنحنيان", height=400)
    ui.show(fig)

# -------------------------------------------------------- cointegration ------
with tabs[3]:
    ui.section("التكامل المشترك ونماذج تصحيح الخطأ مع تعلم الآلة", "Cointegration, ECM & VECM with ML", "teal")
    st.latex(r"\Delta\mathbf y_t=\boldsymbol\alpha\boldsymbol\beta^{\top}\mathbf y_{t-1}+\sum_{i=1}^{p-1}\Gamma_i\Delta\mathbf y_{t-i}+\mathbf u_t"
             r"\quad\Longrightarrow\quad \Delta\mathbf y_t=g_\omega\big(\boldsymbol\beta^{\top}\mathbf y_{t-1},\ \Delta\mathbf y_{t-1},\dots\big)+\mathbf u_t")
    ui.cards([
        {"title": "SPECS", "en": "Smeekes & Wijler (2021, J. Econometrics)", "color": "teal",
         "body": "<b>Single-equation Penalized Error-Correction Selector</b>: انحدار منظّم لنموذج تصحيح خطأ شرطي متناثر في بيانات كبيرة متكاملة، <b>دون اختبار مسبق</b> لرتبة التكامل أو رتبة التكامل المشترك؛ خاصية العرّاف؛ تنبؤ آني للبطالة الهولندية بـ Google Trends. حزمة R: specs."},
        {"title": "VECM آلي بالانكماش", "en": "Liao & Phillips (2015, Econometric Theory)", "color": "violet",
         "body": "اختيار مشترك لرتبة التكامل المشترك وعدد الفجوات وتقدير المعلمات بعقوبات لاسو تكيفية — بديل آلي لاختبارات جوهانسن المتتابعة."},
        {"title": "ECM عصبي لا خطي", "en": "Fang & Yang (2023)", "color": "orange",
         "body": "شبكة هجينة (متكررة خطية + شبكة BP) لتعلم تصحيح خطأ لا خطي؛ تطبيق على الذهب ومؤشر الدولار. شرط: السلاسل تجتاز اختبار التكامل المشترك أولًا."},
        {"title": "شبكات تصحيح الخطأ", "en": "Error Correction Neural Networks (2020)", "color": "rose",
         "body": "إدراج آلية تصحيح الخطأ داخل معمارية الشبكة (ARMA-cell / ECNN) للحفاظ على معلومة المستويات التي يضيعها التفريق."},
        {"title": "تصحيح خطأ بعتبات وأشجار", "en": "Threshold / smooth-transition ECM + ML", "color": "sky",
         "body": "تكيّف غير متماثل نحو التوازن (Balke & Fomby, 1997) يمكن اكتشاف عتباته بالأشجار أو الغابات — مع ضرورة استدلال عتبات صالح."},
        {"title": "ARDL + ML", "en": "Bounds-test caution", "color": "amber",
         "body": "يمكن إسناد مكوّن لاخطي أو البواقي لـ ML، لكن <b>تحسن RMSE لا يثبت التكامل المشترك</b> ولا يعدّل فروض اختبار الحدود لبيساران وشين وسميث (2001)."},
    ], cols=3)
    ui.note("**لماذا لا نفرّق ونرمي البيانات للشبكة؟** لأن التفريق يُضيع معلومة **المستويات** والعلاقة طويلة الأجل. حد تصحيح الخطأ $\\beta^\\top y_{t-1}$ ميزة اقتصادية ثمينة: "
            "أدخله صراحةً كمدخل للمتعلم، أو قيّد المعمارية لتحترمه. كذلك: اختبارات جذور الوحدة تبقى ضرورية — الشبكة لا تعرف أن السلسلة I(1).",
            "المبدأ", "teal", "🔗")

# --------------------------------------------------------------- Granger ----
with tabs[4]:
    ui.section("سببية غرانجر في عصر تعلم الآلة", "Granger causality with ML", "rose")
    ui.ladder([
        ("اختبار اللاخطية المهملة", "Lee, White & Granger (1993, J. Econometrics)", "اختبار مضاعف لاغرانج قائم على شبكة عصبية بأوزان عشوائية؛ حجم مناسب وقوة جيدة."),
        ("غرانجر العصبي", "Tank, Covert, Foti, Shojaie & Fox (2022, IEEE TPAMI)", "cMLP و cLSTM مع عقوبة لاسو المجموعات على أوزان الطبقة الأولى: إن صُفّرت أوزان فجوات $x_j$ ⇒ $x_j$ لا يسبب $y$ بمعنى غرانجر. كود: Neural-GC."),
        ("غرانجر عالي الأبعاد", "Hecq, Margaritella & Smeekes (2023, J. Financial Econometrics)", "اختبار LM لسببية غرانجر في VAR عالي الأبعاد مع إجراء <b>اختيار مزدوج</b> (PDS) لضبط الحجم بعد لاسو؛ تطبيق على علاقة النقود–الدخل وشبكات التقلب."),
        ("PCMCI", "Runge et al. (2019, Science Advances)", "اكتشاف شبكات سببية زمنية في بيانات كبيرة غير خطية باختبارات استقلال شرطي (خطية أو لا معلمية)؛ حزمة tigramite؛ PCMCI+ للروابط المتزامنة."),
        ("غرانجر بالغابات والنواة", "Kernel / RF-based Granger", "مقارنة الأداء التنبؤي خارج العينة مع وبدون فجوات $x$ عبر متعلم مرن + اختبار Diebold–Mariano أو اختبار تبديلي."),
        ("السببية في الكمّيات", "Troster (2018, Econometric Reviews)", "اختبار سببية غرانجر في الكمّيات — يكمّل الأدوات الهجينة للتوزيعات."),
    ])
    ui.section("مختبر: علاقة لا خطية يفوتها اختبار F الخطي", "Lab — nonlinear Granger causality", "pink")
    st.markdown("نولّد $y_t=0.3y_{t-1}+\\gamma\\,(x_{t-1}^2-1)+\\varepsilon_t$: $x$ يسبب $y$ **لا خطيًا** فقط. اختبار F الخطي يكاد لا يرى العلاقة (الارتباط الخطي قريب من الصفر)، "
                "بينما تقارن النسخة الهجينة خطأ تنبؤ غابة عشوائية خارج العينة **مع** و**بدون** فجوات $x$.")
    gam = st.slider("قوة الأثر اللاخطي γ", 0.0, 1.0, 0.6, 0.1, key="gc_g")

    @st.cache_data(show_spinner="اختبار...")
    def granger_lab(g: float, T: int = 800):
        rng = np.random.default_rng(4)
        x = rng.normal(size=T); y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.3 * y[t - 1] + g * (x[t - 1] ** 2 - 1) + rng.normal()
        Yt, ylag, xlag = y[1:], y[:-1], x[:-1]
        Xr = np.column_stack([np.ones(T - 1), ylag]); Xu = np.column_stack([Xr, xlag])
        rss = lambda X: np.sum((Yt - X @ np.linalg.lstsq(X, Yt, rcond=None)[0]) ** 2)
        F = (rss(Xr) - rss(Xu)) / (rss(Xu) / (T - 1 - 3))
        pF = 1 - stats.f.cdf(F, 1, T - 4)
        cut = 500
        rf = lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=10, random_state=0, n_jobs=-1)
        e_r = Yt[cut:] - rf().fit(ylag[:cut, None], Yt[:cut]).predict(ylag[cut:, None])
        e_u = Yt[cut:] - rf().fit(np.column_stack([ylag, xlag])[:cut], Yt[:cut]).predict(np.column_stack([ylag, xlag])[cut:])
        d = e_r ** 2 - e_u ** 2
        dm = d.mean() / np.sqrt(d.var(ddof=1) / len(d))
        pdm = 1 - stats.norm.cdf(dm)
        return F, pF, np.mean(e_r ** 2), np.mean(e_u ** 2), dm, pdm, xlag, Yt

    F, pF, mr, mu, dm, pdm, xl, yt = granger_lab(gam)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F خطي", f"{F:.2f}", f"p = {pF:.3f}", delta_color="off")
    c2.metric("MSE بدون x (غابة)", f"{mr:.3f}")
    c3.metric("MSE مع x (غابة)", f"{mu:.3f}")
    c4.metric("DM أحادي الجانب", f"{dm:.2f}", f"p = {pdm:.4f}", delta_color="off")
    fig = go.Figure(go.Scatter(x=xl, y=yt, mode="markers", marker=dict(size=4, color=ui.COLORS["rose"], opacity=.5)))
    fig.update_xaxes(title="x(t−1)"); fig.update_yaxes(title="y(t)")
    ui.style_fig(fig, "العلاقة على شكل U: الارتباط الخطي ≈ 0 رغم وجود سببية قوية", height=380, legend_h=False)
    ui.show(fig)

# ------------------------------------------------------------- volatility ---
with tabs[5]:
    ui.section("نمذجة التقلب الهجينة", "Volatility: GARCH/HAR + ML", "fuchsia")
    st.latex(r"\sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2+\beta\sigma_{t-1}^2+\underbrace{g_\omega\big(\varepsilon_{t-1},\sigma_{t-1}^2,\mathbf x_{t-1}\big)}_{\text{ANN component}},\qquad "
             r"RV_{t+1}=f_{ML}\big(RV_t,\overline{RV}_{t-4:t},\overline{RV}_{t-21:t},\dots\big)")
    ui.cards([
        {"title": "ANN–GARCH", "en": "Donaldson & Kamstra (1997)", "color": "fuchsia", "body": "إضافة مكوّن شبكي لمعادلة التباين الشرطي لالتقاط عدم التماثل واللاخطية."},
        {"title": "HAR + ML", "en": "Corsi (2009) HAR as the benchmark", "color": "violet", "body": "نموذج HAR للتقلب المحقق مع غابات وتعزيز وشبكات؛ الأشجار والشبكات تتفوق خصوصًا في الآفاق الأطول وفق مراجعات حديثة."},
        {"title": "HAR-RNN", "en": "recurrent extension for VaR", "color": "teal", "body": "يدمج الذاكرة الطويلة لـ HAR مع شبكة متكررة للديناميكيات اللاخطية في قيمة المخاطرة."},
        {"title": "GARCH كمدخل للشبكة", "en": "GARCH-feature networks", "color": "orange", "body": "استخدام مخرجات GARCH (التباين، البواقي المعيارية) كمدخلات LSTM؛ تحسّن متكرر في الدراسات التطبيقية."},
        {"title": "نماذج تأسيسية للتقلب", "en": "TSFM for realized volatility (2025)", "color": "sky", "body": "اختبار نماذج أساس مسبقة التدريب على التقلب المحقق مقابل HAR."},
        {"title": "المحاذير", "en": "Caveats", "color": "rose", "body": "دراسات مفردة غالبًا؛ لا إجماع على التفوق؛ يجب تقييم الخسائر المناسبة للتقلب (QLIKE) وتكاليف التداول وفترات الانهيار."},
    ], cols=3)

# ------------------------------------------------------------- wavelets -----
with tabs[6]:
    ui.section("الهجين الموجي والترددي", "Wavelet & time–frequency hybrids", "emerald")
    st.mermaid_chart(
        """
flowchart LR
    Y[Series y_t] --> W[Decomposition<br/>MODWT / DWT / EMD / VMD]
    W --> D1[Detail D1: high frequency]
    W --> D2[Detail D2..Dj]
    W --> A[Approximation: trend]
    D1 --> M1[LSTM / SVR / RF]
    D2 --> M2[ARIMA or NN]
    A --> M3[ARIMA / ARDL]
    M1 & M2 & M3 --> S[Sum or stack the component forecasts]
"""
    )
    ui.cards([
        {"title": "Wavelet–ANN / Wavelet–LSTM", "en": "decomposition-ensemble forecasting", "color": "emerald", "body": "تفكيك إلى مقاييس زمنية وتنبؤ كل مكوّن بنموذج مناسب؛ شائع في أسعار النفط والصرف."},
        {"title": "WT–ARIMA–LSTM", "en": "NAJEF (2024); JRFM (2023)", "color": "teal", "body": "ثلاثي: موجات + ARIMA للخطي + LSTM/RF/XGB للاخطي؛ تفوق مبلّغ عنه على الذهب واليورو."},
        {"title": "EMD / VMD hybrids", "en": "empirical / variational mode decomposition", "color": "violet", "body": "تفكيك تكيّفي إلى دوال نمطية ثم تعلم آلة لكل نمط."},
        {"title": "TimeKAN", "en": "Scientific Reports (2026)", "color": "orange", "body": "شبكة كولموغوروف–أرنولد مع تفكيك ترددي — مجال تنبؤي لا اختبار سببية أو تكامل."},
        {"title": "QQ + KRLS", "en": "Sim & Zhou (2015) + Hainmueller & Hazlett (2014)", "color": "rose", "body": "مزج الانحدار الكمّي-على-الكمّي مع المربعات الصغرى المنظّمة بالنواة (مثل حزمة qqkrls) لالتقاط اللاخطية عبر الكمّيات."},
        {"title": "الترابط الترددي + ML", "en": "frequency connectedness", "color": "sky", "body": "مؤشرات انتقال الصدمات (Diebold–Yilmaz، Baruník–Křehlík) مع ML لمحدداتها — ليست سببية بنيوية."},
    ], cols=3)
    ui.note("**أخطر عيب في الهجين الموجي**: تطبيق التفكيك على **العينة كاملة** قبل التقسيم. المرشحات ثنائية الجانب تستخدم قيمًا مستقبلية، فتتسرب معلومة المستقبل إلى ميزات التدريب "
            "وتظهر دقة وهمية. الحل: تفكيك داخل كل نافذة متدحرجة أو مرشحات أحادية الجانب (MODWT مع معالجة الحدود).", "تسرب المرشحات", "rose", "🚫")

# ------------------------------------------------------- mixed/breaks -------
with tabs[7]:
    ui.section("التواتر المختلط والانكسارات والأنظمة وجذور الوحدة", "Mixed frequency, breaks, regimes and unit roots with ML", "amber")
    ui.cards([
        {"title": "sg-LASSO-MIDAS", "en": "Babii, Ghysels & Striaukas (2022, JBES)", "color": "amber", "body": "لاسو المجموعات المتناثر لانحدارات MIDAS بقواميس متعددة الحدود؛ نظرية لعمليات τ-mixing ذات ذيول ثقيلة؛ امتداد للبانل (2023). حزمة midasml."},
        {"title": "كشف نقاط التغير بالتعلم العميق", "en": "Li, Fearnhead, Fryzlewicz & Wang (2024, JRSSB)", "color": "violet", "body": "تدريب شبكة لتصنيف وجود تغير ⇒ أداء لا يقل عن CUSUM ويتفوق عند فشل افتراضاته."},
        {"title": "اختبارات انكسار بالشبكات", "en": "Kapetanios (2002)", "color": "teal", "body": "تقريب التوقع الشرطي بشبكة ثم اختبارات انكسار على البواقي لسلاسل لا خطية مستقرة."},
        {"title": "تجميع اختبارات جذر الوحدة", "en": "Cornwall, Chen & Sauley (2021)", "color": "orange", "body": "معاملة اختبارات جذر الوحدة كمتعلمات ضعيفة وتجميعها بالغابات والتعزيز (عرض عمل أولي)."},
        {"title": "تبديل الأنظمة + ML", "en": "Markov-switching / TAR with trees", "color": "rose", "body": "HMM لهاميلتون كنموذج مزيج؛ الأشجار تكتشف عتبات متعددة؛ MRF يعمم المعاملات المتغيرة."},
        {"title": "VAR بمعاملات متغيرة + ML", "en": "TVP and GTVP models", "color": "sky", "body": "MRF و BART-VAR كبدائل مرنة لـ TVP-VAR التقليدي مع الحفاظ على قابلية التفسير."},
    ], cols=3)

with tabs[8]:
    ui.section("جدول شامل للخوارزميات الهجينة في السلاسل الزمنية", "Comprehensive table", "slate")
    big = pd.DataFrame([
        ("ARIMA–ANN", "ARIMA", "MLP على البواقي", "تنبؤ", "Zhang (2003)", "Python/R"),
        ("ARDL–LSTM", "ARDL", "LSTM/GRU", "تنبؤ", "تطبيقات طاقة/مال", "Keras/PyTorch"),
        ("Neural VAR / Deep VAR", "VAR", "شبكات لكل معادلة", "تنبؤ، سببية", "arXiv 1903.09395؛ NeurIPS 2021", "PyTorch"),
        ("BART-VAR", "VAR", "أشجار بايزية", "تنبؤ بالذيول", "Clark et al. (2023)", "R (flexBART…)"),
        ("Lasso-VAR", "VAR", "لاسو هرمي", "VAR كبير", "Nicholson et al. (2020)", "R BigVAR"),
        ("MRF", "انحدار TVP", "غابة عشوائية", "تنبؤ مفسَّر", "Goulet Coulombe (2024)", "R MacroRF"),
        ("SPECS", "ECM شرطي", "انحدار منظّم", "تكامل مشترك متناثر", "Smeekes & Wijler (2021)", "R specs"),
        ("Lasso-VECM", "VECM", "لاسو تكيفي", "اختيار الرتبة", "Liao & Phillips (2015)", "—"),
        ("Neural ECM", "ECM", "شبكة هجينة", "تصحيح خطأ لا خطي", "Fang & Yang (2023)", "—"),
        ("NN nonlinearity test", "اختبار LM", "شبكة بأوزان عشوائية", "اختبار", "Lee, White & Granger (1993)", "R tseries"),
        ("Neural Granger", "سببية غرانجر", "cMLP/cLSTM + group lasso", "اكتشاف سببية", "Tank et al. (2022)", "Python Neural-GC"),
        ("HD Granger PDS", "VAR + LM", "لاسو اختيار مزدوج", "اختبار سببية", "Hecq et al. (2023)", "R"),
        ("PCMCI", "استقلال شرطي", "اختبارات لا معلمية", "شبكات سببية", "Runge et al. (2019)", "Python tigramite"),
        ("ANN–GARCH", "GARCH", "شبكة", "تقلب", "Donaldson & Kamstra (1997)", "—"),
        ("HAR–ML", "HAR", "غابات/شبكات", "تقلب محقق", "مراجعات 2020s", "Python"),
        ("Wavelet–LSTM", "تفكيك موجي", "LSTM", "تنبؤ", "تطبيقات النفط/الصرف", "PyWavelets + Keras"),
        ("QQ-KRLS", "كمّي على كمّي", "KRLS", "لا خطية عبر الكمّيات", "Sim & Zhou؛ Hainmueller & Hazlett", "R qqkrls"),
        ("sg-LASSO-MIDAS", "MIDAS", "sparse-group lasso", "تنبؤ آني", "Babii et al. (2022)", "R midasml"),
        ("Deep change-point", "اختبار انكسار", "مصنّف عصبي", "كشف تغيرات", "Li et al. (2024)", "—"),
    ], columns=["الهجين", "المكوّن القياسي", "المكوّن الذكي", "الغرض", "المرجع", "البرمجيات"])
    st.dataframe(big, hide_index=True, height=600)
    ui.refs("Zhang (2003); Lee, White & Granger (1993); Liao & Phillips (2015); Smeekes & Wijler (2021); Hecq, Margaritella & Smeekes (2023); Tank et al. (2022); "
            "Runge et al. (2019); Clark et al. (2023); Nicholson et al. (2020); Goulet Coulombe (2024); Babii, Ghysels & Striaukas (2022); Koop, Pesaran & Potter (1996).")
