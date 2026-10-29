import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "كيف ندمج؟ المنهجية العملية خطوة بخطوة",
    "How to Integrate — A Step-by-Step Research Workflow",
    "الهندسة السليمة لمشروع قياس اقتصادي مع ذكاء اصطناعي: من السؤال إلى النشر، مع تقسيم العينة والتوفيق المتقاطع ومنع التسرب "
    "واختيار المتعلمات والضبط والتوثيق — وقوالب أكواد جاهزة.",
    "teal", "orange",
)

tabs = st.tabs([
    ":material/route: خط سير البحث",
    ":material/checklist: الخطوات العشر",
    ":material/sync: التوفيق المتقاطع",
    ":material/water_drop: منع التسرب",
    ":material/psychology: اختيار المتعلم",
    ":material/stacks: أنماط الدمج التنبؤي",
    ":material/code: قوالب الأكواد",
])

with tabs[0]:
    ui.section("خط سير مشروع الدمج", "The integration pipeline", "teal")
    st.mermaid_chart(
        """
flowchart LR
    Q[1. Economic question] --> E[2. Estimand & identification<br/>DAG, design]
    E --> D[3. Data plan<br/>timing, vintages, units]
    D --> B[4. Econometric benchmark<br/>OLS, IV, VAR, ARDL...]
    B --> I[5. Integration point<br/>where does ML enter?]
    I --> V[6. Inference & diagnostics<br/>orthogonal score, CI, coverage]
    V --> R[7. Robustness & sensitivity<br/>learners, seeds, prompts]
    R --> P[8. Reporting & replication]
"""
    )
    ui.cards([
        {"title": "ابدأ بالمعنى", "en": "Meaning first", "color": "teal", "body": "المعنى الاقتصادي للمتغيرات، والرسم السببي DAG إن كان السؤال سببيًا، والقيود المحاسبية والنظرية إن كان هيكليًا."},
        {"title": "حدد نقطة الدخول", "en": "Entry point", "color": "violet", "body": "اختيار عوامل؟ تعلم بواقي؟ دالة مساعدة؟ قراءة نصوص؟ حل معادلة توازن؟ تقدير توقعات؟"},
        {"title": "ثلاثة نماذج دائمًا", "en": "Three-way comparison", "color": "orange", "body": "قياس فقط (Econometrics-only)، ذكاء اصطناعي فقط (AI-only)، والهجين (Hybrid). إن لم يتفوق الهجين بوضوح فالبساطة فضيلة."},
    ])

with tabs[1]:
    ui.section("قائمة تصميمية من عشر خطوات", "Ten design steps", "violet")
    ui.ladder([
        ("فرضية قابلة للتكذيب", "Falsifiable hypothesis", "عيّن المتغير التابع والمعالجة/المفسرات والزمن والأفق ومجتمع الدراسة."),
        ("هدف الاستدلال", "Inferential target", "معامل؟ أثر سببي؟ توزيع؟ كمّي؟ دالة سياسة؟ مؤشر تركيبي؟"),
        ("الآلية ومجموعة المعلومات", "Mechanism & information set", "ارسم الآلية الاقتصادية وحدد ما لم يكن متاحًا لحظة القرار."),
        ("طبيعة البيانات", "Data audit", "النوع والتواتر وتواريخ الصدور والمراجعات (Vintages)."),
        ("فحوص ما قبل النمذجة", "Pre-modeling checks", "المفقودات، الشواذ، الانكسارات، الاعتماد، جذور الوحدة والتكامل المشترك عند الحاجة."),
        ("خط أساس قياسي وتقسيم", "Baseline & split", "نموذج قياسي مبرَّر وتقسيم بيانات يمنع التسرب."),
        ("أصغر مهمة لـ ML", "Minimal ML task", "اختر الخوارزمية لأصغر مهمة ذات قيمة مضافة، دون تغيير هدف الاستدلال."),
        ("ربط النظرية بالخوارزمية", "Theory–algorithm link", "حين تكون القيود موثوقة؛ ثم اختبر حذفها (Ablation)."),
        ("عدم اليقين الملائم", "Appropriate uncertainty", "HAC، العناقيد، Block bootstrap، الدالة المتعامدة، معايرة عينة التحقق."),
        ("المقارنة والتوثيق", "Comparison & documentation", "خارج العينة، عبر أنظمة وأماكن مختلفة، مع توثيق كل الإصدارات."),
    ])
    ui.refs("مقتبسة ومعدّلة من موسوعة د. مروان رودان (2026)، الفصل 6.")

with tabs[2]:
    ui.section("التوفيق المتقاطع: كل طيّة تؤدي الدورين", "Cross-fitting animated", "sky")
    st.markdown("نقسم العينة إلى $K$ طيات. في كل دورة: نتعلم الدوال المساعدة $\\hat\\eta$ على $K-1$ طيات (أزرق)، ونحسب البواقي والدالة المتعامدة على الطيّة المتبقية (برتقالي). "
                "ثم ندمج التقديرات من كل الطيات. هكذا **لا تُستخدم مشاهدة في تعلم $\\hat\\eta$ ثم في تقييمها** ⇒ يزول تحيز الإفراط في التوفيق.")
    K = 5
    n = 50
    folds = np.repeat(np.arange(K), n // K)

    def cf_trace(k):
        colors = [ui.COLORS["orange"] if f == k else ui.COLORS["sky"] for f in folds]
        return go.Bar(x=np.arange(n), y=np.ones(n), marker_color=colors, hoverinfo="skip")

    fig = go.Figure(data=[cf_trace(0)],
                    frames=[go.Frame(data=[cf_trace(k)], name=str(k + 1),
                                     layout=go.Layout(title_text=f"الدورة {k + 1}: تعلّم η̂ على الطيات الأخرى ← قيّم ψ على الطيّة {k + 1}")) for k in range(K)])
    fig.update_yaxes(visible=False); fig.update_xaxes(title="المشاهدات مرتبة حسب الطيّة")
    ui.style_fig(fig, "الدورة 1: تعلّم η̂ على الطيات الأخرى ← قيّم ψ على الطيّة 1", height=320, legend_h=False)
    fig.update_layout(bargap=0.05)
    ui.add_play(fig, list(range(1, K + 1)), duration=1000, prefix="الدورة: ", redraw=True)
    ui.show(fig)
    st.latex(r"\hat\theta=\frac1K\sum_{k=1}^{K}\hat\theta_k,\qquad \frac1{n_k}\sum_{i\in I_k}\psi\big(W_i;\hat\theta_k,\hat\eta_{-k}\big)=0")
    ui.cards([
        {"title": "DML1 مقابل DML2", "en": "Aggregation", "color": "sky", "body": "DML1 يأخذ متوسط تقديرات الطيات؛ DML2 يحل معادلة لحظية واحدة مجمّعة — مفضّل في العينات الصغيرة."},
        {"title": "تكرار التقسيم", "en": "Repeated splitting", "color": "violet", "body": "كرر التقسيم S مرة وخذ الوسيط لتقليل حساسية النتيجة لتقسيم واحد (Chernozhukov et al., 2018)."},
        {"title": "طيات البانل والزمن", "en": "Dependent data", "color": "orange", "body": "في البانل: طيات حسب الوحدة. في الزمن: «اترك الجيران» (Semenova et al., 2023) أو كتل مع فجوة."},
    ])

with tabs[3]:
    ui.section("منع التسرب: أخطر عيب في الأبحاث الهجينة", "Data leakage", "rose")
    leak = pd.DataFrame([
        ("التسرب الزمني", "Temporal leakage", "استخدام بيانات لاحقة في التدريب أو في بناء الميزات", "تقسيم زمني صارم، أصل متدحرج"),
        ("تسرب المراجعات", "Revision / vintage leakage", "استخدام القيم النهائية المنقحة بدل المتاحة وقت التنبؤ", "قواعد بيانات Real-time (ALFRED)"),
        ("تسرب المرشحات", "Two-sided filter leakage", "مرشح HP أو تفكيك موجي ثنائي الجانب على العينة كاملة", "مرشحات أحادية الجانب، تفكيك داخل كل نافذة"),
        ("تسرب المعالجة المسبقة", "Preprocessing leakage", "التوحيد/التعويض باستخدام إحصاءات العينة كاملة", "احسب الإحصاءات على التدريب فقط (Pipeline)"),
        ("تسرب الوحدة", "Unit leakage (panel)", "مشاهدات الفرد نفسه في التدريب والاختبار", "Grouped CV حسب الوحدة"),
        ("تسرب الهدف", "Target leakage", "ميزة هي نتيجة للهدف أو مرادفة له", "مراجعة اقتصادية لكل ميزة"),
        ("تسرب التدريب المسبق", "Pretraining contamination", "LLM/TSFM رأى المستقبل في بيانات تدريبه", "نماذج مؤرخة (ChronoGPT, DatedGPT)، اختبارات النظر للأمام"),
        ("تسرب الاختيار", "Selection leakage", "اختيار المواصفة بعد رؤية نتائج الاختبار", "عينة محجوزة نهائية لا تُلمس، تسجيل مسبق"),
    ], columns=["النوع", "English", "كيف يحدث؟", "العلاج"])
    st.table(leak)
    ui.note("تحسّن RMSE «المذهل» في ورقة هجينة سببه في أحيان كثيرة **تسرب** لا قوة الخوارزمية. القاعدة: **كل ما لا تعرفه لحظة القرار ممنوع على النموذج**.",
            "قاعدة", "rose", "🚫")

with tabs[4]:
    ui.section("أي متعلم أختار؟", "Choosing the learner", "emerald")
    c1, c2 = st.columns([1, 2])
    with c1:
        n_size = st.select_slider("حجم العينة", ["< 100", "100–1,000", "1,000–100,000", "> 100,000"], value="100–1,000", key="lr_n")
        p_dim = st.select_slider("عدد المتغيرات", ["قليل", "متوسط", "كبير (p≈n)", "ضخم (p≫n)"], value="متوسط", key="lr_p")
        dtype = st.selectbox("نوع البيانات", ["جدولية", "سلسلة زمنية", "بانل", "نص", "صورة"], key="lr_t")
        nonlin = st.toggle("لا خطية وتفاعلات متوقعة", value=True, key="lr_nl")
    recs = []
    if dtype == "نص":
        recs = ["قواميس/TF-IDF + لاسو كخط أساس", "تضمينات (Embeddings) + لاسو/غابة", "FinBERT أو LLM مع عينة تحقق بشرية"]
    elif dtype == "صورة":
        recs = ["CNN مدربة مسبقًا مع تعلم بالنقل", "ViT/نماذج متعددة الوسائط", "تحقق ميداني + تصحيح خطأ القياس"]
    else:
        if n_size == "< 100":
            recs = ["OLS/ARDL/BVAR أو لاسو مع ضبط حذر", "تجنب الشبكات العميقة", "تنظيم بايزي بقبْلي معلوماتي"]
        elif p_dim in ("كبير (p≈n)", "ضخم (p≫n)"):
            recs = ["Lasso / Elastic Net (خط أساس قوي)", "عوامل (PCA) + انحدار", "غابة عشوائية/تعزيز للتحقق من اللاخطية"]
        elif nonlin:
            recs = ["Random forest (متين وقليل الضبط)", "Gradient boosting (أدق مع ضبط)", "شبكة عصبية إن كانت n كبيرة جدًا"]
        else:
            recs = ["OLS/Lasso — قد يكفي", "Ridge عند التعدد الخطي", "قارن مع غابة كتحقق من المواصفة"]
        if dtype == "سلسلة زمنية":
            recs.append("تقسيم زمني + معيار AR/VAR/BVAR/DFM إلزامي")
        if dtype == "بانل":
            recs.append("Grouped CV + آثار ثابتة (DML-FE لكلارك وبولسيلي)")
    with c2:
        with ui.tinted("emerald"):
            st.markdown("#### التوصية المقترحة")
            for r in recs:
                st.markdown(f"- {r}")
            st.markdown("**وفي DML تحديدًا**: استخدم **تجميعًا (Ensemble / Super learner)** لعدة متعلمات، وأبلغ عن حساسية النتيجة لاختيار المتعلم.")
    ui.note("نظرية DML تتطلب أن يكون **حاصل ضرب** معدلي تقارب المتعلمين أسرع من $n^{-1/2}$ — مثلًا كل منهما أسرع من $n^{-1/4}$. "
            "لاسو في البيئات المتناثرة، والغابات والشبكات في البيئات الملساء تحقق ذلك تحت شروط.", "الشرط النظري", "violet", "📐")

with tabs[5]:
    ui.section("ستة أنماط للدمج التنبؤي", "Six predictive-hybrid patterns", "orange")
    pat = pd.DataFrame([
        ("التوليف المتوازي", "Parallel ensemble", "تنبؤات عدة نماذج ثم متوسط مرجح أو تكديس", "ARIMA + GBM", "ضبط الأوزان داخل العينة نفسها"),
        ("تصحيح البواقي", "Residual correction", "نموذج قياسي أولًا ثم ML يتعلم البواقي", "ARIMA ← غابة على البواقي (Zhang, 2003)", "التعلم على بواقي داخل العينة يضخم النتيجة"),
        ("إثراء الميزات", "Feature augmentation", "استخراج موجات/فورييه/عوامل/تضمينات ثم نموذج", "Wavelet + ARDL أو لاسو", "التفكيك ثنائي الجانب يسرّب المستقبل"),
        ("بوابة الأنظمة", "Regime gate / mixture of experts", "شبكة تختار بين متخصصين حسب الحالة", "VAR في الهدوء / GBM في الأزمة", "تعريف النظام بعد وقوعه"),
        ("متعدد المهام", "Joint multi-task", "أهداف مترابطة في خسارة واحدة", "تضخم + نمو + بطالة", "حساسية أوزان المهام"),
        ("تنبؤ موجّه بالنظرية", "Theory-informed forecast", "تنبؤ مرن مقيّد بعلاقات صحيحة", "BAM وقيود الرتابة", "القيد الخاطئ يولّد تحيزًا"),
    ], columns=["النمط", "English", "التصميم", "مثال", "الخطر"])
    st.table(pat)
    st.latex(r"\hat y_{t+h}=f_{\text{econ}}(\mathcal I_t)+\hat g_{\text{ML}}(\mathcal I_t)\qquad\text{[residual hybrid]}")
    ui.note("في ARDL/VECM: إذا كانت هناك علاقات تكامل مشترك ذات معنى، **لا يغني تحسن RMSE في شبكة عصبية عن اختبار السكون والتكامل المشترك**. "
            "يمكن إسناد مكوّن لاخطي أو حد تصحيح الخطأ إلى ML مع الحفاظ على معنى العلاقة طويلة الأجل — لا تسمية «AI-ARDL» اعتباطًا. "
            "تفاصيل الخوارزميات الهجينة في قسم «الخوارزميات الهجينة المتخصصة».", "ملاحظة جوهرية", "amber", "⚠️")

with tabs[6]:
    ui.section("قوالب أكواد جاهزة", "Code templates", "indigo")
    lang = st.segmented_control("اللغة", ["Python", "R", "Stata"], default="Python", key="code_lang")
    if lang == "Python":
        st.code('''# Double ML for a partially linear model with DoubleML
import numpy as np, pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

df = pd.read_csv("data.csv")                     # y: outcome, d: treatment, X: controls
data = DoubleMLData(df, y_col="y", d_cols="d")
ml_l = RandomForestRegressor(n_estimators=500, min_samples_leaf=5)   # E[Y|X]
ml_m = LassoCV()                                                     # E[D|X]
dml = DoubleMLPLR(data, ml_l, ml_m, n_folds=5, n_rep=10)             # repeated cross-fitting
dml.fit()
print(dml.summary)                               # theta-hat, SE, CI

# Causal forest for heterogeneous effects with EconML
from econml.dml import CausalForestDML
cf = CausalForestDML(model_y=RandomForestRegressor(), model_t=RandomForestRegressor(),
                     n_estimators=2000, min_samples_leaf=10, cv=5)
cf.fit(Y=df.y, T=df.d, X=df[["age", "income"]], W=df.filter(like="w_"))
tau_hat = cf.effect(df[["age", "income"]])
lb, ub = cf.effect_interval(df[["age", "income"]], alpha=0.05)
''', language="python")
    elif lang == "R":
        st.code('''# Double ML and causal forests in R
library(DoubleML); library(mlr3); library(mlr3learners); library(grf)

dml_data <- DoubleMLData$new(df, y_col = "y", d_cols = "d")
learner  <- lrn("regr.ranger", num.trees = 500)
dml_plr  <- DoubleMLPLR$new(dml_data, ml_l = learner$clone(), ml_m = learner$clone(),
                            n_folds = 5, n_rep = 10)
dml_plr$fit(); print(dml_plr)

# Generalized random forest: causal forest + doubly robust ATE
cf <- causal_forest(X = as.matrix(df[, c("age", "income")]), Y = df$y, W = df$d)
average_treatment_effect(cf, target.sample = "all")
best_linear_projection(cf, df[, c("age", "income")])   # heterogeneity summary

# Post-double-selection lasso (Belloni, Chernozhukov & Hansen, 2014)
library(hdm)
pds <- rlassoEffect(x = as.matrix(df[, grep("^w_", names(df))]), y = df$y, d = df$d,
                    method = "double selection")
summary(pds)
''', language="r")
    else:
        st.code('''* Post-double-selection and DML in Stata
* lassopack / pdslasso (Ahrens, Hansen & Schaffer)
pdslasso y d (w1-w200), rlasso robust

* Built-in (Stata 16+): partialing-out and cross-fit partialing-out
dsregress  y d, controls(w1-w200)
xporegress y d, controls(w1-w200) xfolds(5) resample(10)

* ddml (Ahrens, Hansen, Schaffer & Wiemann): DML with stacking
ddml init partial, kfolds(5) reps(10)
ddml E[Y|X]: pystacked y w1-w200, type(reg) methods(ols lassocv rf gradboost)
ddml E[D|X]: pystacked d w1-w200, type(reg) methods(ols lassocv rf gradboost)
ddml crossfit
ddml estimate, robust
''', language="stata")
    ui.refs("DoubleML (Bach et al., 2022 JMLR)، EconML (Microsoft)، grf (Tibshirani, Athey & Wager)، hdm (Chernozhukov, Hansen & Spindler)، "
            "pdslasso/lassopack/ddml (Ahrens, Hansen, Schaffer, Wiemann). تحقق من توثيق الإصدار المثبت لديك قبل الاستخدام.")
