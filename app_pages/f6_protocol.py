import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "فلسفة وآليات الدمج (6): الحدود والبروتوكول والتطبيق والمراجع",
    "Part 6: the Lucas critique, small samples, a research protocol, testing added value, a full Python application, common mistakes, decision matrix, quiz, glossary, references (chapters 38–48)",
    "متى يفشل الدمج؟ كيف نطوّر منهجية هجينة قابلة للاختبار؟ كيف نثبت أن الجزء الآلي أضاف قيمة؟ تطبيق Python كامل بلا تسرب مستقبلي، أخطاء شائعة، "
    "مصفوفة قرار، أسئلة فهم مشروحة، قاموس، ومراجع من 1976 إلى أكتوبر 2026. محاضرة <b>د. مروان رودان</b>، الفصول 38–48.",
    "slate", "violet",
)

tabs = st.tabs([
    ":material/policy: 38 نقد لوكاس",
    ":material/compress: 39 العينات الصغيرة",
    ":material/route: 40 بروتوكول التطوير",
    ":material/fact_check: 41 اختبار القيمة المضافة",
    ":material/code: 42 تطبيق Python كامل",
    ":material/error: 43 أخطاء شائعة",
    ":material/grid_on: 44 مصفوفة القرار",
    ":material/quiz: 45 أسئلة الفهم",
    ":material/translate: 46 قاموس المصطلحات",
    ":material/menu_book: 47 المراجع",
    ":material/flag: 48 الخاتمة الكبرى",
])

# --------------------------------------------------------------------------- 38
with tabs[0]:
    ui.section("38 — التنبؤ مقابل السياسة الاقتصادية: نقد لوكاس وتغير العلاقات", "Forecasting vs policy: the Lucas critique", "slate")
    st.markdown("افترض أن النموذج الهجين يتنبأ بالتضخم جيدًا تحت سياسة نقدية تاريخية. هل يمكن استخدامه مباشرة لتقدير أثر تغيير قواعد السياسة النقدية جذريًا؟ ليس بالضرورة. هناك فرق بين:")
    st.latex(r"E(Y\mid X=x)\qquad\text{و}\qquad E[Y\mid do(X=x)].")
    st.markdown(
        "الأول توقع شرطي في التوزيع المرصود؛ والثاني يحتاج إلى تعريف تدخل سببي. قد يؤدي تغيير السياسة إلى تغيير توقعات الوكلاء وقراراتهم والعلاقات التي "
        "تعلمها النموذج. وهذا يرتبط بإشكالية نقد لوكاس (1976) [م23]. وتناقش مراجعة بحثية قيد النشر لـ Annie Liang كيف يمكن للتعلم الآلي أيضًا أن يختبر "
        "حدود النماذج الاقتصادية ويقترح نماذج هجينة قابلة للتفسير [م7].\n\n"
        "كذلك يجب الانتباه إلى **Distribution Shift** و**Structural Breaks** و**Concept Drift**. قد يتدهور أداء نموذج بعد أزمة اقتصادية أو تغيير نظام تنظيمي حتى لو كان رائعًا في بيانات التدريب."
    )
    st.markdown("#### هل يعالج إدخال النظرية هذه المشكلة تمامًا؟")
    st.markdown("لا. قد يساعد تقييد النموذج ببنية اقتصادية على تحسين الثبات خارج العينة عندما تكون الافتراضات ملائمة، لكنه **لا يضمن** صلاحية التنبؤ تحت أي تدخل أو نظام جديد.")
    ui.refs("[م23] Lucas (1976) Carnegie-Rochester Conference Series on Public Policy 1؛ [م7] Liang (قيد النشر) JEL «Using Machine Learning to Generate, Clarify, and Improve Economic Models».")

# --------------------------------------------------------------------------- 39
with tabs[1]:
    ui.section("39 — هل الدمج مناسب للعينات الصغيرة؟ وما البدائل؟", "Small samples", "amber")
    st.markdown("في الاقتصاد الكلي قد نتعامل مع 70 أو 120 مشاهدة فقط. هل يصح تدريب LSTM عميقة؟ يمكن تقنيًا في بعض الحالات، لكن خطر الإفراط في التكيف مرتفع، خصوصًا عندما يكون عدد المعلمات كبيرًا مقارنة بالمعلومات المتاحة.")
    st.markdown("#### مبدأ الاختيار التدريجي")
    ui.ladder([
        ("نموذج مرجعي", "Baseline", "ابدأ بنموذج مرجعي اقتصادي واضح مثل ARDL أو VAR أو انحدار ديناميكي مناسب."),
        ("ابحث عن قصور", "Diagnose", "اختبر إن كان هناك نمط قابل للتنبؤ في البواقي أو قصور في الشكل الوظيفي."),
        ("بدائل بسيطة", "Simple ML", "جرّب Ridge/Lasso أو أشجارًا صغيرة أو Gradient Boosting مقيد التعقيد."),
        ("شبكات تسلسلية عند الحاجة", "Sequence models", "إذا كانت هناك بنية تسلسلية قوية وعينة مناسبة، فكّر في LSTM أو غيرها مع Regularization وتقييم صارم."),
        ("النظرية حين تقل البيانات", "Theory", "إذا كانت النظرية قوية والبيانات قليلة، ادرس قيودًا بنيوية أو تدريبًا أوليًا، لكن اختبر مدى صحة افتراضات النظرية."),
        ("حجم المعلومات الفعال", "Effective sample size", "لا تخلط عدد الصفوف الاسمي مع **حجم المعلومات المستقلة الفعال**؛ فالارتباط الذاتي والاعتماد المقطعي قد يخفضانه."),
    ])
    st.latex(r"n_{\rm eff}\approx\frac{n}{1+2\sum_{k\ge1}\rho_k}\qquad\text{(لسلسلة AR(1)}:\ n_{\rm eff}\approx n\,\tfrac{1-\rho}{1+\rho}\text{)}")
    st.caption("مثال: 120 مشاهدة شهرية بارتباط ذاتي 0.8 تحمل معلومات تعادل نحو 13 مشاهدة مستقلة فقط.")
    st.markdown("#### لماذا قد يؤدي الدمج إلى نتيجة أسوأ؟")
    st.markdown("لأنه يضيف خطأ تقدير واحتمال Overfitting، وقد ينقل أخطاء النموذج الأول، وقد يفرض قيودًا غير صحيحة، وقد يستخدم متغيرات غير متاحة وقت إصدار التنبؤ. لذلك لا توجد قاعدة **«المزيد من AI = المزيد من الدقة»**.")

# --------------------------------------------------------------------------- 40
with tabs[2]:
    ui.section("40 — بروتوكول التطوير البحثي: من سؤال الاقتصاد إلى نموذج هجين صالح للاختبار", "Research development protocol", "violet")
    st.markdown("عند تطوير منهجية جديدة أو مكتبة Python/Stata، لا نبدأ بكتابة كود يجمع نموذجين. نبدأ ببروتوكول علمي واضح يمكن لباحث آخر إعادة تنفيذه.")
    stages = [
        ("المرحلة 1: اكتب السؤال والهدف بدقة", "هل نريد تنبؤًا لمعدل التضخم بعد شهر؟ أم معاملًا طويل الأجل؟ أم أثر سياسة اقتصادية؟ أم توزيعًا للمخاطر؟ كل هدف يقتضي طريقة تدريب وتقييم مختلفة."),
        ("المرحلة 2: صف بنية البيانات ومجموعة المعلومات", "حدد إن كانت البيانات Time Series أو Panel أو Cross-section، وهل المتغيرات متاحة في وقت التنبؤ، وهل توجد مراجعات لاحقة للبيانات، وماذا يحدث عند القيم المفقودة أو الشواذ."),
        ("المرحلة 3: اكتب النموذج المرجعي قبل الدمج", "قد يكون ARDL أو ECM أو VAR أو Fixed Effects أو نموذج سببي. بيّن ما الذي يفسره، وما المعلمات ذات الأهمية، وما القيود والافتراضات اللازمة."),
        ("المرحلة 4: حدد الشيء الذي سيتعلمه AI", "هل الهدف هو الباقي؟ أم علاقة غير خطية مضافة؟ أم وزن المزج؟ أم معامل متغير؟ أم دالة مساعدة في DML؟ أم حل معادلات اقتصادية؟ لا يكفي أن تقول «سنستخدم LSTM»."),
        ("المرحلة 5: اختر اتجاه النقل وطريقة التدريب", "حدد بصراحة: Sequential أو Joint أو Alternating أو Pretraining. واكتب أي معاملات تُجمد، وأين تنتقل المخرجات، وأين يمكن أن يعود التدرج، وهل يستخدم المكوّن الثاني توقعات مصدرة خارج العينة."),
        ("المرحلة 6: ضع شروط التحديد والاستقرار", "إذا كانت المعلمة الاقتصادية تهمك، فابحث عن تداخلها مع الجزء المرن، وافتراضات تعريفها، وما إذا كانت القيود أو التعامد ضرورية. وإذا كان النموذج ديناميكيًا، افحص الاستقرار والتوازن."),
        ("المرحلة 7: تصميم التدريب والتقييم", "افصل التدريب والتحقق والاختبار بالطريقة المناسبة لبنية البيانات، واضبط المعلمات الفائقة داخل مجموعة التدريب فقط. عند التنبؤ الزمني استخدم Rolling-Origin عند الحاجة."),
        ("المرحلة 8: تقرير النتائج بصورة شفافة", "اعرض نماذج المقارنة، ومقاييس الاختبار، والمجالات الزمانية، وإجراءات معالجة البيانات، وحجم التحسن وعدم يقينه، وحساسية النتائج، والحالات التي يفشل فيها الدمج."),
    ]
    for t, b in stages:
        with st.expander(t, expanded=False):
            st.markdown(b)
    st.markdown("**الشكل (10) — سلسلة القرار البحثي**")
    st.mermaid_chart(
        """
flowchart TB
    A[Economic question and type of target parameter] --> B[Data + timing of information availability]
    B --> C[Integration interface: residuals / variables / forecasts / coefficients / constraints]
    C --> D[Training choice: Sequential / Alternating / Joint / Transfer]
    D --> E[Checks: identification, leakage, stability, uncertainty]
    E --> F[Final validation and interpretation against the economic goal]
"""
    )

# --------------------------------------------------------------------------- 41
with tabs[3]:
    ui.section("41 — كيف نختبر أن الجزء الآلي أضاف قيمة فعلًا؟", "Testing the added value", "emerald")
    st.markdown("لا يكفي عرض RMSE للنموذج الهجين وحده. ينبغي مقارنته بنماذج مرجعية مناسبة:")
    st.markdown(
        "| المقارنة | ما الذي تكشفه؟ |\n|---|---|\n"
        "| Econometrics وحده | هل يوجد تحسن على النموذج القياسي؟ |\n"
        "| AI وحده | هل الدمج أفضل من تعلم مباشر بالمعلومات نفسها؟ |\n"
        "| النموذج الهجين | هل يضيف التعاون قيمة؟ |\n"
        "| Hybrid دون جزء AI | مدى الاعتماد على الجزء المرن |\n"
        "| Hybrid دون الجزء القياسي | مدى قيمة البنية الاقتصادية المضافة |\n"
        "| Hybrid مع مدخلات AI بديلة | هل التحسن بسبب معلومات جديدة أم مرونة زائدة؟ |\n"
        "| Hybrid مع قيد اقتصادي / دون قيد | أثر القيود على التنبؤ والتفسير |"
    )
    st.markdown("#### المقاييس الأساسية")
    st.latex(r"RMSE=\sqrt{\frac1N\sum_i(Y_i-\widehat Y_i)^2},\qquad MAE=\frac1N\sum_i|Y_i-\widehat Y_i|.")
    st.markdown("في مسائل التنبؤ بالكميات قد نستخدم Pinball Loss، وفي التوزيع يمكن استخدام CRPS ومعايرة الاحتمالات، وفي الاستدلال السببي ننظر إلى التحديد وفترات الثقة. **لا توجد «درجة نجاح واحدة» تصلح لجميع أهداف الدمج.**")
    st.markdown("#### هل فرق RMSE وحده كافٍ لإثبات تفوق النموذج؟")
    st.markdown("لا. يجب دراسة استقرار التحسن عبر فترات مختلفة، وأفق التنبؤ، وحساسيته لطريقة التقسيم، وقد تُستخدم اختبارات مقارنة التنبؤ مثل Diebold–Mariano عندما تنطبق شروطها [م24]. في المقارنات بين النماذج المتداخلة أو الآفاق المتعددة، يلزم الانتباه إلى خصائص الاختبار.")
    st.markdown("#### لماذا نفحص Ablation؟")
    st.markdown("لأن نموذجًا هجينًا قد يحقق نتائج جيدة رغم أن أحد مكوناته لا يساهم كثيرًا. حذف المكونات بصورة منظمة يساعد على اختبار قيمتها الإضافية، لكنه لا يثبت وحده التفسير الاقتصادي أو السببية.")
    ui.refs("[م24] Diebold & Mariano (1995) Journal of Business & Economic Statistics 13(3).")

# --------------------------------------------------------------------------- 42
CODE = '''"""Econometrics + AI: leak-safe rolling-origin residual-learning demo.
Synthetic economic data only.  Run: python hybrid_example.py
Requires: numpy, pandas, scikit-learn.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

rng = np.random.default_rng(17)
T = 520
x = np.zeros(T)
y = np.zeros(T)
for t in range(1, T):
    x[t] = 0.65 * x[t - 1] + rng.normal(0, 0.8)
    y[t] = (0.55 * y[t - 1]
            + 0.30 * x[t - 1]
            + 0.70 * (x[t - 1] ** 2 - 1.0)
            + rng.normal(0, 0.45))

# Forecast origin t-1: features known then; outcome observed at t.
X_econ = np.column_stack([y[:-1], x[:-1]])
Z_ml = np.column_stack([y[:-1], x[:-1]])
target = y[1:]
train_end = 360
first_origin = 100

# OOF-like historic base errors from expanding origin, no future leakage.
z_oof, errors = [], []
for i in range(first_origin, train_end):
    model_i = LinearRegression().fit(X_econ[:i], target[:i])
    pred_i = model_i.predict(X_econ[i:i + 1])[0]
    z_oof.append(Z_ml[i])
    errors.append(target[i] - pred_i)

correction = HistGradientBoostingRegressor(
    max_iter=130,
    max_leaf_nodes=7,
    min_samples_leaf=20,
    learning_rate=0.06,
    l2_regularization=0.25,
    random_state=42,
)
correction.fit(np.array(z_oof), np.array(errors))

# Fit benchmark and independent ML using only the historical training segment.
econ = LinearRegression().fit(X_econ[:train_end], target[:train_end])
independent_ml = HistGradientBoostingRegressor(
    max_iter=130,
    max_leaf_nodes=7,
    min_samples_leaf=20,
    learning_rate=0.06,
    l2_regularization=0.25,
    random_state=42,
).fit(Z_ml[:train_end], target[:train_end])

ix = slice(train_end, None)
truth = target[ix]
p_econ = econ.predict(X_econ[ix])
p_ml = independent_ml.predict(Z_ml[ix])
p_hybrid = p_econ + correction.predict(Z_ml[ix])

rows = []
for name, pred in [
    ('Dynamic regression', p_econ),
    ('Standalone ML', p_ml),
    ('Econometrics + residual ML', p_hybrid),
]:
    rows.append({
        'model': name,
        'RMSE': np.sqrt(mean_squared_error(truth, pred)),
        'MAE': mean_absolute_error(truth, pred),
    })
results = pd.DataFrame(rows)
print(results.round(4).to_string(index=False))

pd.DataFrame({
    'actual': truth,
    'econometric': p_econ,
    'standalone_ml': p_ml,
    'hybrid': p_hybrid,
}).to_csv('hybrid_example_predictions.csv', index=False)
print('Saved hybrid_example_predictions.csv (synthetic data).')
'''


@st.cache_data(show_spinner="تشغيل التطبيق...")
def run_example(seed: int = 17, nonlin: float = 0.70):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(seed)
    T = 520
    x, y = np.zeros(T), np.zeros(T)
    for t in range(1, T):
        x[t] = 0.65 * x[t - 1] + rng.normal(0, 0.8)
        y[t] = 0.55 * y[t - 1] + 0.30 * x[t - 1] + nonlin * (x[t - 1] ** 2 - 1.0) + rng.normal(0, 0.45)
    X_econ = np.column_stack([y[:-1], x[:-1]])
    Z_ml = X_econ.copy()
    target = y[1:]
    train_end, first_origin = 360, 100
    z_oof, errors = [], []
    for i in range(first_origin, train_end):
        pred_i = LinearRegression().fit(X_econ[:i], target[:i]).predict(X_econ[i:i + 1])[0]
        z_oof.append(Z_ml[i])
        errors.append(target[i] - pred_i)
    hgb = dict(max_iter=130, max_leaf_nodes=7, min_samples_leaf=20, learning_rate=0.06, l2_regularization=0.25, random_state=42)
    correction = HistGradientBoostingRegressor(**hgb).fit(np.array(z_oof), np.array(errors))
    econ = LinearRegression().fit(X_econ[:train_end], target[:train_end])
    ml = HistGradientBoostingRegressor(**hgb).fit(Z_ml[:train_end], target[:train_end])
    ix = slice(train_end, None)
    truth = target[ix]
    preds = {"الانحدار الديناميكي القياسي": econ.predict(X_econ[ix]), "Gradient Boosting مستقل": ml.predict(Z_ml[ix])}
    preds["النموذج الهجين (تصحيح البواقي)"] = preds["الانحدار الديناميكي القياسي"] + correction.predict(Z_ml[ix])
    rows = [(k, float(np.sqrt(np.mean((truth - v) ** 2))), float(np.mean(np.abs(truth - v)))) for k, v in preds.items()]
    return pd.DataFrame(rows, columns=["النموذج", "RMSE", "MAE"]), truth, preds


with tabs[4]:
    ui.section("42 — تطبيق Python كامل: مراحل التدريب والدمج دون تسرب مستقبلي", "A complete leak-free Python application", "teal")
    st.markdown("سننشئ بيانات اقتصادية **اصطناعية** عن متغير يشبه التضخم ومتغير مفسر يشبه صدمة اقتصادية. الهدف تعليمي: معرفة كيف ينتقل الخطأ من نموذج قياسي إلى خوارزمية تعلم آلي. العملية المولدة للبيانات:")
    st.latex(r"Y_t=0.55Y_{t-1}+0.30X_{t-1}+0.70(X_{t-1}^2-1)+\varepsilon_t.")
    st.markdown("لكن النموذج القياسي الذي يراه الباحث لا يحتوي على $X^2$؛ سنجعله انحدارًا ديناميكيًا خطيًا. ثم نولد أخطاء تنبؤ تاريخية بطريقة **Expanding Origin** داخل بيانات التدريب، ونستخدمها لتدريب Gradient Boosting لتصحيح التنبؤ. ونقارنه أيضًا بـ Gradient Boosting مستقل يدرس $Y$ مباشرة.")
    st.markdown("#### كيف تشغل التطبيق؟")
    st.markdown("احفظ الكود التالي باسم `hybrid_example.py`، ثم ثبّت المكتبات: `pip install numpy pandas scikit-learn` وشغّل `python hybrid_example.py`. في Google Colab يمكن تشغيله في خلية بعد تثبيت المكتبات إذا لزم الأمر.")
    st.code(CODE, language="python")
    st.download_button("تنزيل hybrid_example.py", CODE.encode("utf-8"), file_name="hybrid_example.py", icon=":material/download:")
    st.markdown("#### نتائج التشغيل بعد تنفيذ الكود على الإعدادات المحددة")
    st.markdown(
        "| النموذج | RMSE | MAE |\n|---|---|---|\n"
        "| الانحدار الديناميكي القياسي | 1.4611 | 1.0397 |\n"
        "| Gradient Boosting مستقل | **0.9154** | **0.5847** |\n"
        "| النموذج الهجين (تصحيح البواقي) | 0.9284 | 0.6280 |"
    )
    st.markdown("هذه الأرقام ناتجة عن تنفيذ الكود المدرج مع بذرة عشوائية ثابتة وليست نتائج بيانات اقتصادية حقيقية. في هذه التجربة حسّن الدمج النموذج القياسي بصورة واضحة، **لكنه لم يتفوق على التعلم الآلي المستقل**. وهذه نتيجة تعليمية مقصودة: نجاح إضافة AI إلى نموذج قياسي لا يعني أن الدمج أفضل من AI وحده.")
    ui.section("شغّله هنا مباشرة", "Run it live", "violet")
    c1, c2 = st.columns(2)
    seed = c1.number_input("البذرة العشوائية (17 = إعدادات المحاضرة)", 0, 10_000, 17, key="f6_seed")
    nl = c2.slider("قوة الحد اللاخطي 0.70·(X²−1)", 0.0, 1.5, 0.70, 0.05, key="f6_nl")
    res, truth, preds = run_example(int(seed), float(nl))
    st.dataframe(res.round(4), hide_index=True)
    if int(seed) == 17 and abs(nl - 0.70) < 1e-9:
        st.success("مطابق تمامًا لجدول المحاضرة (1.4611 / 0.9154 / 0.9284).", icon=":material/check_circle:")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=truth, name="الفعلي", mode="lines", line=dict(color="#94a3b8")))
    cols = ["#7c3aed", "#0d9488", "#e11d48"]
    for (k, v), c in zip(preds.items(), cols):
        fig.add_trace(go.Scatter(y=v, name=k, mode="lines", line=dict(color=c, width=1.8)))
    fig.update_xaxes(title="فترة الاختبار")
    ui.style_fig(fig, "التنبؤات على عينة الاختبار (159 فترة)", height=420)
    ui.show(fig)
    st.caption("جرّب: اجعل قوة الحد اللاخطي 0 — يصبح العالم خطيًا، فيقترب القياسي من الأفضل ولا يضيف الدمج شيئًا يذكر.")
    st.markdown("#### لماذا احتفظنا بنموذج AI مستقل؟")
    st.markdown("حتى لا نقع في خطأ شائع: نقارن نموذجنا الهجين بالنموذج القياسي فقط، فنرى تحسنًا كبيرًا ونعلن أن الدمج هو السبب، بينما قد يكون نموذج ML المباشر أفضل أصلًا. يجب أن نسأل دائمًا: هل حصلنا على قيمة **إضافية من التعاون**، أم فقط من استعمال خوارزمية أكثر مرونة؟")
    st.markdown("#### لماذا لم نستخدم بواقي التدريب المباشرة؟")
    st.markdown("داخل الفترة التدريبية قمنا بتقدير النموذج القياسي تكراريًا باستخدام المعلومات السابقة ثم توقع القيمة التالية وتسجيل خطئها. بهذا لا يحصل نموذج التصحيح على أهداف مبنية على تنبؤات تستخدم المستقبل في تقدير معلماتها. وهذا يوضح مبدأ **تصميم الأخطاء خارج نافذة التقدير**.")
    st.markdown("#### كيف نحول الكود إلى بحث فعلي؟")
    st.markdown("نستبدل البيانات الاصطناعية ببيانات منشورة لها تواريخ توفر فعلية، ونحدد أفق التنبؤ بدقة، ونُجري اختيار المعلمات الفائقة داخل تدريب زمني متداخل، ونكرر الاختبارات عبر نوافذ وأزمات وفترات مختلفة، وندرس عدم اليقين في فروق الأداء. لا يكفي نسخ الكود وتغيير اسم المتغير التابع.")

# --------------------------------------------------------------------------- 43
with tabs[5]:
    ui.section("43 — أخطاء شائعة جدًا في بحوث دمج القياس الاقتصادي مع AI", "Very common mistakes", "rose")
    mistakes = [
        ("«أرسلنا البواقي إلى LSTM» دون تحديد Input وTarget.", "يجب بيان ما إذا كانت البواقي أهدافًا أم خصائص، وأيها معروف عند زمن التنبؤ."),
        ("استخدام الباقي المستقبلي الحقيقي في المدخلات.", "هذا تسرب مباشر للمعلومات."),
        ("استخراج البواقي من نموذج مقدّر بكامل العينة ثم ادعاء تقييم تنبؤي مستقل.", "يتطلب الأمر محاكاة إصدار التنبؤ بالمعلومات التاريخية."),
        ("الخلط بين Training Loss وTest Error.", "انخفاض خسارة التدريب لا يثبت تحسن التنبؤ بالمستقبل."),
        ("الاعتقاد أن LSTM تتعلم اللاخطية وARDL يتعلم الخطية فقط.", "كلا الوصفين أبسط من الواقع؛ فقد يتداخل ما يتعلمه المكوّنان."),
        ("تفسير معاملات الجزء القياسي بعد تدريب مشترك دون بحث التحديد.", "قد تعيد الشبكة تعلم العلاقة نفسها."),
        ("الاعتقاد أن التعامد الهندسي يضمن السببية.", "لا؛ هناك فرق بين الإسقاط الجبري وNeyman Orthogonality والتعريف السببي."),
        ("استخدام DML كأنه خوارزمية تعطي سببية تلقائيًا.", "يظل اختيار الضوابط والافتراضات والتصميم أساسيًا."),
        ("اعتبار مؤشر LLM قيمة خالية من الخطأ.", "أخطاء القياس والتسرب والاختلاف بين Prompts يمكن أن تؤثر في الانحدار."),
        ("الاعتقاد أن إدخال قيد اقتصادي واحد يحفظ كل بنية النموذج.", "الاستقرار والتوازن ومطابقة النظرية تتطلب فحوصًا مستقلة."),
        ("مقارنة النموذج الهجين بالقياس الاقتصادي فقط.", "ينبغي مقارنته أيضًا بنموذج AI مستقل وبمرجع بسيط مناسب."),
        ("اختيار نموذج بالغ التعقيد لعينة قصيرة دون Regularization ومقارنة مرجعية.", "يزيد خطر الإفراط في التكيف."),
        ("نقل أخطاء معيارية من OLS بعد تعديل دالة الهدف.", "التغير في إجراء التقدير قد يغيّر الاستدلال كله."),
        ("القول إن صيغة اكتشفها Symbolic Regression قانون اقتصادي.", "اكتشاف علاقة وصفية لا يساوي تعريف قانون سببي أو هيكلي."),
        ("اعتبار الأوزان المتعلمة في Gating احتمالات صحة النموذج.", "هي أوزان محسنة لغرض أو خسارة معينة، وليست إثباتًا نظريًا."),
        ("إهمال تعدد الآفاق والبيانات المنقحة.", "يجب التمييز بين تنبؤ خطوة وأفق 12 شهرًا، وبين البيانات المنشورة في وقتها والبيانات المنقحة لاحقًا."),
        ("وصف كل نظام فيه جزآن بأنه End-to-End.", "يلزم تحقق مسار التدرج المشترك أو تحديث متكامل فعلًا."),
        ("تجاهل أن الخطأ قد يكون غير متنبأ في المتوسط لكنه متنبأ في التباين.", "قد يكون هدف المخاطر أهم من تصحيح القيمة المتوقعة."),
    ]
    for i, (m, why) in enumerate(mistakes, 1):
        st.markdown(f"**{i}. {m}** {why}")

# --------------------------------------------------------------------------- 44
with tabs[6]:
    ui.section("44 — مصفوفة القرار: أي آلية دمج تناسب سؤالك؟", "Decision matrix", "indigo")
    st.markdown(
        "| سؤال البحث | خيار دمج يمكن البدء به | شيء يجب التأكد منه |\n|---|---|---|\n"
        "| أريد تحسين تنبؤ التضخم | Residual Learning / Stacking | توفر معلومات إضافية أو نمط قابل للتعلم؛ تقييم زمني |\n"
        "| لدي نصوص اقتصادية وأحتاج مؤشرًا | NLP/LLM → Econometrics | قياس المفهوم وعينة تحقق وخطأ قياس |\n"
        "| أريد تفسير معامل مهم مع ضوابط كثيرة | Partialling-Out / DML | شروط التعريف وOrthogonal Score وCross-Fitting |\n"
        "| أريد أثر سياسة لا تنطبق عليها العشوائية | IV/Deep IV عند توفر أداة | صلاحية الأداة وقوتها والفرضيات السببية |\n"
        "| أريد آثارًا تختلف بين الأفراد | Causal Forest / Parameter Learning | تعريف الأثر وفروق كافية في البيانات |\n"
        "| أريد علاقة طويلة الأجل مع ديناميكيات مرنة | ECM + شبكة مقيدة أو تصحيح تنبؤ | التكامل المشترك والتوازن والاستقرار |\n"
        "| لدي عينة صغيرة ونظرية هيكلية | Regularization أو Transfer Learning | معقولية النموذج الهيكلي وتعميمه |\n"
        "| أريد قيود مشتقات الطلب | Theory-Constrained ML | صلاحية القيد وهل هو ناعم أم صلب |\n"
        "| أريد حل نموذج DSGE أو نمو | Neural Economic Solver | معادلات التوازن والجدوى والتوقعات |\n"
        "| أريد اكتشاف صيغة وظيفية | Symbolic Regression → Econometric Validation | تجنب اختيار الصيغة على اختبار النموذج النهائي |\n"
        "| أريد تقدير المخاطر لا المتوسط فقط | Distributional Hybrid / Quantile ML | المعايرة وذيل التوزيع والتغطية |\n"
        "| أريد تنبؤ بانل لوحدات جديدة | Panel-Aware ML + Econometrics | فصل الوحدات والزمن والاعتماد المقطعي |"
    )
    ui.note("الاختيار يبدأ من **السؤال الاقتصادي وشروطه**، ثم يحدد موضع دخول ML، ثم يحدد طريقة التدريب والتقييم. لا توجد خوارزمية واحدة تهيمن على كل الأهداف.",
            "خلاصة المصفوفة", "indigo", "🧭")

# --------------------------------------------------------------------------- 45
with tabs[7]:
    ui.section("45 — أسئلة اختبار الفهم مع الإجابات المشروحة", "Check your understanding", "teal")
    st.caption("فكّر في الإجابة أولًا، ثم افتح السؤال لترى الإجابة المشروحة.")
    qa = [
        ("هل مجرد استعمال ARDL وLSTM في الورقة يعني أنهما تدربا معًا؟",
         "لا. قد يقدر الباحث ARDL أولًا ويثبت معاملاته، ثم يدرب LSTM على البواقي. هذا **تدريب متتابع**. يصبح Joint Training فقط إذا كانت معاملات المكوّنين تدخل فعليًا في إجراء تحسين مشترك أو تحديث متبادل محدد."),
        ("لماذا لا نستطيع إدخال الباقي المستقبلي الحقيقي إلى LSTM؟",
         "لأن الباقي الحقيقي يحتاج إلى معرفة Y المستقبلية؛ وعند إصدار التنبؤ لا تكون معروفة. نستخدم خصائص تاريخية وبواقي سابقة متاحة ونطلب توقع الباقي القادم، ثم نضيف **الباقي المتوقع** فقط إلى التنبؤ الأساسي."),
        ("هل البواقي الكبيرة تعني بالضرورة وجود فرصة لتعلمها؟",
         "لا. إذا كان $E(e\\mid Z)=0$ بالنسبة إلى مجموعة المدخلات، فأفضل تصحيح لمتوسط الباقي وفق MSE هو صفر. قد تظل البواقي مفيدة لتعلم التباين أو الكميات الشرطية."),
        ("ما الفرق بين Stacking وResidual Learning؟",
         "Stacking يجمع توقعات نموذجين أو أكثر. أما Residual Learning فيجعل نموذجًا ثانيًا يتعلم خطأ تنبؤ النموذج الأول. لا يعني أي منهما بالضرورة تدريبًا مشتركًا."),
        ("لماذا قد تكون β غير قابلة للتفسير في نموذج βX+g(X)؟",
         "لأن الشبكة قد تتعلم مكوّنًا خطيًا يماثل βX، فيمكن إعادة توزيع المعامل بين المكوّنين دون تغيير مجموع التنبؤ. نحتاج إلى شروط تمييز أو قيود أو تصميم تقدير ملائم."),
        ("هل Neyman Orthogonality هو جعل X غير مرتبط بمخرجات الشبكة؟",
         "ليس بالضرورة. تعامد Neyman هو خاصية لمعادلة التقدير تقلل الحساسية من الدرجة الأولى لأخطاء تقدير الدوال المساعدة. أما الإسقاط $(I-P_X)$ فيفصل مخرجات النموذج عن فضاء أعمدة X داخل العينة."),
        ("ماذا يتعلم AI في DML؟",
         "يتعلم عادة دوالًا مساعدة مثل $E(Y\\mid X)$ و$E(D\\mid X)$، ثم يُستخدم تقديرها ضمن درجة متعامدة لاستخراج المعلمة المستهدفة. لا يلزم أن يتنبأ AI بالأثر السببي مباشرة."),
        ("إذا أعطينا الشبكة قيدًا اقتصاديًا في Loss، هل ستطبقه دائمًا؟",
         "في Soft Constraint قد تقبل مخالفات مقابل تحسين الملاءمة، بحسب وزن العقوبة ومجال التطبيق. أما Hard Constraint فيُبنى داخل المعمارية بطريقة تضمن خاصية محددة ضمن شروطها."),
        ("هل معامل تعديل الخطأ السالب يضمن استقرار ECM عصبي؟",
         "لا. يجب دراسة الديناميكية كاملة. حتى في مثال أحادي بسيط يتوقف الاستقرار المحلي على $|1+\\alpha+g'(0)|$، لا على إشارة α وحدها."),
        ("لماذا لا يعني نجاح التنبؤ صلاحية تحليل سياسة اقتصادية جديدة؟",
         "لأن التدخل قد يغير السلوك والعلاقات التاريخية التي تعلمها النموذج. التنبؤ الشرطي ليس هو نفسه الأثر السببي للتدخل، وقد نحتاج إلى بنية هيكلية وفرضيات ثبات مناسبة."),
        ("هل الأفضل دائمًا أن يتفوق Hybrid على كل نموذج منفرد؟",
         "هذه فرضية اختبار وليست قاعدة. قد يحسن Hybrid النموذج القياسي ولكنه لا يتفوق على ML المستقل؛ وهو ما أظهره المثال البرمجي المرفق. يجب عرض المقارنات جميعها وعدم انتقاء مرجع واحد."),
        ("كيف أعرف أن ورقة بحثية وصفت ميكانيزم الدمج بدقة؟",
         "ابحث عن تعريف واضح للمدخلات والأهداف، واتجاه انتقال المعلومات، وطريقة التدريب، ومعاملات التثبيت والتحديث، ودالة الخسارة، وتوقيت توفر البيانات، وشروط الاستدلال والتقييم المستقل. إذا غاب أحدها فقد يصعب إعادة إنتاج النتائج أو فهم معناها."),
    ]
    for i, (q, a) in enumerate(qa, 1):
        with st.expander(f"س{i}. {q}"):
            st.markdown(a)

# --------------------------------------------------------------------------- 46
GLOSS = [
    ("Econometrics", "القياس الاقتصادي", "منهجيات لتقدير العلاقات الاقتصادية واختبار الفرضيات وبناء الاستدلال والتنبؤ تحت افتراضات محددة."),
    ("Machine Learning", "التعلم الآلي", "خوارزميات تتعلم أنماطًا أو دوالًا أو سياسات من البيانات أو من أهداف تدريب محددة، وغالبًا ما تهتم بالتعميم خارج العينة."),
    ("Hybrid Model", "نموذج هجين", "معمارية تجمع مكونات قياسية ومرنة عبر واجهة معلومات أو تدريب أو هدف مشترك واضح."),
    ("Feature / Input", "الخاصية / المدخل", "معلومة تقدم للنموذج عند إصدار التوقع أو إجراء التقدير، ويجب التحقق من توفرها فعليًا في تلك اللحظة."),
    ("Target / Label", "الهدف", "المخرج التاريخي الذي تحاول الخوارزمية تعلم توقعه؛ قد يكون Y أو الباقي أو التصنيف أو معلمة أخرى."),
    ("Residual / Forecast Error", "الباقي / خطأ التنبؤ", "الفرق بين المحقق والمقدر؛ بواقي الملاءمة داخل العينة تختلف منهجيًا عن أخطاء توقع أُصدرت قبل تحقق القيم."),
    ("Loss Function", "دالة الخسارة", "قياس عددي يربط أداء النموذج بهدف التدريب، مثل MSE أو Pinball Loss أو عقوبة انتهاك قيود."),
    ("Gradient / Backpropagation", "التدرج / الانتشار العكسي", "مشتقات الخسارة بالنسبة إلى المعلمات، وطريقة حسابها عبر شبكة قابلة للتفاضل باستخدام قاعدة السلسلة."),
    ("Epoch / Batch / Iteration", "حقبة / دفعة / خطوة تحديث", "تسميات لمراحل التدريب الحسابي؛ لا تساوي الفترات الزمنية في البيانات الاقتصادية."),
    ("Learning Rate", "معدل التعلم", "معامل يضبط حجم خطوة تحديث الأوزان في خوارزميات التحسين القائمة على التدرجات."),
    ("Overfitting / Generalization", "الإفراط في التكيف / التعميم", "التكيف مع تفاصيل عينة التدريب على نحو يضعف الأداء على بيانات جديدة، مقابل القدرة على التنبؤ خارج العينة."),
    ("Regularization", "الانتظام أو التنظيم", "قيود أو عقوبات أو تصميمات تقلل التعقيد الفعلي للنموذج وقد تساعد على استقرار تقديره."),
    ("Rolling / Expanding Origin", "أصل متحرك / متوسع", "إصدار تنبؤات تاريخية باستخدام ما كان متاحًا حتى أصل التنبؤ فقط؛ النافذة المتحركة تثبت طولها، والمتوسعة تزيد طولها."),
    ("Out-of-Fold Predictions", "توقعات خارج الطية", "توقعات لمشاهدات لم تدخل في تدريب النموذج المستخدم لإنتاجها. يستخدم هذا المبدأ في Stacking وCross-Fitting."),
    ("Stacking / Gating", "التكديس / البوابة", "الأول تجميع توقعات الخبراء، والثاني تعلم أوزان تتغير بحسب متغيرات الحالة أو المدخلات."),
    ("Identifiability", "قابلية التحديد", "هل تميز البيانات والافتراضات قيمة المعلمة أو مساهمة المكوّن بصورة فريدة أو كافية للاستدلال؟"),
    ("Component Orthogonalization", "التعامد الهندسي للمكونات", "إجراء هندسي لتقليل تداخل مكونات النموذج، مثل إسقاط مخرجات الشبكة خارج فضاء X."),
    ("Neyman Orthogonality", "تعامد نيمان", "خاصية تجعل متوسط معادلة تقدير المعلمة المستهدفة غير حساس من الدرجة الأولى لأخطاء محلية في الدوال المساعدة."),
    ("Nuisance Functions", "الدوال المساعدة", "مكونات قد يكون تقديرها ضروريًا لاستخراج معلمة الاهتمام، مثل E(Y|X) أو احتمال المعالجة."),
    ("Cross-Fitting", "التقسيم المتقاطع", "تقدير الدوال المساعدة في طيات منفصلة عن المشاهدات التي تدخل بها في معادلة تقدير المعلمة المستهدفة."),
    ("DML — Double/Debiased Machine Learning", "التعلم الآلي المزدوج", "إطار قياسي يستخدم درجات متعامدة وCross-Fitting لتقدير معلمات منخفضة البعد مع دوال مساعدة معقدة."),
    ("End-to-End / Stop Gradient", "تدريب متكامل / إيقاف التدرج", "الأول تدريب يسمح بمرور إشارات الخسارة عبر المكونات القابلة للتفاضل؛ الثاني يوقف هذا المرور عند نقطة معينة."),
    ("Alternating Optimization", "التحسين بالتناوب", "تحديث مجموعة معلمات مع تثبيت أخرى ثم تبادل الأدوار ضمن هدف أو أهداف محددة."),
    ("Soft / Hard Constraints", "قيود ناعمة / صلبة", "القيود الناعمة تعاقب المخالفة في الخسارة؛ والصلبة تدخل في تصميم الدالة بحيث تضمن خاصية ضمن المجال المعرّف."),
    ("GMM / Moment Conditions", "العزوم المعممة / شروط اللحظات", "تقدير يستغل علاقات لحظية مفترضة مثل E[Zu(θ)]=0، ويمكن دمج وظائف تعلم مرنة فيها ضمن شروط تعريف واستدلال."),
    ("Instrumental Variables / Deep IV", "المتغيرات الآلية / Deep IV", "نهج للتعامل مع بعض أنواع الداخلية عند توفر أداة صالحة؛ Deep IV يوظف شبكات مرنة في مرحلتي تقدير مناسبتين."),
    ("ECM / Cointegration", "نموذج تصحيح الخطأ / التكامل المشترك", "ECM يصف الديناميكيات حول علاقة توازن، والتكامل المشترك يربط سلاسل غير ساكنة بعلاقة خطية ساكنة تحت شروط مناسبة."),
    ("Latent State / State-Space", "حالة كامنة / فضاء الحالة", "متغير حالة لا يلاحظ مباشرة وتُقدّر ديناميكيته وصلته بالمشاهدات من خلال نموذج حالة ومعادلات قياس."),
    ("Pretraining / Fine-tuning", "تدريب أولي / ضبط دقيق", "تدريب أولي على بيانات أو مهمة، ثم تكييف الأوزان مع بيانات أو هدف جديد."),
    ("Symbolic Regression", "الانحدار الرمزي", "بحث خوارزمي عن تعبير رياضي يمثل البيانات، وقد يكون خطوة لاكتشاف أشكال وظيفية مرشحة تخضع لاحقًا للتقييم الاقتصادي."),
    ("Distribution Shift / Structural Break", "تحول التوزيع / انكسار هيكلي", "تغير توزيع البيانات أو العلاقات التي تربطها، ما قد يضعف نقل الأداء التنبؤي من الماضي إلى المستقبل."),
    ("Calibration / Validation Sample", "المعايرة / عينة التحقق", "المعايرة تتعلق بتوافق التنبؤات الاحتمالية أو مؤشرات القياس مع الواقع؛ وعينة التحقق تساعد على قياس وتصحيح أخطاء المؤشرات المولدة."),
    ("Covariate Measurement Error", "خطأ قياس المتغير المفسر", "فرق بين المتغير المراد قياسه والمؤشر المستعمل؛ قد يولد انحيازًا أو فترات ثقة غير صالحة إذا أُهمل."),
    ("Ablation Study", "دراسة الحذف المنظم", "إزالة مكوّنات النموذج أو ميزاتها بصورة منظمة لاختبار مساهماتها في أداء أو خاصية محددة."),
    ("Neural Solver", "الحلّال العصبي", "شبكة تُستخدم لتقريب حل دالة سياسة أو علاقة توازن عبر تقليل بواقي المعادلات والشروط الاقتصادية."),
]

with tabs[8]:
    ui.section("46 — قاموس المصطلحات: من المفاهيم الأولى إلى المكونات المتقدمة", "Glossary", "sky")
    q = st.text_input("ابحث في المصطلحات (عربي أو إنجليزي)", key="f6_gq", placeholder="مثلًا: Cross-Fitting أو التحديد")
    gdf = pd.DataFrame(GLOSS, columns=["English", "العربية", "التعريف"])
    if q.strip():
        mask = gdf.apply(lambda r: r.astype(str).str.contains(q.strip(), case=False, regex=False).any(), axis=1)
        gdf = gdf[mask]
    st.caption(f"{len(gdf)} من {len(GLOSS)} مصطلحًا")
    st.dataframe(gdf, hide_index=True, height=min(720, 40 + 36 * max(1, len(gdf))))

# --------------------------------------------------------------------------- 47
REFS = [
    ("م1", "Mullainathan, S. & Spiess, J. (2017)", "Machine Learning: An Applied Econometric Approach", "Journal of Economic Perspectives", "https://doi.org/10.1257/jep.31.2.87"),
    ("م2", "Athey, S. & Imbens, G. W. (2019)", "Machine Learning Methods That Economists Should Know About", "Annual Review of Economics", "https://doi.org/10.1146/annurev-economics-080217-053433"),
    ("م3", "Zhang, G. P. (2003)", "Time Series Forecasting Using a Hybrid ARIMA and Neural Network Model", "Neurocomputing 50, 159–175", "https://doi.org/10.1016/S0925-2312(01)00702-0"),
    ("م4", "Robinson, P. M. (1988)", "Root-N-Consistent Semiparametric Regression", "Econometrica 56(4)", "https://doi.org/10.2307/1912705"),
    ("م5", "Chernozhukov et al. (2018)", "Double/debiased Machine Learning for Treatment and Structural Parameters", "The Econometrics Journal", "https://doi.org/10.1111/ectj.12097"),
    ("م6", "Dell, M. (2025)", "Deep Learning for Economists", "Journal of Economic Literature 63(1)", ""),
    ("م7", "Liang, A. (قيد النشر بحسب صفحة AEA حتى أكتوبر 2026)", "Using Machine Learning to Generate, Clarify, and Improve Economic Models", "Journal of Economic Literature (Forthcoming)", ""),
    ("م8", "Rügamer et al. (2023)", "Semi-Structured Distributional Regression", "The American Statistician (نشر إلكتروني مبكر 2023)", "https://doi.org/10.1080/00031305.2022.2164054"),
    ("م9", "Rügamer, D. (2023)", "A New PHO-rmula for Improved Performance of Semi-Structured Networks", "ICML / PMLR", ""),
    ("م10", "Ludwig, J., Mullainathan, S. & Rambachan, A. (2025)", "Large Language Models: An Applied Econometric Framework", "NBER w33344", "https://www.nber.org/papers/w33344"),
    ("م11", "Dell, M. & Rambachan, A. (2026)", "The Measurement Revolution? Credible Measurement and Inference in the Age of AI", "NBER w35744 · سبتمبر 2026", "https://www.nber.org/papers/w35744"),
    ("م12", "Battaglia, L., Christensen, T., Hansen, S. & Sacher, S. (2024/2025)", "Inference for Regression with Variables Generated by AI or Machine Learning", "Cowles Foundation DP 2421", ""),
    ("م13", "Hyndman, R. J. & Athanasopoulos, G.", "Forecasting: Principles and Practice — Time Series Cross-Validation", "كتاب مفتوح", "https://otexts.com/fpp3/tscv.html"),
    ("م14", "Duan, J. & Pelger, M. (2026)", "Inference with AI-Generated Covariates", "NBER w35481 · يوليو 2026", "https://www.nber.org/papers/w35481"),
    ("م15", "Christensen, T. & Hansen, S. (2026)", "Performing Valid Inference with AI/ML-Generated Covariates", "AEA Papers and Proceedings 116", ""),
    ("م16", "Hartford, J., Lewis, G., Leyton-Brown, K. & Taddy, M. (2017)", "Deep IV: A Flexible Approach for Counterfactual Prediction", "ICML / PMLR", "https://proceedings.mlr.press/v70/hartford17a.html"),
    ("م17", "Bennett, A., Kallus, N. & Schnabel, T. (2019)", "Deep Generalized Method of Moments for Instrumental Variable Analysis", "arXiv", "https://arxiv.org/abs/1905.12495"),
    ("م18", "Wager, S. & Athey, S. (2018)", "Estimation and Inference of Heterogeneous Treatment Effects Using Random Forests", "JASA", "https://doi.org/10.1080/01621459.2017.1319839"),
    ("م19", "Athey, S., Tibshirani, J. & Wager, S. (2019)", "Generalized Random Forests", "Annals of Statistics", "https://doi.org/10.1214/18-AOS1709"),
    ("م20", "Buckmann, M. & Potjagailo, G. (2025)", "Infusing Economically Motivated Structure into Machine Learning Methods", "Bank of England SWP 1144", ""),
    ("م21", "Chen, H., Cheng, Y., Liu, Y. & Tang, K. (2026)", "Teaching Economics to the Machines", "NBER Working Paper 34713 · يناير 2026", "https://www.nber.org/papers/w34713"),
    ("م22", "Fernández-Villaverde, J. (2026)", "Deep Learning for Solving Economic Models", "Journal of Economic Literature 64(3)", ""),
    ("م23", "Lucas, R. E. (1976)", "Econometric Policy Evaluation: A Critique", "Carnegie-Rochester Conference Series on Public Policy", "https://doi.org/10.1016/S0167-2231(76)80003-6"),
    ("م24", "Diebold, F. X. & Mariano, R. S. (1995)", "Comparing Predictive Accuracy", "Journal of Business & Economic Statistics", "https://doi.org/10.1080/07350015.1995.10524599"),
]

with tabs[9]:
    ui.section("47 — المراجع العلمية الأصلية والمعاصرة: من 1976 إلى أكتوبر 2026", "References", "violet")
    st.markdown("هذه روابط لمصادر أصلية وموثوقة؛ وضعت أمام كل مرجع رمزه المستعمل داخل المحاضرة. قد تكون بعض الصفحات مفتوحة لقراءة الملخص فقط وفق سياسة الناشر. ومتى وُجدت ورقة عمل حديثة لا تعني الإشارة إليها أنها محكمة أو مقبولة للنشر ما لم يُذكر ذلك صراحة.")
    for code, au, ti, ve, link in REFS:
        lk = f" — [{link.replace('https://', '')}]({link})" if link else ""
        st.markdown(f"**[{code}] {au}** · *{ti}* · {ve}{lk}")
    ui.link("app_pages/p22_references.py", "قائمة مراجع المنصة الكاملة المصنّفة", ":material/menu_book:")

# --------------------------------------------------------------------------- 48
with tabs[10]:
    ui.section("48 — الخاتمة الكبرى: كيف تفكر كمن يصمّم منهجية جديدة؟", "Final word", "violet")
    st.markdown("بعد هذه المحاضرة، عندما تواجه ورقة تجمع نموذجًا قياسيًا وخوارزمية ذكاء اصطناعي، لا تبدأ بسؤال «ما اسم النموذج؟». ابدأ بالأسئلة التالية:")
    ui.ladder([
        ("ما سؤال البحث؟", "Question", "هل هو Forecasting أم Inference أم Causality أم Structural Modeling؟"),
        ("ما المعلومات التي تنتقل؟", "What flows", "هل هي بيانات أو تنبؤات أو بواقي أو معاملات أو حالات أو أوزان أو قيود أو تدرجات؟"),
        ("متى تصبح هذه المعلومات متاحة؟", "When", "هل استخدم النموذج شيئًا لم يكن معروفًا عند إصدار التنبؤ؟"),
        ("كيف يتم التدريب؟", "How", "هل المكوّن الأول مجمد أم قابل للتحديث؟ هل يوجد End-to-End أم مجرد استخدام متتابع للمخرجات؟"),
        ("هل النموذج محدد؟", "Identified?", "هل نستطيع تمييز المعلمات الاقتصادية عن المعلومات التي تمتصها الشبكة؟"),
        ("كيف تم التحقق؟", "Validated?", "هل تفوق النظام على كل من النموذج القياسي والنموذج الآلي المستقل؟ هل صلحت فترات الثقة؟ هل احترمت القيود والاستقرار؟"),
    ])
    st.markdown("هذه الأسئلة الستة تحول فكرة الدمج من تجربة برمجية إلى **تصميم اقتصادي وإحصائي يمكن الدفاع عنه**.")
    ui.note("أكثر النماذج الهجينة تعقيدًا ليست بالضرورة أفضلها. القيمة الحقيقية للدمج تأتي من تقسيم واضح للوظائف، وانتقال معلومات يمكن تتبعه، وتدريب ملائم، وتحديد صحيح للمعلمات، وتقييم يناسب السؤال الاقتصادي.",
            "الرسالة الأخيرة", "violet", "🎯")
    ui.note("الدمج بين القياس الاقتصادي والذكاء الاصطناعي هو **تصميم منظم للعلاقة بين البيانات، والبنية الاقتصادية، والتعلم، والاستدلال**، بحيث نعرف بدقة ماذا يستقبل كل مكوّن، وماذا يتعلم، وكيف يتغير، ولماذا يضيف قيمة إلى الهدف الاقتصادي المستهدف.",
            "ملخص في جملة واحدة", "teal", "✍️")
    st.markdown("**إعداد وتقديم: الدكتور مروان رودان** — موسوعة آليات دمج القياس الاقتصادي والذكاء الاصطناعي، مادة تعليمية مرجعية، © 2026.")
