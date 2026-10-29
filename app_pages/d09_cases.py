import pandas as pd
import streamlit as st

from core import ui

ui.hero(
    "حالات تطبيقية كاملة من الصفر",
    "Five complete integration projects — from the question to the paper, step by step",
    "كل ما سبق نظري ومجزأ بحسب الموضع والمستوى. هنا نجمعه في <b>خمسة مشاريع كاملة</b>، كل منها يمر بالمسار نفسه: السؤال والبيانات، "
    "النموذج المرجعي، تشخيص قصوره، قرار الدمج (أين؟ أي مستوى؟ أي اتجاه؟ أي اقتران؟)، الخطوات والكود، قراءة النتائج، ما قد يخطئ، "
    "وكيف يُكتب في الورقة.",
    "violet", "orange",
)


def case(c: dict) -> None:
    ui.section(c["title"], c["en"], c["color"])
    st.markdown(f"**السؤال البحثي:** {c['question']}")
    st.markdown(f"**البيانات:** {c['data']}")
    cols = st.columns(4)
    for col, (k, v) in zip(cols, c["decision"].items()):
        with col:
            with ui.tinted("slate"):
                st.caption(k)
                st.markdown(f"**{v}**")
    st.markdown("#### 1) النموذج المرجعي")
    st.markdown(c["baseline"])
    if c.get("baseline_eq"):
        st.latex(c["baseline_eq"])
    st.markdown("#### 2) تشخيص القصور: لماذا لا يكفي؟")
    st.markdown(c["diagnosis"])
    st.markdown("#### 3) تصميم الدمج")
    st.markdown(c["design"])
    if c.get("design_eq"):
        st.latex(c["design_eq"])
    st.markdown("#### 4) الخطوات")
    ui.ladder(c["steps"])
    st.markdown("#### 5) الكود")
    st.code(c["code"], language="python")
    with ui.tinted("teal"):
        st.markdown("#### 6) كيف تقرأ النتائج")
        st.markdown(c["read"])
    c1, c2 = st.columns(2)
    with c1:
        with ui.tinted("rose"):
            st.markdown("#### 7) ما قد يخطئ")
            st.markdown("\n".join(f"- {x}" for x in c["wrong"]))
    with c2:
        with ui.tinted("amber"):
            st.markdown("#### 8) كيف تكتبه في الورقة")
            st.markdown(c["paper"])
    ui.refs(c["refs"])


tabs = st.tabs([
    ":material/map: خريطة الحالات",
    ":material/trending_up: ① التنبؤ بالتضخم",
    ":material/school: ② عائد التعليم بضوابط كثيرة",
    ":material/candlestick_chart: ③ تقلب الأسهم",
    ":material/newspaper: ④ مؤشر من الأخبار",
    ":material/shopping_cart: ⑤ منحنى طلب بقيود نظرية",
    ":material/description: قالب منهجي لأي مشروع",
])

with tabs[0]:
    ui.section("خمس حالات تغطي المواضع والمستويات الرئيسية", "Case map", "violet")
    st.dataframe(pd.DataFrame([
        ("① التنبؤ بالتضخم", "البواقي + التوليف", "L2", "القياس ← الذكي", "ضعيف (متسلسل) + متوازٍ", "RMSE خارج العينة، DM"),
        ("② عائد التعليم", "الاستدلال", "L3", "الذكي ← القياس", "متداخل", "θ̂ وفترة ثقة"),
        ("③ تقلب الأسهم", "الدالة + الهدف", "L2/L5", "تكافلي", "مندمج (معمارية)", "الأرجحية، VaR"),
        ("④ مؤشر من الأخبار", "المعلومات", "L4 + L3", "الذكي ← القياس", "متسلسل مع تصحيح", "معامل مصحح، PPI"),
        ("⑤ منحنى الطلب", "الهدف (قيد نظري)", "L5", "القياس ← الذكي", "مندمج (خسارة)", "خطأ الاختبار، احترام القيد"),
    ], columns=["الحالة", "موضع الدمج", "المستوى", "الاتجاه", "الاقتران", "المخرَج المحاسَب عليه"]), hide_index=True)
    ui.note(
        "اقرأ الحالات بالترتيب أو اختر الأقرب لمشروعك. لاحظ أن **قرار الدمج** (المربعات الأربع أعلى كل حالة) يُتخذ **قبل** كتابة أي كود، "
        "وهو الذي يحدد طريقة التقييم وما يجب حمايته.",
        "طريقة الاستخدام", "sky", "🧭",
    )

with tabs[1]:
    case(dict(
        title="① التنبؤ بالتضخم: من AR إلى هجين بواقي وتوليف", en="Inflation forecasting", color="sky",
        question="هل يحسّن الدمج بين نموذج قياسي وتعلم آلي التنبؤ بالتضخم الشهري خطوة واحدة إلى الأمام؟",
        data="تضخم شهري (مؤشر أسعار المستهلك، تغير لوغاريتمي سنوي)، فجوة البطالة، أسعار النفط، 1990–2024. إصدارات البيانات المتاحة في الوقت الحقيقي.",
        decision={"الموضع": "البواقي + التوليف", "المستوى": "L2", "الاتجاه": "القياس في خدمة التنبؤ", "الاقتران": "متسلسل + متوازٍ"},
        baseline="منحنى فيليبس بانحدار ذاتي، مقدَّر بالمربعات الصغرى في نافذة متوسعة:",
        baseline_eq=r"\pi_{t+1}=\alpha+\rho_1\pi_t+\rho_2\pi_{t-1}+\kappa\,u^{gap}_t+\delta\,\Delta oil_t+e_{t+1}",
        diagnosis=("بواقي النموذج على فترة التدريب: Ljung–Box مقبول (لا ارتباط خطي متبقٍ)، لكن اختبار Teräsvirta **مرفوض** (p = 0.01) واختبار "
                   "McLeod–Li مرفوض: توجد لاخطية في المتوسط وتجمّع في التباين. والأداء يتدهور بوضوح في فترات الصدمات (2008، 2021–2022)."),
        design=("**هجين متسلسل:** غابة عشوائية على البواقي بمدخلات متأخرة (بواقي، تضخم، فجوة، نفط). **ثم توليف متوازٍ** بين الهجين وغابة "
                "مستقلة ومرجع AR، بأوزان مقلصة نحو المتساوية. كل شيء يُعاد تقديره في كل شهر من فترة الاختبار."),
        design_eq=r"\hat\pi^{hyb}_{t+1}=\hat\pi^{PC}_{t+1}+\hat h\big(e_t,e_{t-1},\pi_t,u^{gap}_t,\Delta oil_t\big),\qquad \hat\pi^{c}=\textstyle\sum_kw_k\hat\pi^{(k)}",
        steps=[
            ("تقسيم زمني", "Split", "تدريب أولي 1990–2009، تقييم متدحرج 2010–2024 بنافذة متوسعة."),
            ("المرجع", "Baseline", "منحنى فيليبس بالمربعات الصغرى في كل نافذة؛ وAR(2) كمرجع أبسط."),
            ("تشخيص البواقي", "Diagnose", "Ljung–Box، McLeod–Li، Teräsvirta على بواقي نافذة التدريب الأولى فقط."),
            ("متعلم البواقي", "Residual learner", "غابة على بواقي التدريب بمدخلات متأخرة؛ المعلمات الفائقة باختبار زمني داخل التدريب."),
            ("التوليف", "Combine", "متوسط بسيط + أوزان مقلصة من أداء آخر 36 شهرًا."),
            ("التقييم", "Evaluate", "RMSE، اختبار Clark–West (النماذج متداخلة)، مجموعة الثقة للنماذج، وتحليل فرعي للفترات."),
        ],
        code='''import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

def pc_design(df):                       # Phillips-curve regressors known at t
    return sm.add_constant(df[["pi", "pi_l1", "ugap", "doil"]])

preds = []
for t in range(start, len(df) - 1):      # expanding window, one-step ahead
    tr = df.iloc[:t]
    ols = sm.OLS(tr["pi_next"], pc_design(tr)).fit()
    e = tr["pi_next"] - ols.fittedvalues                     # in-window residuals only
    F = pd.concat([e.shift(1), e.shift(2), tr[["pi", "ugap", "doil"]]], axis=1).dropna()
    rf = RandomForestRegressor(500, min_samples_leaf=10, random_state=0).fit(F, e.loc[F.index])
    x_now = pc_design(df.iloc[[t]])
    f_now = np.r_[e.iloc[-1], e.iloc[-2], df[["pi", "ugap", "doil"]].iloc[t]].reshape(1, -1)
    pc = ols.predict(x_now)[0]
    preds.append(dict(date=df.index[t + 1], actual=df["pi_next"].iloc[t],
                      pc=pc, hybrid=pc + rf.predict(f_now)[0]))
out = pd.DataFrame(preds).set_index("date")
rmse = lambda c: np.sqrt(((out["actual"] - out[c]) ** 2).mean())
print({c: rmse(c) for c in ["pc", "hybrid"]})''',
        read=("قارن RMSE الهجين بمنحنى فيليبس، ثم اختبر الفرق بـ **Clark–West** لأن الهجين يحتوي النموذج القياسي (متداخل). ارسم **الفرق التراكمي في "
              "مربع الخطأ** عبر الزمن: إذا صعد بثبات فالمكسب منتظم؛ إذا قفز في 2021–2022 فقط فالمكسب ظرفي (صدمة) ويجب قوله صراحة. وأخيرًا "
              "تأكد أن التوليف البسيط ليس أفضل من الهجين المعقد."),
        wrong=["حساب البواقي بنموذج مقدَّر على كل العينة (تسرب).", "استخدام القيم المراجَعة بدل إصدارات الوقت الحقيقي.",
               "اختبار DM بدل Clark–West لنماذج متداخلة (منحاز ضد الهجين).", "تجربة 20 متعلمًا وتقرير الأفضل دون تصحيح (MCS)."],
        paper=("«نتبع Zhang (2003) في هجين متسلسل يقدّر منحنى فيليبس ثم يتعلم بواقيه بغابة عشوائية. تُبرر الخطوة الثانية برفض اختبار Teräsvirta "
               "للخطية في البواقي. كل المكونات تُعاد تقديرها في نافذة متوسعة باستخدام إصدارات البيانات في الوقت الحقيقي، ونقيّم بإحصاءة "
               "Clark–West ومجموعة الثقة للنماذج، ونبلغ عن الأداء في فترات فرعية.»"),
        refs="Zhang (2003)؛ Teräsvirta, Lin & Granger (1993)؛ Clark & West (2007)؛ Hansen, Lunde & Nason (2011)؛ Medeiros, Vasconcelos, Veiga & Zilberman (2021) JBES 39(1) «Forecasting inflation in a data-rich environment».",
    ))

with tabs[2]:
    case(dict(
        title="② عائد التعليم مع 200 ضابط: من OLS إلى DML", en="Returns to schooling with many controls", color="indigo",
        question="ما الأثر السببي لسنة تعليم إضافية على الأجر، مع ضبط خلفية أسرية ومناطقية غنية؟",
        data="مسح أسر: 12,000 فرد، لوغاريتم الأجر، سنوات التعليم، و200 ضابط محتمل (تعليم الوالدين، المنطقة، حجم الأسرة، تفاعلات، متغيرات مقطعة).",
        decision={"الموضع": "الاستدلال (الأجزاء المزعجة)", "المستوى": "L3", "الاتجاه": "الذكي في خدمة القياس", "الاقتران": "متداخل"},
        baseline="انحدار مينسر مع عدد قليل من الضوابط المختارة يدويًا:",
        baseline_eq=r"\ln w_i=\theta\,educ_i+x_i'\beta+u_i",
        diagnosis=("مع 200 ضابط، OLS على كلها غير دقيق (تباين كبير) وقد تكون العلاقات لاخطية؛ والاختيار اليدوي اعتباطي ونتائجه تتغير بحسب الاختيار "
                   "(ثلاث مواصفات معقولة تعطي 0.06 و0.08 و0.10). Lasso المفرد يحيّز θ كما رأينا في صفحة المعلومات."),
        design=("**DML للنموذج جزئي الخطية:** التعلم الآلي يقدّر $\\ell(X)=\\mathbb E[\\ln w\\mid X]$ و$m(X)=\\mathbb E[educ\\mid X]$ بتقسيم متقاطع، ثم ينحدر "
                "باقيًا على باقٍ. المتعلمات: توليف (Lasso، غابة، تعزيز). **افتراض التعرّف** (لا إرباك غير مشاهَد بشرط X) يبقى افتراضًا يُناقش."),
        design_eq=r"\hat\theta=\frac{\sum_i\big(educ_i-\hat m^{(-k)}(X_i)\big)\big(\ln w_i-\hat\ell^{(-k)}(X_i)\big)}{\sum_i\big(educ_i-\hat m^{(-k)}(X_i)\big)^2}",
        steps=[
            ("تحديد المعلمة والتعرّف", "Estimand", "θ = متوسط أثر سنة تعليم، تحت عدم الإرباك بشرط X. ناقش متغير القدرة المحذوف."),
            ("استبعاد الضوابط السيئة", "Bad controls", "لا مهنة ولا قطاع حالي (نتائج التعليم نفسه)."),
            ("اختيار المتعلمات", "Learners", "Lasso، غابة، تعزيز، وتوليفها. ضبط المعلمات داخل كل طية."),
            ("التقسيم المتقاطع", "Cross-fit", "5 طيات، تكرار 20 مرة، الوسيط."),
            ("الاستدلال", "Inference", "خطأ معياري من العزم المتعامد، عنقودي بحسب المنطقة إن لزم."),
            ("المتانة والحساسية", "Robustness", "مقارنة المتعلمات، وتحليل حساسية للإرباك غير المشاهد (Chernozhukov et al. 2022 «omitted variable bias»)."),
        ],
        code='''import doubleml as dml
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV

data = dml.DoubleMLData(df, y_col="lnwage", d_cols="educ", x_cols=controls)
results = {}
for name, learner in {"lasso": LassoCV(cv=5),
                      "forest": RandomForestRegressor(500, min_samples_leaf=5),
                      "boost": GradientBoostingRegressor()}.items():
    m = dml.DoubleMLPLR(data, ml_l=learner, ml_m=learner, n_folds=5, n_rep=20)
    m.fit()
    results[name] = (m.coef[0], m.se[0])
print(results)
# sensitivity to unobserved confounding (Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis)
m.sensitivity_analysis(cf_y=0.03, cf_d=0.03)
print(m.sensitivity_summary)''',
        read=("θ̂ يُقرأ كمعامل انحدار عادي: 0.085 تعني نحو 8.5% زيادة في الأجر لكل سنة. **تطابق** التقديرات عبر المتعلمات دليل استقرار؛ اختلافها "
              "الكبير يعني أن بعض المتعلمات لا تحقق شروط المعدلات. تحليل الحساسية يجيب: «كم يجب أن يكون الإرباك غير المشاهد قويًا ليجعل الأثر صفرًا؟»"),
        wrong=["تقديم DML كحل لمشكلة القدرة غير المشاهدة؛ هو يعالج الضوابط المشاهدة فقط.", "إدخال المهنة كضابط (ضابط سيئ).",
               "تقسيم عشوائي مع أفراد من الأسرة نفسها في طيات مختلفة.", "تقرير تقدير تقسيم واحد بدل الوسيط عبر التكرارات."],
        paper=("«نقدّر θ بالتعلم الآلي المزدوج (Chernozhukov et al. 2018) في نموذج جزئي الخطية، مع تقدير الأجزاء المزعجة بتوليف متعلمات وتقسيم متقاطع "
               "خماسي مكرر 20 مرة. نبلغ عن التقدير لكل متعلم وعن تحليل الحساسية للإرباك غير المشاهد. التعرّف يعتمد على عدم الإرباك بشرط الضوابط، "
               "ونناقش حدوده في القسم X.»"),
        refs="Mincer (1974)؛ Card (1999) Handbook of Labor Economics 3A؛ Chernozhukov et al. (2018)؛ Bach et al. (2022) JMLR DoubleML؛ Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022) «Long story short: omitted variable bias in causal ML».",
    ))

with tabs[3]:
    case(dict(
        title="③ تقلب الأسهم: من GARCH إلى GARCH عصبي بنيوي", en="Equity volatility", color="orange",
        question="هل يحسّن نموذج يدمج معادلة GARCH داخل شبكة تكرارية التنبؤ بالتقلب وقياس المخاطر، مع بقاء المعلمات مفسرة؟",
        data="عوائد يومية لمؤشر S&P 500 (آخر 4000 يوم)، مع التقلب المحقق كمعيار تقييم.",
        decision={"الموضع": "الدالة + دالة الهدف", "المستوى": "L2 / L5", "الاتجاه": "تكافلي", "الاقتران": "مندمج (معمارية)"},
        baseline="GJR-GARCH(1,1) بتوزيع ستيودنت بالأرجحية العظمى:",
        baseline_eq=r"\sigma_t^2=\omega+\big(\alpha+\gamma\,\mathbb 1\{\epsilon_{t-1}<0\}\big)\epsilon_{t-1}^2+\beta\sigma_{t-1}^2",
        diagnosis=("البواقي المعيارية تجتاز Ljung–Box لكنها تفشل في اختبارات اختراق VaR في الأزمات (تجمّع الاختراقات): الاستجابة للصدمات ليست ثابتة كما "
                   "تفترض المعادلة. نريد مرونة دون خسارة تفسير α وβ وγ."),
        design=("**GARCH-LSTM البنيوي (Zhao et al. 2024):** معادلة GJR تصبح بوابة إخراج، وخلية LSTM تعدّلها ضربيًا ضمن حدود تضمن موجبية التباين. "
                "تُدرَّب بالأرجحية (ستيودنت) لا بمربع الخطأ. عند $w=0$ يعود النموذج GJR تمامًا، فيُقرأ $w$ كمقياس لإسهام الشبكة."),
        design_eq=r"\sigma^2_t=\underbrace{\omega+(\alpha+\gamma\mathbb 1_{\epsilon<0})\epsilon_{t-1}^2+\beta\sigma^2_{t-1}}_{o_t\ \text{(GJR)}}\cdot\big(1+w\tanh c_t\big),\quad |w|<1",
        steps=[
            ("المرجع", "Baseline", "GARCH وGJR بالأرجحية؛ احسب AIC/BIC ومتوسط الخسارة QLIKE."),
            ("النموذج المندمج", "Fused model", "StructuralGARCHLSTM بنواة GJR، وGARCHNN للمقارنة."),
            ("التدريب", "Train", "أرجحية ستيودنت، إيقاف مبكر على 20% أخيرة من التدريب، بذور متعددة."),
            ("التنبؤ", "Forecast", "خطوة واحدة دقيقة، وأفق 10 أيام بمحاكاة مونت كارلو عبر المعادلة."),
            ("التقييم", "Evaluate", "QLIKE مقابل التقلب المحقق، اختبارات Kupiec وChristoffersen لـ VaR، وDM."),
            ("التفسير", "Interpret", "قارن α وβ وγ بين GJR والمندمج، واقرأ w."),
        ],
        code='''import garchai
from garchai.datasets import load_returns

r = load_returns("sp500").iloc[-4000:].to_numpy(float)
cfg = garchai.ModelConfig(distribution="student-t", seed=0)
train = garchai.TrainConfig(max_epochs=300, lr=0.01)

base = garchai.GJRGARCH(config=cfg);                         base.fit(r, train=train)
fused = garchai.StructuralGARCHLSTM(kernel="gjr", config=cfg); fused.fit(r, train=train)

for m in (base, fused):
    res = m.result_
    print(type(m).__name__, res.loglikelihood, res.parameters)
    print(" 1-day vol:", m.forecast(horizon=1).volatility,
          " 10-day path:", m.forecast(horizon=10, n_simulations=20000, seed=1).volatility)
print("coupling w =", float(fused.coupling))   # 0 -> pure GJR''',
        read=("إذا تحسنت الأرجحية بوضوح (فرق AIC كبير رغم المعلمات الإضافية) وتحسنت اختبارات VaR في الأزمات، والمعاملات α وβ وγ بقيت قريبة "
              "من GJR، فالشبكة أضافت **تعديلًا** مفيدًا دون أن تغيّر البنية. إذا كان $w\\approx0$ فالبيانات لا تحتاج الجزء العصبي."),
        wrong=["تدريب بمربع الخطأ على العوائد بدل الأرجحية (يتجاهل الذيول).", "مقارنة الأرجحية داخل العينة فقط لنموذج بمعلمات أكثر.",
               "عدم تثبيت البذور: الشبكات تعطي نتائج مختلفة بين التشغيلات.", "تقييم التقلب بمربع العائد (مقياس مشوش جدًا) بدل التقلب المحقق."],
        paper=("«نقدّر GARCH-LSTM البنيوي لـ Zhao et al. (2024)، حيث معادلة GJR-GARCH بوابة إخراج تعدّلها خلية LSTM ضمن حدود تضمن موجبية التباين، "
               "بالأرجحية العظمى لتوزيع ستيودنت. نبلغ عن المعاملات القياسية ومعامل الاقتران w، ونقيّم التنبؤات بخسارة QLIKE واختبارات تغطية VaR.»"),
        refs="Glosten, Jagannathan & Runkle (1993) J. Finance 48(5)؛ Patton (2011) J. Econometrics 160(1) QLIKE؛ Christoffersen (1998) International Economic Review 39(4)؛ Zhao et al. (2024) AAAI.",
    ))

with tabs[4]:
    case(dict(
        title="④ مؤشر معنويات من الأخبار وأثره على الاستثمار", en="A text-based index in a regression", color="amber",
        question="هل تؤثر معنويات الأخبار الاقتصادية على قرارات الاستثمار للشركات، وما حجم الأثر بعد تصحيح خطأ القياس؟",
        data="300,000 مقال اقتصادي 2005–2023، و6,000 شركة–سنة مع الاستثمار والضوابط المالية. 1,000 مقال مرمَّزة بشريًا بعينة عشوائية.",
        decision={"الموضع": "المعلومات (إنشاء متغير)", "المستوى": "L4 ثم L3", "الاتجاه": "الذكي في خدمة القياس", "الاقتران": "متسلسل مع تصحيح"},
        baseline="انحدار الاستثمار على مؤشر معنويات معجمي (Loughran–McDonald) مع آثار ثابتة:",
        baseline_eq=r"I_{it}/K_{i,t-1}=\beta\,S_{t}+x_{it}'\gamma+\alpha_i+\tau_t+u_{it}",
        diagnosis=("المعجم يتجاهل السياق والنفي، ودقته على العينة المرمَّزة 64% فقط. نموذج لغوي يصل إلى 86%. لكن حتى 86% تعني خطأ قياس، والخطأ "
                   "**مرتبط** بطول المقال ونوعه، وقد يكون مرتبطًا بالفترة (النموذج يعرف ما حدث بعد 2008)."),
        design=("**المرحلة 1 (L4):** نموذج لغوي بمطالبة مثبتة يعطي درجة لكل مقال، تُجمَّع شهريًا. **المرحلة 2 (L3/تصحيح):** تقدير β بـ "
                "PPI مستخدمًا العينة المرمَّزة لتصحيح التحيز، مع تحليل حساسية للمطالبة والنموذج. واختبار تسرب المستقبل بإخفاء الأسماء والتواريخ."),
        design_eq=r"\hat\beta^{PP}=\hat\beta\big(\hat S^{\text{all}}\big)-\Big[\hat\beta\big(\hat S^{\text{lab}}\big)-\hat\beta\big(S^{\text{lab}}\big)\Big]",
        steps=[
            ("تعريف المفهوم", "Construct", "«المعنويات» = تقييم كاتب المقال لاتجاه الاقتصاد خلال 12 شهرًا، على مقياس من −1 إلى 1."),
            ("الترميز البشري", "Gold standard", "1,000 مقال بعينة عشوائية طبقية، مرمِّزان مستقلان، قياس الاتفاق."),
            ("القياس الآلي", "Measure", "نموذج لغوي بمطالبة وإصدار مثبتين مسبقًا وتوثيقهما؛ تجميع شهري."),
            ("فحص الخطأ", "Error audit", "دقة، وانحدار الخطأ على خصائص المقال والفترة؛ اختبار إخفاء الهوية للتسرب."),
            ("التقدير المصحح", "Corrected estimate", "PPI أو DSL لـ β مع الآثار الثابتة وأخطاء عنقودية."),
            ("الحساسية", "Sensitivity", "مطالبتان بديلتان ونموذجان مختلفان؛ مقارنة بالمعجم."),
        ],
        code='''import numpy as np
# s_hat: LLM sentiment for every article; s_lab: human codes on the random labelled subset
monthly_hat = articles.groupby("month")["s_hat"].mean()
panel["S_hat"] = panel["month"].map(monthly_hat)

def beta_fe(df, s_col):                       # within-firm, within-year OLS slope on sentiment
    d = df.copy()
    for c in [s_col, "inv"]:
        d[c] = d[c] - d.groupby("firm")[c].transform("mean") - d.groupby("year")[c].transform("mean") + d[c].mean()
    return np.polyfit(d[s_col], d["inv"], 1)[0]

b_all = beta_fe(panel, "S_hat")
lab = panel[panel["month"].isin(labelled_months)]           # months with human-coded articles
rectifier = beta_fe(lab, "S_hat") - beta_fe(lab, "S_true")   # how far the AI measure moves beta
beta_pp = b_all - rectifier
# standard errors: bootstrap firms and labelled articles jointly (or use ppi_py / DSL)''',
        read=("قارن ثلاثة أرقام: β بالمعجم، β بالنموذج اللغوي دون تصحيح، وβ المصحح. إذا كان المصحح **أكبر** من غير المصحح فقد كان هناك تخفيف؛ "
              "إذا كان أصغر فالخطأ غير كلاسيكي ضخّم الأثر. فترة الثقة المصححة أوسع من الساذجة لكنها **صادقة**."),
        wrong=["معاملة المؤشر المولَّد كأنه بلا خطأ.", "تعديل المطالبة بعد رؤية نتائج الانحدار.",
               "عينة ترميز غير عشوائية (المقالات الأسهل فقط).", "تسرب المستقبل: النموذج يعرف أزمات لاحقة للمقال."],
        paper=("«نقيس المعنويات بنموذج لغوي بمطالبة مثبتة مسبقًا (الملحق A)، ونتحقق منها على 1,000 مقال مرمَّز بشريًا بعينة عشوائية (دقة 86%). لأن "
               "خطأ القياس يرتبط بخصائص المقال، نصحح تقدير β بالاستدلال المدعوم بالتنبؤ (Angelopoulos et al. 2023)، ونختبر تسرب المستقبل بإخفاء "
               "الأسماء والتواريخ.»"),
        refs="Loughran & McDonald (2011)؛ Baker, Bloom & Davis (2016) QJE 131(4)؛ Angelopoulos et al. (2023) Science؛ Egami et al. (2023)؛ Ludwig, Mullainathan & Rambachan (2025).",
    ))

with tabs[5]:
    case(dict(
        title="⑤ منحنى طلب بقيود نظرية: شبكة رتيبة", en="Demand estimation under shape restrictions", color="emerald",
        question="ما شكل منحنى الطلب على الوقود بحسب السعر والدخل، مع ضمان احترام قانون الطلب في كل النقاط؟",
        data="5,000 مشاهدة أسرة: كمية الوقود، السعر المحلي، الدخل، حجم الأسرة، المنطقة. بيانات قليلة عند الأسعار المرتفعة جدًا.",
        decision={"الموضع": "دالة الهدف والمعمارية", "المستوى": "L5", "الاتجاه": "القياس في خدمة الذكي", "الاقتران": "مندمج (قيد)"},
        baseline="طلب لوغاريتمي خطي، أو شرائح غير معلمية:",
        baseline_eq=r"\ln q_i=\alpha+\epsilon_p\ln p_i+\epsilon_y\ln y_i+x_i'\gamma+u_i",
        diagnosis=("النموذج اللوغاريتمي يفرض مرونة ثابتة، والبيانات تشير إلى مرونة متغيرة بحسب الدخل. الشبكة الحرة تعطي منحنى متعرجًا فيه "
                   "مقاطع **صاعدة** في المناطق قليلة البيانات (طلب يزداد بالسعر)، وهذا مرفوض اقتصاديًا ويضر التنبؤ في تلك المناطق."),
        design=("شبكة **رتيبة بالبناء** في السعر (أوزان غير سالبة على مسار السعر) ومرنة في الدخل والخصائص، أو شبكة عادية مع **عقوبة** على الميل الموجب "
                "في نقاط اصطناعية تغطي المناطق الفارغة. اختيار وزن العقوبة خارج العينة."),
        design_eq=r"\hat f=\arg\min_f\ \tfrac1n\sum_i\big(\ln q_i-f(\ln p_i,\ln y_i,x_i)\big)^2+\lambda\,\tfrac1M\sum_m\big[\partial_{\ln p}f(\tilde z_m)\big]_+^2",
        steps=[
            ("القيد", "Restriction", "∂f/∂ln p ≤ 0 لكل المدخلات (قانون الطلب، غير جيفن للوقود)."),
            ("نقاط النظرية", "Collocation points", "شبكة اصطناعية من الأسعار والدخول تغطي المنطقة الكاملة بما فيها الفارغة."),
            ("التدريب", "Train", "شبكة 2×64 مع العقوبة، λ من شبكة قيم بتحقق متقاطع."),
            ("التحقق من القيد", "Check", "نسبة النقاط ذات الميل الموجب على شبكة اختبار كثيفة."),
            ("المرونات", "Elasticities", "∂f/∂ln p بالتفاضل التلقائي لكل أسرة، ومتوسطها حسب شرائح الدخل."),
            ("المقارنة", "Compare", "خطأ الاختبار والمرونات مقابل اللوغاريتمي والشبكة الحرة."),
        ],
        code='''import torch, torch.nn as nn
net = nn.Sequential(nn.Linear(k, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
Zc = collocation_grid(requires_grad=True)          # synthetic (ln p, ln y, x) points, column 0 = ln p
for epoch in range(3000):
    opt.zero_grad()
    fit = ((net(Z_train).squeeze() - lnq_train) ** 2).mean()
    dq_dp = torch.autograd.grad(net(Zc).sum(), Zc, create_graph=True)[0][:, 0]
    theory = torch.relu(dq_dp).pow(2).mean()       # penalise any upward-sloping demand
    (fit + lam * theory).backward()
    opt.step()
# elasticity for each household = d ln q / d ln p at its own data point
Zt = Z_test.clone().requires_grad_(True)
elast = torch.autograd.grad(net(Zt).sum(), Zt)[0][:, 0]''',
        read=("المرونة المقدرة لكل أسرة تُعرض بحسب شرائح الدخل: الأسر الفقيرة عادةً أكثر مرونة. تحقق أن نسبة المخالفات قريبة من الصفر، وأن خطأ الاختبار "
              "**في المناطق قليلة البيانات** انخفض مقارنة بالشبكة الحرة؛ هذا هو مكسب النظرية الحقيقي."),
        wrong=["فرض الرتابة على سلعة قد لا تحققها.", "اختيار λ بخطأ التدريب.",
               "عقوبة على نقاط البيانات فقط فتبقى المناطق الفارغة بلا قيد.", "قراءة المرونة كأثر سببي دون معالجة داخلية السعر (تحتاج أداة)."],
        paper=("«نقدّر الطلب بشبكة عصبية مع عقوبة على الميل الموجب في السعر عند نقاط اصطناعية تغطي فضاء المدخلات (L5). يُختار وزن العقوبة بالتحقق "
               "المتقاطع. نبلغ عن نسبة النقاط المخالفة (0.2%) والمرونات حسب شرائح الدخل. ولأن السعر قد يكون داخليًا، نستخدم ضريبة الوقود المحلية أداةً في "
               "النسخة الثانية من التقدير.»"),
        refs="Hausman & Newey (1995) Econometrica 63(6)؛ Blundell, Horowitz & Parey (2012) Quantitative Economics 3(1)؛ Chetverikov, Santos & Shaikh (2018)؛ Raissi et al. (2019).",
    ))

with tabs[6]:
    ui.section("قالب منهجي لأي مشروع دمج", "A methods template", "violet")
    st.markdown("انسخ هذا القالب في ملاحظاتك قبل البدء، واملأه **قبل** كتابة الكود:")
    st.code(
        """1. السؤال والمخرَج المحاسَب عليه:   [معلمة + فترة ثقة | تنبؤ + خطأ خارج العينة | قرار + رفاه]
2. المعلمة/الهدف بدقة:                [مثال: متوسط أثر سنة تعليم على ln الأجر]
3. الاستراتيجية التعرّفية (إن سببي):    [عدم إرباك بشرط X | أداة | فروق في الفروق | انقطاع]
4. النموذج المرجعي القياسي:           [المعادلة]
5. التشخيص الذي يبرر الدمج:           [الاختبارات ونتائجها]
6. قرار الدمج:
   - الموضع:   [بيانات | معلومات | دالة | معلمات | بواقي | هدف | استدلال | توليف/قرار]
   - المستوى:  [L0 … L7]
   - الاتجاه:  [ذكي→قياس | قياس→ذكي | تكافلي]
   - الاقتران: [متسلسل | مشترك | مندمج] والمعمارية [متسلسل | متوازٍ | متداخل | تكراري]
7. ما يجب حمايته:                     [التعامد/التقسيم | التقسيم الزمني | تصحيح خطأ القياس | صحة القيد]
8. خطة التقييم:                       [المقاييس، الاختبارات، الفترات الفرعية، المراجع]
9. المتانة والحساسية:                  [متعلمات، بذور، مطالبات، λ، تقسيمات]
10. إعادة الإنتاج:                     [الإصدارات، البذور، الكود، البيانات]""",
        language="text",
    )
    ui.note(
        "**القاعدة الذهبية:** إذا لم تستطع ملء البندين 6 و7 بدقة، فأنت لم تصمم دمجًا بعد؛ أنت تجرب أدوات. الأوراق التي تُقبل في المجلات "
        "الجيدة تبرر كل بند من هذه البنود صراحة.",
        "قبل أن تكتب سطرًا من الكود", "violet", "📝",
    )
