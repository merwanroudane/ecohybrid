import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

from core import ui

ui.hero(
    "التنبؤ والسلاسل الزمنية في عصر تعلم الآلة",
    "Forecasting & Time Series — From Box–Jenkins to Machine Learning and Foundation Models",
    "التنبؤ هو الميدان الذي يتفوق فيه تعلم الآلة طبيعيًا لأنه سؤال ŷ خالص. لكن السلاسل الاقتصادية قصيرة وغير مستقرة وتتعرض للمراجعات، "
    "لذلك يبقى الحس القياسي حاسمًا: التقسيم الزمني، البيانات الآنية، الاختبارات الإحصائية للمقارنة، والمعايير القوية.",
    "orange", "sky",
)

tabs = st.tabs([
    ":material/history: من ARIMA إلى ML",
    ":material/insights: ماذا تعلّمنا؟",
    ":material/schedule: التنبؤ الآني",
    ":material/memory: التعلم العميق للسلاسل",
    ":material/science: مختبر التنبؤ المتدحرج",
    ":material/straighten: فترات التنبؤ والتقييم",
    ":material/account_balance_wallet: التمويل وتسعير الأصول",
])

with tabs[0]:
    ui.section("تطور أدوات التنبؤ الاقتصادي", "Evolution of economic forecasting", "orange")
    ui.ladder([
        ("النماذج الهيكلية الكبيرة", "1950s–1970s", "نماذج كلاين–غولدبرغر بمئات المعادلات؛ هزّها نقد لوكاس وتضخم السبعينيات."),
        ("ARIMA و VAR", "1970–1990", "Box–Jenkins (1970)، Sims (1980)، BVAR مع قبْلي مينيسوتا (Litterman, 1986) = أول انكماش منظّم."),
        ("الشبكات العصبية الأولى", "1988–2000", "White (1988)، Swanson & White (1997)؛ نتائج مختلطة ومشكلات إفراط في التوفيق."),
        ("البيانات الكثيرة", "2000–2012", "مؤشرات الانتشار (Stock & Watson, 2002)، FAVAR، BVAR الكبيرة (Bańbura et al., 2010)، التنبؤ الآني."),
        ("تعلم الآلة", "2012–2020", "لاسو، غابات، تعزيز، شبكات؛ Medeiros et al. (2021): الغابة العشوائية تتفوق في التنبؤ بالتضخم الأمريكي."),
        ("الهجين والتفسير", "2020–2024", "Goulet Coulombe et al. (2022)، الغابة الكلية (MRF)، BART-VAR، فضيلة التعقيد."),
        ("النماذج التأسيسية", "2024–2026", "TimesFM، Chronos، Moirai، BISTRO (BIS 2026)، MACROCAST — مع قلق التلوث والمراجعات."),
    ])
    st.latex(r"\hat y_{t+h\mid t}=\hat f_h\big(y_t,y_{t-1},\dots,\mathbf x_t,\mathbf x_{t-1},\dots\big),\qquad \hat f_h=\arg\min_f \sum_t L\big(y_{t+h},f(\cdot)\big)+\lambda\,\Omega(f)")
    st.markdown("**التنبؤ المباشر (Direct)** يقدّر نموذجًا لكل أفق $h$؛ **التكراري (Iterated)** يقدّر نموذج خطوة واحدة ويكرره. تعلم الآلة يُستخدم غالبًا بالتنبؤ المباشر.")

with tabs[1]:
    ui.section("ماذا تعلّمنا من الأدبيات التجريبية؟", "Lessons from the empirical literature", "violet")
    ui.cards([
        {"title": "اللاخطية هي المفتاح", "en": "Goulet Coulombe, Leroux, Stevanovic & Surprenant (2022, JAE)", "color": "violet",
         "body": "فككوا مكاسب ML إلى أربع خصائص: <b>اللاخطية</b> هي مصدر المكسب الحقيقي، ونموذج العوامل يبقى أفضل تنظيم، و K-fold أفضل ممارسة للضبط، "
                 "وخسارة L2 أفضل من ε-insensitive. المكاسب تظهر في فترات عدم اليقين والضغط المالي وانفجار الفقاعات."},
        {"title": "الغابات للتضخم", "en": "Medeiros, Vasconcelos, Veiga & Zilberman (2021, JBES)", "color": "emerald",
         "body": "الغابة العشوائية تتفوق على المعايير التقليدية في التنبؤ بالتضخم الأمريكي، بفضل التقاط اللاخطيات وتفاعلات عدم اليقين."},
        {"title": "الاقتصاد كغابة عشوائية", "en": "Goulet Coulombe (2024, JAE)", "color": "teal",
         "body": "MRF: انحدار خطي بمعاملات متغيرة زمنيًا معممة (GTVPs) تُقدَّر بغابة ⇒ تفسير مباشر؛ تنبأ بارتفاع البطالة 2008."},
        {"title": "فضيلة التعقيد", "en": "Kelly, Malamud & Zhou (2024, JF)", "color": "fuchsia",
         "body": "نماذج بمعلمات أكثر من المشاهدات تتنبأ بعوائد السوق أفضل نظريًا وتجريبيًا؛ R² خارج العينة مقياس ضعيف للقيمة الاقتصادية."},
        {"title": "لكن البساطة تنافس", "en": "IMF (2025) GDP nowcasting", "color": "amber",
         "body": "نماذج Bridge والعوامل الديناميكية تفوقت غالبًا على ML المعقد في السلاسل القصيرة؛ Lasso/Elastic Net أكثر تنافسية من الشبكات العميقة."},
        {"title": "BART للذيول", "en": "Clark, Huber, Koop, Marcellino & Pfarrhofer (2023, IER)", "color": "rose",
         "body": "BART-VAR متعدد المتغيرات يلتقط اللاخطيات في توزيعات التنبؤ ويحسّن التنبؤ بمخاطر الذيول مقارنة بـ BVAR-SV."},
    ], cols=3)

with tabs[2]:
    ui.section("التنبؤ الآني: تقدير الحاضر قبل نشر بياناته", "Nowcasting", "teal")
    st.mermaid_chart(
        """
flowchart LR
    M[Monthly indicators<br/>IP, retail, PMI, jobs] --> R[Ragged-edge panel]
    W[Weekly/daily data<br/>claims, card spending, Google Trends] --> R
    T[Text & satellite<br/>news, night lights] --> R
    R --> F{Model}
    F --> B[Bridge equations]
    F --> DFM[Dynamic factor model + Kalman]
    F --> MI[MIDAS / sg-LASSO-MIDAS]
    F --> ML[RF, GBM, NN with release calendar]
    B & DFM & MI & ML --> N[Nowcast of current-quarter GDP]
"""
    )
    ui.cards([
        {"title": "MIDAS", "en": "Ghysels, Santa-Clara & Valkanov (2004)", "color": "teal", "body": "انحدار بتواترات مختلطة بأوزان فجوات مرنة (Almon، Beta)."},
        {"title": "sg-LASSO-MIDAS", "en": "Babii, Ghysels & Striaukas (2022, JBES)", "color": "violet", "body": "لاسو المجموعات المتناثر مع قواميس كثيرات حدود متعامدة؛ تفوق على تنبؤات NY Fed، والنص إضافة مفيدة. حزمة midasml."},
        {"title": "DFM الآني", "en": "Giannone, Reichlin & Small (2008)", "color": "orange", "body": "عوامل ديناميكية مع مرشح كالمان يتعامل مع الحافة غير المتساوية (Ragged edge)."},
        {"title": "BART بتواتر مختلط", "en": "Huber, Koop, Onorante, Pfarrhofer & Schreiner (2023, JoE)", "color": "rose", "body": "VAR لا معلمي بأشجار انحدار جمعية للتنبؤ الآني خلال جائحة كوفيد."},
    ], cols=4)
    ui.note("التقييم الصحيح للتنبؤ الآني يعيد بناء **مجموعة المعلومات كما كانت متاحة فعلًا** في كل تاريخ (Real-time vintages) — لا القيم المنقحة لاحقًا. "
            "وإلا كان التحسن المعلن مضللًا (Pseudo real-time).", "تحذير", "amber", "⚠️")

with tabs[3]:
    ui.section("الشبكات العميقة للسلاسل الزمنية", "Deep learning for time series", "indigo")
    dl = pd.DataFrame([
        ("RNN / LSTM / GRU", "حالة كامنة تتحدث عبر الزمن مع بوابات ذاكرة", "تشبه نماذج فضاء الحالة غير الخطية", "عينات قصيرة، صعوبة التدريب"),
        ("TCN", "التفافات سببية متوسعة (Dilated)", "مرشحات خطية متعددة المقاييس", "تحديد طول الذاكرة"),
        ("DeepAR", "تنبؤ احتمالي تكراري عبر سلاسل كثيرة (Salinas et al., 2020)", "بانل سلاسل مع معلمات مشتركة", "افتراض توزيعي"),
        ("N-BEATS / N-HiTS", "كتل اتجاه وموسمية قابلة للتفسير", "تفكيك الاتجاه والموسمية", "أفضل للسلاسل الكثيرة"),
        ("TFT", "انتباه مع متغيرات ثابتة ومعروفة مستقبلًا (Lim et al., 2021)", "انحدار ديناميكي متعدد الآفاق", "يحتاج بيانات كثيرة"),
        ("PatchTST / iTransformer", "تقطيع السلسلة إلى رقع ومعالجة المتغيرات كرموز", "VAR بانتباه", "حساسية للضبط"),
        ("Neural ODE / SDE", "ديناميكيات مستمرة متعلَّمة", "نماذج الانتشار في التمويل", "تفسير اقتصادي محدود"),
    ], columns=["المعمارية", "الفكرة", "النظير القياسي", "المحذور"])
    st.table(dl)
    ui.note("في الاقتصاد الكلي: عدد المشاهدات الفصلية بعد الحرب ~300 فقط. الشبكات العميقة تنجح أكثر في **البيانات عالية التواتر** (التمويل، الطاقة، المبيعات) "
            "أو عند **التدريب عبر سلاسل كثيرة** (Global models) — وهذا ما تستغله النماذج التأسيسية.", "متى تنجح؟", "indigo", "💡")

with tabs[4]:
    ui.section("مختبر متحرك: AR خطي مقابل غابة عشوائية في التنبؤ المتدحرج", "Rolling-origin forecasting lab", "pink")
    st.markdown("سلسلة مولّدة من **نموذج عتبة ذاتي الانحدار (TAR)** — ديناميكية مختلفة في الركود والتوسع. نقارن تنبؤ خطوة واحدة متدحرجًا من AR(2) خطي "
                "وغابة عشوائية على الفجوات. شغّل الحركة لترى التنبؤات تتقدم زمنيًا مع RMSE التراكمي.")
    thr = st.slider("قوة اللاخطية (الفرق بين النظامين)", 0.0, 1.2, 0.8, 0.1, key="tar_k")

    @st.cache_data(show_spinner="تشغيل التنبؤ المتدحرج...")
    def rolling_lab(k: float):
        rng = np.random.default_rng(3)
        T = 420
        y = np.zeros(T)
        for t in range(2, T):
            if y[t - 1] < 0:
                y[t] = -0.2 + (0.4 + k) * y[t - 1] - 0.3 * y[t - 2] + rng.normal(0, .8)
            else:
                y[t] = 0.3 + (0.4 - k / 2) * y[t - 1] + 0.1 * y[t - 2] + rng.normal(0, .8)
        lags = np.column_stack([y[1:-1], y[:-2]]); target = y[2:]
        start = 250
        ar, rf = [], []
        for t in range(start, len(target)):
            Xtr, ytr = lags[:t], target[:t]
            Z = np.column_stack([np.ones(len(Xtr)), Xtr])
            b = np.linalg.lstsq(Z, ytr, rcond=None)[0]
            ar.append(np.r_[1, lags[t]] @ b)
            if (t - start) % 10 == 0:
                model = RandomForestRegressor(n_estimators=150, min_samples_leaf=8, random_state=0, n_jobs=-1).fit(Xtr, ytr)
            rf.append(model.predict(lags[t:t + 1])[0])
        idx = np.arange(start, len(target))
        return idx, target[idx], np.array(ar), np.array(rf), target

    idx, actual, ar, rf, full = rolling_lab(thr)
    steps = list(range(10, len(idx) + 1, 10))

    def roll_tr(m):
        r_ar = np.sqrt(np.mean((actual[:m] - ar[:m]) ** 2)); r_rf = np.sqrt(np.mean((actual[:m] - rf[:m]) ** 2))
        return [go.Scatter(x=idx[:m], y=actual[:m], mode="lines", line=dict(color="#94a3b8", width=2), name="الفعلي"),
                go.Scatter(x=idx[:m], y=ar[:m], mode="lines", line=dict(color=ui.COLORS["sky"], width=2.5), name=f"AR(2) — RMSE {r_ar:.3f}"),
                go.Scatter(x=idx[:m], y=rf[:m], mode="lines", line=dict(color=ui.COLORS["rose"], width=2.5), name=f"Random forest — RMSE {r_rf:.3f}")]

    fig = go.Figure(data=roll_tr(steps[0]), frames=[go.Frame(data=roll_tr(m), name=str(m)) for m in steps])
    fig.update_xaxes(title="الزمن", range=[idx[0] - 2, idx[-1] + 2]); fig.update_yaxes(title="y", range=[actual.min() - .5, actual.max() + .5])
    ui.style_fig(fig, "تنبؤات خطوة واحدة متدحرجة (نافذة متوسعة)")
    ui.add_play(fig, steps, duration=350, prefix="عدد التنبؤات: ", redraw=True)
    ui.show(fig)
    e_ar, e_rf = actual - ar, actual - rf
    dlt = e_ar ** 2 - e_rf ** 2
    dm = dlt.mean() / np.sqrt(np.var(dlt, ddof=1) / len(dlt))
    pval = 2 * (1 - stats.norm.cdf(abs(dm)))
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE: AR(2)", f"{np.sqrt(np.mean(e_ar ** 2)):.3f}")
    c2.metric("RMSE: Random forest", f"{np.sqrt(np.mean(e_rf ** 2)):.3f}")
    c3.metric("إحصاءة Diebold–Mariano", f"{dm:.2f}", f"p = {pval:.3f}", delta_color="off")
    ui.note("اجعل «قوة اللاخطية» = 0: يصبح النموذج قريبًا من الخطي ويتعادل AR مع الغابة أو يتفوق عليها (الغابة أكثر تباينًا). "
            "زد اللاخطية: تتفوق الغابة. هذا بالضبط استنتاج غوليه كولومب وزملائه: **المكسب يأتي من اللاخطية**.", "التجربة", "pink", "🧪")

with tabs[5]:
    ui.section("فترات التنبؤ وتقييم التنبؤات", "Prediction intervals & forecast evaluation", "emerald")
    st.markdown("**التنبؤ المطابق (Conformal prediction)**: فترات تنبؤ بتغطية مضمونة في العينات المحدودة دون افتراض توزيعي، لأي متعلم — "
                "نستخدم بواقي عينة معايرة (Calibration) ونأخذ الكمّي المناسب. للبيانات الزمنية: نسخ تكيفية (Adaptive conformal) ونسخ للمتحكمات التركيبية (Chernozhukov, Wüthrich & Zhu, 2021).")
    alpha = st.slider("مستوى الدلالة α (التغطية المستهدفة = 1−α)", 0.05, 0.3, 0.1, 0.05, key="conf_a")
    q = np.quantile(np.abs(e_rf[:60]), 1 - alpha)
    cover = np.mean(np.abs(e_rf[60:]) <= q)
    fig = go.Figure([
        go.Scatter(x=idx[60:], y=rf[60:] + q, mode="lines", line=dict(width=0), showlegend=False),
        go.Scatter(x=idx[60:], y=rf[60:] - q, mode="lines", fill="tonexty", fillcolor="rgba(225,29,72,0.15)", line=dict(width=0), name=f"فترة مطابقة {1 - alpha:.0%}"),
        go.Scatter(x=idx[60:], y=actual[60:], mode="markers", marker=dict(size=5, color="#334155"), name="الفعلي"),
        go.Scatter(x=idx[60:], y=rf[60:], mode="lines", line=dict(color=ui.COLORS["rose"]), name="تنبؤ الغابة"),
    ])
    ui.style_fig(fig, f"التغطية الفعلية خارج عينة المعايرة = {cover:.1%} (المستهدف {1 - alpha:.0%})", height=420)
    ui.show(fig)
    ev = pd.DataFrame([
        ("Diebold–Mariano (1995)", "تساوي الدقة التنبؤية لنموذجين", "صالح للنماذج غير المتداخلة؛ تصحيح HAC للآفاق h>1"),
        ("Clark–West (2007)", "مقارنة نماذج متداخلة", "يصحح التحيز في MSPE للنموذج الأكبر"),
        ("Giacomini–White (2006)", "القدرة التنبؤية المشروطة", "يسمح بتقدير متدحرج ومقارنة طرق لا نماذج"),
        ("Model Confidence Set (Hansen, Lunde & Nason, 2011)", "مجموعة النماذج الأفضل إحصائيًا", "يعالج تعدد المقارنات بين عشرات النماذج"),
        ("Forecast encompassing", "هل يحوي تنبؤ معلومات تنبؤ آخر؟", "أساس التوليف بين القياسي و ML"),
        ("PIT / CRPS / Log score", "تقييم التنبؤات التوزيعية", "ضروري للتنبؤ بالمخاطر والذيول"),
    ], columns=["الاختبار", "يختبر", "ملاحظات"])
    st.table(ev)

with tabs[6]:
    ui.section("تعلم الآلة في التمويل وتسعير الأصول", "Machine learning in empirical asset pricing", "fuchsia")
    ui.cards([
        {"title": "تسعير الأصول تجريبيًا بالـ ML", "en": "Gu, Kelly & Xiu (2020, RFS)", "color": "fuchsia",
         "body": "مقارنة شاملة: الشبكات والأشجار تضاعف تقريبًا الأداء الاقتصادي لاستراتيجيات المحافظ مقارنة بالانحدار؛ أهم المتنبئات: الزخم والسيولة والتقلب."},
        {"title": "نماذج المشفّر الذاتي", "en": "Gu, Kelly & Xiu (2021, J. Econometrics)", "color": "violet",
         "body": "عوامل كامنة بأحمال لا خطية تعتمد على خصائص الأسهم؛ أخطاء تسعير خارج العينة أصغر بكثير من النماذج العاملية الرائدة."},
        {"title": "التعلم العميق مع عدم المراجحة", "en": "Chen, Pelger & Zhu (2024, Management Science)", "color": "teal",
         "body": "شبكة لعامل الخصم العشوائي SDF تُدرَّب على شروط عدم المراجحة E[M R]=0 بصيغة تنافسية — النظرية داخل الخسارة."},
        {"title": "نماذج تسعير بالذكاء الاصطناعي", "en": "Kelly et al. (2025–2026) AIPM", "color": "orange",
         "body": "محوّلات داخل SDF تستفيد من المعلومات عبر الأصول؛ ورقة عمل NBER مع تحديث 2026."},
        {"title": "التقلب والمخاطر", "en": "HAR-RNN, GARCH-NN hybrids", "color": "rose",
         "body": "هجين HAR مع شبكات متكررة للتقلب المحقق وقيمة المخاطرة؛ تفاصيل في صفحة هجين السلاسل الزمنية."},
        {"title": "التحوط العميق", "en": "Buehler et al. (2019) Deep hedging", "color": "sky",
         "body": "تعلم معزز/شبكات لاستراتيجيات تحوط مع تكاليف معاملات وقيود سيولة."},
    ], cols=3)
    ui.refs("Stock & Watson (2002); Giannone, Reichlin & Small (2008); Medeiros et al. (2021) JBES; Goulet Coulombe et al. (2022) JAE; Goulet Coulombe (2024) JAE; "
            "Babii, Ghysels & Striaukas (2022) JBES; Kelly, Malamud & Zhou (2024) JF; Gu, Kelly & Xiu (2020, 2021); Chen, Pelger & Zhu (2024); Diebold & Mariano (1995).")
