import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "الدمج على مستوى المكونات: التفكيك ثم النمذجة ثم إعادة التركيب",
    "Component-level integration — decompose, model each component, recombine (and avoid the two-sided filter trap)",
    "لا نعالج السلسلة ككتلة واحدة: نفصلها إلى مكونات ذات خصائص مختلفة (اتجاه، دورة، موسمية، ضوضاء، أو نطاقات ترددية)، ونختار لكل "
    "مكوّن النموذج الملائم له، ثم نعيد التركيب. هذه هي البنية <b>المتوازية–المتسلسلة</b>. قوتها حقيقية، لكن فيها أشهر فخ تسرب في أدبيات "
    "الهجن: <b>المرشحات ثنائية الجانب تستخدم المستقبل</b>.",
    "cyan", "violet",
)

tabs = st.tabs([
    ":material/lightbulb: الفكرة والبنية",
    ":material/table_chart: طرق التفكيك",
    ":material/model_training: نموذج لكل مكوّن",
    ":material/join_full: إعادة التركيب",
    ":material/warning: فخ المرشحات ثنائية الجانب",
    ":material/science: مختبر: دقة مستحيلة = تسرب",
    ":material/library_books: من مكتبات المؤلف",
    ":material/checklist: قائمة التحقق",
])

# --------------------------------------------------------------------------- 1
with tabs[0]:
    ui.section("لماذا نفكك قبل أن ننمذج؟", "Why decompose", "cyan")
    st.latex(r"y_t=T_t+C_t+S_t+R_t\qquad\text{(جمعي)},\qquad \ln y_t=\ln T_t+\ln C_t+\ln S_t+\ln R_t\qquad\text{(ضربي)}")
    st.markdown(
        "- **الاتجاه** $T_t$ بطيء، شبه حتمي أو تكاملي: نماذج خطية أو انجراف عشوائي تعمل جيدًا وتستقرئ بأمان.\n"
        "- **الدورة** $C_t$ متوسطة التردد وقد تكون لاخطية أو تعتمد على النظام: هنا تفيد ARDL أو الشبكات.\n"
        "- **الموسمية** $S_t$ منتظمة: متغيرات وهمية أو حدود فورييه تكفي غالبًا.\n"
        "- **الضوضاء** $R_t$ قد تكون غير قابلة للتنبؤ (فلا ننمذجها) أو تحمل لاخطية صغيرة.\n\n"
        "نموذج واحد على السلسلة الخام يُجبر على تعلم كل هذه الديناميكيات معًا؛ والشبكات بالذات **ضعيفة في استقراء الاتجاه** خارج مدى التدريب. "
        "وجد Zhang & Qi (2005) أن الشبكات العصبية تتنبأ أفضل بكثير عندما تُزال منها الاتجاهات والموسمية مسبقًا."
    )
    st.mermaid_chart(
        """
flowchart LR
    Y[Original series y_t] --> D{Decompose<br/>causally}
    D --> T[Trend T_t] --> MT[Linear / drift model]
    D --> C[Cycle C_t] --> MC[ARDL / LSTM / GBM]
    D --> S[Seasonal S_t] --> MS[Dummies / Fourier]
    D --> R[Irregular R_t] --> MR[ML or nothing]
    MT --> F[Recombine]
    MC --> F
    MS --> F
    MR --> F
    F --> O[Forecast y_t+h]
"""
    )
    ui.note(
        "**موقعه في التصنيفات:** في تصنيف نقطة الاتصال هو **المستوى 5 (المكونات)**؛ في سلّم L0–L7 هو **L2 (الدمج التنبؤي)**؛ وفي معماريات "
        "Hajirahimi & Khashei (2019) هو البنية **المتوازية–المتسلسلة**: معالجة أولية، ثم نماذج متوازية، ثم دمج.",
        "أين يقع؟", "cyan", "🧭",
    )
    ui.refs("Zhang & Qi (2005) European J. Operational Research 160(2)؛ Hajirahimi & Khashei (2019) Engineering Applications of AI 86؛ Hyndman & Athanasopoulos (2021) Forecasting: Principles and Practice, ch. 3.")

# --------------------------------------------------------------------------- 2
with tabs[1]:
    ui.section("طرق التفكيك: ماذا تفصل؟ وهل هي سببية؟", "Decomposition methods", "violet")
    st.dataframe(pd.DataFrame([
        ("المتوسط المتحرك الكلاسيكي", "اتجاه-دورة + موسمية + باقٍ", "المركزي لا (يستخدم المستقبل)؛ المتأخر نعم", "ARIMA للاتجاه، وهميات للموسمية"),
        ("STL", "اتجاه + موسمية + باقٍ (Loess)", "لا على كامل العينة؛ نعم إذا أُعيد على الماضي فقط", "ARIMA للاتجاه، XGBoost للباقي"),
        ("مرشح Hodrick–Prescott", "اتجاه + دورة", "لا (ثنائي الجانب)؛ النسخة أحادية الجانب موجودة", "دورة الأعمال ثم نموذج للدورة"),
        ("انحدار هاملتون (2018)", "مكوّن متوقع + دورة", "نعم (يستخدم الماضي فقط)", "بديل سببي لـ HP"),
        ("حدود فورييه", "ترددات دورية ناعمة وانكسارات تدريجية", "نعم إذا قُدّرت على التدريب", "ARDL/جذر الوحدة مع فورييه + ML للباقي"),
        ("الموجات (DWT/MODWT)", "نطاقات زمنية–ترددية متعددة", "لا عند الحواف إلا بتفكيك متدحرج أو مرشحات سببية", "نموذج مختلف لكل نطاق"),
        ("EMD / EEMD / CEEMDAN", "دوال النمط الجوهري (IMFs)", "لا (تكيفي على كامل السلسلة)", "LSTM أو SVR لكل IMF"),
        ("VMD", "أنماط بنطاقات ترددية محددة", "لا على كامل العينة", "نموذج لكل نمط"),
    ], columns=["الطريقة", "ماذا تفصل", "سببية؟ (تستخدم الماضي فقط)", "الاستخدام النموذجي"]), hide_index=True)
    st.markdown("**تفكيك الموجات المنفصل غير المعشَّر (MODWT)** يكتب السلسلة كمجموع تفاصيل في مقاييس زمنية مختلفة وتقريب نهائي:")
    st.latex(r"y_t=\sum_{j=1}^{J}D_{j,t}+A_{J,t},\qquad D_j\ \text{تقلبات بمدة تقريبية}\ [2^{j},2^{j+1}]\ \text{فترة}")
    st.markdown("و**EMD** يستخرج تكيفيًا دوال نمط جوهري $c_k(t)$ وباقيًا اتجاهيًا: $y_t=\\sum_kc_k(t)+r(t)$، وCEEMDAN يضيف ضوضاء مُتحكَّمًا بها لتقليل خلط الأنماط.")
    ui.refs("Cleveland et al. (1990) J. Official Statistics 6(1) STL؛ Hodrick & Prescott (1997) JMCB 29(1)؛ Hamilton (2018) REStat 100(5)؛ Percival & Walden (2000) Wavelet Methods for Time Series Analysis؛ "
            "Huang et al. (1998) Proc. R. Soc. A 454 EMD؛ Torres et al. (2011) ICASSP CEEMDAN؛ Dragomiretskiy & Zosso (2014) IEEE TSP VMD.")

# --------------------------------------------------------------------------- 3
with tabs[2]:
    ui.section("مطابقة النموذج مع طبيعة المكوّن", "Matching models to components", "teal")
    ui.cards([
        {"title": "الاتجاه / التقريب منخفض التردد", "color": "violet",
         "body": "نماذج قياسية بسيطة: انجراف، اتجاه خطي محلي، ARIMA. الاستقراء الآمن أهم من المرونة. هنا يبقى التفسير الاقتصادي (النمو طويل الأجل)."},
        {"title": "الدورة / النطاقات المتوسطة", "color": "teal",
         "body": "ARDL مع متغيرات خارجية، أو شبكات وأشجار عند وجود لاخطية أو تبديل أنظمة. أغلب مكسب الهجين يأتي من هنا."},
        {"title": "الموسمية", "color": "amber",
         "body": "متغيرات وهمية أو حدود فورييه؛ نادرًا ما تحتاج تعلمًا آليًا إلا مع موسمية متغيرة أو متعددة (يومية وأسبوعية وسنوية)."},
        {"title": "التفاصيل عالية التردد / الضوضاء", "color": "rose",
         "body": "إما لا شيء (التنبؤ بالصفر)، أو متعلم صغير منظَّم جدًا. نمذجة الضوضاء بشبكة كبيرة هي أسرع طريق للإفراط في التوفيق."},
    ], cols=2)
    ui.note(
        "**ميزة تفسيرية إضافية:** التفكيك الترددي يجيب عن سؤال اقتصادي لا يطرحه النموذج الكلي: «هل أثر سعر النفط على التضخم قصير الأجل أم طويل "
        "الأجل؟». مكتبة wkrls تقدّر أثرًا حديًا لاخطيًا **لكل نطاق ترددي ولكل مشاهدة** بدمج MODWT مع KRLS.",
        "التفكيك كأداة تفسير", "teal", "🔍",
    )

# --------------------------------------------------------------------------- 4
with tabs[3]:
    ui.section("إعادة التركيب ليست خطوة شكلية", "Recombination", "indigo")
    st.latex(r"\hat y_{t+h}=\sum_{k=1}^{K}\hat c^{(k)}_{t+h}\qquad\text{أو}\qquad \hat y_{t+h}=g\big(\hat c^{(1)}_{t+h},\dots,\hat c^{(K)}_{t+h}\big)")
    st.markdown(
        "- **الجمع البسيط** صحيح فقط إذا كان التفكيك جمعيًا ومكتملًا (مجموع المكونات يساوي السلسلة).\n"
        "- **التكديس** يتعلم أوزان المكونات، لكنه يحتاج تنبؤات خارج الطية **سابقة زمنيًا**.\n"
        "- **عدم اليقين:** تباين المجموع يتضمن التغايرات بين أخطاء المكونات: $\\operatorname{Var}(\\sum e_k)=\\sum_k\\operatorname{Var}(e_k)+2\\sum_{k<l}\\operatorname{Cov}(e_k,e_l)$. "
        "تجاهل التغاير يعطي فترات تنبؤ ضيقة زيفًا.\n"
        "- **الاتساق الزمني:** يجب أن تُحسب كل المكونات عند الأصل $t$ نفسه ومن المعلومات نفسها $\\mathcal I_t$."
    )

# --------------------------------------------------------------------------- 5
with tabs[4]:
    ui.section("الفخ: المرشح ثنائي الجانب يرى المستقبل", "Two-sided filters leak the future", "rose")
    st.markdown("المتوسط المتحرك **المركزي** بعرض $2k+1$ يعطي اتجاهًا عند $t$ باستخدام قيم **لاحقة**:")
    st.latex(r"\hat T^{\text{centred}}_t=\frac1{2k+1}\sum_{j=-k}^{k}y_{t+j}\qquad\text{مقابل}\qquad \hat T^{\text{trailing}}_t=\frac1{2k+1}\sum_{j=0}^{2k}y_{t-j}")
    st.markdown(
        "إذا فككنا **كامل السلسلة** ثم قسمناها إلى تدريب واختبار، فإن مكوّنات آخر نقاط التدريب ونقاط الاختبار تحتوي معلومات من $y_{t+1},\\dots,y_{t+k}$. "
        "النموذج الذي يتنبأ بالمكوّنات ثم يجمعها يحصل عمليًا على جزء من الجواب. النتيجة المعروفة في أدبيات «التفكيك–التجميع»: دقة تبدو "
        "أفضل بكثير مما يمكن تحقيقه فعلًا، وقد تنزل **حتى تحت حد الضوضاء نفسه**، ثم تنهار عند التطبيق الحقيقي."
    )
    ui.cards([
        {"title": "المرشحات المعرَّضة", "color": "rose",
         "body": "المتوسط المركزي، HP ثنائي الجانب، STL وEMD وVMD على كامل العينة، الموجات قرب الحافة الأخيرة، والتعيير على كامل السلسلة."},
        {"title": "الحل 1: تفكيك متدحرج", "color": "emerald",
         "body": "عند كل أصل t: فكّك $y_1,\\dots,y_t$ فقط، واستخدم قيمة المكوّنات عند t. مكلف حسابيًا لكنه صادق."},
        {"title": "الحل 2: مرشحات سببية", "color": "teal",
         "body": "متوسطات متأخرة، HP أحادي الجانب، انحدار هاملتون، موجات بمرشحات سببية (à trous). تأخر في الطور مقابل الصدق."},
        {"title": "الحل 3: الاختبار الذاتي", "color": "amber",
         "body": "إذا نزل RMSE خارج العينة تحت انحراف الضوضاء المعروف أو تحت أي حد معقول، اشتبه في التسرب قبل أن تحتفل."},
    ], cols=2)
    ui.refs("Hamilton (2018) «Why you should never use the Hodrick–Prescott filter» REStat 100(5)؛ Percival & Walden (2000) عن آثار الحواف؛ موسوعة آلية الدمج — د. مروان رودان (2026).")

# --------------------------------------------------------------------------- 6
with tabs[5]:
    ui.section("مختبر: التفكيك المركزي مقابل السببي", "Leaky vs causal decomposition", "rose")
    st.markdown(
        "سلسلة = اتجاه بطيء + دورة + ضوضاء بانحراف معروف. نفكك بمتوسط متحرك بعرض $2k+1$، ونتنبأ بالاتجاه بانجراف وبالباقي (دورة + ضوضاء) "
        "بنموذج AR(2) مقدَّر على التدريب، ثم نجمع. المقارنة على آخر 30% خطوة واحدة إلى الأمام."
    )
    c1, c2, c3 = st.columns(3)
    k = c1.slider("نصف عرض المرشح k", 2, 12, 2, key="cmp_k")
    sd = c2.slider("انحراف الضوضاء الحقيقي", 0.2, 1.5, 0.6, 0.1, key="cmp_sd")
    amp = c3.slider("سعة الدورة", 0.0, 3.0, 1.5, 0.25, key="cmp_amp")

    def run(k, sd, amp, T=600, seed=9):
        rng = np.random.default_rng(seed)
        t = np.arange(T)
        y = 0.02 * t + 2 * np.sin(2 * np.pi * t / 200) + amp * np.sin(2 * np.pi * t / 32) + rng.normal(0, sd, T)
        cut = int(0.7 * T)
        w = 2 * k + 1
        pad = np.r_[np.full(k, np.nan), y, np.full(k, np.nan)]
        centred = np.array([np.nanmean(pad[i:i + w]) for i in range(T)])          # uses y[t-k..t+k] — leaks
        trailing = np.array([y[max(0, i - w + 1):i + 1].mean() for i in range(T)])  # uses y[t-2k..t] — causal

        def forecast(trend):
            rem = y - trend
            Z = np.column_stack([np.ones(T - 2), rem[1:-1], rem[:-2]])
            b = np.linalg.lstsq(Z[:cut - 2], rem[2:cut], rcond=None)[0]
            # forecast y[t+1] made at t: trend drift + AR(2) on remainder
            tr_hat = trend[1:-1] + (trend[1:-1] - trend[:-2])
            rm_hat = Z @ b
            return tr_hat + rm_hat                                               # predictions for y[2:]

        tgt = y[2:]
        oos = slice(cut - 2, None)
        Zd = np.column_stack([np.ones(T - 2), y[1:-1], y[:-2]])
        bd = np.linalg.lstsq(Zd[:cut - 2], tgt[:cut - 2], rcond=None)[0]
        out = {}
        for name, pred in {"تفكيك مركزي (ثنائي الجانب، على كامل العينة)": forecast(centred),
                           "تفكيك سببي (متأخر)": forecast(trailing),
                           "AR(2) مباشرة دون تفكيك": Zd @ bd}.items():
            out[name] = float(np.sqrt(np.mean((tgt[oos] - pred[oos]) ** 2)))
        return out, t, y, centred, trailing, cut

    out, t, y, centred, trailing, cut = run(k, sd, amp)
    tab = pd.DataFrame(list(out.items()), columns=["الطريقة", "RMSE خارج العينة"])
    tab["نسبة إلى انحراف الضوضاء"] = tab["RMSE خارج العينة"] / sd
    st.dataframe(tab.round(3), hide_index=True)
    leak = out["تفكيك مركزي (ثنائي الجانب، على كامل العينة)"]
    if leak < sd:
        st.error(f"التفكيك المركزي أعطى RMSE = {leak:.3f}، أي **أقل من انحراف الضوضاء الحقيقي {sd:.2f}**. هذا مستحيل لأي متنبئ صادق: إنه تسرب من المستقبل.",
                 icon=":material/report:")
    fig = go.Figure()
    s = slice(cut - 80, cut + 60)
    fig.add_trace(go.Scatter(x=t[s], y=y[s], mode="markers", name="y", marker=dict(size=4, color="#cbd5e1")))
    fig.add_trace(go.Scatter(x=t[s], y=centred[s], name="اتجاه مركزي (يرى المستقبل)", line=dict(color="#e11d48", width=2.5)))
    fig.add_trace(go.Scatter(x=t[s], y=trailing[s], name="اتجاه متأخر (سببي)", line=dict(color="#0d9488", width=2.5)))
    fig.add_vline(x=cut, line_dash="dash", line_color="#475569", annotation_text="بداية الاختبار")
    ui.style_fig(fig, "المرشح المركزي يسبق البيانات؛ المتأخر يتبعها بتأخر طور", height=430)
    ui.show(fig)
    ui.note(
        "**القراءة:** المرشح المركزي يبدو أنعم وأدق لأنه «يعرف» ما سيأتي، فيحصل تنبؤه المجمَّع على أفضلية لا يستحقها؛ ومع مرشح ضيق (k صغير) "
        "ينزل خطؤه **تحت حد الضوضاء** وهو ما لا يستطيعه أي متنبئ صادق. المرشح السببي يتأخر قليلًا في الطور لكنه صادق، ونتيجته قريبة من AR مباشر: "
        "هنا التفكيك الصادق لا يضيف كثيرًا، وهذه نتيجة صحيحة تستحق أن تُذكر في الورقة بدل إخفائها.",
        "ماذا نتعلم؟", "rose", "🧪",
    )

# --------------------------------------------------------------------------- 7
with tabs[6]:
    ui.section("التفكيك في مكتبات د. مروان رودان", "Decomposition in the author's libraries", "violet")
    ui.cards([
        {"title": "wkrls", "en": "MODWT × KRLS", "color": "teal",
         "body": "تفكيك موجي MODWT للمتغيرات ثم KRLS لكل نطاق: أثر حدي لاخطي لكل نطاق ترددي ولكل مشاهدة (قصير/متوسط/طويل الأجل). مثال على التفكيك "
                 "كأداة <b>تفسير</b> لا تنبؤ فقط."},
        {"title": "peltwtcn", "en": "PELT + wavelet denoising + TCN/LSTM", "color": "orange",
         "body": "كشف الانكسارات بـ PELT، تنقية موجية، ثم شبكة عميقة. تفرّق المكتبة صراحة بين وضع <b>paper</b> (تنقية على كامل العينة لمطابقة الورقة) "
                 "ووضع <b>causal</b> (تنقية متدحرجة بلا معلومات مستقبلية)، ومنذ الإصدار 1.1.0 ينمذج الوضع السببي الفروق تلقائيًا."},
        {"title": "hybridecon", "en": "Hybrid volatility and macro", "color": "rose",
         "body": "يجمع نماذج قياسية للتقلب مع متعلمات مرنة؛ التفكيك إلى مكون التقلب ومكون المتوسط مثال على الدمج بالمكونات."},
        {"title": "المختبر التطبيقي", "en": "Run them on your data", "color": "violet",
         "body": "شغّل wkrls وpeltwtcn من صفحتي «مختبر النواة والموجات» و«مختبر التقلب» على بياناتك، واختر الوضع السببي عند التقييم خارج العينة."},
    ], cols=2)
    ui.link("app_pages/lab_kernel.py", "مختبر النواة والموجات والكمّيات (wkrls)", ":material/blur_on:")
    ui.link("app_pages/lab_vol.py", "مختبر التقلب والتنبؤ الهجين (peltwtcn)", ":material/candlestick_chart:")

# --------------------------------------------------------------------------- 8
with tabs[7]:
    ui.section("قائمة تحقق للدمج على مستوى المكونات", "Checklist", "slate")
    st.markdown(
        "- [ ] بررتُ التفكيك: للمكونات ديناميكيات مختلفة فعلًا (طيف، ACF لكل مكوّن).\n"
        "- [ ] التفكيك **سببي** أو **متدحرج** عند كل أصل تنبؤ؛ لا تفكيك لكامل العينة قبل التقسيم.\n"
        "- [ ] نموذج كل مكوّن مختار بحسب طبيعته، والضوضاء لا تُنمذج بشبكة كبيرة.\n"
        "- [ ] إعادة التركيب متسقة (مجموع المكونات = السلسلة) وفترات التنبؤ تأخذ التغايرات في الحسبان.\n"
        "- [ ] قارنتُ بنموذج مباشر دون تفكيك وبنموذج قياسي مرجعي.\n"
        "- [ ] فحصتُ أن RMSE خارج العينة ليس أقل من حد معقول للضوضاء (اختبار ذاتي للتسرب)."
    )
