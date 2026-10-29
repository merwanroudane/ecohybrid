import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "مكتبات د. مروان رودان الهجينة على PyPI",
    "Dr Merwan Roudane's AI × Econometrics Libraries — Theory, Reference Papers and Usage",
    "تطبيقات Python مفتوحة المصدر لأحدث خوارزميات الدمج بين الذكاء الاصطناعي والقياس الاقتصادي في البانل والسلاسل الزمنية والسببية والتقلب. "
    "كل مكتبة تنفّذ <b>ورقة مرجعية</b> بأمانة (معادلةً بمعادلة مع محاكاة الورقة وبياناتها)، ثم تضيف ما يحتاجه الباحث التطبيقي: "
    "تشخيصات، جداول وأشكال بمستوى المجلات، وأدلة خطوة بخطوة. المصدر: <span class='ltr'>pypi.org/user/merwan.roudane</span>.",
    "fuchsia", "teal",
)

LIBS = [
    # name, family, level, econometric core, AI component, reference, one-line idea
    ("pragdml", "السببية و DML", "L3 + L4", "DML، دالة ريس، تحليل حساسية OVB", "تمثيلات AI (تضمينات) كضوابط", "Aradillas Fernández, Chernozhukov, Cinelli, Klaassen, Newey, Spindler et al. (2026); Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (REStat 2026)",
     "DML حين تكون الضوابط تمثيلات تعلّمها الذكاء الاصطناعي: الفجوة بين الهدف الحقيقي والمقدّر = حاصل ضرب خطأي التمثيل."),
    ("ardldml", "السببية و DML", "L3", "اختبار حدود ARDL للتكامل المشترك", "تعامد بمتعلمات + Wild bootstrap نظامي مقيّد", "DML-Bounds — امتداد لـ Pesaran, Shin & Smith (2001)",
     "اختبار حدود ARDL مع عشرات الضوابط المستمرة: إزالة أثرها بالتعامد، وقيم حرجة بالـ bootstrap لا بالجداول."),
    ("tsdml", "السببية و DML", "L3", "النموذج شبه الخطي، الإسقاطات المحلية، HAC", "DML للسلاسل: توفيق متقاطع عكسي وضبط «منطقة غولديلوكس»", "Ciganovic, D'Amario & Tancioni (2026), The Econometrics Journal — arXiv 2603.10999",
     "DML للسلاسل الزمنية دون تمزيق البنية الزمنية: تعلّم على كتل معكوسة زمنيًا بلا كتل فاصلة محذوفة."),
    ("oracle-varx", "السببية و DML", "L3", "VARX، التحكم في معدل الاكتشاف الزائف (FDR)", "DML متعامد + اختيار فجوات تكيفي", "Tan, Zanoria, Chen, Lyu & Cucuringu (2026) — arXiv 2610.05618",
     "اكتشاف «من يسبب من، وبأي فجوة» في سلاسل مُربَكة، مع معامل وخطأ معياري وضمان FDR لكل حافة."),
    ("dynpanelai", "البانل + ML", "L3", "Difference GMM، تصحيحات التحيز، Anderson–Hsiao", "AB-LASSO، DML بطيات زمنية، لاسو متعامد، انكماش، AC-GATE العصبي", "Arellano & Bond (1991); Chernozhukov, Fernández-Val, Huang & Wang (2024); Semenova et al. (2023); Kock & Tang (2019); Sneller (2026); Xu (2026)",
     "ثمانية منهجيات للبانل الديناميكي عالي الأبعاد خلف واجهة واحدة — يطابق Difference GMM نتائج xtabond2 بفارق أقل من 0.2%."),
    ("mllpanel", "البانل + ML", "L3", "آثار ثابتة + اعتماد مقطعي (عوامل مشتركة)", "أي متعلم ML للدالة المجهولة + DML بطيات على الزمن", "Yang, Long & Cai (2024) — Machine Learning Based Panel Data Models; Mundlak (1978); Pesaran (2006)",
     "دالة مجهولة + آثار فردية مرتبطة + عوامل مشتركة معًا، دون متغير بديل أو أداة: المتوسطات الزمنية والمقطعية تمتص المتغيرات الكامنة."),
    ("deeppanel", "البانل + ML", "L2 + L4", "بانل مُجمَّع، اختبار قابلية التجميع، PVAR", "شبكات عميقة مجمّعة + LLM للتنبؤ الإقليمي بالتضخم + تنبؤ مطابق", "Chronopoulos, Chrysikou, Kapetanios, Mitchell & Raftapostolos (2023; Econometric Reviews 2026); Gao, Sun, Wang, Liu & Hsiao (2026)",
     "تقدير البانل بشبكات عميقة مع مشتقات جزئية واختبار تجميع لا خطي، ونموذج LDPM يدمج مخرجات نماذج اللغة."),
    ("mlpaneldata", "البانل + ML", "L2", "Pooled، FE، RE، Mundlak، فروق أولى", "شبكات قابلة للتفسير بمرشحات التغير المستمر + شبكة بانل عميقة + هجين خطي–عصبي", "Yang, Zheng & E (2020) arXiv 2010.05311; Chronopoulos et al. (2023) arXiv 2305.19921",
     "مقدّر هجين متكامل: نموذج بانل خطي ثم مكوّن عصبي على البواقي، مع بطارية كاملة من الاختبارات القبلية والبعدية."),
    ("panelxai", "البانل + ML", "L2 + التفسير", "بانل ديناميكي بفجوات وآثار ثابتة و CCE", "SHAP مهيكل (متغير × فجوة × وحدة × زمن)، انجراف التفسير بين الأنظمة", "منهجية المؤلف — تبني على SHAP و CCE (Pesaran 2006)",
     "تفسير AI يحترم بنية البانل الاقتصادية: فصل مساهمة الديناميكيات الخاصة عن العوامل المشتركة، وانقلاب إشارات الأثر بين الأنظمة."),
    ("drgct", "سببية غرانجر", "L3", "اختبار سببية غرانجر", "شبكات عميقة + صيغة مزدوجة المتانة", "Hui, Liu & Song (2025) — arXiv 2509.15798",
     "اختبار غرانجر غير خطي مزدوج المتانة بالتعلم العميق، مع كل تصاميم المحاكاة وتطبيق السعر–الحجم في ثلاثة أسواق."),
    ("shapneurgc", "سببية غرانجر", "L3", "سببية غرانجر متعددة المتغيرات", "قيم شابلي المعلوماتية كتنظيم لشبكة غرانجر العصبية", "Yang, Zhu, Tian, Gao & Li (2026), ICML — PMLR 306",
     "اكتشاف سببي زمني بإسناد عالمي للميزات؛ تطبيق على انتقال السياسة النقدية الأمريكية (10 متغيرات، 1960–2026)."),
    ("castor-causal", "سببية غرانجر", "L3", "أنظمة متعددة وانكسارات", "تعلم بنية سببية زمنية لكل نظام بشكل مشترك", "Rahmani & Frossard (2025), AISTATS — PMLR 258",
     "يكتشف عدد الأنظمة وحدودها ورسمًا سببيًا (لحظيًا ومتأخرًا) لكل نظام دون معرفة مسبقة."),
    ("samovar-var", "VAR والسلاسل العصبية", "L2 + L5", "VAR ديناميكي، استجابات نبضية، Diebold–Mariano", "محوّل بانتباه خطي يكافئ نموذج VAR", "Lu & Yang (2025), ICML — arXiv 2502.07244",
     "محوّل خطي هو نفسه نموذج VAR: تنبؤ متعدد المتغيرات مع استرجاع مصفوفات الفجوات والاستجابات النبضية من الشبكة المدرّبة."),
    ("gpvar", "VAR والسلاسل العصبية", "L2", "VAR بتقلب عشوائي، GIRF", "عمليات غاوسية لدوال الفجوات", "Hauzenberger, Huber, Marcellino & Petz (2022), JBES",
     "VAR لا معلمي بعمليات غاوسية يلتقط عدم تماثل صدمات عدم اليقين حسب الإشارة والحجم والزمن (بيانات FRED-QD ومؤشر JLN)."),
    ("tvwnn", "VAR والسلاسل العصبية", "L2", "معاملات متغيرة زمنيًا، مرشح كالمان، EM", "شبكة أمامية تتعلم أساسًا منخفض الأبعاد", "Rudd, Bondell & Silver (2026), Journal of Forecasting 45(1)",
     "شبكة عصبية بأوزان تتبع مسيرة عشوائية: لا خطية الشبكة + قدرة نماذج TVP على تتبع التغير البنيوي."),
    ("hybridmetrics", "VAR والسلاسل العصبية", "L2", "ARDL، MIDAS، ARDL-MIDAS", "ARDL-MIDAS-Transformer، ARDL-MIDAS-DNN، DeepTVAR، DeepVARwT، غرانجر العصبي", "Chalkiadakis, Peters & Ames (2023), Digital Finance; Zhang (2003)",
     "مكتبة موحّدة للهجين القياسي–العميق مع بطارية اختبارات قبلية وبعدية وتشخيصات الاستقرار."),
    ("bartardl", "VAR والسلاسل العصبية", "L2 + L3", "ARDL بفجوات موزعة", "أشجار الانحدار البايزية الجمعية BART للمتوسط الشرطي", "Mahdavi, Ehsani, Ahelegbey & Mohammadpour (2024), Applied Mathematics 15",
     "ARDL لا معلمي يلتقط التفاعلات واللاخطية، مع مقارنة بلاسو والشبكة المرنة والشبكات البايزية."),
    ("garchai", "التقلب والمخاطر", "L2 + L5", "GARCH وعائلته، VaR والاختبار الخلفي", "نماذج تقلب عصبية قابلة للتفاضل ومقيّدة قياسيًا", "24 ورقة GARCH-AI منشورة (مع سجل مصدر لكل نموذج)",
     "24 نموذج GARCH-AI في مكتبة PyTorch واحدة، مطابقة لحزمة arch بدقة 4.4e-15، وتصرّح حين لا يتفوق النموذج على خط أساسه."),
    ("hybridecon", "التقلب والمخاطر", "L2", "GARCH، GARCH-MIDAS، HAR-RV، BEKK", "هجين التعلم العميق للتقلب والمخاطر والتنبؤ الكلي", "Engle, Ghysels & Sohn (2013); Corsi (2009); Engle & Kroner (1995) وأدبيات الهجين 2023–2026",
     "واجهة واحدة لكل نماذج الهجين المسحية مع التشخيص والاختيار والتقييم: fit → forecast → diagnose → report."),
    ("harlstmgarch", "التقلب والمخاطر", "L2", "HAR (Corsi 2009) + GARCH", "LSTM مضبوط بالتحسين البايزي + مجموعة ثقة النماذج", "Ben Romdhane & Boubaker (2026), JRFM 19 — doi:10.3390/jrfm19010077",
     "إطار هجين ثلاثي المراحل للتقلب المحقق في أسواق الطاقة (يذكر الملف التعريفي Belhimer Hocine مؤلفًا للحزمة)."),
    ("peltwtcn", "التقلب والمخاطر", "L2", "كشف الانكسارات: PELT، ICSS، Bai–Perron", "تنقية موجية + تعلم عميق لتنبؤ سعر الكربون", "Ren et al. (2025) — arXiv 2511.04988",
     "PELT ← موجات ← تعلم عميق على 6113 مشاهدة يومية حقيقية لسوق الكربون الأوروبي، مع تدقيق تكرار الورقة."),
    ("wkrls", "النواة والموجات والكمّيات", "L2", "MODWT (Percival & Walden 2000)", "KRLS (Hainmueller & Hazlett 2014)", "تركيب المؤلف: MODWT × KRLS",
     "آثار حدية لا خطية لكل نطاق ترددي × كل مشاهدة: كيف يتغير أثر المتغير بين المدى القصير والمتوسط والطويل."),
    ("qqkrls", "النواة والموجات والكمّيات", "L2", "الانحدار الكمّي على الكمّي (Sim & Zhou 2015)", "KRLS (Hainmueller & Hazlett 2014)", "Adebayo, Ozkan & Eweade (2024), J. Cleaner Production 440",
     "QQKRLS: علاقات لا خطية عبر كمّيات المتغيرين مع أشكال حرارية بأسلوب MATLAB (منشور أيضًا على CRAN)."),
    ("nonparmtvcsq", "النواة والموجات والكمّيات", "مكمّل", "سببية في الكمّيات، Wald المتدحرج", "تقدير لا معلمي بالنواة", "Balcilar et al. (2018); Shi, Phillips & Hurn; Otero & Baum (tvgc)",
     "سببية لا معلمية في الكمّيات (متوسط/تباين)، متغيرة زمنيًا، وسببية غرانجر المتغيرة زمنيًا."),
]
df = pd.DataFrame(LIBS, columns=["المكتبة", "العائلة", "المستوى", "المكوّن القياسي", "المكوّن الذكي", "الورقة المرجعية", "الفكرة"])
FAM_COLOR = {"السببية و DML": "violet", "البانل + ML": "emerald", "سببية غرانجر": "rose", "VAR والسلاسل العصبية": "sky",
             "التقلب والمخاطر": "orange", "النواة والموجات والكمّيات": "teal"}

with st.container(border=True):
    st.markdown("**المختبر التطبيقي:** شغّل هذه المكتبات مباشرة على بياناتك — ارفع الملف، اختر المتغيرات والخيارات، ونفّذ.")
    lc = st.columns(7)
    for col, (f, lab, ic) in zip(lc, [("lab_data", "مدير البيانات", "upload_file"), ("lab_causal", "DML والسببية", "hub"),
                                       ("lab_panel", "البانل", "grid_view"), ("lab_granger", "غرانجر", "share"),
                                       ("lab_var", "VAR · ARDL", "ssid_chart"), ("lab_vol", "التقلب", "candlestick_chart"),
                                       ("lab_kernel", "النواة والكمّيات", "blur_on")]):
        ui.link(f"app_pages/{f}.py", lab, f":material/{ic}:", col)

tabs = st.tabs([
    ":material/hub: نظرة عامة",
    ":material/functions: السببية و DML",
    ":material/grid_view: البانل + ML",
    ":material/compare_arrows: سببية غرانجر واكتشاف السببية",
    ":material/ssid_chart: VAR والسلاسل العصبية",
    ":material/show_chart: التقلب والمخاطر",
    ":material/waves: النواة والموجات والكمّيات",
    ":material/travel_explore: اختر مكتبتك",
    ":material/table_view: الجدول الشامل",
    ":material/library_books: المكتبات المكمّلة (99)",
])


def lib_cards(family: str) -> None:
    sub = df[df["العائلة"] == family]
    ui.cards([{"title": r["المكتبة"], "en": r["الورقة المرجعية"], "color": FAM_COLOR[family],
               "body": f"{r['الفكرة']}<br><b>القياسي:</b> {r['المكوّن القياسي']}<br><b>الذكي:</b> {r['المكوّن الذكي']}<br>"
                       f"<span class='ltr' style='font-size:.8rem;color:#64748b'>pip install {r['المكتبة']} · Level {r['المستوى']}</span>"}
              for _, r in sub.iterrows()], cols=2)


with tabs[0]:
    ui.section("خريطة المكتبات حسب العائلة", "Map of the libraries", "fuchsia")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("مشاريع على PyPI", "99")
    c2.metric("مكتبات AI × قياس اقتصادي", len(df))
    c3.metric("عائلات منهجية", df["العائلة"].nunique())
    c4.metric("أحدث إصدار", "أكتوبر 2026")
    st.mermaid_chart(
        """
mindmap
  root((Roudane hybrid libraries))
    Causal and DML
      pragdml
      ardldml
      tsdml
      oracle-varx
    Panel + ML
      dynpanelai
      mllpanel
      deeppanel
      mlpaneldata
      panelxai
    Granger and discovery
      drgct
      shapneurgc
      castor-causal
    VAR and neural TS
      samovar-var
      gpvar
      tvwnn
      hybridmetrics
      bartardl
    Volatility and risk
      garchai
      hybridecon
      harlstmgarch
      peltwtcn
    Kernel wavelet quantile
      wkrls
      qqkrls
      nonparmtvcsq
"""
    )
    ui.note("**المبدأ المشترك بين المكتبات**: (1) تنفيذ أمين للورقة المرجعية — معادلةً بمعادلة، مع إعادة إنتاج تصاميم المحاكاة وتطبيقاتها على **بيانات حقيقية مضمّنة**؛ "
            "(2) توثيق صريح لكل انحراف عن الورقة ولأخطائها المطبعية؛ (3) **صدق منهجي**: المكتبات تحذّر حين تخرج عن افتراضاتها "
            "(مثل dynpanelai حين يكون √N/T كبيرًا، و garchai حين لا يتفوق النموذج على خط أساسه)؛ (4) جداول وأشكال بمستوى المجلات وأدلة للباحث المبتدئ.",
            "فلسفة البناء", "fuchsia", "🧭")
    cnt = df.groupby("العائلة").size()
    fig = go.Figure(go.Bar(x=cnt.index, y=cnt.values, marker_color=[ui.COLORS[FAM_COLOR[f]] for f in cnt.index], text=cnt.values, textposition="outside"))
    ui.style_fig(fig, "عدد المكتبات في كل عائلة", height=360, legend_h=False)
    ui.show(fig)

with tabs[1]:
    ui.section("السببية والتعلم المزدوج: DML في البيانات الحديثة والسلاسل الزمنية", "Causal inference & DML", "violet")
    lib_cards("السببية و DML")
    ui.section("الأفكار النظرية المفتاحية", "Key theoretical ideas", "indigo")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**pragdml — الفجوة حاصل ضرب لا مجموع**")
        st.latex(r"\theta^*-\theta_\lambda=\mathbb E\big[(g^*-g_\lambda)(\alpha^*-\alpha_\lambda)\big]")
        st.markdown("ضغط المتغير الغني إلى تضمين يغيّر **الهدف السكاني** نفسه؛ يكفي أن يحفظ التمثيل أحد الكائنين (الانحدار أو وزن التوازن) ليبقى الهدف قريبًا. "
                    "وعند فقدان معلومات جوهري: **منطقة تعرّف** مع استدلال بجذر n لحدودها و**قيمة متانة (Robustness value)**؛ على بيانات 401(k): الأثر ≈ 7,855$ وقيمة المتانة 4.2%.")
        st.markdown("**tsdml — التوفيق المتقاطع العكسي**")
        st.latex(r"y_t=\theta_0 d_t+g_0(X_t)+\varepsilon_t,\qquad d_t=m_0(X_t)+\xi_t,\qquad \mathbb E[\xi_t\mid X_t,\mathcal F_{aux,k}]=0")
        st.markdown("قابلية الانعكاس الزمني للعمليات المستقرة تسمح بالتعلم على كتل «مستقبلية» معكوسة دون حذف كتل فاصلة (67% من العينة عند K=6 مقابل 56% في NLO)، "
                    "واختيار العقوبة في منطقة مستقرة من منحنى خطأ التحقق يخفض التحيز في العينات الصغيرة بنحو 24%.")
    with c2:
        st.markdown("**ardldml — العدد الفعلي للمتكاملات**")
        st.latex(r"\tilde k=k-r,\qquad \tilde k=k\Rightarrow I(1)\text{ bound},\quad 0<\tilde k<k\Rightarrow\text{inside},\quad \tilde k=0\Rightarrow I(0)\text{ bound}")
        st.markdown("إزالة أثر ضوابط **متكاملة** قد تمتص الاتجاه العشوائي المختبَر؛ لذلك تُحسب القيم الحرجة بـ **Wild bootstrap نظامي مقيّد** يحفظ ارتباط المعادلتين، "
                    "مع تشخيص «امتصاص الاتجاه» (trend_absorption) قبل تفسير عدم الرفض.")
        st.markdown("**oracle-varx — حواف سببية بضمان FDR**")
        st.markdown("تقدير متعامد بالـ DML لكل حافة $x_i(t-\\ell)\\to x_j(t)$ في سلاسل مُربَكة، مع اختيار فجوات تكيفي وتحكم في معدل الاكتشاف الزائف — "
                    "يجيب: من يسبب من؟ بأي فجوة؟ ومتى؟")
    st.code("""pip install pragdml ardldml tsdml oracle-varx

from tsdml import Calibrator, ReverseCrossFitting
cal = Calibrator(metric="goldilocks_zone", n_blocks=6).calibrate(X, y, d, ...)
model = ReverseCrossFitting(n_blocks=6, block_specific_learners=cal.block_specific_learners_).fit(X, y, d)
model.summary()""", language="python")

with tabs[2]:
    ui.section("بيانات البانل مع تعلم الآلة والتعلم العميق", "Panel data + ML", "emerald")
    lib_cards("البانل + ML")
    ui.section("dynpanelai: أي مقدّر أستخدم؟", "Decision tree from dynpanelai", "teal")
    st.mermaid_chart(
        """
flowchart TB
    A{Is T short, under about 15?} -->|Yes| B[diff_gmm collapse=True or bias corrections]
    A -->|No| C{Many or nonlinear controls?}
    C -->|No| D{Many moment conditions m2/NT large?}
    D -->|Yes| E[ABLasso]
    D -->|No| F[diff_gmm]
    C -->|Yes| G{What do you want?}
    G --> H[One treatment effect: DMLDynamicPanel]
    G --> I[Effects across groups: OrthogonalLasso]
    G --> J[All coefficients uniformly: PanelLasso]
    G --> K[A forecast: PenalizedFE]
    G --> L[Who responds over what horizon: ACGate]
"""
    )
    st.markdown("**mllpanel — الحيلة النظرية** (Yang, Long & Cai, 2024):")
    st.latex(r"y_{it}=f(x_{it})+\alpha_i+\gamma_i'\lambda_t+\varepsilon_{it}\ \Longrightarrow\ y_{it}=f(x_{it})+\beta'z_{it}+e_{it},\quad "
             r"z_{it}=\big(\bar y_t-\bar y,\ \bar x_t-\bar x,\ \bar y_i-\bar y,\ \bar x_i-\bar x\big)")
    st.markdown("المتوسطات المقطعية تمتد على العوامل $\\lambda_t$ (Pesaran 2006) والمتوسطات الزمنية على الآثار الثابتة $\\alpha_i$ (Mundlak 1978)، فيصبح النموذج شبه خطي يُقدَّر بـ DML "
                "مع **طيات على الزمن** لا على المشاهدات — كي لا يتسرب $\\lambda_t$ عبر التقسيم. تطبيق: دالة إنتاج لـ 738 شركة إسبانية دون فرض كوب–دوغلاس.")

with tabs[3]:
    ui.section("سببية غرانجر واكتشاف البنية السببية الزمنية", "Granger causality & temporal causal discovery", "rose")
    lib_cards("سببية غرانجر")
    ui.cards([
        {"title": "oracle-varx", "en": "DML + FDR edges", "color": "violet", "body": "انظر تبويب السببية و DML — اكتشاف الفجوات السببية في سلاسل مُربَكة."},
        {"title": "hybridmetrics", "en": "Neural Granger causality", "color": "sky", "body": "سببية غرانجر العصبية مع اختبارات قبلية وبعدية كاملة."},
        {"title": "nonparmtvcsq", "en": "Quantile & time-varying causality", "color": "teal", "body": "السببية في الكمّيات والسببية المتغيرة زمنيًا كأدوات مكمّلة."},
    ], cols=3)
    ui.note("هذه المكتبات تغطي الطيف الكامل الذي عرضناه في صفحة هجين السلاسل الزمنية: من **الاختبار الإحصائي** (DRGCT المزدوج المتانة) إلى **الاكتشاف المنظّم** "
            "(SRNGC بشابلي) إلى **اكتشاف بنى متعددة الأنظمة** (CASTOR) و**الاستدلال على الحواف** (ORACLE-VARX).", "التكامل بين الأدوات", "rose", "🔗")

with tabs[4]:
    ui.section("VAR والسلاسل الزمنية العصبية", "VAR & neural time series", "sky")
    lib_cards("VAR والسلاسل العصبية")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**tvwnn — شبكة بأوزان متغيرة زمنيًا**")
        st.latex(r"y_t=\boldsymbol\beta_t'\,f_w(u_t)+e_t,\qquad \boldsymbol\beta_t=\boldsymbol\beta_{t-1}+\eta_t")
        st.markdown("الشبكة تتعلم أساسًا منخفض الأبعاد، والأوزان عليه تُقدَّر بمرشح/مملّس كالمان داخل خوارزمية EM تتناوب مع خطوات Adam.")
    with c2:
        st.markdown("**gpvar — VAR بعمليات غاوسية وتقلب عشوائي**")
        st.latex(r"\mathbf y_t=\mathbf f(\mathbf y_{t-1},\dots,\mathbf y_{t-p})+\boldsymbol\varepsilon_t,\qquad f_j\sim\mathcal{GP}\big(0,\ k_{\kappa,\xi}\big),\quad \log\omega_{jt}\sim AR(1)")
        st.markdown("اللاخطية تجعل الاستجابات **معممة** (GIRF) تعتمد على التاريخ والإشارة والحجم — الرسالة التجريبية الأساسية حول صدمات عدم اليقين.")
    st.markdown("**samovar-var**: يثبت Lu & Yang (2025) أن طبقة انتباه خطية ذاتية الانحدار **تكافئ** نموذج VAR ديناميكيًا؛ المكتبة تعيد قراءة مصفوفات الفجوات والاستجابات النبضية "
                "من الشبكة المدرّبة وتقارنها بـ VAR/AR مع اختبارات Diebold–Mariano — مثال مثالي على **L5: النظرية داخل المعمارية**.")

with tabs[5]:
    ui.section("التقلب والمخاطر والتنبؤ بالطاقة", "Volatility, risk and energy forecasting", "orange")
    lib_cards("التقلب والمخاطر")
    st.latex(r"\underbrace{RV_{t+1}^{HAR}}_{\text{Corsi}}\ \to\ \underbrace{\hat e_t\ \text{via LSTM (Bayesian-optimized)}}_{\text{nonlinear}}\ \to\ \underbrace{\sigma^2_t\ \text{via GARCH}}_{\text{clustering}}\ \to\ \text{MCS evaluation}")

with tabs[6]:
    ui.section("النواة والموجات والكمّيات", "Kernel, wavelet and quantile hybrids", "teal")
    lib_cards("النواة والموجات والكمّيات")
    st.latex(r"\text{WKRLS: }\ y=\sum_{j}\ f_j\big(\mathbf x^{(j)}\big)+\varepsilon,\quad \mathbf x^{(j)}=\text{MODWT band } j,\quad \frac{\partial \hat f_j}{\partial x_k^{(j)}}\Big|_{i}\ \text{for every band}\times\text{observation}")
    ui.note("أدوات مكمّلة من المكتبات نفسها: **mqqr** (الكمّي على الكمّي متعدد المتغيرات والسببية)، **wqr** (الانحدار الكمّي الموجي والوساطة)، "
            "**frequencyquantile** (كمّيات متعددة الترددات)، **qvar-model** (VAR كمّي) — وهي الأساس القياسي الذي تُبنى عليه هجائن النواة.", "أدوات مكمّلة", "teal", "🧰")

with tabs[7]:
    ui.section("اختر المكتبة المناسبة لمشكلتك", "Library finder", "indigo")
    data = st.segmented_control("نوع البيانات", ["سلسلة زمنية", "بانل", "مقطعية/تجريبية"], default="سلسلة زمنية", key="lf_data")
    goal = st.segmented_control("الهدف", ["أثر سببي", "سببية/اكتشاف", "تنبؤ", "تقلب/مخاطر", "تكامل مشترك", "آثار لا خطية/ترددية"], default="أثر سببي", key="lf_goal")
    rules = {
        ("سلسلة زمنية", "أثر سببي"): ["tsdml", "oracle-varx"],
        ("سلسلة زمنية", "سببية/اكتشاف"): ["drgct", "shapneurgc", "castor-causal", "oracle-varx", "nonparmtvcsq"],
        ("سلسلة زمنية", "تنبؤ"): ["samovar-var", "gpvar", "tvwnn", "hybridmetrics", "bartardl", "peltwtcn"],
        ("سلسلة زمنية", "تقلب/مخاطر"): ["garchai", "hybridecon", "harlstmgarch"],
        ("سلسلة زمنية", "تكامل مشترك"): ["ardldml", "bartardl"],
        ("سلسلة زمنية", "آثار لا خطية/ترددية"): ["wkrls", "qqkrls", "gpvar"],
        ("بانل", "أثر سببي"): ["dynpanelai", "mllpanel"],
        ("بانل", "سببية/اكتشاف"): ["panelxai", "dynpanelai"],
        ("بانل", "تنبؤ"): ["deeppanel", "mlpaneldata", "dynpanelai"],
        ("بانل", "تقلب/مخاطر"): ["hybridecon"],
        ("بانل", "تكامل مشترك"): ["ardldml"],
        ("بانل", "آثار لا خطية/ترددية"): ["mllpanel", "mlpaneldata", "panelxai"],
        ("مقطعية/تجريبية", "أثر سببي"): ["pragdml"],
        ("مقطعية/تجريبية", "سببية/اكتشاف"): ["pragdml"],
        ("مقطعية/تجريبية", "تنبؤ"): ["pragdml"],
        ("مقطعية/تجريبية", "تقلب/مخاطر"): ["hybridecon"],
        ("مقطعية/تجريبية", "تكامل مشترك"): ["ardldml"],
        ("مقطعية/تجريبية", "آثار لا خطية/ترددية"): ["qqkrls", "wkrls"],
    }
    picks = rules.get((data, goal), [])
    sub = df[df["المكتبة"].isin(picks)]
    if sub.empty:
        st.info("لا توجد مكتبة مخصصة لهذا التركيب بعد — وهي فجوة محتملة لتطوير خوارزمية جديدة.")
    else:
        ui.cards([{"title": r["المكتبة"], "en": r["الورقة المرجعية"], "color": FAM_COLOR[r["العائلة"]],
                   "body": f"{r['الفكرة']}<br><span class='ltr' style='font-size:.8rem;color:#64748b'>pip install {r['المكتبة']}</span>"}
                  for _, r in sub.iterrows()], cols=2)

with tabs[8]:
    ui.section("الجدول الشامل: المكتبة × المكوّن القياسي × المكوّن الذكي × المرجع", "Comprehensive table", "slate")
    st.dataframe(df, hide_index=True, height=720,
                 column_config={"الفكرة": st.column_config.TextColumn(width="large"), "الورقة المرجعية": st.column_config.TextColumn(width="large")})
    st.caption("المستويات L2–L5 وفق سلّم الدمج في صفحة «مستويات الدمج». الروابط والتوثيق الكامل لكل مكتبة على PyPI و GitHub (merwanroudane).")

with tabs[9]:
    ui.section("المكتبات القياسية المكمّلة: الأساس الذي يُبنى عليه الهجين", "The 99-project econometric toolkit", "amber")
    st.markdown("إلى جانب مكتبات الدمج، يضم الملف على PyPI عشرات المكتبات القياسية المتخصصة التي تمثل **المكوّن القياسي** لأي هجين: اختبارات جذر الوحدة غير الخطية، "
                "التكامل المشترك بالانكسارات والدوال الفورييه والعتبات، عائلة NARDL، السببية غير المتماثلة، وتكامل البانل المشترك.")
    fams = pd.DataFrame([
        ("جذور الوحدة غير الخطية والفورييه والكمّية", "funitroot، fourier-gls، fractionaldouble، fwadf، qadf، qfadf، hybridnonlinur، tarur، rals-unitroot، kapetanios-test، narayanpop، boundedtest، nearkpss، urvol، polynomial-unitroot، kmtest", "Becker–Enders–Lee؛ Koenker & Xiao (2004)؛ Li & Zheng (2018)؛ Kapetanios (2005)؛ Narayan & Popp"),
        ("التكامل المشترك بالانكسارات والفورييه والزمن المتغير", "fourier-johansen، fjohansen، fouriercoint، tvcoint، tvrcoint، tvccointreg، segmcoint، selectbreakcoint، vecmbreak، cointhatemij، pmct، cointsmall، pycointbreak، co-eco، robcointeg", "Kurita & Shintani (2025)؛ Bierens & Martins (2010)؛ Hatemi-J؛ Hall, Swamy & Tavlas؛ Hassler & Breitung"),
        ("العتبات والكمّيات واللاخطية", "threshcoint، quantilecoint، qardl، qardlar، qvar-model، mqqr، qqkrls، wqr، frequencyquantile", "Hansen & Seo؛ Xiao (2009)؛ Cho, Kim & Shin (QARDL)؛ Sim & Zhou (2015)"),
        ("عائلة ARDL / NARDL", "qiyasi-bootardl، nardl-fourier، twostep-nardl، mtnardl، wavenardl، bartardl، ardldml", "Pesaran, Shin & Smith (2001)؛ Shin, Yu & Greenwood-Nimmo (2014)؛ McNown et al. bootstrap ARDL"),
        ("السببية", "asycaus، asymcaus، asymcauss، dasycaus، nonparmtvcsq، drgct، shapneurgc", "Hatemi-J (2012)؛ Balcilar et al.؛ Shi–Phillips–Hurn"),
        ("البانل", "multicoint، pycupfm، pydcce، pybayescointur، pywaveletpanel، dynpanelai، mllpanel، deeppanel، mlpaneldata، panelxai", "Pesaran (2006)؛ Bai, Kao & Ng (2009)؛ Chudik & Pesaran"),
        ("الأنظمة المتكاملة من الرتب العليا", "rbfmvar، haldrup، mixvecm، cpr-econometrics", "Chang & Phillips (RBFM-VAR)؛ Haldrup (1994)"),
        ("الأدوات و MCP", "econenv، eviews-mcp، gauss-lang-mcp، python-session-mcp، rpython-bridge، smartprep، visualmetrics، tseffects", "بيئات متعددة المحركات وربط النماذج اللغوية بمحركات القياس"),
    ], columns=["العائلة", "أمثلة من المكتبات", "مراجع تأسيسية"])
    st.table(fams)
    ui.note("هذه البنية تجسّد فكرة الدليل الأساسية: **الهجين الجيد يبدأ بمكوّن قياسي متين** (اختبار جذر وحدة صحيح، تكامل مشترك مع انكسارات، تحديد الأنظمة) "
            "ثم يُسند للذكاء الاصطناعي أصغر مهمة ذات قيمة مضافة.", "الربط بفلسفة الدليل", "amber", "🧩")
    ui.refs("المصدر: الملف العام لـ Dr Merwan Roudane على PyPI (99 مشروعًا حتى 28 أكتوبر 2026) ووصف كل حزمة وملفاتها التعريفية على PyPI و GitHub.")
