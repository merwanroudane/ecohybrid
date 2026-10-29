import pandas as pd
import streamlit as st

from core import ui

ui.hero(
    "الأدوات والبرمجيات",
    "Software Ecosystem for AI × Econometrics — Python, R, Stata, Julia",
    "خريطة الحزم البرمجية لكل فرع من فروع الدمج، مع ملاحظات الاستخدام. ابدأ دائمًا بتطبيق المؤلفين الأصليين للأوراق الحديثة، "
    "ثم انقله إلى بيئتك البحثية مع اختبارات مطابقة.",
    "teal", "violet",
)

tabs = st.tabs([":material/code: Python", ":material/data_object: R", ":material/terminal: Stata", ":material/apps: حسب المهمة", ":material/dataset: بيانات للتدريب"])

with tabs[0]:
    py = pd.DataFrame([
        ("scikit-learn", "التعلم الآلي العام", "Lasso، الغابات، التعزيز، SVM، التحقق المتقاطع، Pipelines"),
        ("statsmodels / linearmodels", "القياس الاقتصادي", "OLS، IV، بانل، VAR، ARIMA، فضاء الحالة — خطوط أساس قياسية"),
        ("DoubleML", "DML", "PLR، IRM، PLIV، IIVM، DiD، SSM مع أي متعلم (Bach et al., JMLR 2022)"),
        ("EconML (Microsoft)", "الأثر غير المتجانس", "Causal forests DML، DR-learner، Orthogonal IV، Deep IV، تعلم السياسات"),
        ("CausalML (Uber)", "Uplift / meta-learners", "S/T/X/R-learners، أشجار الرفع"),
        ("DoWhy / PyWhy", "سير العمل السببي", "النمذجة بالـ DAG، التعرّف، التقدير، التفنيد"),
        ("pyfixest", "آثار ثابتة سريعة", "انحدارات بآثار ثابتة متعددة وأخطاء عنقودية"),
        ("PyTorch / JAX", "التعلم العميق والتفاضل الآلي", "شبكات موجّهة بالنظرية، DEQN، PINN، قيود في الخسارة"),
        ("tigramite", "اكتشاف السببية الزمنية", "PCMCI و PCMCI+"),
        ("Neural-GC", "غرانجر العصبي", "cMLP و cLSTM (Tank et al.)"),
        ("PySR", "الانحدار الرمزي", "بحث تطوري عن معادلات صريحة"),
        ("ValidMLInference", "المتغيرات المولّدة", "تصحيح الاستدلال مع متغيرات AI (Christensen & Hansen)"),
        ("ppi_py", "PPI", "الاستدلال المدعوم بالتنبؤ (Angelopoulos et al.)"),
        ("transformers (Hugging Face)", "نماذج اللغة", "BERT، FinBERT، تضمينات، نماذج مفتوحة"),
        ("chronos / timesfm / BISTRO", "نماذج أساس للسلاسل", "تنبؤ بلا تدريب — مع تقييم آني"),
        ("TabPFN", "نموذج أساس جدولي", "تنبؤ سريع للبيانات الصغيرة والمتوسطة"),
        ("gpboost / merf", "ML بآثار مختلطة", "تعزيز/غابات مع آثار عشوائية"),
        ("PyWavelets", "التفكيك الموجي", "DWT/MODWT لبناء الهجين الموجي"),
    ], columns=["الحزمة", "الفرع", "الاستخدام"])
    st.dataframe(py, hide_index=True, height=650)

with tabs[1]:
    r = pd.DataFrame([
        ("grf", "الغابات العشوائية المعممة", "causal_forest، instrumental_forest، quantile_forest، policy evaluation"),
        ("DoubleML (R)", "DML", "نسخة R عبر mlr3"),
        ("hdm", "لاسو عالي الأبعاد", "rlasso، الاختيار المزدوج، IV عالي الأبعاد"),
        ("policytree", "تعلم السياسات", "أشجار قرار مثلى لتعيين المعالجة"),
        ("causalweight", "الوساطة والانتقاء", "DML للوساطة، الانتقاء، المعالجات الديناميكية"),
        ("xtdml", "DML للبانل", "آثار ثابتة (Clarke & Polselli)"),
        ("specs", "SPECS", "تكامل مشترك منظّم (Smeekes & Wijler)"),
        ("midasml", "sg-LASSO MIDAS", "تنبؤ آني بتواترات مختلطة"),
        ("BigVAR", "Lasso-VAR", "VAR كبير بتنظيم هرمي"),
        ("did / DRDID / synthdid", "DiD الحديث", "Callaway–Sant'Anna، DR-DiD، Synthetic DiD"),
        ("quantregForest", "غابات كمّية", "Meinshausen (2006)"),
        ("bartCause / bcf", "BART السببي", "BART و Bayesian causal forests"),
        ("ranger / xgboost / glmnet", "متعلمات", "غابات سريعة، تعزيز، شبكة مرنة"),
        ("qqkrls / QuantileOnQuantile", "الكمّي على الكمّي", "QQR مع KRLS (د. مروان رودان)"),
    ], columns=["الحزمة", "الفرع", "الاستخدام"])
    st.dataframe(r, hide_index=True, height=540)

with tabs[2]:
    s = pd.DataFrame([
        ("lasso / dsregress / poregress / xporegress", "مدمج منذ Stata 16", "لاسو، الاختيار المزدوج، إزالة الأثر، الإزالة بالتوفيق المتقاطع"),
        ("lassopack / pdslasso", "Ahrens, Hansen & Schaffer", "لاسو منظّم، PDS، IV عالي الأبعاد"),
        ("ddml", "Ahrens, Hansen, Schaffer & Wiemann", "DML مع التكديس (pystacked)"),
        ("pystacked", "تكديس متعلمات Python", "دمج scikit-learn داخل Stata"),
        ("rforest / boost", "مجتمع Stata", "غابات وتعزيز"),
        ("csdid / drdid", "DiD الحديث", "Callaway–Sant'Anna و DR-DiD"),
        ("python: (Stata 16+)", "تكامل Python", "تشغيل PyTorch/EconML من داخل do-files"),
    ], columns=["الأمر/الحزمة", "المصدر", "الاستخدام"])
    st.dataframe(s, hide_index=True)

with tabs[3]:
    task = pd.DataFrame([
        ("ATE بضوابط كثيرة", "DoubleML، EconML", "DoubleML، hdm", "ddml، xporegress"),
        ("CATE وتعلم السياسات", "EconML، CausalML", "grf، policytree", "—"),
        ("DiD/SC مع ML", "DoubleML (DID)، pysyncon", "did، DRDID، synthdid", "csdid، drdid"),
        ("IV عالي الأبعاد / Deep IV", "EconML", "hdm", "pdslasso، ivlasso"),
        ("تنبؤ كلي بالـ ML", "scikit-learn، sktime، darts", "glmnet، ranger، BigVAR", "—"),
        ("تكامل مشترك منظّم", "—", "specs", "—"),
        ("سببية غرانجر غير خطية", "Neural-GC، tigramite", "—", "—"),
        ("النص كبيانات", "transformers، gensim، scikit-learn", "quanteda، text2vec", "—"),
        ("تصحيح المتغيرات المولّدة", "ValidMLInference، ppi_py", "—", "—"),
        ("حل النماذج الهيكلية", "PyTorch، JAX", "—", "—"),
        ("التعلم المعزز", "gymnasium، stable-baselines3", "ReinforcementLearning", "—"),
    ], columns=["المهمة", "Python", "R", "Stata"])
    st.table(task)
    ui.note("**خطأ شائع في بناء مكتبة جديدة**: توفر كود بحثي للتنبؤ لا يكفي لوصفه أداة استدلال قياسي. قبل النشر: مواصفات المعلمة الهدف، شروط التعرّف، "
            "الخوارزمية، تكرار النتائج، مونت كارلو للحجم والقوة والتغطية، حالات فشل موثقة، واختبارات وأمثلة قابلة للتكرار.", "تنبيه", "amber", "⚠️")

with tabs[4]:
    ui.cards([
        {"title": "FRED-MD / FRED-QD", "en": "McCracken & Ng", "color": "teal", "body": "مئات السلاسل الكلية الأمريكية الشهرية/الفصلية — معيار التنبؤ الكلي بالـ ML."},
        {"title": "ALFRED", "en": "Real-time vintages", "color": "violet", "body": "الإصدارات التاريخية للبيانات لتقييم آني صادق."},
        {"title": "LaLonde / NSW", "en": "Job training", "color": "orange", "body": "البيانات الكلاسيكية لتقييم البرامج ومقارنة المقدّرات السببية."},
        {"title": "401(k) eligibility", "en": "Chernozhukov & Hansen", "color": "rose", "body": "مثال DML الشهير (حزمة DoubleML تتضمنه)."},
        {"title": "ACIC challenges", "en": "Causal inference competitions", "color": "sky", "body": "بيانات شبه اصطناعية لمقارنة مقدّرات الأثر غير المتجانس."},
        {"title": "Loughran–McDonald", "en": "Financial sentiment lexicon", "color": "emerald", "body": "قاموس نبرة مالي مع وثائق 10-K."},
        {"title": "EPU index", "en": "policyuncertainty.com", "color": "amber", "body": "مؤشر عدم اليقين في السياسة الاقتصادية لعشرات الدول."},
        {"title": "VIIRS night lights", "en": "Earth Observation Group", "color": "indigo", "body": "صور الأضواء الليلية الشهرية منذ 2012."},
    ], cols=4)
