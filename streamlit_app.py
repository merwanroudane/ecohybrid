import streamlit as st

from core import ui

st.set_page_config(
    page_title="AI × Econometrics | دليل الدمج",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "**AI × Econometrics** — دليل الدمج بين الذكاء الاصطناعي والقياس الاقتصادي.\n\n"
                         "إعداد وتطوير: **د. مروان رودان (Dr Merwan Roudane)** · https://github.com/merwanroudane"},
)
ui.setup()


def P(file: str, title: str, icon: str, default: bool = False) -> st.Page:
    return st.Page(f"app_pages/{file}", title=title, icon=f":material/{icon}:", default=default)


pages = {
    "المدخل": [
        P("home.py", "الرئيسية وخريطة الدليل", "home", default=True),
    ],
    "المحاضرة التأسيسية: لماذا ندمج؟": [
        P("g1_why_breiman.py", "1 · السؤال والدوافع وبريمان وفاريان (01–05)", "psychology"),
        P("g2_exchange.py", "2 · ماذا يأخذ كل مجال من الآخر؟ (06–08)", "sync_alt"),
        P("g3_data_revolution.py", "3 · ثورة البيانات وقياس يولّده AI (09–14)", "dataset"),
        P("g4_mechanism_roadmap.py", "4 · من الدوافع إلى الآلية وخريطة الباحث (15–22)", "route"),
    ],
    "الجذور والتاريخ": [
        P("p01_pre_integration.py", "ما قبل الدمج: الجذور والقيود", "history_edu"),
        P("p02_timeline.py", "الخط الزمني وموجات الدمج", "timeline"),
        P("p03_big_data.py", "ثورة البيانات الضخمة وغير المهيكلة", "database"),
    ],
    "الأسس النظرية": [
        P("p04_two_cultures.py", "الثقافتان: التنبؤ مقابل الاستدلال", "balance"),
        P("p05_ml_foundations.py", "مفاهيم تعلم الآلة للاقتصاديين", "school"),
        P("p06_algorithms.py", "الخوارزميات بعين قياسية", "account_tree"),
    ],
    "الدمج: المفهوم والمستويات والمنهج": [
        P("p07_concept.py", "مفهوم الدمج ودوافعه", "merge"),
        P("p08_levels.py", "مستويات الدمج وتصنيفاته", "stacks"),
        P("p09_how.py", "كيف ندمج؟ المنهجية العملية", "route"),
    ],
    "الدمج بالتفصيل: من الصفر إلى الأحدث": [
        P("d01_from_zero.py", "الدمج من الصفر: ماذا ندمج وأين؟", "start"),
        P("d02_levels_deep.py", "المستويات L0–L7 بالتفصيل", "stairs"),
        P("d10_mechanism.py", "آلية الدمج الفعلية: الإشارات والواجهات", "cable"),
        P("d03_residuals.py", "الدمج عبر البواقي", "merge"),
        P("d11_components.py", "الدمج على مستوى المكونات (التفكيك)", "waterfall_chart"),
        P("d04_information.py", "الدمج في المعلومات والمتغيرات", "dataset"),
        P("d05_function.py", "الدمج في الشكل الدالي والمعلمات", "function"),
        P("d06_objective.py", "الدمج في دالة الهدف والتقدير", "target"),
        P("d07_inference.py", "الدمج في الاستدلال", "verified"),
        P("d08_combination.py", "الدمج في التوليف والقرار", "layers"),
        P("d09_cases.py", "حالات تطبيقية كاملة من الصفر", "cases"),
    ],
    "محاضرة: فلسفة وآليات الدمج (48 فصلًا)": [
        P("f1_philosophy.py", "1 · الأسس والمفاهيم والتدريب (01–06)", "school"),
        P("f2_residuals.py", "2 · البواقي والتنبؤات والبوابات (07–13)", "merge"),
        P("f3_parameters.py", "3 · المعلمات والتحديد والتدريب المشترك (14–22)", "tune"),
        P("f4_causal.py", "4 · DML والسببية والنظرية (23–28)", "hub"),
        P("f5_structures.py", "5 · ECM وLSTM والبانل والحلّالات (29–37)", "account_tree"),
        P("f6_protocol.py", "6 · الحدود والبروتوكول والتطبيق (38–48)", "fact_check"),
    ],
    "المنهجيات والتطبيقات": [
        P("p10_causal_ml.py", "الاستدلال السببي بتعلم الآلة", "hub"),
        P("p11_heterogeneity.py", "الأثر غير المتجانس وتعلم السياسات", "diversity_3"),
        P("p12_forecasting.py", "التنبؤ والسلاسل الزمنية", "trending_up"),
        P("p13_unstructured.py", "النص والصورة كبيانات", "text_snippet"),
        P("p14_structural.py", "النماذج الهيكلية والذكاء الاصطناعي", "architecture"),
        P("p15_genai.py", "الذكاء التوليدي ونماذج اللغة الكبيرة", "smart_toy"),
        P("p16_econ_for_ai.py", "القياس الاقتصادي في خدمة الذكاء الاصطناعي", "sync_alt"),
    ],
    "الخوارزميات الهجينة المتخصصة": [
        P("p30_author_libs.py", "مكتبات د. مروان رودان الهجينة (PyPI)", "deployed_code"),
        P("p23_hybrid_ts.py", "هجين السلاسل الزمنية: VAR · VECM · ARDL · Granger", "ssid_chart"),
        P("p24_hybrid_panel.py", "هجين البانل والسببية: DML · RDD · DiD · IV", "grid_view"),
        P("p25_rl.py", "التعلم المعزز والاقتصاد القياسي", "sports_esports"),
        P("p26_theory_nn.py", "الشبكات الموجّهة بالنظرية الاقتصادية", "neurology"),
        P("p27_measurement.py", "ثورة القياس: المتغيرات المولّدة بالذكاء الاصطناعي", "straighten"),
        P("p28_foundation.py", "النماذج التأسيسية والرمزية واكتشاف السببية", "auto_awesome"),
    ],
    "المختبر التطبيقي لمكتبات المؤلف": [
        P("lab_data.py", "مدير البيانات (رفع وتحويل)", "upload_file"),
        P("lab_causal.py", "مختبر DML والسببية", "hub"),
        P("lab_panel.py", "مختبر البانل الهجين", "grid_view"),
        P("lab_granger.py", "مختبر سببية غرانجر العصبية", "share"),
        P("lab_var.py", "مختبر VAR · ARDL الهجين", "ssid_chart"),
        P("lab_vol.py", "مختبر التقلب والتنبؤ الهجين", "candlestick_chart"),
        P("lab_kernel.py", "مختبر النواة والموجات والكمّيات", "blur_on"),
    ],
    "التقييم والقرار": [
        P("p17_when.py", "متى ندمج؟ ومتى لا ندمج؟", "rule"),
        P("p18_critiques.py", "الانتقادات والقيود والمخاطر", "report"),
        P("p19_frontier.py", "أين وصلنا اليوم؟ والآفاق", "rocket_launch"),
        P("p29_evaluation.py", "التقييم والتشخيص وقائمة التحقق", "fact_check"),
    ],
    "المصادر والأدوات": [
        P("p20_tools.py", "الأدوات والبرمجيات", "construction"),
        P("p21_glossary.py", "مسرد المصطلحات", "translate"),
        P("p22_references.py", "المراجع والقراءات", "menu_book"),
    ],
}

page = st.navigation(pages, position="sidebar", expanded=True)

with st.sidebar:
    st.markdown("### :violet[AI] × :orange[Econometrics]")
    st.caption("دليل نظري شامل للباحثين حول الدمج بين الذكاء الاصطناعي والقياس الاقتصادي، "
               "من الجذور إلى آخر التطورات (2026).")
    ui.author_card()

page.run()
ui.footer()
