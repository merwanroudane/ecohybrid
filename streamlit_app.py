import streamlit as st

from core import ui

st.set_page_config(
    page_title="AI × Econometrics | دليل الدمج",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.setup()


def P(file: str, title: str, icon: str, default: bool = False) -> st.Page:
    return st.Page(f"app_pages/{file}", title=title, icon=f":material/{icon}:", default=default)


pages = {
    "المدخل": [
        P("home.py", "الرئيسية وخريطة الدليل", "home", default=True),
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

page.run()
ui.footer()
