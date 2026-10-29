import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui
from core.lab.page import TRANSFORMS, read_upload

ui.hero(
    "مدير البيانات للمختبر التطبيقي",
    "Data manager — upload, inspect and transform your data once, use it in every library",
    "ارفع ملفك (CSV أو Excel أو Stata أو Parquet) مرة واحدة، وعاين المتغيرات، وطبّق التحويلات الشائعة (لوغاريتم، فروق، معدلات نمو، فجوات، تعيير)، "
    "ثم استخدمه مباشرة في مختبر أي مكتبة من مكتبات د. مروان رودان الهجينة.",
    "indigo", "teal",
)

tabs = st.tabs([":material/upload_file: رفع البيانات", ":material/table_view: المعاينة والوصف", ":material/transform: التحويلات", ":material/show_chart: الرسم"])

with tabs[0]:
    up = st.file_uploader("ارفع ملف البيانات", type=["csv", "txt", "xlsx", "xls", "dta", "parquet"], key="lab_upload")
    if up is not None and st.session_state.get("lab_df_name") != up.name:
        try:
            df = read_upload(up)
            st.session_state["lab_df"] = df
            st.session_state["lab_df_name"] = up.name
            st.success(f"تم تحميل {up.name}: {df.shape[0]} صفًا × {df.shape[1]} عمودًا", icon=":material/check_circle:")
        except Exception as e:  # noqa: BLE001
            st.error(f"تعذّرت قراءة الملف: {e}")
    df = st.session_state.get("lab_df")
    if df is not None:
        st.info(f"البيانات الحالية: **{st.session_state.get('lab_df_name')}** — {df.shape[0]} × {df.shape[1]}. اذهب إلى أي صفحة مختبر واختر «بياناتي المرفوعة».", icon=":material/database:")
        if st.button("حذف البيانات المرفوعة", icon=":material/delete:"):
            for k in ("lab_df", "lab_df_name"):
                st.session_state.pop(k, None)
            st.rerun()
    ui.note("**تنسيق البيانات:**\n- **سلاسل زمنية**: صف لكل فترة مرتبة زمنيًا، وعمود لكل متغير (وعمود تاريخ اختياري).\n"
            "- **بانل**: صيغة طويلة — عمود للوحدة، عمود للزمن، وأعمدة للمتغيرات (صف لكل وحدة–فترة).\n"
            "- **مقطعية**: صف لكل مشاهدة.\n- الأعمدة النصية لا تظهر في قوائم المتغيرات الرقمية.", "إرشادات", "sky", "📋")

df = st.session_state.get("lab_df")

with tabs[1]:
    if df is None:
        st.caption("ارفع بيانات أولًا.")
    else:
        st.dataframe(df.head(300), height=320)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**الإحصاء الوصفي**")
            st.dataframe(df.describe().T, height=320)
        with c2:
            st.markdown("**القيم المفقودة والأنواع**")
            st.dataframe(pd.DataFrame({"النوع": df.dtypes.astype(str), "مفقود": df.isna().sum(), "% مفقود": (100 * df.isna().mean()).round(2)}), height=320)

with tabs[2]:
    if df is None:
        st.caption("ارفع بيانات أولًا.")
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        with st.form("lab_tr_form"):
            c1, c2, c3 = st.columns(3)
            cols = c1.multiselect("المتغيرات", num_cols, key="tr_cols")
            how = c2.selectbox("التحويل", list(TRANSFORMS), key="tr_how")
            grp = c3.selectbox("تجميع حسب الوحدة (للبانل)", ["— بلا —"] + list(df.columns), key="tr_grp")
            ok = st.form_submit_button("تطبيق التحويل", icon=":material/auto_fix_high:")
        if ok and cols:
            g = None if grp == "— بلا —" else df[grp]
            suffix = {"لوغاريتم ln(x)": "ln", "فرق أول Δx": "d", "معدل نمو لوغاريتمي 100·Δln(x)": "g", "نسبة تغير %": "pct",
                      "تعيير (z-score)": "z", "فجوة أولى x(t−1)": "L1"}[how]
            for c in cols:
                df[f"{c}_{suffix}"] = TRANSFORMS[how](df[c], g)
            st.session_state["lab_df"] = df
            st.success(f"أضيفت {len(cols)} أعمدة جديدة بلاحقة _{suffix}")
        c1, c2 = st.columns(2)
        if c1.button("حذف الصفوف ذات القيم المفقودة", icon=":material/cleaning_services:"):
            st.session_state["lab_df"] = df.dropna().reset_index(drop=True)
            st.rerun()
        c2.download_button("تنزيل البيانات المعدّلة CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name="lab_data.csv", icon=":material/download:")

with tabs[3]:
    if df is None:
        st.caption("ارفع بيانات أولًا.")
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        pick = st.multiselect("متغيرات للرسم", num_cols, default=num_cols[:3], key="plot_cols")
        if pick:
            fig = go.Figure([go.Scatter(y=df[c], name=c, mode="lines") for c in pick])
            ui.style_fig(fig, "السلاسل المختارة", height=420)
            ui.show(fig)
