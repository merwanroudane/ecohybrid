import importlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import glossary_data, ui

# reload so content pushed to the repo shows without restarting the app
G = importlib.reload(glossary_data).G

ui.hero(
    "مسرد المصطلحات الإنجليزية–العربية",
    "English–Arabic Glossary of AI × Econometrics",
    f"أكثر من {len(G)} مصطلحًا مصنّفة في عشرة مجالات، مع تعريف مختصر لكل مصطلح. ابحث بالعربية أو الإنجليزية، أو صفِّ حسب المجال.",
    "violet", "teal",
)

df = pd.DataFrame(G, columns=["English", "العربية", "المجال", "التعريف"])
tabs = st.tabs([":material/search: البحث في المسرد", ":material/style: بطاقات حسب المجال", ":material/pie_chart: توزيع المصطلحات"])

with tabs[0]:
    c1, c2 = st.columns([2, 1])
    q = c1.text_input("ابحث", placeholder="مثال: orthogonality، تسرب، Lasso، بلمان ...", key="gl_q")
    cats = c2.multiselect("المجال", sorted(df["المجال"].unique()), key="gl_c")
    sub = df.copy()
    if q:
        mask = sub.apply(lambda r: q.lower() in " ".join(r.astype(str)).lower(), axis=1)
        sub = sub[mask]
    if cats:
        sub = sub[sub["المجال"].isin(cats)]
    st.caption(f"{len(sub)} مصطلحًا مطابقًا")
    st.dataframe(sub, hide_index=True, height=620,
                 column_config={"التعريف": st.column_config.TextColumn(width="large"), "English": st.column_config.TextColumn(width="medium")})

with tabs[1]:
    cat = st.selectbox("اختر المجال", sorted(df["المجال"].unique()), key="gl_card")
    sub = df[df["المجال"] == cat]
    ui.cards([{"title": r["العربية"], "en": r["English"], "body": r["التعريف"]} for _, r in sub.iterrows()], cols=3)

with tabs[2]:
    cnt = df["المجال"].value_counts()
    fig = go.Figure(go.Pie(labels=cnt.index, values=cnt.values, hole=.45, marker=dict(colors=ui.PLOT_COLORS)))
    ui.style_fig(fig, "توزيع المصطلحات حسب المجال", height=480)
    ui.show(fig)
