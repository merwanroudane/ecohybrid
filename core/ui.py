"""Shared look-and-feel helpers: RTL layout, right sidebar, coloured cards,
callouts, term chips, Plotly styling and play/pause animation controls."""

from __future__ import annotations

import re
from contextlib import contextmanager
from html import escape

import plotly.graph_objects as go
import streamlit as st

from core.mathhtml import tex_to_html

# Inline $...$ inside raw-HTML components is not seen by Streamlit's KaTeX, and st.html
# strips MathML, so it is converted to plain HTML (Unicode, <sub>/<sup>, CSS fractions).
_INLINE_MATH = re.compile(r"(?<![\\$])\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)")


def _math(html: str) -> str:
    if "$" not in html:
        return html
    return _INLINE_MATH.sub(lambda m: f'<span class="imath">{tex_to_html(m.group(1))}</span>', html)


# ---------------------------------------------------------------- palette ---
COLORS = {
    "violet": "#7c3aed", "teal": "#0d9488", "sky": "#0284c7", "rose": "#e11d48",
    "amber": "#d97706", "emerald": "#059669", "indigo": "#4f46e5", "pink": "#db2777",
    "orange": "#ea580c", "cyan": "#0891b2", "lime": "#65a30d", "fuchsia": "#c026d3",
    "slate": "#475569",
}
TINTS = {
    "violet": "#f5f0ff", "teal": "#ecfdf9", "sky": "#eff8ff", "rose": "#fff1f3",
    "amber": "#fff8eb", "emerald": "#ecfdf5", "indigo": "#eef0ff", "pink": "#fdf0f7",
    "orange": "#fff4ec", "cyan": "#ecfbfe", "lime": "#f5fbe9", "fuchsia": "#fcf0fe",
    "slate": "#f4f6f9",
}
CYCLE = ["violet", "teal", "orange", "sky", "rose", "emerald", "pink", "amber",
         "indigo", "cyan", "lime", "fuchsia"]
PLOT_COLORS = [COLORS[c] for c in CYCLE]


def color_at(i: int) -> str:
    return CYCLE[i % len(CYCLE)]


def rgba(hex_color: str, alpha: float) -> str:
    """'#rrggbb' -> 'rgba(r,g,b,a)' (Plotly rejects 8-digit hex)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def E(text: str) -> str:
    """Isolate an English fragment inside Arabic HTML so brackets do not flip."""
    return f'<span class="ltr">{escape(text)}</span>'


# ------------------------------------------------------------- global CSS ---
def _css() -> str:
    tint_rules = "\n".join(
        f"""div[class*="st-key-tint-{name}-"] {{
            background: linear-gradient(135deg, {TINTS[name]} 0%, #ffffff 120%);
            border: 1px solid {COLORS[name]}33;
            border-right: 6px solid {COLORS[name]};
            border-radius: 16px; padding: 18px 20px 10px 20px;
            box-shadow: 0 2px 10px {COLORS[name]}14;
        }}
        div[class*="st-key-tint-{name}-"] h3, div[class*="st-key-tint-{name}-"] h4,
        div[class*="st-key-tint-{name}-"] strong {{ color: {COLORS[name]}; }}"""
        for name in COLORS
    )
    return f"""
<style>
/* ---- right-to-left reading + sidebar on the right ---- */
[data-testid="stAppViewContainer"] {{ flex-direction: row-reverse; }}
[data-testid="stMain"], [data-testid="stSidebar"] {{ direction: rtl; text-align: right; }}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #f5f3ff 0%, #ecfeff 55%, #fff7ed 100%);
    border-left: 1px solid #ddd6fe; border-right: none;
}}
[data-testid="stSidebarNav"] a span {{ font-size: 0.93rem; }}
[data-testid="stSidebarNavSeparator"] {{ display: none; }}
[data-testid="stNavSectionHeader"] {{
    color: #6d28d9 !important; font-weight: 800; letter-spacing: .2px;
    border-bottom: 2px dashed #c4b5fd; padding-bottom: 2px;
}}
/* keep maths, code, charts and diagrams left-to-right */
.katex, .katex-display, [data-testid="stLatex"], pre, code,
[data-testid="stPlotlyChart"], [data-testid="stCode"], .stMermaid, svg {{
    direction: ltr; unicode-bidi: isolate;
}}
[data-testid="stLatex"] {{ text-align: center; }}
[data-testid="stMermaidChart"] {{ direction: ltr; text-align: center; }}
[data-testid="stMermaidChart"] img, [data-testid="stMermaidChart"] svg {{ max-width: 100% !important; height: auto !important; }}
[data-testid="stMarkdownContainer"] ul, [data-testid="stMarkdownContainer"] ol {{
    padding-right: 1.4em; padding-left: 0;
}}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
    line-height: 1.95;
}}
[data-testid="stTable"] table, [data-testid="stDataFrame"] {{ direction: rtl; }}
[data-testid="stExpander"] summary {{ direction: rtl; }}
/* tabs: coloured pills */
[data-testid="stTabs"] [role="tablist"] {{ gap: 6px; flex-wrap: wrap; }}
[data-testid="stTabs"] button[role="tab"] {{
    background: #f5f3ff; border-radius: 10px 10px 0 0; padding: 6px 14px;
    border: 1px solid #ede9fe;
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, #ede9fe, #e0f2fe); color: #5b21b6;
}}
/* hero banner */
.hero {{
    border-radius: 22px; padding: 26px 30px 22px 30px; margin: 0 0 18px 0;
    color: #1e1b4b; position: relative; overflow: hidden;
    border: 1px solid #ffffff;
}}
.hero h1 {{ font-family: Cairo, sans-serif; font-weight: 800; font-size: 2.0rem;
    margin: 0 0 4px 0; line-height: 1.45; }}
.hero .en {{ direction: ltr; text-align: right; font-family: 'Inter', sans-serif;
    font-weight: 600; letter-spacing: .4px; opacity: .78; font-size: 1.02rem; }}
.hero p {{ margin: 10px 0 0 0; font-size: 1.04rem; line-height: 1.9; max-width: 980px; }}
.hero .blob {{ position: absolute; border-radius: 50%; filter: blur(2px); opacity: .35; }}
/* section header */
.sec {{ display: flex; align-items: center; gap: 12px; margin: 26px 0 10px 0; }}
.sec .bar {{ width: 8px; height: 34px; border-radius: 6px; }}
.sec .t {{ font-family: Cairo, sans-serif; font-weight: 800; font-size: 1.38rem; }}
.sec .e {{ direction: ltr; font-size: .9rem; opacity: .7; font-weight: 600; }}
/* card grid */
.grid {{ display: grid; gap: 14px; margin: 6px 0 14px 0; }}
.card {{
    border-radius: 16px; padding: 16px 16px 12px 16px; background: #fff;
    border: 1px solid #ece8fb; box-shadow: 0 3px 12px rgba(80,60,160,.06);
    border-top: 5px solid var(--c); transition: transform .15s ease;
}}
.card:hover {{ transform: translateY(-2px); }}
.card .ct {{ font-family: Cairo, sans-serif; font-weight: 800; color: var(--c);
    font-size: 1.07rem; margin-bottom: 2px; }}
.card .ce {{ direction: ltr; text-align: right; font-size: .8rem; color: #64748b;
    font-weight: 600; margin-bottom: 6px; }}
.card .cb {{ font-size: .95rem; line-height: 1.85; color: #312e81; }}
.card .cb ul {{ padding-right: 1.1em; margin: 4px 0; }}
/* term chips */
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 12px 0; }}
.chip {{ border-radius: 999px; padding: 4px 12px; font-size: .86rem;
    border: 1px solid var(--c); background: var(--t); color: #1e1b4b; }}
.chip b {{ color: var(--c); direction: ltr; unicode-bidi: isolate; }}
/* step ladder */
.ladder {{ display: flex; flex-direction: column; gap: 8px; margin: 8px 0 14px 0; }}
.rung {{ display: flex; align-items: stretch; gap: 0; border-radius: 14px; overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,.05); background: #fff; border: 1px solid #eee; }}
.rung .n {{ min-width: 64px; display: flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 800; font-size: 1.25rem; font-family: Cairo, sans-serif; }}
.rung .b {{ padding: 10px 14px; }}
.rung .b .h {{ font-weight: 800; font-family: Cairo, sans-serif; }}
.rung .b .s {{ font-size: .9rem; color: #475569; line-height: 1.8; }}
/* small reference line */
.refline {{ font-size: .82rem; color: #64748b; border-top: 1px dashed #ddd6fe;
    margin-top: 8px; padding-top: 6px; }}
.ltr {{ direction: ltr; unicode-bidi: isolate; display: inline-block; }}
.imath {{ direction: ltr; unicode-bidi: isolate; display: inline-block; }}
.imath {{ font-family: "Cambria Math", "STIX Two Math", "Latin Modern Math", "Times New Roman", serif; font-size: 1.05em; white-space: nowrap; }}
.imath .mf, .imath .mt {{ font-style: normal; }}
.imath .mt {{ font-family: inherit; }}
.imath sub, .imath sup {{ font-size: .72em; }}
.imath .mfrac {{ display: inline-flex; flex-direction: column; vertical-align: middle; text-align: center; font-size: .92em; margin: 0 .12em; }}
.imath .mfrac > span:first-child {{ border-bottom: 1px solid currentColor; padding: 0 .15em; }}
.imath .mover {{ text-decoration: overline; }}
.foot {{ text-align: center; color: #7c7aa3; font-size: .85rem; margin-top: 40px;
    padding-top: 12px; border-top: 1px dashed #ddd6fe; line-height: 2; }}
.foot a, .author a, .byline a {{ color: #6d28d9; text-decoration: none; font-weight: 600; }}
.foot a:hover, .author a:hover, .byline a:hover {{ text-decoration: underline; }}
/* developer credit */
.author {{ background: linear-gradient(135deg, #ede9fe, #ccfbf1); border: 1px solid #ddd6fe;
    border-radius: 14px; padding: 12px 14px; margin-top: 6px; }}
.author .lbl {{ font-size: .75rem; color: #6d28d9; font-weight: 700; letter-spacing: .3px; }}
.author .nm {{ font-family: Cairo, sans-serif; font-size: 1.15rem; font-weight: 800; color: #1e1b4b; }}
.author .en {{ font-size: .85rem; color: #475569; }}
.author .links {{ font-size: .82rem; margin-top: 4px; }}
.byline {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin: -4px 0 14px;
    padding: 10px 16px; background: #ffffff; border: 1px solid #ede9fe; border-radius: 14px; font-size: .92rem; color: #475569; }}
.byline b {{ color: #1e1b4b; font-family: Cairo, sans-serif; }}
{tint_rules}
</style>
"""


def setup() -> None:
    """Inject global CSS and reset the per-run key counter."""
    st.session_state["_ui_k"] = 0
    st.html(_css())


def _key(prefix: str) -> str:
    st.session_state["_ui_k"] = st.session_state.get("_ui_k", 0) + 1
    return f"{prefix}-{st.session_state['_ui_k']}"


# ------------------------------------------------------------- components ---
def hero(title: str, en: str, desc: str, c1: str = "violet", c2: str = "sky") -> None:
    st.html(
        f"""<div class="hero" style="background: linear-gradient(120deg, {TINTS[c1]} 0%,
        #ffffff 45%, {TINTS[c2]} 100%); box-shadow: 0 8px 26px {COLORS[c1]}1f;">
        <div class="blob" style="width:220px;height:220px;background:{COLORS[c1]}55;left:-60px;top:-80px"></div>
        <div class="blob" style="width:160px;height:160px;background:{COLORS[c2]}55;left:140px;bottom:-90px"></div>
        <h1 style="color:{COLORS[c1]}">{title}</h1>
        <div class="en">{escape(en)}</div>
        <p>{_math(desc)}</p></div>"""
    )


def section(title: str, en: str = "", color: str = "violet") -> None:
    st.html(
        f"""<div class="sec"><div class="bar" style="background:linear-gradient(180deg,
        {COLORS[color]}, {COLORS[color]}55)"></div><div><div class="t" style="color:{COLORS[color]}">
        {title}</div><div class="e">{escape(en)}</div></div></div>"""
    )


def cards(items: list[dict], cols: int = 3) -> None:
    """items: dicts with keys title, body (HTML), optional en, color, icon."""
    parts = []
    for i, it in enumerate(items):
        c = COLORS[it.get("color", color_at(i))]
        icon = it.get("icon", "")
        en = f'<div class="ce">{escape(it["en"])}</div>' if it.get("en") else ""
        parts.append(
            f'<div class="card" style="--c:{c}"><div class="ct">{icon} {_math(it["title"])}</div>'
            f'{en}<div class="cb">{_math(it["body"])}</div></div>'
        )
    st.html(
        f'<div class="grid" style="grid-template-columns: repeat(auto-fit, minmax({int(900 / cols) - 40}px, 1fr));">'
        + "".join(parts) + "</div>"
    )


def chips(pairs: list[tuple[str, str]]) -> None:
    """Show English term / Arabic gloss pairs as coloured chips."""
    out = []
    for i, (en, ar) in enumerate(pairs):
        n = color_at(i)
        out.append(f'<span class="chip" style="--c:{COLORS[n]};--t:{TINTS[n]}"><b>{escape(en)}</b> · {_math(ar)}</span>')
    st.html('<div class="chips">' + "".join(out) + "</div>")


def ladder(steps: list[tuple[str, str, str]], start: int = 1) -> None:
    """steps: (heading, english, sub-text)."""
    rows = []
    for i, (h, en, s) in enumerate(steps):
        n = color_at(i)
        rows.append(
            f'<div class="rung"><div class="n" style="background:linear-gradient(160deg,{COLORS[n]},{COLORS[n]}bb)">'
            f'{i + start}</div><div class="b"><div class="h" style="color:{COLORS[n]}">{_math(h)} '
            f'<span class="ltr" style="font-size:.82rem;color:#64748b">({escape(en)})</span></div>'
            f'<div class="s">{_math(s)}</div></div></div>'
        )
    st.html('<div class="ladder">' + "".join(rows) + "</div>")


@contextmanager
def tinted(color: str = "violet"):
    """A coloured container whose body is ordinary Streamlit content."""
    with st.container(key=_key(f"tint-{color}")):
        yield


def note(body: str, title: str = "", color: str = "sky", icon: str = "") -> None:
    with tinted(color):
        if title:
            st.markdown(f"#### {icon} {title}")
        st.markdown(body)


def refs(text: str) -> None:
    st.html(f'<div class="refline">📚 {text}</div>')


AUTHOR_AR = "د. مروان رودان"
AUTHOR_EN = "Dr Merwan Roudane"
AUTHOR_GITHUB = "https://github.com/merwanroudane"
AUTHOR_PYPI = "https://pypi.org/user/merwan.roudane/"


def _author_links() -> str:
    return (f'<a href="{AUTHOR_GITHUB}" target="_blank" rel="noopener">GitHub</a> · '
            f'<a href="{AUTHOR_PYPI}" target="_blank" rel="noopener">PyPI</a>')


def author_card() -> None:
    """Developer credit for the sidebar."""
    st.html(f'<div class="author"><div class="lbl">إعداد وتطوير</div><div class="nm">{AUTHOR_AR}</div>'
            f'<div class="en"><span class="ltr">{AUTHOR_EN}</span></div>'
            f'<div class="links"><span class="ltr">{_author_links()}</span></div></div>')


def byline() -> None:
    """Developer credit under a page hero."""
    st.html(f'<div class="byline">إعداد وتطوير: <b>{AUTHOR_AR}</b> <span class="ltr">({AUTHOR_EN})</span>'
            f'<span class="ltr">{_author_links()}</span></div>')


def footer() -> None:
    st.html('<div class="foot">دليل الدمج بين الذكاء الاصطناعي والقياس الاقتصادي · '
            '<span class="ltr">AI × Econometrics Integration Guide</span><br>'
            f'إعداد وتطوير: <b>{AUTHOR_AR}</b> <span class="ltr">({AUTHOR_EN}) © 2026 · {_author_links()}</span></div>')


# ------------------------------------------------------------------ plots ---
def style_fig(fig: go.Figure, title: str = "", height: int = 460, legend_h: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16, family="Cairo, sans-serif")),
        height=height,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fbfaff",
        font=dict(family="IBM Plex Sans Arabic, Inter, sans-serif", size=13, color="#1e1b4b"),
        colorway=PLOT_COLORS,
        margin=dict(l=40, r=30, t=60 if title else 30, b=40),
        hoverlabel=dict(font_family="IBM Plex Sans Arabic, Inter, sans-serif"),
    )
    if legend_h:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    fig.update_xaxes(gridcolor="#ede9fe", zerolinecolor="#ddd6fe")
    fig.update_yaxes(gridcolor="#ede9fe", zerolinecolor="#ddd6fe")
    return fig


def add_play(fig: go.Figure, names: list, duration: int = 450, prefix: str = "",
             redraw: bool = False, transition: int | None = None) -> go.Figure:
    """Attach Play / Pause buttons and a frame slider to an animated figure."""
    tr = int(duration * 0.6) if transition is None else transition
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=-0.30, xanchor="left", yanchor="top",
            pad=dict(r=6, t=4), showactive=False, bgcolor="#ede9fe", bordercolor="#c4b5fd",
            font=dict(size=13, color="#4c1d95"),
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": duration, "redraw": redraw},
                                  "fromcurrent": True,
                                  "transition": {"duration": tr, "easing": "cubic-in-out"}}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate", "transition": {"duration": 0}}]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.22, len=0.78, y=-0.22, yanchor="top", pad=dict(t=10, b=4),
            currentvalue=dict(prefix=prefix, xanchor="right", font=dict(size=13, color="#6d28d9")),
            bgcolor="#ede9fe", activebgcolor="#a78bfa", bordercolor="#c4b5fd",
            steps=[dict(method="animate", label=str(n),
                        args=[[str(n)], {"frame": {"duration": 0, "redraw": redraw},
                                         "mode": "immediate", "transition": {"duration": 0}}])
                   for n in names],
        )],
    )
    fig.update_layout(margin=dict(b=150), height=max(fig.layout.height or 460, 500))
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, config={"displaylogo": False, "responsive": True})


def link(page: str, label: str, icon: str | None = None, container=None) -> None:
    """st.page_link that degrades silently when the page is run outside st.navigation."""
    try:
        (container or st).page_link(page, label=label, icon=icon)
    except st.errors.StreamlitPageNotFoundError:
        pass
