import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Lasso

from core import ui

ui.hero(
    "الثقافتان: التنبؤ مقابل الاستدلال",
    "The Two Cultures — Prediction (ŷ) vs. Estimation & Inference (β̂)",
    "قلب الدمج النظري: تعلم الآلة مُصمَّم ليجيب عن سؤال <b>ŷ</b> (ما القيمة المتوقعة؟)، والقياس الاقتصادي مُصمَّم ليجيب "
    "عن سؤال <b>β̂</b> (ما الأثر؟ ولماذا؟). فهم الفرق — ومتى يلتقيان — هو أهم درس في هذا الدليل.",
    "indigo", "teal",
)

tabs = st.tabs([
    ":material/diversity_2: ثقافتا بريمان",
    ":material/functions: ŷ مقابل β̂",
    ":material/gavel: مسائل السياسات التنبؤية",
    ":material/psychology: التفسير مقابل التنبؤ",
    ":material/science: مختبر: عدم استقرار الاختيار",
    ":material/join: المصالحة بين الثقافتين",
])

with tabs[0]:
    ui.section("مقال بريمان الشهير (2001)", "Breiman (2001): Statistical modeling — the two cultures", "indigo")
    st.markdown(
        """
قسّم **ليو بريمان (Leo Breiman)** — مخترع الغابات العشوائية — الإحصائيين إلى ثقافتين: تفترض الأولى أن الطبيعة تولّد البيانات وفق
**نموذج عشوائي معلوم الشكل** نقدّر معلماته، وتعتبر الثانية آلية التوليد **صندوقًا أسود** ونبحث عن **خوارزمية** تتنبأ جيدًا بالمخرجات.
قدّر بريمان أن ~98% من الإحصائيين آنذاك ينتمون للثقافة الأولى، ودعا إلى تبني الثانية.
        """
    )
    st.mermaid_chart(
        """
flowchart LR
    subgraph DM[Data Modeling Culture - Econometrics]
      direction LR
      x1[x] --> N1[y = f x,beta + e<br/>known form] --> y1[y]
    end
    subgraph AM[Algorithmic Modeling Culture - Machine Learning]
      direction LR
      x2[x] --> N2[Nature = black box<br/>learn f-hat to predict] --> y2[y]
    end
"""
    )
    comp = pd.DataFrame([
        ("الهدف", "فهم الآلية وتقدير المعلمات", "دقة التنبؤ"),
        ("النموذج", "معلمي محدد مسبقًا من النظرية", "خوارزمي مرن يُتعلَّم من البيانات"),
        ("معيار النجاح", "اختبارات الفرضيات، الملاءمة، الدلالة", "الخطأ خارج العينة"),
        ("التحقق", "التشخيص داخل العينة", "عينة اختبار/تحقق متقاطع"),
        ("الأبعاد", "p صغير", "p كبير وأحيانًا p ≫ n"),
        ("التفسير", "معاملات قابلة للتفسير", "صندوق أسود غالبًا"),
        ("الخطر النموذجي", "سوء المواصفة", "الإفراط في التوفيق/التسرب"),
    ], columns=["البعد", "ثقافة نمذجة البيانات (القياس)", "الثقافة الخوارزمية (تعلم الآلة)"])
    st.table(comp)
    ui.note("ردّ عليه آنذاك كوكس (Cox) وإفرون (Efron): الثقافة الخوارزمية تُهمل **التفسير والآلية والاستدلال**. "
            "وبعد عشرين سنة قال **إمبنز وآثي (2021)** إن الاقتصاد القياسي المعاصر يتجه إلى **ثقافة ثالثة** تستخدم الخوارزميات "
            "داخل إطار سببي-استدلالي — وهي جوهر «الدمج».", "الجدل", "indigo", "🗣️")

with tabs[1]:
    ui.section("مولايناثان وسبيس: مشكلة ŷ ومشكلة β̂", "Mullainathan & Spiess (2017)", "teal")
    c1, c2 = st.columns(2)
    with c1:
        ui.note(r"""
$$\min_{f\in\mathcal F}\ \mathbb{E}\big[(y-f(x))^2\big]$$
نبحث عن دالة $\hat f$ تتنبأ جيدًا **خارج العينة**. لا يهم إن كانت المعاملات «صحيحة»؛ يهم أن يكون $\hat y=\hat f(x)$ قريبًا من $y$.
""", "مشكلة التنبؤ ŷ", "teal", "🎯")
    with c2:
        ui.note(r"""
$$y = \beta D + g(X) + \varepsilon,\quad \text{target: } \beta$$
نبحث عن معلمة **مفسَّرة اقتصاديًا** ومُقدَّرة **بلا تحيز** مع **فترة ثقة صالحة**. دقة التنبؤ وحدها لا تضمن ذلك.
""", "مشكلة التقدير β̂", "violet", "🔬")
    st.markdown(
        """
**الرسائل الأساسية للمقال:**
1. خوارزميات تعلم الآلة **ممتازة في ŷ** لأنها تختار التعقيد المناسب عبر **التنظيم (Regularization)** و**الضبط التجريبي (Empirical tuning)**.
2. لكنها **سيئة كأداة لـ β̂ إن استُخدمت بسذاجة**: المعاملات التي «تختارها» غير مستقرة (متغيرات مترابطة تتبادل الأدوار)، و**التنظيم يحيّزها نحو الصفر**.
3. **أين نستخدم ŷ في مسائل β̂؟** في المراحل التي هي **مسائل تنبؤ مقنّعة**: المرحلة الأولى في IV، درجة الميل (Propensity score)،
   المتغيرات الضابطة، المتغيرات المستخلصة من الصور والنصوص، اختبار النظريات بقدرتها التنبؤية.
        """
    )
    st.latex(r"\underbrace{\mathbb{E}[(y-\hat f(x))^2]}_{\text{what ML optimizes}}\;\neq\;"
             r"\underbrace{\mathbb{E}[(\hat\beta-\beta)^2]}_{\text{what econometrics cares about}}")
    ui.chips([("Prediction", "التنبؤ"), ("Estimation", "التقدير"), ("Inference", "الاستدلال"), ("Regularization", "التنظيم"),
              ("Empirical tuning", "الضبط التجريبي"), ("Masked prediction problem", "مسألة تنبؤ مقنّعة")])

with tabs[2]:
    ui.section("عندما يكون التنبؤ هو سؤال السياسة نفسه", "Prediction policy problems — Kleinberg et al. (2015)", "orange")
    c1, c2 = st.columns(2)
    with c1:
        ui.note("هل يزيد **رقص المطر** من هطول المطر؟ سؤال سببي: نحتاج إلى $\\partial y/\\partial D$. لا يفيد التنبؤ هنا.",
                "رقصة المطر — سؤال سببي", "sky", "💃")
    with c2:
        ui.note("هل أحمل **مظلة** اليوم؟ يكفيني **التنبؤ** باحتمال المطر؛ المظلة لا تسبب المطر. القرار يعتمد على $\\hat y$.",
                "المظلة — سؤال تنبؤي", "orange", "☂️")
    st.markdown(
        """
كثير من قرارات السياسة العامة هي في الحقيقة **مسائل تنبؤ**:

| القرار | ما الذي نتنبأ به؟ | الدراسة |
|---|---|---|
| الإفراج بكفالة قبل المحاكمة | خطر الهروب أو العودة للجريمة | Kleinberg et al. (2018) QJE |
| جراحة استبدال الورك لكبار السن | احتمال الوفاة خلال سنة (الجراحة بلا فائدة) | Kleinberg et al. (2015) AER P&P |
| استهداف برامج مكافحة الفقر | الدخل/الاستهلاك من مؤشرات بسيطة (Proxy means test) | تطبيقات البنك الدولي |
| التفتيش الصحي على المطاعم | احتمال المخالفة | تطبيقات بلدية |
| توظيف المعلمين | الأداء المستقبلي | دراسات اقتصاد التعليم |

**لكن انتبه:** حتى مسائل التنبؤ تحتاج إلى حس قياسي: **المشاهدات المختارة (Selective labels)** — لا نرى نتيجة من لم يُفرج عنه —
و**الأثر السببي للقرار نفسه**، و**العدالة**، و**التحول التوزيعي** إذا غيّرت السياسة سلوك الناس.
        """
    )

with tabs[3]:
    ui.section("التفسير، التنبؤ، الإسناد", "Shmueli (2010) & Efron (2020)", "fuchsia")
    ui.cards([
        {"title": "النمذجة التفسيرية", "en": "Explanatory modeling", "icon": "🧩",
         "body": "اختبار فرضيات سببية من النظرية؛ النموذج يعكس الآلية؛ التحيز أخطر من التباين."},
        {"title": "النمذجة التنبؤية", "en": "Predictive modeling", "icon": "🔮",
         "body": "أدنى خطأ متوقع على بيانات جديدة؛ قد نقبل تحيزًا لخفض التباين؛ النموذج «الخاطئ» قد يتنبأ أفضل."},
        {"title": "النمذجة الوصفية", "en": "Descriptive modeling", "icon": "📋",
         "body": "تلخيص البنية الإحصائية دون ادعاء سببي أو تنبؤي."},
    ])
    st.markdown(
        """
يرى **إفرون (2020)** أن خوارزميات التنبؤ الحديثة ممتازة في **التنبؤ**، أما **التقدير** (Estimation) و**الإسناد** (Attribution:
أي المتغيرات مسؤولة؟) فهما أصعب بكثير، لأن الخوارزمية قد تتنبأ جيدًا عبر **متغيرات بديلة مترابطة** دون أن «تعرف» أيها المسبب.
وتؤكد **شموئيلي (2010)** أن الخلط بين التفسير والتنبؤ شائع في العلوم الاجتماعية، وأن **القدرة التنبؤية** معيار مهم أيضًا لتقييم النظريات.
        """
    )
    st.latex(r"\text{EPE}(x_0)=\underbrace{\sigma^2}_{\text{irreducible}}+\underbrace{\big(\mathbb E\hat f(x_0)-f(x_0)\big)^2}_{\text{Bias}^2}"
             r"+\underbrace{\operatorname{Var}\big(\hat f(x_0)\big)}_{\text{Variance}}")
    st.markdown("النموذج التفسيري يركّز على **إزالة التحيز** في معلمة محددة؛ التنبؤي يقبل التحيز إذا انخفض **مجموع** الحدّين.")

with tabs[4]:
    ui.section("مختبر: لاسو يتنبأ جيدًا لكنه «يغيّر رأيه» في المتغيرات", "Lab — LASSO selection instability", "rose")
    st.markdown(
        "20 متغيرًا مترابطًا في **أربع مجموعات** (ارتباط داخلي $\\rho$)، والمتغيرات المؤثرة فعلًا: $x_1, x_6, x_{11}$. "
        "نسحب عينات Bootstrap متتالية ونقدّر لاسو في كل منها. شغّل الحركة: **المتغيرات المختارة تتبدل** من عينة إلى أخرى، "
        "بينما تبقى **دقة التنبؤ خارج العينة** شبه ثابتة. هذا ما عناه مولايناثان وسبيس: *لا تقرأ معاملات لاسو كأنها β̂*."
    )
    rho = st.slider("الارتباط داخل المجموعة ρ", 0.0, 0.95, 0.85, 0.05, key="tc_rho")

    @st.cache_data(show_spinner=False)
    def lasso_instability(rho: float, draws: int = 30):
        rng = np.random.default_rng(11)
        p, n = 20, 150
        cov = np.eye(p)
        for g in range(4):
            idx = range(g * 5, g * 5 + 5)
            for i in idx:
                for j in idx:
                    if i != j:
                        cov[i, j] = rho
        L = np.linalg.cholesky(cov)
        beta = np.zeros(p); beta[[0, 5, 10]] = [1.0, 0.8, 0.6]
        X = rng.standard_normal((n, p)) @ L.T; y = X @ beta + rng.standard_normal(n)
        Xt = rng.standard_normal((3000, p)) @ L.T; yt = Xt @ beta + rng.standard_normal(3000)
        coefs, r2 = [], []
        for _ in range(draws):
            b = rng.integers(0, n, n)
            m = Lasso(alpha=0.08).fit(X[b], y[b])
            coefs.append(m.coef_)
            r2.append(1 - np.mean((yt - m.predict(Xt)) ** 2) / np.var(yt))
        return np.array(coefs), np.array(r2)

    coefs, r2 = lasso_instability(rho)
    labels = [f"x{i + 1}" for i in range(20)]
    colors = [ui.COLORS["rose"] if i in (0, 5, 10) else ui.COLORS["sky"] for i in range(20)]
    fig = go.Figure(
        data=[go.Bar(x=labels, y=coefs[0], marker_color=colors)],
        frames=[go.Frame(data=[go.Bar(x=labels, y=coefs[k], marker_color=colors)], name=str(k + 1),
                         layout=go.Layout(title_text=f"عينة Bootstrap رقم {k + 1} — R² خارج العينة = {r2[k]:.3f}"))
                for k in range(len(coefs))],
    )
    fig.update_yaxes(title="معامل لاسو", range=[-0.3, 1.4])
    fig.update_xaxes(title="المتغيرات (الأحمر = المؤثر فعلًا)")
    ui.style_fig(fig, f"عينة Bootstrap رقم 1 — R² خارج العينة = {r2[0]:.3f}", legend_h=False)
    ui.add_play(fig, list(range(1, len(coefs) + 1)), duration=600, prefix="العينة: ", redraw=True)
    ui.show(fig)
    c1, c2 = st.columns(2)
    with c1:
        sel = (np.abs(coefs) > 1e-8).mean(0)
        f2 = go.Figure(go.Bar(x=labels, y=sel, marker_color=colors))
        f2.update_yaxes(title="نسبة مرات الاختيار", tickformat=".0%", range=[0, 1.05])
        ui.style_fig(f2, "تكرار اختيار كل متغير عبر العينات", height=360)
        ui.show(f2)
    with c2:
        f3 = go.Figure(go.Scatter(x=list(range(1, len(r2) + 1)), y=r2, mode="lines+markers", line=dict(color=ui.COLORS["teal"], width=3)))
        f3.update_yaxes(title="R² خارج العينة", range=[0, 1])
        f3.update_xaxes(title="العينة")
        ui.style_fig(f3, "دقة التنبؤ مستقرة رغم تبدّل المتغيرات", height=360)
        ui.show(f3)
    ui.note("ارفع $\\rho$ إلى 0.9 فما فوق: يزداد تبادل الأدوار بين $x_1$ و $x_2$... داخل المجموعة الواحدة. **التنبؤ مستقر — الإسناد غير مستقر.** "
            "لذلك يستخدم DML لاسو فقط **لتقدير الأجزاء المساعدة**، لا لقراءة المعاملات.", "الخلاصة", "rose", "🎯")

with tabs[5]:
    ui.section("نحو ثقافة ثالثة: الدمج", "Toward a third culture", "emerald")
    st.mermaid_chart(
        """
flowchart TB
    E[Econometrics<br/>identification, inference, theory] --> T((Third culture<br/>Causal ML / Integrated econometrics))
    M[Machine learning<br/>prediction, flexibility, scale] --> T
    T --> A[Use ML for nuisance & measurement]
    T --> B[Keep design-based identification]
    T --> C[Orthogonal scores + cross-fitting]
    T --> D[Report uncertainty honestly]
"""
    )
    ui.ladder([
        ("ميّز نوع السؤال", "Classify the question", "سؤال تنبؤي بحت؟ سببي؟ هيكلي؟ أم مختلط يحتوي «مسائل تنبؤ مقنّعة»؟"),
        ("أعطِ كل ثقافة دورها", "Division of labor", "التعرّف والمعلمة المستهدفة للقياس؛ الدوال المساعدة عالية الأبعاد لتعلم الآلة."),
        ("احمِ الاستدلال", "Protect inference", "تعامد نيمان، التوفيق المتقاطع، تصحيح التحيز، فترات ثقة صالحة."),
        ("قيّم بالتنبؤ أيضًا", "Evaluate predictively too", "حتى النظريات الاقتصادية يمكن اختبارها بقدرتها التنبؤية (Fudenberg et al. «completeness»)."),
    ])
    ui.refs("Breiman (2001) Stat. Science; Shmueli (2010) Stat. Science; Kleinberg et al. (2015) AER P&P; Mullainathan & Spiess (2017) JEP; "
            "Kleinberg et al. (2018) QJE; Efron (2020) JASA; Imbens & Athey (2021) Observational Studies.")
