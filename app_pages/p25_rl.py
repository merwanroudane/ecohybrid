import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ui

ui.hero(
    "التعلم المعزز والاقتصاد القياسي",
    "Reinforcement Learning, Dynamic Programming, Bandits and Structural Econometrics",
    "التعلم المعزز هو أقرب فروع الذكاء الاصطناعي إلى الاقتصاد: عون يختار أفعالًا في حالات ويتلقى مكافآت ويعظّم قيمة متوقعة مخصومة — "
    "وهي البرمجة الديناميكية لبلمان التي بنى عليها الاقتصاديون نماذج البحث والاستثمار والاستهلاك. هنا نربط المفاهيم ونعرض التطبيقات والمحاذير.",
    "lime", "violet",
)

tabs = st.tabs([
    ":material/school: الأساسيات",
    ":material/compare_arrows: RL والتقدير الهيكلي",
    ":material/science: مختبر: البحث عن عمل",
    ":material/casino: البانديت والتجارب التكيفية",
    ":material/history: التقييم دون اتصال",
    ":material/apps: التطبيقات الاقتصادية",
    ":material/warning: المحاذير",
])

with tabs[0]:
    ui.section("عملية القرار الماركوفية ومعادلة بلمان", "MDP, Bellman equation and core algorithms", "lime")
    st.latex(r"\pi^*=\arg\max_\pi\ \mathbb E_\pi\Big[\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)\Big],\qquad s_{t+1}\sim P(\cdot\mid s_t,a_t)")
    st.latex(r"Q^*(s,a)=r(s,a)+\gamma\,\mathbb E\big[\max_{a'}Q^*(s',a')\mid s,a\big]\qquad\text{Q-learning: } Q\leftarrow Q+\eta\big[r+\gamma\max_{a'}Q(s',a')-Q\big]")
    mapping = pd.DataFrame([
        ("الحالة s", "State", "متغيرات الحالة (رأس المال، الثروة، الصدمة)"),
        ("الفعل a", "Action", "متغير التحكم (الاستهلاك، الاستثمار، السعر)"),
        ("المكافأة r", "Reward", "المنفعة الآنية / الربح"),
        ("معامل الخصم γ", "Discount factor", "β في النماذج الاقتصادية"),
        ("السياسة π", "Policy", "دالة السياسة / قاعدة القرار"),
        ("دالة القيمة V", "Value function", "دالة القيمة في البرمجة الديناميكية"),
        ("نموذج الانتقال P", "Transition model", "قانون الحركة والتوقعات"),
        ("الاستكشاف", "Exploration", "التجريب — غائب غالبًا عن نماذج التوقعات الرشيدة"),
    ], columns=["مفهوم RL", "English", "المقابل الاقتصادي"])
    st.table(mapping)
    ui.cards([
        {"title": "قائم على النموذج / بلا نموذج", "en": "Model-based vs. model-free", "color": "lime", "body": "الاقتصاد تقليديًا قائم على النموذج (نعرف P)؛ RL بلا نموذج يتعلم من التجربة مباشرة."},
        {"title": "طرق القيمة", "en": "Q-learning, DQN (Mnih et al., 2015)", "color": "teal", "body": "تعلم Q ثم السياسة الجشعة؛ DQN يقرّب Q بشبكة عميقة."},
        {"title": "تدرج السياسة", "en": "REINFORCE, PPO, actor-critic", "color": "violet", "body": "تحسين سياسة معلمية مباشرة؛ مناسب للأفعال المستمرة."},
        {"title": "متعدد الأعوان", "en": "Multi-agent RL", "color": "orange", "body": "تفاعل استراتيجي وتعلم مشترك — صلة بنظرية الألعاب التطورية والتعلم في الألعاب."},
    ], cols=4)

with tabs[1]:
    ui.section("التعلم المعزز والتقدير الهيكلي: وجهان لعملة واحدة", "RL ↔ dynamic discrete choice", "teal")
    st.mermaid_chart(
        """
flowchart LR
    subgraph RL[Reinforcement learning]
      R1[Known reward r] --> R2[Learn optimal policy pi*]
    end
    subgraph IRL[Inverse RL]
      I1[Observed behaviour] --> I2[Recover reward / preferences]
    end
    subgraph SE[Structural econometrics]
      S1[Observed choices] --> S2[Estimate utility parameters theta<br/>NFXP, CCP, MPEC]
    end
    I2 <-->|same problem| S2
"""
    )
    ui.cards([
        {"title": "محرك الحافلة لروست", "en": "Rust (1987, Econometrica) — NFXP", "color": "teal", "body": "تقدير تكلفة الصيانة واستبدال المحرك بحل البرمجة الديناميكية داخل حلقة الإمكان — أساس الاختيار الديناميكي المتقطع."},
        {"title": "احتمالات الاختيار الشرطية", "en": "Hotz & Miller (1993) — CCP", "color": "violet", "body": "تجنب حل النموذج كاملًا بقلب احتمالات الاختيار المقدّرة — نظير «شبكة السياسة» في AlphaGo."},
        {"title": "التعلم المعزز العكسي", "en": "Inverse RL (Ng & Russell, 2000)", "color": "orange", "body": "استرجاع دالة المكافأة من السلوك = تقدير التفضيلات الهيكلية. مسح Rust & Rawat (2026) يربط الأدبيتين."},
        {"title": "الذكاء الاصطناعي كتقدير هيكلي", "en": "Igami (2020, Econometrics Journal)", "color": "rose", "body": "Deep Blue = دالة قيمة معايرة؛ Bonanza = NFXP؛ AlphaGo = CCP + محاكاة الاختيار الشرطي."},
        {"title": "حل النماذج بالـ RL", "en": "Deep RL for dynamic models", "color": "sky", "body": "استخدام تدرج السياسة والشبكات لحل نماذج غير متجانسة عالية الأبعاد بديلًا عن الشبكات النقطية."},
        {"title": "مسح حديث", "en": "Charpentier, Élie & Remlinger (2023, Computational Economics); Rawat (2026)", "color": "emerald", "body": "مراجعات لتطبيقات RL في الاقتصاد ونظرية الألعاب والتمويل والتقدير الهيكلي."},
    ], cols=3)

with tabs[2]:
    ui.section("مختبر متحرك: عامل يتعلم «الأجر الاحتياطي» بالتعلم المعزز", "Q-learning in the McCall job-search model", "pink")
    st.markdown("نموذج ماكول (1970): كل فترة يأتي عرض أجر $w$ من توزيع معروف؛ القبول يعطي $w/(1-\\beta)$ إلى الأبد، والرفض يعطي إعانة $c$ وعرضًا جديدًا. "
                "الحل الاقتصادي **أجر احتياطي** $w^*$: اقبل إن $w\\ge w^*$. هنا العامل **لا يعرف** التوزيع ويتعلم بالتجربة (Q-learning). شغّل الحركة لترى سياسته تقترب من $w^*$.")
    c_ben = st.slider("إعانة البطالة c", 10.0, 40.0, 25.0, 1.0, key="mc_c")

    @st.cache_data(show_spinner="تدريب العامل...")
    def mccall(c: float, beta: float = 0.95):
        w = np.linspace(10, 60, 26); p = np.exp(-0.5 * ((w - 32) / 9) ** 2); p /= p.sum()
        v = w / (1 - beta)
        for _ in range(2000):
            v = np.maximum(w / (1 - beta), c + beta * (p @ v))
        wstar = (1 - beta) * (c + beta * (p @ v))
        rng = np.random.default_rng(0)
        Q = np.zeros((len(w), 2)); Q[:, 1] = w / (1 - beta)
        snaps, keep, eps_n = [], [100, 300, 1000, 3000, 8000, 20000, 50000, 120000], 0
        s = rng.choice(len(w), p=p)
        for t in range(1, max(keep) + 1):
            eps = max(0.05, 1 - t / 60000)
            a = rng.integers(2) if rng.uniform() < eps else int(Q[s, 1] > Q[s, 0])
            lr = 0.5 / (1 + t / 5000)
            if a == 1:
                Q[s, 1] += lr * (w[s] / (1 - beta) - Q[s, 1])
                s = rng.choice(len(w), p=p)
            else:
                s2 = rng.choice(len(w), p=p)
                Q[s, 0] += lr * (c + beta * Q[s2].max() - Q[s, 0])
                s = s2
            if t in keep:
                snaps.append((t, (Q[:, 1] > Q[:, 0]).astype(float), Q[:, 0].copy()))
        return w, p, wstar, snaps

    w, p, wstar, snaps = mccall(c_ben)

    def mc_tr(i):
        t, acc, q0 = snaps[i]
        return [go.Bar(x=w, y=p / p.max(), marker_color="#e2e8f0", name="توزيع العروض (مُعيَّر)"),
                go.Scatter(x=w, y=acc, mode="lines+markers", line=dict(color=ui.COLORS["pink"], width=4, shape="hv"), name="سياسة العامل المتعلَّمة (1 = قبول)")]

    fig = go.Figure(data=mc_tr(0), frames=[go.Frame(data=mc_tr(i), name=str(snaps[i][0])) for i in range(len(snaps))])
    fig.add_vline(x=wstar, line_dash="dash", line_color=ui.COLORS["violet"], annotation_text=f"w* الأمثل = {wstar:.1f}")
    fig.update_xaxes(title="عرض الأجر w"); fig.update_yaxes(range=[-0.05, 1.15], title="قرار القبول")
    ui.style_fig(fig, "التعلم المعزز يكتشف قاعدة الأجر الاحتياطي")
    ui.add_play(fig, [s[0] for s in snaps], duration=900, prefix="عدد التجارب: ", redraw=True)
    ui.show(fig)
    ui.note("ارفع الإعانة $c$: يرتفع الأجر الاحتياطي الأمثل ويتعلم العامل أن يكون أكثر انتقائية — نتيجة اقتصادية كلاسيكية يعيد التعلم المعزز اكتشافها **دون معرفة التوزيع**. "
            "هذه هي الفكرة: RL = برمجة ديناميكية **بالعينات** بدل التوقعات المحسوبة.", "القراءة", "pink", "💼")

with tabs[3]:
    ui.section("البانديت متعدد الأذرع والتجارب التكيفية", "Multi-armed bandits & adaptive experiments", "orange")
    st.latex(r"\text{Thompson sampling: } \ \tilde p_k\sim\text{Beta}(1+S_k,\,1+F_k),\quad a=\arg\max_k\tilde p_k;\qquad "
             r"\text{Exploration sampling: } \ q_k\propto \pi_k(1-\pi_k)")
    st.markdown("أربع سياسات (معالجات) باحتمالات نجاح مجهولة. تُجرى التجربة على **موجات** من 100 مشارك؛ بعد كل موجة نحدّث الاحتمالات البعدية ونعيد التخصيص. "
                "شغّل الحركة لترى الحصص تتركز على المعالجة الأفضل (D).")

    @st.cache_data(show_spinner=False)
    def bandit(waves: int = 15, n: int = 100):
        rng = np.random.default_rng(3)
        true = np.array([0.30, 0.42, 0.48, 0.55]); K = 4
        S = np.zeros(K); F = np.zeros(K); shares_t, shares_e = [], []
        for _ in range(waves):
            draws = rng.beta(1 + S, 1 + F, (4000, K)); pi = np.bincount(draws.argmax(1), minlength=K) / 4000
            q = pi * (1 - pi); q = q / q.sum() if q.sum() > 0 else np.ones(K) / K
            shares_t.append(pi); shares_e.append(q)
            alloc = rng.multinomial(n, q)
            succ = rng.binomial(alloc, true); S += succ; F += alloc - succ
        return np.array(shares_t), np.array(shares_e), true

    sh_t, sh_e, true = bandit()
    arms = ["A (0.30)", "B (0.42)", "C (0.48)", "D (0.55)"]
    view = st.segmented_control("ماذا نعرض؟", ["احتمال أن تكون كل معالجة الأفضل", "حصص التخصيص (Exploration sampling)"],
                                default="احتمال أن تكون كل معالجة الأفضل", key="band_v")
    data = sh_t if view.startswith("احتمال") else sh_e
    fig = go.Figure(data=[go.Bar(x=arms, y=data[0], marker_color=ui.PLOT_COLORS[:4])],
                    frames=[go.Frame(data=[go.Bar(x=arms, y=data[k], marker_color=ui.PLOT_COLORS[:4])], name=str(k + 1)) for k in range(len(data))])
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    ui.style_fig(fig, view, height=420, legend_h=False)
    ui.add_play(fig, list(range(1, len(data) + 1)), duration=700, prefix="الموجة: ", redraw=True)
    ui.show(fig)
    ui.cards([
        {"title": "التجارب لاختيار السياسة", "en": "Kasy & Sautmann (2021, Econometrica)", "color": "orange",
         "body": "الهدف اختيار <b>أفضل سياسة في النهاية</b> لا تعظيم النتائج داخل العينة؛ «عيّنة الاستكشاف» تخصص بدالة مقعرة متزايدة في احتمال الأفضلية؛ تطبيق على ستة أساليب لتوظيف مرشدين زراعيين في الهند."},
        {"title": "الاستدلال بعد التكيف", "en": "Inference after adaptive designs", "color": "violet",
         "body": "التخصيص التكيفي يحيّز المقدّرات الساذجة ⇒ أوزان تصحيحية ومقدّرات AIPW التكيفية (Hadad et al., 2021)."},
        {"title": "البانديت السياقي", "en": "Contextual bandits", "color": "teal",
         "body": "التخصيص حسب خصائص الفرد = تعلم سياسات استهداف آنيًا؛ صلة بتعلم السياسات في صفحة الأثر غير المتجانس."},
    ])

with tabs[4]:
    ui.section("التعلم المعزز دون اتصال وتقييم السياسات", "Offline RL & off-policy evaluation", "sky")
    st.markdown(
        """
في الاقتصاد نادرًا ما نستطيع «التجريب» على الاقتصاد. **التعلم المعزز دون اتصال (Offline RL)** يتعلم سياسات من بيانات تاريخية جُمعت تحت سياسات سابقة —
وهي مسألة **استدلال سببي ديناميكي**:
- **أنظمة المعالجة الديناميكية (Dynamic treatment regimes)** في الإحصاء الحيوي: تسلسل قرارات مثلى (Murphy, 2003).
- **التقييم خارج السياسة** بمقدّرات مزدوجة المتانة لسلاسل القرارات.
- **الشرط الحاسم**: **الدعم** — لا يمكن تقييم أفعال لا توجد في البيانات التاريخية، و**عدم الإرباك المتسلسل**.
        """
    )
    st.latex(r"\hat V^{DR}(\pi)=\frac1n\sum_i\sum_{t}\gamma^t\Big[\rho_{i,1:t}\big(r_{it}-\hat Q(s_{it},a_{it})\big)+\rho_{i,1:t-1}\hat V(s_{it})\Big],\qquad \rho_{i,1:t}=\prod_{k\le t}\frac{\pi(a_{ik}\mid s_{ik})}{\pi_0(a_{ik}\mid s_{ik})}")

with tabs[5]:
    ui.section("تطبيقات التعلم المعزز في الاقتصاد والتمويل", "Economic applications", "violet")
    ui.cards([
        {"title": "التواطؤ الخوارزمي", "en": "Calvano, Calzolari, Denicolò & Pastorello (2020, AER)", "color": "rose",
         "body": "خوارزميات Q-learning تتنافس في احتكار قلة متكرر فتتعلم أسعارًا فوق تنافسية بعقوبات «انتقامية» دون تواصل — قضية سياسة منافسة حية."},
        {"title": "الاقتصادي الاصطناعي", "en": "Zheng et al. (2022, Science Advances) — AI Economist", "color": "violet",
         "body": "تعلم معزز ثنائي المستوى: أعوان يتعلمون العمل والادخار، ومخطط يتعلم جداول الضرائب التي توازن المساواة والإنتاجية."},
        {"title": "السياسة النقدية", "en": "RL-based policy rules", "color": "teal",
         "body": "دراسات في بنوك مركزية تتعلم قواعد فائدة بالتعلم المعزز في نماذج مقدّرة؛ تُقارن بقاعدة تايلور."},
        {"title": "تصميم المزادات والآليات", "en": "Dütting et al. (2019) Optimal auctions through deep learning", "color": "orange",
         "body": "شبكات تتعلم آليات مزاد قريبة من المثلى مع قيود التوافق التحفيزي."},
        {"title": "التحوط والتنفيذ", "en": "Deep hedging; optimal execution", "color": "sky",
         "body": "سياسات تحوط مع تكاليف وسيولة؛ تنفيذ الأوامر الكبيرة بأقل أثر سوقي؛ صناعة السوق."},
        {"title": "الأعوان المتعلمون في الماكرو", "en": "Learning agents in macro", "color": "emerald",
         "body": "استبدال التوقعات الرشيدة بتعلم تكيفي (Evans & Honkapohja) — والتعلم المعزز تعميم حديث له."},
    ], cols=3)

with tabs[6]:
    ui.section("محاذير التعلم المعزز في الاقتصاد", "Pitfalls", "rose")
    ui.ladder([
        ("سوء تحديد المكافأة", "Reward misspecification", "تعظيم ربح اسمي بدل الرفاهية أو المخاطر يولّد سياسات ضارة («قانون غودهارت» للمكافأة)."),
        ("فجوة المحاكاة والواقع", "Sim-to-real gap", "سياسة ممتازة في محاكي مقدّر قد تفشل في الاقتصاد الحقيقي إذا كان نموذج الانتقال خاطئًا."),
        ("نقد لوكاس", "Policy invariance", "الأعوان الحقيقيون يغيّرون سلوكهم ردًا على السياسة الجديدة — المحاكي الثابت لا يلتقط ذلك."),
        ("الأفعال خارج الدعم", "Off-support actions", "التعلم دون اتصال لا يستطيع تقييم ما لم يُجرَّب تاريخيًا."),
        ("الأمان والقيود", "Safety & constraints", "قيود الميزانية والعدالة والمخاطر يجب إدراجها صراحة (Constrained RL)."),
        ("عدم الاستقرار متعدد الأعوان", "Multi-agent instability", "التعلم المشترك قد يقود إلى تواطؤ أو دورات لا تتقارب."),
    ])
    ui.refs("Bellman (1957); Rust (1987); Hotz & Miller (1993); Watkins (1989); Mnih et al. (2015) Nature; Igami (2020); Kasy & Sautmann (2021) Econometrica; "
            "Calvano et al. (2020) AER; Zheng et al. (2022) Science Advances; Charpentier, Élie & Remlinger (2023) Computational Economics; Peña & Tsay (2021) ch. 8.")
