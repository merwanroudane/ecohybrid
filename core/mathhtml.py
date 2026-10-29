"""Inline LaTeX -> sanitizer-safe HTML (Unicode symbols, <sub>/<sup>, CSS fractions).

Streamlit's st.html strips MathML and SVG, and Streamlit's KaTeX only reaches
markdown, so inline formulas inside raw-HTML components are converted here.
Covers the subset of LaTeX used in the lecture pages.
"""

from __future__ import annotations

import re
from functools import lru_cache
from html import escape

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
    "rho": "ρ", "varrho": "ϱ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "ϕ", "varphi": "φ", "chi": "χ", "psi": "ψ",
    "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ",
    "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω", "ell": "ℓ",
}
SYMBOLS = {
    "mid": " ∣ ", "le": " ≤ ", "leq": " ≤ ", "ge": " ≥ ", "geq": " ≥ ", "neq": " ≠ ", "ne": " ≠ ", "approx": " ≈ ",
    "to": " → ", "rightarrow": " → ", "Rightarrow": " ⇒ ", "Longrightarrow": " ⟹ ", "leftarrow": " ← ", "Leftrightarrow": " ⇔ ",
    "infty": "∞", "in": " ∈ ", "notin": " ∉ ", "subset": " ⊂ ", "subseteq": " ⊆ ", "cup": " ∪ ", "cap": " ∩ ",
    "times": " × ", "cdot": "·", "cdots": "⋯", "dots": "…", "ldots": "…", "pm": " ± ", "partial": "∂", "nabla": "∇",
    "sum": "∑", "prod": "∏", "int": "∫", "propto": " ∝ ", "equiv": " ≡ ", "sim": " ∼ ", "perp": " ⊥ ", "forall": "∀",
    "exists": "∃", "varnothing": "∅", "emptyset": "∅", "lceil": "⌈", "rceil": "⌉", "lfloor": "⌊", "rfloor": "⌋",
    "langle": "⟨", "rangle": "⟩", "|": "‖", "{": "{", "}": "}", "%": "%", "_": "_", "&": "&", "odot": "⊙", "circ": "∘",
    "star": "⋆", "ast": "∗", "top": "⊤", "prime": "′", "setminus": " ∖ ", "lvert": "|", "rvert": "|", "vert": "|",
    "ll": " ≪ ", "gg": " ≫ ", "mapsto": " ↦ ", "iota": "ι", "hbar": "ħ", "Re": "ℜ", "Im": "ℑ",
    ",": "\u2009", ";": " ", ":": " ", "quad": "  ", "qquad": "    ", "!": "", " ": " ",
}
FUNCS = {"arg", "min", "max", "log", "ln", "exp", "sup", "inf", "lim", "det", "Pr", "sin", "cos", "tan", "tanh", "plim",
         "median", "Var", "Cov", "Med", "rank", "tr", "diag", "sign", "argmin", "argmax"}
ACCENTS = {"hat": "\u0302", "widehat": "\u0302", "tilde": "\u0303", "widetilde": "\u0303", "bar": "\u0304",
           "dot": "\u0307", "ddot": "\u0308", "vec": "\u20d7", "check": "\u030c"}
BB = {"E": "𝔼", "R": "ℝ", "N": "ℕ", "P": "ℙ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "1": "𝟙", "V": "𝕍", "I": "𝕀"}
CAL = {"A": "𝒜", "B": "ℬ", "C": "𝒞", "D": "𝒟", "E": "ℰ", "F": "ℱ", "G": "𝒢", "H": "ℋ", "I": "ℐ", "J": "𝒥", "K": "𝒦",
       "L": "ℒ", "M": "ℳ", "N": "𝒩", "O": "𝒪", "P": "𝒫", "Q": "𝒬", "R": "ℛ", "S": "𝒮", "T": "𝒯", "U": "𝒰", "V": "𝒱",
       "W": "𝒲", "X": "𝒳", "Y": "𝒴", "Z": "𝒵"}
IGNORE = {"left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "displaystyle", "textstyle",
          "limits", "nolimits", "scriptstyle"}

_TOKEN = re.compile(r"\\([A-Za-z]+|.)|([{}^_])|(\s+)|(.)", re.S)


def _tokens(s: str) -> list[tuple[str, str]]:
    out = []
    for cmd, sym, ws, ch in _TOKEN.findall(s):
        if cmd:
            out.append(("cmd", cmd))
        elif sym:
            out.append((sym, sym))
        elif ws:
            continue
        else:
            out.append(("ch", ch))
    return out


class _P:
    def __init__(self, toks):
        self.t, self.i = toks, 0
        self.tight = False          # inside a sub/superscript: no spacing around operators

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def group(self) -> str:
        """One argument: a {...} group or a single token."""
        tok = self.peek()
        if tok is None:
            return ""
        if tok[0] == "{":
            self.take()
            out = self.seq(until="}")
            self.take()
            return out
        return self.atom()

    def raw_group(self) -> str:
        tok = self.peek()
        if tok and tok[0] == "{":
            self.take()
            depth, buf = 1, []
            while self.peek() is not None:
                k, v = self.take()
                if k == "{":
                    depth += 1
                elif k == "}":
                    depth -= 1
                    if depth == 0:
                        break
                buf.append(("\\" + v) if k == "cmd" else v)
            return "".join(buf)
        k, v = self.take() if tok else ("ch", "")
        return v

    def seq(self, until: str | None = None) -> str:
        out = []
        while self.peek() is not None and not (until and self.peek()[0] == until):
            out.append(self.atom())
        return "".join(out)

    def atom(self) -> str:
        k, v = self.take()
        base = self._base(k, v)
        # scripts
        while self.peek() is not None and self.peek()[0] in ("^", "_"):
            s = self.take()[0]
            saved, self.tight = self.tight, True
            arg = self.group()
            self.tight = saved
            base += f"<sup>{arg}</sup>" if s == "^" else f"<sub>{arg}</sub>"
        return base

    def _accent_arg(self) -> str:
        """The accented symbol only; any ^/_ that follows belongs to the accented result."""
        tok = self.peek()
        if tok is None:
            return ""
        if tok[0] == "{":
            return self.group()
        return self._base(*self.take())

    def _base(self, k: str, v: str) -> str:
        out = self._render(k, v)
        return out.strip() if self.tight else out

    def _render(self, k: str, v: str) -> str:
        if k == "ch":
            if v.isalpha():
                return f"<i>{escape(v)}</i>"
            if v == "'":
                return "′"
            if v in "=<>":
                return f" {escape(v)} "
            if v in "+":
                return " + "
            if v == "-":
                return " − "
            return escape(v)
        if k == "{":
            out = self.seq(until="}")
            self.take()
            return out
        if k in ("^", "_"):
            return ""
        if k == "}":
            return ""
        # commands
        if v in GREEK:
            return f"<i>{GREEK[v]}</i>" if v[0].islower() else GREEK[v]
        if v in SYMBOLS:
            return SYMBOLS[v]
        if v in FUNCS:
            return f'<span class="mf">{v}</span>'
        if v in IGNORE:
            return ""
        if v in ACCENTS:
            arg = self._accent_arg()
            plain = re.sub(r"<[^>]+>", "", arg)
            if len(plain) == 1:
                return arg.replace(plain, plain + ACCENTS[v], 1)
            return f'<span class="mover">{arg}</span>'
        if v in ("overline",):
            return f'<span class="mover">{self.group()}</span>'
        if v == "mathbb":
            raw = self.raw_group()
            return "".join(BB.get(c, c) for c in raw)
        if v == "mathcal":
            raw = self.raw_group()
            return "".join(CAL.get(c, c) for c in raw)
        if v in ("mathbf", "boldsymbol", "bm"):
            return f"<b>{self.group()}</b>"
        if v in ("mathrm", "operatorname", "textrm", "mathsf"):
            return f'<span class="mf">{escape(self.raw_group())}</span>'
        if v in ("text", "textit", "mbox"):
            return f'<span class="mt">{escape(self.raw_group())}</span>'
        if v in ("frac", "tfrac", "dfrac", "cfrac"):
            num, den = self.group(), self.group()
            return f'<span class="mfrac"><span>{num}</span><span>{den}</span></span>'
        if v == "binom":
            a, b = self.group(), self.group()
            return f"({a} {b})"
        if v == "sqrt":
            arg = self.group()
            return f'√<span class="mover">{arg}</span>' if len(re.sub(r"<[^>]+>", "", arg)) > 1 else f"√{arg}"
        if v in ("xrightarrow", "xleftarrow"):
            arg = self.group()
            return f" →<sup>{arg}</sup> " if v == "xrightarrow" else f" ←<sup>{arg}</sup> "
        if v in ("overset", "stackrel"):
            top, base = self.group(), self.group()
            return f"{base}<sup>{top}</sup>"
        if v in ("underbrace", "overbrace", "underline"):
            return self.group()
        return escape("\\" + v)


@lru_cache(maxsize=4096)
def tex_to_html(tex: str) -> str:
    try:
        body = _P(_tokens(tex)).seq()
    except Exception:  # noqa: BLE001
        return escape(tex)
    return re.sub(r"\s{2,}", " ", body).strip()
