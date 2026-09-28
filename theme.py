"""
Visual styling: colour that carries meaning rather than decoration.

Every colour maps to something you need to judge quickly — a grade, a data
status, a verdict, a direction. Green and red carry the usual trading sense,
and are always paired with a word or symbol so the meaning survives for anyone
who can't rely on colour alone.
"""

GREEN = "#00E5A0"
RED = "#FF5C7A"
AMBER = "#FFB65C"
BLUE = "#5CC8FF"
GREY = "#8A93AB"
PURPLE = "#B58CFF"

GRADE_COLOURS = {"A+": GREEN, "B": AMBER, "C": GREY, "—": GREY}
DIRECTION_COLOURS = {"Long": GREEN, "Short": RED}

CSS = """
<style>
/* Headline numbers: bigger, tighter, easier to scan on a phone. */
[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; }
[data-testid="stMetricLabel"] { opacity: .75; font-size: .78rem;
    text-transform: uppercase; letter-spacing: .06em; }

/* Metric cards with a subtle accent edge. */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.015));
    border: 1px solid rgba(255,255,255,.08);
    border-left: 3px solid #00E5A0;
    border-radius: 12px; padding: .7rem .9rem;
}

/* Bordered containers used for trade plans and reviews. */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px; border-color: rgba(255,255,255,.10) !important;
}

h1 { font-weight: 800; letter-spacing: -.02em; }
h5 { color: #9FB0D0; text-transform: uppercase; letter-spacing: .07em;
     font-size: .82rem; margin-top: 1.1rem; }

/* Tabs: clearer which one is active. */
.stTabs [data-baseweb="tab"] { font-weight: 600; }
.stTabs [aria-selected="true"] { color: #00E5A0 !important; }

.stButton > button { border-radius: 10px; font-weight: 600; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

.pill { display: inline-block; padding: .16rem .6rem; border-radius: 999px;
        font-size: .76rem; font-weight: 700; letter-spacing: .03em; }
</style>
"""


def pill(text: str, colour: str) -> str:
    """A coloured badge. Always contains its own text, so colour is never the
    only thing carrying the meaning."""
    return (f'<span class="pill" style="background:{colour}22;color:{colour};'
            f'border:1px solid {colour}55">{text}</span>')


def grade_pill(grade: str) -> str:
    return pill(grade, GRADE_COLOURS.get(grade, GREY))


def gauge(value: float, low_label: str, high_label: str, colour: str) -> str:
    """A simple 0-100 bar, for indices like Fear & Greed or Altcoin Season."""
    v = max(0.0, min(100.0, float(value)))
    return (
        f'<div style="margin:.35rem 0 .1rem">'
        f'<div style="height:10px;border-radius:999px;background:rgba(255,255,255,.08);'
        f'position:relative">'
        f'<div style="position:absolute;left:0;top:0;bottom:0;width:{v}%;'
        f'border-radius:999px;background:{colour}"></div></div>'
        f'<div style="display:flex;justify-content:space-between;font-size:.72rem;'
        f'opacity:.6;margin-top:.25rem"><span>{low_label}</span>'
        f'<span>{high_label}</span></div></div>')
