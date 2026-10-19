"""Ink & Cobalt chart styling for Plotly, Altair and Matplotlib.

Use the same file in Streamlit apps and in Quarto posts so charts match everywhere.

    import brand_charts
    brand_charts.use()          # sets Plotly, Altair and Matplotlib defaults

In Streamlit, pass theme=None so your template wins over Streamlit's:
    st.plotly_chart(fig, theme=None)
    st.altair_chart(chart, theme=None)
"""

INK = "#121417"
COBALT = "#2F5BD3"
COBALT_DEEP = "#1E43A8"
MUTED = "#4A5263"
RULE = "#DFE2E8"
SURFACE = "#F3F4F6"
PAPER = "#FCFCFB"

CATEGORICAL = ["#2F5BD3", "#D08A1E", "#12806F", "#B8456A", "#8DB4F5", "#6B7280"]
SEQUENTIAL = ["#EEF2FC", "#D9E2F8", "#BFCEF3", "#9FB4EC", "#7C98E4",
              "#5A7EDC", "#2F5BD3", "#2549B0", "#1A3687", "#0F1E3D"]

BODY_FONT = "Mozilla Text, sans-serif"
TITLE_FONT = "Mozilla Headline, sans-serif"


def plotly_template():
    import plotly.graph_objects as go
    return go.layout.Template(
        layout=dict(
            font=dict(family=BODY_FONT, size=14, color=INK),
            title=dict(font=dict(family=TITLE_FONT, size=22, color=INK), x=0, xanchor="left"),
            paper_bgcolor=PAPER,
            plot_bgcolor=PAPER,
            colorway=CATEGORICAL,
            colorscale=dict(sequential=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)]),
            xaxis=dict(gridcolor=RULE, linecolor="#C9CED8", zeroline=False, ticks="outside",
                       tickcolor="#C9CED8", title=dict(font=dict(color=MUTED))),
            yaxis=dict(gridcolor=RULE, linecolor="rgba(0,0,0,0)", zeroline=False,
                       title=dict(font=dict(color=MUTED))),
            legend=dict(bgcolor="rgba(0,0,0,0)", title=dict(font=dict(color=MUTED))),
            hoverlabel=dict(bgcolor=INK, font=dict(family=BODY_FONT, color=PAPER), bordercolor=INK),
            margin=dict(l=56, r=24, t=56, b=48),
        )
    )


def altair_theme():
    return {
        "config": {
            "background": PAPER,
            "font": BODY_FONT,
            "title": {"font": TITLE_FONT, "fontSize": 20, "color": INK, "anchor": "start"},
            "axis": {"labelColor": MUTED, "titleColor": MUTED, "gridColor": RULE,
                     "domainColor": "#C9CED8", "tickColor": "#C9CED8", "labelFontSize": 12},
            "legend": {"labelColor": INK, "titleColor": MUTED},
            "range": {"category": CATEGORICAL, "ramp": SEQUENTIAL, "heatmap": SEQUENTIAL},
            "view": {"stroke": None},
            "mark": {"color": COBALT},
        }
    }


def matplotlib_rc():
    from cycler import cycler
    return {
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.edgecolor": "#C9CED8", "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=CATEGORICAL),
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "font.family": ["Mozilla Text", "DejaVu Sans"], "font.size": 11,
        "axes.titlesize": 15, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "image.cmap": "Blues",
    }


def use():
    """Apply the brand to every charting library that's installed."""
    try:
        import plotly.io as pio
        pio.templates["inkcobalt"] = plotly_template()
        pio.templates.default = "inkcobalt"
    except ImportError:
        pass
    try:
        import altair as alt
        if hasattr(alt, "theme") and hasattr(alt.theme, "register"):   # Altair 5.5+
            alt.theme.register("inkcobalt", enable=True)(lambda: alt.theme.ThemeConfig(altair_theme()))
        else:
            alt.themes.register("inkcobalt", altair_theme)
            alt.themes.enable("inkcobalt")
    except ImportError:
        pass
    try:
        import matplotlib as mpl
        mpl.rcParams.update(matplotlib_rc())
    except ImportError:
        pass
