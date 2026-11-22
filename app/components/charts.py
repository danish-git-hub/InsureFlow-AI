"""Plotly charts with the InsureFlow brand applied consistently."""

from __future__ import annotations

import plotly.graph_objects as go

NAVY, NAVY2, BLUE, TEAL = "#102A43", "#243B53", "#2563EB", "#0F766E"
GREEN, AMBER, RED, MUTED = "#15803D", "#D97706", "#DC2626", "#627D98"
SERIES = [TEAL, BLUE, NAVY2, "#14B8A6", "#60A5FA", AMBER, "#7C3AED", MUTED]

STATUS_COLORS = {
    "ELIGIBLE_FOR_HUMAN_REVIEW": GREEN,
    "REVIEW_REQUIRED": AMBER,
    "ADDITIONAL_INFORMATION_REQUIRED": BLUE,
    "POLICY_NOT_FOUND": RED,
    "SUBMITTED": MUTED,
    "HUMAN_APPROVED": TEAL,
    "HUMAN_REJECTED": NAVY2,
    "HUMAN_INFO_REQUESTED": "#60A5FA",
}
RISK_COLORS = {"LOW": GREEN, "MEDIUM": AMBER, "HIGH": RED, "UNASSESSED": "#CBD5E1"}


def pretty(label: str) -> str:
    return label.replace("_", " ").title() if label and label.isupper() else label


def style(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=12, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=NAVY, size=12), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", font_color=NAVY, bordercolor="#D9E2EC"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#D9E2EC", tickfont_color=MUTED)
    fig.update_yaxes(gridcolor="#E9EEF4", zeroline=False, tickfont_color=MUTED)
    return fig


def donut(counts: dict[str, int], colors: dict[str, str] | None = None, height: int = 320) -> go.Figure:
    labels = list(counts)
    fig = go.Figure(go.Pie(
        labels=[pretty(x) for x in labels], values=list(counts.values()), hole=0.62, sort=False,
        marker=dict(colors=[(colors or {}).get(x, SERIES[i % len(SERIES)]) for i, x in enumerate(labels)],
                    line=dict(color="white", width=2)),
        textinfo="value", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{sum(counts.values())}</b><br><span style='font-size:11px'>claims</span>",
                       showarrow=False, font=dict(size=20, color=NAVY))
    style(fig, height)
    fig.update_layout(legend=dict(orientation="v", y=0.5, yanchor="middle", x=1.02))
    return fig


def hbar(counts: dict[str, int | float], color: str = TEAL, height: int = 320, fmt: str = "%{x}") -> go.Figure:
    items = sorted(counts.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[pretty(k) for k, _ in items], orientation="h",
                           marker_color=color, texttemplate=fmt, textposition="outside",
                           hovertemplate="%{y}: %{x}<extra></extra>"))
    return style(fig, height, legend=False)


def monthly_trend(rows: list[dict], height: int = 320) -> go.Figure:
    months = [r["month"] for r in rows]
    fig = go.Figure()
    fig.add_bar(x=months, y=[r["claims"] for r in rows], name="Claims", marker_color=TEAL,
                hovertemplate="%{x}: %{y} claims<extra></extra>")
    fig.add_scatter(x=months, y=[r["amount"] for r in rows], name="Claim amount (INR)", yaxis="y2", mode="lines+markers",
                    line=dict(color=BLUE, width=3), hovertemplate="%{x}: INR %{y:,.0f}<extra></extra>")
    fig.update_layout(yaxis=dict(title="Claims"), yaxis2=dict(title="INR", overlaying="y", side="right",
                                                              showgrid=False, tickformat=".2s"))
    return style(fig, height)


def amount_area(rows: list[dict], height: int = 320) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[r["month"] for r in rows], y=[r["amount"] for r in rows], mode="lines+markers",
                               fill="tozeroy", line=dict(color=TEAL, width=3), fillcolor="rgba(15,118,110,.15)",
                               hovertemplate="%{x}: INR %{y:,.0f}<extra></extra>"))
    fig.update_yaxes(tickformat=".2s")
    return style(fig, height, legend=False)


def gauge(score: float, height: int = 220) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(suffix=" / 100", font=dict(size=26, color=NAVY)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED), bar=dict(color=NAVY, thickness=0.25),
                   steps=[dict(range=[0, 25], color="#DCFCE7"), dict(range=[25, 50], color="#FEF3C7"),
                          dict(range=[50, 100], color="#FEE2E2")]),
    ))
    return style(fig, height, legend=False)
