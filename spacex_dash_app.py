# spacex_dash_app.py — stable Dash layout (Bootstrap grid) for SpaceX Capstone
import os, math
import numpy as np
import pandas as pd
import plotly.express as px
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# ---------- Load data ----------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "launches_clean.csv")
df = pd.read_csv(DATA_PATH, parse_dates=["date_utc"])

# Coerce/clean
df["payload_mass_kg"] = pd.to_numeric(df["payload_mass_kg"], errors="coerce")
df["launch_success"]  = df["launch_success"].map({True:1, False:0, "True":1, "False":0, 1:1, 0:0}).astype("Int64")
for c in ["booster_version","launch_site","orbit"]:
    if c in df.columns:
        df[c] = df[c].astype(str)

# Per-launch table for mission-level stats
df_launch = (df[["flight_number","launch_site","launch_success"]]
             .drop_duplicates(subset=["flight_number"])
             .dropna(subset=["launch_site","launch_success"]))

# ---------- Figure 1: Success count by site ----------
success_counts_by_site = (df_launch[df_launch["launch_success"]==1]
                          .groupby("launch_site")["launch_success"]
                          .size().reset_index(name="success_count"))

fig1 = px.pie(
    success_counts_by_site,
    values="success_count", names="launch_site",
    title="(Dashboard 1) Launch Success Count — All Launch Sites",
    hole=0.25
)
fig1.update_traces(textposition="inside", textinfo="percent+label")
fig1.update_layout(height=420, margin=dict(t=60, l=40, r=20, b=30))

exp1 = ("This pie shows **successful launches** by site. Larger slices indicate sites with more successful missions.")

# ---------- Figure 2: Highest success-ratio site ----------
site_perf = (df_launch.groupby("launch_site")
             .agg(total=("launch_success","size"),
                  successes=("launch_success","sum"))
             .reset_index())
site_perf["success_ratio"] = site_perf["successes"] / site_perf["total"]
best_row = site_perf.sort_values("success_ratio", ascending=False).iloc[0]
best_site = best_row["launch_site"]
best_succ = int(best_row["successes"])
best_fail = int(best_row["total"] - best_row["successes"])
best_ratio_pct = round(best_row["success_ratio"]*100, 1)

fig2 = px.pie(
    pd.DataFrame({"Outcome":["Success","Failure"], "Count":[best_succ, best_fail]}),
    names="Outcome", values="Count",
    title=f"(Dashboard 2) Outcomes at Highest Success-Ratio Site — {best_site} ({best_ratio_pct}%)",
    hole=0.25, color="Outcome",
    color_discrete_map={"Success":"#2ca02c","Failure":"#d62728"}
)
fig2.update_traces(textposition="inside", textinfo="percent+label")
fig2.update_layout(height=420, margin=dict(t=60, l=40, r=20, b=30))

exp2 = (f"**{best_site}** has the highest success ratio (**{best_ratio_pct}%**). The pie splits its missions into success vs failure.")

# ---------- Figure 3: Scatter with payload slider ----------
df_scatter = df.dropna(subset=["payload_mass_kg","launch_success"]).copy()
df_scatter["Outcome"] = np.where(df_scatter["launch_success"]==1, "Success (1)", "Failure (0)")

if len(df_scatter):
    pmin = int(max(0, math.floor(df_scatter["payload_mass_kg"].min()/100.0)*100))
    pmax = int(math.ceil(df_scatter["payload_mass_kg"].max()/100.0)*100)
else:
    pmin, pmax = 0, 25000
if pmax < pmin:
    pmin, pmax = 0, 25000
initial_range = [pmin, pmax]

def make_scatter(filtered: pd.DataFrame):
    if filtered.empty:
        fig = px.scatter(title="(Dashboard 3) Payload vs Launch Outcome — No data in selected range")
        fig.update_layout(height=480, margin=dict(t=60,l=40,r=20,b=40))
        return fig
    fig = px.scatter(
        filtered, x="payload_mass_kg", y="Outcome",
        color="booster_version",
        hover_data=["flight_number","launch_site","orbit"],
        title="(Dashboard 3) Payload vs Launch Outcome — All Sites",
        labels={"payload_mass_kg":"Payload Mass (kg)", "Outcome":"Launch Outcome"},
        render_mode="auto"
    )
    fig.update_yaxes(categoryorder="array", categoryarray=["Failure (0)","Success (1)"])
    fig.update_traces(marker=dict(opacity=0.85, size=9))
    fig.update_layout(height=480, margin=dict(t=60,l=40,r=20,b=40))
    return fig

fig3 = make_scatter(df_scatter[(df_scatter["payload_mass_kg"]>=pmin) & (df_scatter["payload_mass_kg"]<=pmax)])

# ---------- App ----------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def slider_marks(lo, hi, n=6):
    if hi <= lo:
        return {int(lo): str(int(lo))}
    ticks = np.linspace(lo, hi, num=n)
    return {int(v): str(int(v)) for v in ticks}

app.layout = dbc.Container([
    html.H2("SpaceX Falcon 9 — Interactive Dashboard", className="my-3"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id="pie-success-all", figure=fig1, config={"displayModeBar": False}, style={"height":"460px"}),
            html.P(exp1, className="text-muted")
        ], md=6),
        dbc.Col([
            dcc.Graph(id="pie-best-site", figure=fig2, config={"displayModeBar": False}, style={"height":"460px"}),
            html.P(exp2, className="text-muted")
        ], md=6),
    ], align="start", className="g-3"),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id="scatter-outcome", figure=fig3, style={"height":"520px"}),
            html.Label("Payload Mass Range (kg):"),
            dcc.RangeSlider(
                id="payload-slider",
                min=pmin, max=pmax, step=100, value=initial_range,
                marks=slider_marks(pmin, pmax, n=6),
                tooltip={"placement":"bottom", "always_visible": False}
            ),
            html.Div(id="scatter-expl", className="mt-2 text-muted",
                     children="Use the slider to filter payload range and observe which booster versions perform best.")
        ], md=12),
    ], className="g-3 mb-4"),
], fluid=True, style={"maxWidth":"1200px"})

# ---------- Callbacks ----------
@app.callback(
    Output("scatter-outcome","figure"),
    Output("scatter-expl","children"),
    Input("payload-slider","value")
)
def update_scatter(payload_range):
    lo, hi = payload_range if isinstance(payload_range, (list, tuple)) else (pmin, pmax)
    subset = df_scatter[(df_scatter["payload_mass_kg"]>=lo) & (df_scatter["payload_mass_kg"]<=hi)]
    fig = make_scatter(subset)

    if subset.empty:
        return fig, "No data in this range."

    perf = (subset.groupby("booster_version")["launch_success"]
            .mean().sort_values(ascending=False))
    top_line = ""
    if len(perf) > 0:
        top_ver = perf.index[0]
        top_rate = round(perf.iloc[0]*100, 1)
        top_line = f"In **{lo:,}–{hi:,} kg**, booster version **{top_ver}** has the highest success rate (**{top_rate}%**)."

    total = int(len(subset))
    overall = round((int(subset['launch_success'].sum())/total)*100, 1) if total else 0.0
    expl = f"Filtered **{total}** payload rows; overall success rate **{overall}%**. " + top_line
    return fig, expl

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
