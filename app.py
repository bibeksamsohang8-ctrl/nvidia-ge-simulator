import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

MONTHS = 36
FEATURES = ["shock", "cap_cut", "lead", "elast", "stick"]
NAMES = {
    "shock": "AI demand shock",
    "cap_cut": "TSMC capacity cut",
    "lead": "Capacity lead time",
    "elast": "Demand elasticity",
    "stick": "Price stickiness",
}


def simulate(shock, cap_cut, lead, elast, stick, expansion=0.3):
    """Toy market: demand shock + fixed capacity + sticky posted price."""
    t = np.arange(MONTHS)
    d_scale = 1 + shock * np.clip(t / 3, 0, 1)
    start, target = 100 * (1 - cap_cut), 100 * (1 + expansion)
    cap = start + (target - start) * np.clip((t - lead) / 6, 0, 1)
    p_clear = (100 * d_scale / cap) ** (1 / elast)  # price where demand = capacity

    p = np.ones(MONTHS)
    dem, sold, backlog = np.zeros(MONTHS), np.zeros(MONTHS), np.zeros(MONTHS)
    for i in range(MONTHS):
        if i > 0:
            p[i] = p[i - 1] + (1 - stick) * (p_clear[i] - p[i - 1])
        dem[i] = 100 * d_scale[i] * p[i] ** (-elast)
        sold[i] = min(dem[i], cap[i])
        prev = backlog[i - 1] if i > 0 else 0
        backlog[i] = (prev + dem[i] - sold[i]) * 0.9  # 10% of queued orders cancel
    eff = 0.5 * (p + p_clear)  # price buyers really face (list + resale)
    cloud = 1 + 0.6 * (eff - 1)  # GPUs ~60% of cloud cost
    startup = 1 + 0.5 * (cloud - 1)  # compute ~50% of startup cost
    return pd.DataFrame({
        "month": t, "demand": dem, "capacity": cap, "sold": sold,
        "posted_price": p, "clearing_price": p_clear, "backlog": backlog,
        "wait_months": backlog / cap, "gpu_price": eff,
        "cloud_cost": cloud, "startup_cost": startup,
    })


@st.cache_resource
def train_model(n=600):
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "shock": rng.uniform(0.5, 3, n), "cap_cut": rng.uniform(0, 0.4, n),
        "lead": rng.uniform(3, 18, n), "elast": rng.uniform(0.4, 2, n),
        "stick": rng.uniform(0, 1, n),
    })
    y = np.array([simulate(**r).wait_months.max() for r in X.to_dict("records")])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestRegressor(n_estimators=200, random_state=0).fit(Xtr, ytr)
    return model, model.score(Xte, yte)


def line_fig(df, cols, labels, ytitle, lead):
    fig = go.Figure()
    for c, l in zip(cols, labels):
        fig.add_trace(go.Scatter(x=df.month, y=df[c], name=l, mode="lines"))
    fig.add_vline(x=lead, line_dash="dot", annotation_text="new capacity")
    fig.update_layout(xaxis_title="Month", yaxis_title=ytitle, height=380,
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=-0.25))
    return fig


st.set_page_config(page_title="NVIDIA GE & rationing", layout="wide")
st.title("NVIDIA: general equilibrium and supply chain rationing")
st.caption("Illustrative simulation, not calibrated to real NVIDIA data.")

with st.sidebar:
    st.header("Scenario")
    shock = st.slider("AI demand shock (+%)", 50, 300, 150, 10) / 100
    cap_cut = st.slider("TSMC capacity cut (%)", 0, 40, 20, 5) / 100
    lead = st.slider("Capacity lead time (months)", 3, 18, 12)
    elast = st.slider("Demand elasticity", 0.4, 2.0, 0.8, 0.1)
    stick = st.slider("Price stickiness (0 = flexible, 1 = fixed)", 0.0, 1.0, 0.85, 0.05)

df = simulate(shock, cap_cut, lead, elast, stick)
flex = simulate(shock, cap_cut, lead, elast, 0.0)
model, r2 = train_model()
pred = model.predict(pd.DataFrame([dict(shock=shock, cap_cut=cap_cut, lead=lead,
                                        elast=elast, stick=stick)]))[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Peak market-clearing price", f"{df.clearing_price.max():.1f}x")
c2.metric("Peak wait (months)", f"{df.wait_months.max():.1f}")
c3.metric("Peak cloud cost", f"{df.cloud_cost.max():.2f}x")
c4.metric("ML predicted peak wait", f"{pred:.1f}")

tab1, tab2, tab3 = st.tabs(["Ripple effects (Analyst 1)",
                            "Clearing failure and rationing (Analyst 2)", "ML insights"])

with tab1:
    st.plotly_chart(line_fig(df, ["gpu_price", "cloud_cost", "startup_cost"],
                             ["GPU price faced", "Cloud server cost", "Startup compute cost"],
                             "Index (start = 1)", lead), use_container_width=True)
    st.write("A shock at the foundry passes through GPU prices to cloud costs and then to startups. "
             "The effect shrinks at each step because each cost is only part of the next one.")

with tab2:
    st.plotly_chart(line_fig(df, ["demand", "capacity", "sold"],
                             ["Demand at list price", "Capacity", "Units sold"],
                             "Units per month", lead), use_container_width=True)
    st.plotly_chart(line_fig(df, ["posted_price", "clearing_price"],
                             ["Posted list price", "Market-clearing price"],
                             "Price index", lead), use_container_width=True)
    st.plotly_chart(line_fig(df, ["wait_months"], ["Queue wait"], "Months", lead),
                    use_container_width=True)
    st.write("The gap between posted and clearing price is why waitlists and allocation queues appear. "
             "Capacity cannot rise until the lead time has passed, so price alone cannot clear the market.")

with tab3:
    st.write(f"A random forest trained on 600 simulated scenarios predicts peak queue wait "
             f"(test R² = {r2:.2f}).")
    imp = pd.Series(model.feature_importances_, index=[NAMES[f] for f in FEATURES]).sort_values()
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Feature importance")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Peak wait if prices were fully flexible", f"{flex.wait_months.max():.1f} months",
              delta=f"{flex.wait_months.max() - df.wait_months.max():.1f} vs current", delta_color="off")
