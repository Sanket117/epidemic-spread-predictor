import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Epidemic Spread Predictor",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df       = pd.read_csv("data/processed/covid_merged.csv",        parse_dates=["date"])
    risk     = pd.read_csv("data/processed/risk_scores.csv",         parse_dates=["date"])
    snapshot = pd.read_csv("data/processed/latest_risk_snapshot.csv")
    return df, risk, snapshot

df, risk, snapshot = load_data()

ALL_COUNTRIES = sorted(df["country"].unique().tolist())

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Coronavirus_icon.svg/240px-Coronavirus_icon.svg.png", width=60)
st.sidebar.title("Epidemic Spread Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🌍 Global Overview", "📈 Country Analysis", "🔥 Hotspot Detection", "🗺️ Risk Map", "📊 Model Forecast"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")
selected_countries = st.sidebar.multiselect(
    "Select countries", ALL_COUNTRIES,
    default=["US", "India", "Brazil", "United Kingdom", "Germany"]
)
date_range = st.sidebar.date_input(
    "Date range",
    value=[df["date"].min().date(), df["date"].max().date()],
    min_value=df["date"].min().date(),
    max_value=df["date"].max().date()
)

# Apply date filter
if len(date_range) == 2:
    df_filtered = df[
        (df["date"] >= pd.Timestamp(date_range[0])) &
        (df["date"] <= pd.Timestamp(date_range[1]))
    ]
else:
    df_filtered = df.copy()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Global Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🌍 Global Overview":
    st.title("🌍 Global COVID-19 Overview")
    st.markdown("Tracking epidemic spread across 201 countries from March 2020 to March 2023.")

    # KPI cards
    total_cases  = int(df.groupby("country")["confirmed"].max().sum())
    total_deaths = int(df.groupby("country")["deaths"].max().sum())
    total_countries = df["country"].nunique()
    peak_day = df.groupby("date")["new_cases"].sum().idxmax()
    peak_val = int(df.groupby("date")["new_cases"].sum().max())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Confirmed Cases",  f"{total_cases:,}")
    c2.metric("Total Deaths",           f"{total_deaths:,}")
    c3.metric("Countries Tracked",      f"{total_countries}")
    c4.metric("Peak Daily Cases",       f"{peak_val:,}", f"{peak_day.date()}")

    st.markdown("---")

    # Global daily cases
    global_daily = df_filtered.groupby("date")["new_cases"].sum().reset_index()
    global_7day  = global_daily["new_cases"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=global_daily["date"], y=global_daily["new_cases"],
        name="Daily cases", marker_color="rgba(91,141,217,0.4)"
    ))
    fig.add_trace(go.Scatter(
        x=global_daily["date"], y=global_7day,
        name="7-day average", line=dict(color="#1a5fa8", width=2)
    ))
    fig.update_layout(
        title="Global daily new COVID-19 cases",
        xaxis_title="Date", yaxis_title="New cases",
        hovermode="x unified", height=400
    )
    st.plotly_chart(fig, width="stretch")

    # Top 10 countries
    col1, col2 = st.columns(2)

    with col1:
        top10 = (
            df.groupby("country")["confirmed"].max()
            .sort_values(ascending=False).head(10).reset_index()
        )
        fig2 = px.bar(
            top10, x="confirmed", y="country", orientation="h",
            title="Top 10 countries by total cases",
            color="confirmed", color_continuous_scale="Blues",
            labels={"confirmed": "Total cases", "country": ""}
        )
        fig2.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
        st.plotly_chart(fig2, width="stretch")

    with col2:
        top10d = (
            df.groupby("country")["deaths"].max()
            .sort_values(ascending=False).head(10).reset_index()
        )
        fig3 = px.bar(
            top10d, x="deaths", y="country", orientation="h",
            title="Top 10 countries by total deaths",
            color="deaths", color_continuous_scale="Reds",
            labels={"deaths": "Total deaths", "country": ""}
        )
        fig3.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
        st.plotly_chart(fig3, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Country Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Country Analysis":
    st.title("📈 Country-level Analysis")

    country = st.selectbox("Select a country", ALL_COUNTRIES, index=ALL_COUNTRIES.index("India"))
    subset  = df_filtered[df_filtered["country"] == country]

    if subset.empty:
        st.warning("No data for selected country in this date range.")
    else:
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Confirmed",   f"{int(subset['confirmed'].max()):,}")
        c2.metric("Total Deaths",      f"{int(subset['deaths'].max()):,}")
        c3.metric("Peak Daily Cases",  f"{int(subset['new_cases_7day'].max()):,}")
        vax = subset["total_vaccinations_per_hundred"].max()
        c4.metric("Vaccinations / 100 people", f"{vax:.1f}" if pd.notna(vax) else "N/A")

        st.markdown("---")

        # Cases + deaths dual axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=subset["date"], y=subset["new_cases_7day"],
            name="New cases (7-day avg)", fill="tozeroy",
            line=dict(color="#5b8dd9", width=1.5)
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=subset["date"], y=subset["new_deaths"],
            name="Daily deaths", line=dict(color="#e05c5c", width=1.5)
        ), secondary_y=True)
        fig.update_layout(
            title=f"Cases & deaths over time — {country}",
            hovermode="x unified", height=420
        )
        fig.update_yaxes(title_text="New cases", secondary_y=False)
        fig.update_yaxes(title_text="Deaths",    secondary_y=True)
        st.plotly_chart(fig, width="stretch")

        col1, col2 = st.columns(2)

        with col1:
            # Vaccination progress
            vax_data = subset.dropna(subset=["total_vaccinations_per_hundred"])
            if not vax_data.empty:
                fig4 = px.area(
                    vax_data, x="date",
                    y=["total_vaccinations_per_hundred", "people_fully_vaccinated_per_hundred"],
                    title="Vaccination progress",
                    labels={"value": "Per 100 people", "variable": ""},
                    color_discrete_map={
                        "total_vaccinations_per_hundred": "#4caf7d",
                        "people_fully_vaccinated_per_hundred": "#1a6e3c"
                    }
                )
                fig4.update_layout(height=320)
                st.plotly_chart(fig4, width="stretch")
            else:
                st.info("No vaccination data available for this country.")

        with col2:
            # Reproduction rate
            rt_data = subset.dropna(subset=["reproduction_rate"])
            if not rt_data.empty:
                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(
                    x=rt_data["date"], y=rt_data["reproduction_rate"],
                    fill="tozeroy", line=dict(color="#f0a830", width=1.5),
                    name="Rt"
                ))
                fig5.add_hline(y=1.0, line_dash="dash", line_color="red",
                               annotation_text="Rt = 1 (threshold)")
                fig5.update_layout(
                    title=f"Reproduction rate (Rt) — {country}",
                    yaxis_title="Rt", height=320
                )
                st.plotly_chart(fig5, width="stretch")
            else:
                st.info("No reproduction rate data available.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Hotspot Detection
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔥 Hotspot Detection":
    st.title("🔥 Hotspot Detection")
    st.markdown("Countries flagged as outbreak hotspots based on case growth, Rt, and burden metrics.")

    col1, col2, col3 = st.columns(3)
    growth_thresh = col1.slider("Growth rate threshold",  0.1, 2.0, 0.5, 0.1)
    cases_thresh  = col2.slider("Min cases per million",  10,  500, 100, 10)
    rt_thresh     = col3.slider("Min reproduction rate",  1.0, 2.0, 1.2, 0.1)

    hotspots = risk[
        (risk["case_growth_rate"] > growth_thresh) &
        (risk["cases_per_million"] > cases_thresh) &
        (risk["reproduction_rate"].fillna(rt_thresh) > rt_thresh)
    ].copy()

    if len(date_range) == 2:
        hotspots = hotspots[
            (hotspots["date"] >= pd.Timestamp(date_range[0])) &
            (hotspots["date"] <= pd.Timestamp(date_range[1]))
        ]

    st.markdown(f"**{len(hotspots):,} hotspot-weeks detected** in selected period")

    # Hotspot frequency per country
    freq = (
        hotspots.groupby("country").size()
        .sort_values(ascending=False).head(20).reset_index()
    )
    freq.columns = ["country", "hotspot_weeks"]

    fig = px.bar(
        freq, x="hotspot_weeks", y="country", orientation="h",
        title="Countries with most hotspot weeks",
        color="hotspot_weeks", color_continuous_scale="Oranges",
        labels={"hotspot_weeks": "Hotspot weeks", "country": ""}
    )
    fig.update_layout(coloraxis_showscale=False, height=500)
    st.plotly_chart(fig, width="stretch")

    # Hotspot timeline heatmap
    st.subheader("Hotspot intensity heatmap — top 15 countries")
    top15 = freq["country"].head(15).tolist()
    heat_data = (
        risk[risk["country"].isin(top15)]
        .pivot_table(index="country", columns="date", values="risk_score", aggfunc="mean")
    )
    fig2 = px.imshow(
        heat_data,
        color_continuous_scale="YlOrRd",
        labels=dict(color="Risk score"),
        title="Risk score heatmap over time",
        aspect="auto"
    )
    fig2.update_layout(height=450)
    st.plotly_chart(fig2, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Risk Map
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Risk Map":
    st.title("🗺️ Global Risk Map")

    # Date selector for the map
    available_dates = sorted(risk["date"].dt.date.unique())
    selected_date = st.select_slider(
        "Select date", options=available_dates, value=available_dates[-1]
    )

    map_data = risk[risk["date"] == pd.Timestamp(selected_date)].dropna(subset=["risk_score"])

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = px.choropleth(
            map_data,
            locations="country",
            locationmode="country names",
            color="risk_score",
            color_continuous_scale="YlOrRd",
            range_color=[0, 100],
            title=f"Global outbreak risk score — {selected_date}",
            labels={"risk_score": "Risk score"},
            hover_data={
                "risk_score": True,
                "risk_tier": True,
                "cases_per_million": ":.1f",
                "reproduction_rate": ":.2f",
            }
        )
        fig.update_layout(
            geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
            height=520,
            coloraxis_colorbar=dict(title="Risk score", tickvals=[0,25,50,75,100])
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("**Risk tier breakdown**")
        tier_counts = map_data["risk_tier"].value_counts()
        colors_map  = {"Critical": "#e05c5c", "High": "#f0a830",
                       "Moderate": "#5b8dd9", "Low": "#4caf7d"}
        for tier in ["Critical", "High", "Moderate", "Low"]:
            count = tier_counts.get(tier, 0)
            color = colors_map[tier]
            st.markdown(
                f"<div style='background:{color};padding:8px 12px;border-radius:6px;"
                f"margin-bottom:8px;color:white;font-weight:500'>"
                f"{tier}: {count} countries</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("**Top 5 highest risk**")
        top5 = map_data.nlargest(5, "risk_score")[["country", "risk_score"]]
        for _, row in top5.iterrows():
            st.markdown(f"**{row['country']}** — {row['risk_score']:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Model Forecast
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Forecast":
    st.title("📊 Outbreak Forecast")
    st.markdown("Prophet time-series forecast with configurable horizon.")

    col1, col2, col3 = st.columns(3)
    country      = col1.selectbox("Country", ALL_COUNTRIES, index=ALL_COUNTRIES.index("India"))
    forecast_days= col2.slider("Forecast horizon (days)", 14, 90, 60)
    test_days    = col3.slider("Test period (days)", 14, 60, 30)

    if st.button("▶ Run forecast", type="primary"):
        from prophet import Prophet
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        with st.spinner(f"Training Prophet model for {country}..."):
            subset = df[df["country"] == country].sort_values("date")
            subset = subset[subset["new_cases_7day"] >= 0].dropna(subset=["new_cases_7day"])

            prophet_df = subset[["date", "new_cases_7day"]].rename(
                columns={"date": "ds", "new_cases_7day": "y"}
            )

            split = len(prophet_df) - test_days
            train = prophet_df.iloc[:split]
            test  = prophet_df.iloc[split:]

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.3,
                seasonality_mode="multiplicative"
            )
            model.fit(train)

            future   = model.make_future_dataframe(periods=test_days + forecast_days)
            forecast = model.predict(future)
            forecast["yhat"]       = forecast["yhat"].clip(lower=0)
            forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
            forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

            test_fc = forecast.iloc[split: split + test_days]
            mae  = mean_absolute_error(test["y"].values, test_fc["yhat"].values)
            rmse = np.sqrt(mean_squared_error(test["y"].values, test_fc["yhat"].values))

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",  f"{mae:,.0f} cases/day")
        m2.metric("RMSE", f"{rmse:,.0f} cases/day")
        m3.metric("Forecast horizon", f"{forecast_days} days")

        # Plot
        split_date  = test["ds"].iloc[0]
        future_start= test["ds"].iloc[-1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=train["ds"], y=train["y"],
            name="Training data", line=dict(color="#5b8dd9", width=1.2)
        ))
        fig.add_trace(go.Scatter(
            x=test["ds"], y=test["y"],
            name="Actual (test)", line=dict(color="#2e7d32", width=2, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat"],
            name="Forecast", line=dict(color="#e05c5c", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
            y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(224,92,92,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% confidence"
        ))
        fig.add_trace(go.Scatter(
            x=[split_date, split_date],
            y=[0, train["y"].max() * 1.2],
            mode="lines",
            line=dict(color="gray", dash="dot", width=1.5),
            name="Train/test split",
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[future_start, future_start],
            y=[0, train["y"].max() * 1.2],
            mode="lines",
            line=dict(color="orange", dash="dot", width=1.5),
            name="Forecast start",
            showlegend=True
    ))
        fig.update_layout(
            title=f"Prophet forecast — {country}",
            xaxis_title="Date", yaxis_title="New cases (7-day avg)",
            hovermode="x unified", height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        st.subheader("Forecast values (next period)")
        future_only = forecast[forecast["ds"] > future_start][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        future_only.columns = ["Date", "Forecast", "Lower bound", "Upper bound"]
        future_only["Date"] = future_only["Date"].dt.date
        future_only[["Forecast", "Lower bound", "Upper bound"]] = \
            future_only[["Forecast", "Lower bound", "Upper bound"]].round(0).astype(int)
        st.dataframe(future_only.reset_index(drop=True), width="stretch")
