"""SalesPulse AI: demonstration sales analytics dashboard.
Run: streamlit run app.py
"""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="SalesPulse AI", page_icon="📊", layout="wide")
st.title("📊 SalesPulse AI")
st.caption("Sales intelligence and explainable short-term forecasts | Demonstration product")

REQUIRED = ["order_id", "date", "product", "category", "region", "quantity", "unit_price"]

@st.cache_data
def sample_data():
    return pd.read_csv(Path(__file__).parent / "sample_sales.csv")

def validate(raw):
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["quantity", "unit_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[REQUIRED].isna().any().any():
        raise ValueError("Required columns contain blank or invalid values.")
    if (df["quantity"] <= 0).any() or (df["unit_price"] < 0).any():
        raise ValueError("Quantity must be positive and unit_price nonnegative.")
    if df["order_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Order IDs cannot be empty.")
    # An order can have multiple product lines, so order_id need not be unique.
    df["revenue"] = df["quantity"] * df["unit_price"]
    df["date"] = df["date"].dt.normalize()
    return df

def daily_series(df):
    # Include zero-revenue days, essential for meaningful forecasts.
    s = df.groupby("date")["revenue"].sum().sort_index()
    return s.asfreq("D", fill_value=0)

def features(dates, start):
    d = pd.DatetimeIndex(dates)
    return pd.DataFrame({"day_index":(d-start).days.astype(float),
                         "day_of_week":d.dayofweek.astype(float),
                         "day_of_month":d.day.astype(float),
                         "month":d.month.astype(float),
                         "is_weekend":(d.dayofweek>=5).astype(float)})

def forecast_sales(s, horizon):
    """Time-based holdout evaluation; refit on all historical observations."""
    if len(s) < 45:
        return None, "At least 45 calendar days of data are needed to train and evaluate a forecast."
    split = max(30, int(len(s)*0.8))
    if len(s)-split < 7:
        split = len(s)-7
    X = features(s.index, s.index.min())
    model = RandomForestRegressor(n_estimators=160, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(X.iloc[:split], s.iloc[:split])
    pred_test = np.maximum(0, model.predict(X.iloc[split:]))
    mae = mean_absolute_error(s.iloc[split:], pred_test)
    baseline = np.repeat(s.iloc[:split].tail(7).mean(), len(s)-split)
    baseline_mae = mean_absolute_error(s.iloc[split:], baseline)
    model.fit(X, s)
    future_dates = pd.date_range(s.index.max()+pd.Timedelta(days=1), periods=horizon, freq="D")
    pred = np.maximum(0, model.predict(features(future_dates, s.index.min())))
    result = pd.DataFrame({"date":future_dates,"predicted_revenue":pred})
    return (result, mae, baseline_mae, len(s)-split), None

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload your own CSV", type=["csv"])
    st.download_button("Download sample CSV", sample_data().to_csv(index=False).encode(), "sample_sales.csv", "text/csv")
    st.markdown("**Required columns:** " + ", ".join(f"`{c}`" for c in REQUIRED))
    st.caption("For demonstrations, use synthetic or authorised non-sensitive business data.")
    horizon = st.slider("Forecast horizon (days)", 7, 30, 14)

try:
    raw = pd.read_csv(uploaded) if uploaded is not None else sample_data()
    df = validate(raw)
except Exception as exc:
    st.error(f"Could not load sales data: {exc}")
    st.stop()

st.info("Showing uploaded business data." if uploaded else "Showing SYNTHETIC sample data — not real customer sales.")
regions = sorted(df["region"].astype(str).unique())
categories = sorted(df["category"].astype(str).unique())
left, right = st.columns(2)
with left:
    selected_regions = st.multiselect("Regions", regions, default=regions)
with right:
    selected_categories = st.multiselect("Categories", categories, default=categories)
df = df[df["region"].astype(str).isin(selected_regions) & df["category"].astype(str).isin(selected_categories)]
if df.empty:
    st.warning("No rows match the selected filters.")
    st.stop()

min_date, max_date = df["date"].min().date(), df["date"].max().date()
range_value = st.date_input("Date range", value=(min_date,max_date), min_value=min_date, max_value=max_date)
if not isinstance(range_value, (list,tuple)) or len(range_value)!=2:
    st.info("Select both a start and end date.")
    st.stop()
df = df[df["date"].between(pd.Timestamp(range_value[0]),pd.Timestamp(range_value[1]))]
if df.empty:
    st.warning("No sales in this date range.")
    st.stop()

revenue = df["revenue"].sum()
orders = df["order_id"].nunique()
units = df["quantity"].sum()
k1,k2,k3,k4 = st.columns(4)
k1.metric("Total revenue",f"£{revenue:,.2f}")
k2.metric("Unique orders",f"{orders:,}")
k3.metric("Average order value",f"£{revenue/orders:,.2f}")
k4.metric("Units sold",f"{units:,.0f}")

tab1,tab2,tab3 = st.tabs(["📈 Sales trends", "🛍️ Business breakdown", "🤖 AI forecast"])
with tab1:
    cadence = st.selectbox("Aggregate by",["Daily","Weekly","Monthly"],index=1)
    rule = {"Daily":"D","Weekly":"W-MON","Monthly":"MS"}[cadence]
    trend = df.set_index("date")["revenue"].resample(rule).sum().reset_index()
    st.plotly_chart(px.line(trend,x="date",y="revenue",markers=True,title="Revenue over time",labels={"revenue":"Revenue (£)","date":"Date"}),use_container_width=True)
    st.dataframe(trend.rename(columns={"date":"Period","revenue":"Revenue (£)"}),use_container_width=True,hide_index=True)
with tab2:
    a,b = st.columns(2)
    with a:
        category = df.groupby("category",as_index=False)["revenue"].sum().sort_values("revenue",ascending=False)
        st.plotly_chart(px.bar(category,x="category",y="revenue",title="Revenue by category",labels={"revenue":"Revenue (£)"}),use_container_width=True)
    with b:
        region = df.groupby("region",as_index=False)["revenue"].sum().sort_values("revenue",ascending=False)
        st.plotly_chart(px.bar(region,x="region",y="revenue",title="Revenue by region",labels={"revenue":"Revenue (£)"}),use_container_width=True)
    products = df.groupby("product",as_index=False).agg(revenue=("revenue","sum"),units=("quantity","sum")).sort_values("revenue",ascending=False)
    st.subheader("Top products")
    st.dataframe(products.head(15),hide_index=True,use_container_width=True)
with tab3:
    st.caption("Forecast uses a Random Forest trained on calendar features, evaluated on the latest 20% of historical days. Short-term demonstration only; no guaranteed accuracy.")
    daily = daily_series(df)
    result, error = forecast_sales(daily,horizon)
    if error:
        st.warning(error)
    else:
        predicted, mae, baseline_mae, test_days = result
        a,b,c = st.columns(3)
        a.metric("Forecast total",f"£{predicted['predicted_revenue'].sum():,.0f}")
        b.metric("Holdout MAE",f"£{mae:,.0f}",help="Mean absolute error over the most recent held-out days; lower is better")
        c.metric("7-day mean baseline MAE",f"£{baseline_mae:,.0f}",help="A simple comparison forecast")
        if mae >= baseline_mae:
            st.warning("The ML model did not outperform the simple baseline on held-out data. Do not present it as superior.")
        chart = pd.concat([pd.DataFrame({"date":daily.index,"revenue":daily.values,"series":"Historical"}),
                           pd.DataFrame({"date":predicted["date"],"revenue":predicted["predicted_revenue"],"series":"Forecast"})],ignore_index=True)
        st.plotly_chart(px.line(chart,x="date",y="revenue",color="series",title="Historical and projected daily revenue",labels={"revenue":"Revenue (£)"}),use_container_width=True)
        st.caption(f"Evaluation window: last {test_days} days. Model does not include promotions, holidays, stockouts or external drivers.")
        st.download_button("Download forecast CSV",predicted.to_csv(index=False).encode(),"sales_forecast.csv","text/csv")

summary = df.groupby(["region","category"],as_index=False).agg(revenue=("revenue","sum"),units=("quantity","sum"))
st.divider()
st.download_button("⬇️ Download filtered sales summary",summary.to_csv(index=False).encode(),"sales_summary.csv","text/csv")
st.caption("© SalesPulse AI | Demo only. Verify business figures before operational decisions.")
