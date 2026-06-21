import streamlit as st
import pandas as pd
from pathlib import Path

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Aircraft Health Monitoring",
    page_icon="✈️",
    layout="wide"
)

# -------------------------------
# LOAD DATA
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_train.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# -------------------------------
# CUSTOM CSS
# -------------------------------

st.markdown("""
<style>

.stApp{
background: linear-gradient(
135deg,
#020617,
#0f172a,
#111827
);
color:white;
}

.main-title{
text-align:center;
font-size:55px;
font-weight:bold;
background: linear-gradient(
90deg,
#38bdf8,
#06b6d4,
#22c55e
);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
animation: glow 2s infinite alternate;
}

@keyframes glow{
from{
filter:drop-shadow(0 0 10px #38bdf8);
}
to{
filter:drop-shadow(0 0 25px #22c55e);
}
}

.subtitle{
text-align:center;
font-size:20px;
color:#cbd5e1;
margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------
# HEADER
# -------------------------------

st.markdown("""
<h1 class='main-title'>
✈ Aircraft Health Monitoring & Failure Prediction
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p class='subtitle'>
AI Powered Predictive Maintenance using Deep Learning
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# ENGINE SELECT
# -------------------------------

engine_list = sorted(df["engine_id"].unique())

engine_id = st.selectbox(
    "Select Engine",
    engine_list
)

engine_data = df[df["engine_id"] == engine_id]

latest = engine_data.iloc[len(engine_data)//2]

rul = int(latest["RUL"])

max_cycle = int(engine_data["max_cycle"].iloc[0])

health_score = round(
    (rul/max_cycle)*100
)

health_score = max(
    0,
    min(health_score,100)
)

failure_probability = 100 - health_score

# -------------------------------
# STATUS
# -------------------------------

if health_score > 70:
    status = "🟢 Healthy"
    recommendation = "No Maintenance Required"

elif health_score > 40:
    status = "🟡 Warning"
    recommendation = "Schedule Maintenance"

else:
    status = "🔴 Critical"
    recommendation = "Immediate Maintenance Required"

# -------------------------------
# KPI CARDS
# -------------------------------

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric("Health Score", f"{health_score}%")

with c2:
    st.metric("Remaining Life", f"{rul} Cycles")

with c3:
    st.metric("Failure Risk", f"{failure_probability}%")

with c4:
    st.metric("Status", status)

st.markdown("---")

st.header("📊 Executive Dashboard")

st.success(f"""
✈ Engine {engine_id}

Health Score: {health_score}%

Remaining Useful Life: {rul} Cycles

Failure Probability: {failure_probability}%

Status: {status}

Recommended Action:
{recommendation}
""")

# -------------------------------
# EXECUTIVE SUMMARY
# -------------------------------

st.header("📖 Executive Summary")

st.info(f"""
Selected Engine: {engine_id}

Health Score: {health_score}%

Remaining Useful Life: {rul} Cycles

Failure Probability: {failure_probability}%

Recommendation: {recommendation}
""")
st.header("🚦 Engine Status")

if health_score > 70:
    st.success("🟢 Engine Operating Normally")

elif health_score > 40:
    st.warning("🟡 Maintenance Should Be Planned")

else:
    st.error("🔴 Immediate Inspection Required")
# -------------------------------
# LAYMAN EXPLANATION
# -------------------------------

st.header("💡 Understanding This Dashboard")

st.info(f"""
Imagine an aircraft engine like a car engine.

As it runs, components gradually wear out.

This AI system analyzes sensor data and predicts:

✔ Engine Health

✔ Remaining Useful Life

✔ Failure Risk

✔ Maintenance Requirements

Current Prediction:

Health Score:
{health_score}%

Remaining Life:
{rul} cycles

Failure Probability:
{failure_probability}%
""")

# -------------------------------
# HEALTH BAR
# -------------------------------

st.header("💚 Engine Health")

st.progress(health_score)

st.header("❓ What is Remaining Useful Life (RUL)?")

st.info(f"""
Remaining Useful Life (RUL) means how many operating cycles
the engine can continue operating before maintenance becomes necessary.

Current Prediction:
{rul} cycles remaining.

Higher RUL = Healthier Engine

Lower RUL = Higher Failure Risk
""")

# -------------------------------
# HEALTH TREND
# -------------------------------

st.header("📈 Engine Health Trend")

trend = pd.DataFrame({
    "Cycle": engine_data["cycle"],
    "RUL": engine_data["RUL"]
})

st.line_chart(
    trend.set_index("Cycle")
)

# -------------------------------
# SENSOR DATA
# -------------------------------

st.header("📊 Sensor Monitoring")

sensor_df = pd.DataFrame({
    "Sensor":[f"Sensor {i}" for i in range(1,22)],
    "Value":[float(latest[f"sensor_{i}"]) for i in range(1,22)]
})

st.bar_chart(
    sensor_df.set_index("Sensor")
)

# -------------------------------
# MOST RISKY ENGINE
# -------------------------------

st.header("🚨 Fleet Risk Analysis")

fleet = (
    df.groupby("engine_id")["RUL"]
    .max()
    .reset_index()
)

fleet["Health"] = (
    fleet["RUL"] / 120 * 100
).clip(0,100)

risk_engine = fleet.sort_values(
    "Health"
).iloc[0]

st.error(f"""
Most Risky Engine:
{int(risk_engine['engine_id'])}

Health Score:
{risk_engine['Health']:.1f}%

Remaining Life:
{int(risk_engine['RUL'])} cycles
""")

st.header("💰 Business Impact")

st.write("""
Without Predictive Maintenance:

❌ Unexpected Failures

❌ Expensive Repairs

❌ Flight Delays

❌ Safety Risks

With AI-Based Monitoring:

✅ Early Fault Detection

✅ Reduced Maintenance Cost

✅ Improved Safety

✅ Better Resource Planning

✅ Increased Aircraft Availability
""")

# -------------------------------
# PROJECT DETAILS
# -------------------------------

st.header("🤖 Project Details")

st.write("""
Dataset:
NASA CMAPSS Aircraft Engine Dataset

Model:
LSTM Deep Learning

Purpose:
Remaining Useful Life Prediction

Applications:
• Predictive Maintenance
• Aviation Safety
• Failure Prevention
• Cost Optimization

RMSE:
30.46
""")

st.markdown("---")

st.markdown("""
### 👩‍💻 Developed By

Konduru Greeshma

Python | TensorFlow | LSTM | Streamlit
""")

