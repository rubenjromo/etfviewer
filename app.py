import streamlit as st
import pandas as pd
import numpy as np
import requests
import pycountry
import math
import plotly.express as px
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import time

# =============================
# CONFIG STREAMLIT
# =============================
st.set_page_config(
    page_title="ETF Portfolio Analyzer",
    layout="wide"
)

st.title("📊 ETF Portfolio Analyzer — Multiregión")

# --------------------------------------------
# USER INPUTS
# --------------------------------------------
st.sidebar.header("Configuración del Portafolio")

default_portfolio = {
    "SCHD": 25,
    "IDVO": 25,
    "CGDG": 50
}

tickers_text = st.sidebar.text_area(
    "Tickers y pesos (formato: TICKER: PESO, uno por línea)",
    "\n".join([f"{k}: {v}" for k,v in default_portfolio.items()])
)

portfolio_value = st.sidebar.number_input(
    "Valor total invertido (USD)",
    min_value=1.0,
    value=10000.0,
    step=100.0
)

FINNWORLDS_API_KEY = "45f81fc8790e6e351032baab1a264a533f8ebe74"
FINNWORLDS_BASE = "https://api.finnworlds.com/api/v1/etfholdings"

# Parse portfolio dictionary
portfolio = {}
for line in tickers_text.split("\n"):
    if ":" in line:
        t, w = line.split(":")
        try:
            portfolio[t.strip().upper()] = float(w)
        except:
            pass

if not portfolio:
    st.error("No se detectaron ETFs válidos.")
    st.stop()

# =============================
# NORMALIZAR PESOS
# =============================
total_w = sum(portfolio.values())
if abs(total_w - 100.0) < 1e-6:
    portfolio = {k: v/100.0 for k,v in portfolio.items()}
else:
    portfolio = {k: v/total_w for k,v in portfolio.items()}

st.sidebar.write("### Pesos normalizados")
st.sidebar.json(portfolio)

# =============================
# HELPERS
# =============================
def iso2_from_name_or_code(raw):
    if raw is None: return None
    s = str(raw).strip()
    if not s: return None
    s_up = s.upper()

    if len(s_up) == 2: return s_up
    if len(s_up) == 3:
        c = pycountry.countries.get(alpha_3=s_up)
        return c.alpha_2 if c else None

    c = pycountry.countries.get(name=s)
    if c: return c.alpha_2

    aliases = {
        "USA":"US","UNITED STATES":"US","UK":"GB",
        "GREAT BRITAIN":"GB","ENGLAND":"GB","HONG KONG":"HK",
        "CHINA":"CN","TAIWAN":"TW","SOUTH KOREA":"KR"
    }
    return aliases.get(s_up, None)

def normalize_percent_column(df):
    arr = df["percent_raw"].fillna(0).astype(float).values
    if arr.sum() == 0:
        df["percent_frac"] = 0
    else:
        df["percent_frac"] = arr / arr.sum()
    return df

# =============================
# LOAD ETF HOLDINGS
# =============================
@st.cache_data(show_spinner=False)
def load_etf_holdings(ticker):
    url = f"{FINNWORLDS_BASE}?key={FINNWORLDS_API_KEY}&ticker={ticker}"
    r = requests.get(url, timeout=20)

    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json().get("result", {})
    rows = []

    for out in data.get("output", []):
        for h in out.get("holdings", []):
            sec = h.get("investment_security", h)
            rows.append({
                "ticker": ticker,
                "name": sec.get("name", "Unknown"),
                "percent_raw": sec.get("percent_value", 0),
                "country_raw": sec.get("invested_country")
            })

    return pd.DataFrame(rows)

# =============================
# PROCESS ALL ETFs
# =============================
all_frames = []
etf_metrics_list = []

st.subheader("⏳ Descargando información…")

progress = st.progress(0)
step = 1 / len(portfolio)

for i, (ticker, weight) in enumerate(portfolio.items()):
    df = load_etf_holdings(ticker)

    if df.empty:
        continue

    df = normalize_percent_column(df)
    df["weighted_frac"] = df["percent_frac"] * weight
    df["etf_weight"] = weight
    all_frames.append(df)

    # ETF metrics (Yield, Price, CAGR)
    t = yf.Ticker(ticker)
    info = t.info

    yield_raw = info.get("yield") or info.get("dividendYield")
    price = info.get("regularMarketPrice")
    if price is None:
        price = t.history(period="1d")["Close"].iloc[-1]

    hist = t.history(period="5y")
    if len(hist) > 2:
        s = hist["Close"].iloc[0]
        e = hist["Close"].iloc[-1]
        years = (hist.index[-1] - hist.index[0]).days / 365
        cagr = (e/s)**(1/years) - 1
    else:
        cagr = None

    etf_metrics_list.append({
        "Ticker": ticker,
        "ETF Weight (frac)": weight,
        "Yield %": (yield_raw or 0)*100,
        "Price": price,
        "5Y CAGR %": (cagr or 0)*100
    })

    progress.progress((i+1)*step)

if not all_frames:
    st.error("No se pudo obtener holdings de ningún ETF.")
    st.stop()

combined = pd.concat(all_frames)

# =============================
# AGGREGATIONS
# =============================
company_agg = combined.groupby("name").agg(
    weighted_frac=("weighted_frac","sum"),
    count_etfs=("ticker","nunique")
).sort_values("weighted_frac", ascending=False)
company_agg["weighted_percent"] = company_agg["weighted_frac"] * 100

combined["iso2"] = combined["country_raw"].apply(iso2_from_name_or_code)

country_agg = combined.groupby("iso2").agg(
    weighted_frac=("weighted_frac","sum"),
    count_holdings=("name","count")
).sort_values("weighted_frac", ascending=False)
country_agg["weighted_percent"] = country_agg["weighted_frac"] * 100

# =============================
# REGION GROUPING
# =============================
def continent_from_iso2(iso2):
    if iso2 == "US": return "USA"
    if iso2 == "CA": return "Canada"

    europe = {"ES","FR","DE","IT","NL","SE","FI","NO","PL","PT","IE","GB","AT","CH","BE","DK","CZ"}
    asia = {"CN","JP","HK","KR","IN","TW","SG","ID","MY","TH"}
    america_rest = {"MX","BR","AR","CO","CL","PE","EC","UY","PA","CR","GT","DO"}
    africa = {"ZA","EG","MA","NG","TZ","KE"}
    oceania = {"AU","NZ"}

    if iso2 in europe: return "Europa"
    if iso2 in asia: return "ASIA"
    if iso2 in america_rest: return "Resto de América"
    if iso2 in africa: return "África"
    if iso2 in oceania: return "Oceanía"

    return "Otros"

combined["region_group"] = combined["iso2"].apply(continent_from_iso2)

region_agg = combined.groupby("region_group").agg(
    weighted_frac=("weighted_frac","sum")
).sort_values("weighted_frac", ascending=False)
region_agg["weighted_percent"] = region_agg["weighted_frac"] * 100

# =============================
# ETF METRICS TABLE
# =============================
st.subheader("📌 ETF Metrics (per ETF)")
st.dataframe(pd.DataFrame(etf_metrics_list), use_container_width=True)

# =============================
# PORTFOLIO SUMMARY
# =============================
st.subheader("📌 Portfolio Summary")

etf_df = pd.DataFrame(etf_metrics_list)

weighted_yield = (etf_df["Yield %"] * etf_df["ETF Weight (frac)"]).sum()
weighted_cagr = (etf_df["5Y CAGR %"] * etf_df["ETF Weight (frac)"]).sum()
annual_dividends = portfolio_value * weighted_yield / 100

col1, col2, col3 = st.columns(3)
col1.metric("Weighted Dividend Yield", f"{weighted_yield:.2f}%")
col2.metric("Weighted 5Y CAGR", f"{weighted_cagr:.2f}%")
col3.metric("Est. Annual Dividend Income", f"${annual_dividends:,.2f}")

# =============================
# TOP 100 HOLDINGS TABLE
# =============================
st.subheader("📌 Holdings ponderados (Top 100)")
show_company = company_agg.head(100)[["weighted_percent","count_etfs"]]
st.dataframe(show_company, use_container_width=True)

# =============================
# COUNTRY EXPOSURE TABLE
# =============================
st.subheader("📌 Exposición por país (Top 100)")
st.dataframe(country_agg.head(100)[["weighted_percent","count_holdings"]],
             use_container_width=True)

# =============================
# REGION BAR CHART
# =============================
st.subheader("📊 Exposición por Región / Continente")

fig, ax = plt.subplots(figsize=(10,5))
bars = ax.bar(region_agg.index, region_agg["weighted_percent"])
ax.set_ylabel("Exposición (%)")
ax.set_title("Exposición por Región")
plt.xticks(rotation=30)
st.pyplot(fig)

# =============================
# PIE CHART – TOP 10 HOLDINGS
# =============================
st.subheader("🥧 Top 10 Holdings + Others")

top10 = company_agg.head(10)
others = company_agg["weighted_percent"].iloc[10:].sum()

labels = list(top10.index) + ["Otros"]
values = list(top10["weighted_percent"]) + [others]

fig_pie = px.pie(
    names=labels,
    values=values,
    title="Top 10 Holdings",
    hole=0.3
)
st.plotly_chart(fig_pie, use_container_width=True)

# =============================
# PIE CHART – INCOME CONTRIBUTION
# =============================
st.subheader("💵 Annual Dividend Income Contribution (by ETF)")

income_df = pd.DataFrame({
    "Ticker": etf_df["Ticker"],
    "Income ($)": (etf_df["Yield %"] / 100) * etf_df["ETF Weight (frac)"] * portfolio_value
})

fig_inc = px.pie(
    income_df,
    names="Ticker",
    values="Income ($)",
    hole=0.3,
    title="Income Contribution by ETF"
)
st.plotly_chart(fig_inc, use_container_width=True)
