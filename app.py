import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import yfinance as yf

st.set_page_config(page_title="ETF Portfolio Analyzer PRO", layout="wide")

API_KEY = "45f81fc8790e6e351032baab1a264a533f8ebe74"
BASE_URL = "https://api.finnworlds.com/api/v1/etfholdings"

# ----------------------------------------------------------
# CONTINENT MAPPING
# ----------------------------------------------------------
continent_map = {
    "US": "USA",
    "CA": "Canada",
    "MX": "Rest of America",
    "AR": "Rest of America",
    "CL": "Rest of America",
    "BR": "Rest of America",
    "CO": "Rest of America",
    "UY": "Rest of America",
    "PE": "Rest of America",
    "BO": "Rest of America",
    "PY": "Rest of America",
    "EC": "Rest of America",

    # Europe
    "GB": "Europe", "FR": "Europe", "DE": "Europe", "NL": "Europe", "SE": "Europe",
    "ES": "Europe", "IT": "Europe", "FI": "Europe", "NO": "Europe",
    "DK": "Europe", "IE": "Europe", "PL": "Europe", "CH": "Europe",
    "PT": "Europe", "GR": "Europe", "AT": "Europe", "CZ": "Europe",

    # Asia
    "JP": "Asia", "CN": "Asia", "HK": "Asia", "SG": "Asia", "KR": "Asia",
    "TW": "Asia", "IN": "Asia", "MY": "Asia", "TH": "Asia",

    # Africa
    "ZA": "Africa", "EG": "Africa", "MA": "Africa", "NG": "Africa",

    # Oceania
    "AU": "Oceania", "NZ": "Oceania"
}

# ----------------------------------------------------------
# SCRAPE SECTORS FROM YAHOO
# ----------------------------------------------------------
@st.cache_data(ttl=3600)
def scrape_sector(ticker):
    """
    Scrape sector using Yahoo Finance key statistics.
    """
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/profile?p={ticker}"
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")
        spans = soup.find_all("span")
        for s in spans:
            if "Sector" in s.text:
                return s.find_next("span").text
    except:
        pass
    return "Unknown"


# ----------------------------------------------------------
# DOWNLOAD ETF HOLDINGS
# ----------------------------------------------------------
@st.cache_data(ttl=3600)
def get_etf_holdings(ticker):
    params = {"key": API_KEY, "ticker": ticker}
    try:
        r = requests.get(BASE_URL, params=params).json()
    except:
        return None

    try:
        raw = r["result"]["output"][0]["holdings"]
        df = pd.DataFrame([
            {
                "name": h["investment_security"]["name"],
                "pct": float(h["investment_security"]["percent_value"]),
                "country": h["investment_security"]["invested_country"],
                "ticker": h["investment_security"].get("cusip", "")
            }
            for h in raw
        ])
        return df
    except:
        return None


# ----------------------------------------------------------
# LAYOUT
# ----------------------------------------------------------
st.title("📊 ETF Portfolio Analyzer PRO")

st.sidebar.header("Configuración del Portafolio")

n = st.sidebar.number_input("Cantidad de ETFs", 1, 20, 3)

tickers = []
weights = []

for i in range(n):
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        t = st.text_input(f"ETF {i+1} (ej: SCHD)", key=f"t{i}")
    with col2:
        w = st.number_input(f"Peso {i+1}", 0.0, 1.0, 0.0, key=f"w{i}")
    tickers.append(t.upper())
    weights.append(w)

total_invested = st.sidebar.number_input("💰 Total invertido (USD)", 0.0, 1e12, 10000.0)

if sum(weights) == 0:
    st.warning("Asigna pesos a los ETFs.")
    st.stop()

weights = [w / sum(weights) for w in weights]


# ----------------------------------------------------------
# DOWNLOAD ALL HOLDINGS
# ----------------------------------------------------------
all_frames = []
for t, w in zip(tickers, weights):
    df = get_etf_holdings(t)
    if df is None:
        st.error(f"No se pudo obtener holdings de {t}")
        st.stop()

    df["weighted"] = df["pct"] * w
    df["etf"] = t
    all_frames.append(df)

full = pd.concat(all_frames, ignore_index=True)

# ----------------------------------------------------------
# AGGREGATION
# ----------------------------------------------------------
company = full.groupby("name", as_index=False).agg(
    weighted_percent=("weighted", "sum"),
    count_etfs=("etf", "nunique")
).sort_values("weighted_percent", ascending=False)

country = full.groupby("country", as_index=False).agg(
    weighted_percent=("weighted", "sum"),
    count_holdings=("name", "count")
).sort_values("weighted_percent", ascending=False)

# CONTINENT AGG
country["continent"] = country["country"].map(continent_map).fillna("Other")

continent = country.groupby("continent", as_index=False).agg(
    weighted_percent=("weighted_percent", "sum")
).sort_values("weighted_percent", ascending=False)

# SECTORS
unique_tickers = full["ticker"].dropna().unique().tolist()
sector_map = {t: scrape_sector(t) for t in unique_tickers}

full["sector"] = full["ticker"].map(sector_map).fillna("Unknown")

sectors = full.groupby("sector", as_index=False).agg(
    weighted_percent=("weighted", "sum")
).sort_values("weighted_percent", ascending=False)


# ----------------------------------------------------------
# TABS
# ----------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏢 Holdings",
    "🌍 Regiones / Continentes",
    "📦 Sectores",
    "💸 Dividendos",
    "📈 Proyección 20 años",
    "🆚 Comparación",
    "🔥 Heatmap Correlación"
])

# ================= TAB 1 ======================
with tab1:
    st.subheader("Top 100 Holdings Ponderados")
    st.dataframe(company.head(100))

    csv = company.to_csv(index=False).encode()
    st.download_button("⬇ Descargar CSV Completo de Holdings", csv, "holdings.csv")

# ================= TAB 2 ======================
with tab2:
    st.subheader("Exposición por Continente")
    fig = px.bar(continent, x="continent", y="weighted_percent")
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3 ======================
with tab3:
    st.subheader("Exposición por Sectores")
    fig = px.bar(sectors, x="sector", y="weighted_percent")
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 4 ======================
with tab4:
    st.subheader("Dividendos Proyectados")
    # Placeholder – You can integrate real dividend scraping here
    st.info("Funcionalidad de dividendos próximamente.")

# ================= TAB 5 ======================
with tab5:
    st.subheader("Proyección a 20 años")
    CAGR = 0.08
    years = np.arange(0, 21)
    projection = total_invested * (1 + CAGR) ** years

    fig = px.line(x=years, y=projection, labels={"x": "Años", "y": "Valor (USD)"})
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 6 ======================
with tab6:
    st.subheader("Comparación entre portafolios futuros")
    st.info("Pronto podrás guardar portafolios y compararlos.")

# ================= TAB 7 ======================
with tab7:
    st.subheader("Heatmap Correlación")
    st.info("Pronto incluiré correlaciones reales entre ETFs y sectores.")



