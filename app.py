# app.py — Professional Streamlit ETF Portfolio Analyzer
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pycountry
import math
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import time

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="ETF Portfolio Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("ETF Portfolio Analyzer")

# --------------------------
# Sidebar - Inputs
# --------------------------
st.sidebar.header("1) Configura tu portafolio")

st.sidebar.markdown("""
Ingrese los ETFs y pesos:


Los pesos pueden sumar 100 o cualquier número — se normaliza.
""")

default_text = "SCHD: 25\nVOO: 75"
tickers_text = st.sidebar.text_area("Tickers y Pesos", value=default_text, height=140)
portfolio_value = st.sidebar.number_input("Valor total invertido (USD)", min_value=1.0, value=10000.0, step=100.0)

st.sidebar.markdown("---")
FINNWORLDS_API_KEY = "45f81fc8790e6e351032baab1a264a533f8ebe74"
FINNWORLDS_BASE = "https://api.finnworlds.com/api/v1/etfholdings"


# --------------------------
# Parse Portfolio
# --------------------------
def parse_portfolio(text):
    out = {}
    for line in text.strip().splitlines():
        if ":" in line:
            t, w = line.split(":", 1)
            try:
                out[t.strip().upper()] = float(w.strip())
            except:
                continue
    return out

raw_portfolio = parse_portfolio(tickers_text)

if not raw_portfolio:
    st.sidebar.error("Error: No se detectaron tickers válidos.")
    st.stop()

# Normalize weights
total_w = sum(raw_portfolio.values())
if abs(total_w - 100.0) < 1e-6:
    portfolio = {k: v/100.0 for k, v in raw_portfolio.items()}
else:
    s = sum(raw_portfolio.values())
    portfolio = {k: v/s for k, v in raw_portfolio.items()}

st.sidebar.subheader("Pesos Normalizados")
st.sidebar.json({k: round(v, 6) for k, v in portfolio.items()})


# --------------------------
# Helpers
# --------------------------
@st.cache_data(ttl=3600)
def iso2_from_name_or_code(raw):
    if raw is None: return None
    s = str(raw).strip()
    if not s: return None
    s_up = s.upper()

    # Code direct
    if len(s_up) == 2: return s_up

    # 3-letter code
    if len(s_up) == 3:
        c = pycountry.countries.get(alpha_3=s_up)
        if c: return c.alpha_2

    # Name
    try:
        c = pycountry.countries.get(name=s)
        if c: return c.alpha_2
    except:
        pass

    # Aliases
    aliases = {
        "USA": "US", "UNITED STATES": "US",
        "UK": "GB", "GREAT BRITAIN": "GB",
        "HONG KONG": "HK", "CHINA": "CN", "TAIWAN": "TW"
    }
    return aliases.get(s_up, None)


def normalize_percent_column(df):
    arr = df["percent_raw"].fillna(0).astype(float).values
    if arr.sum() == 0:
        df["percent_frac"] = 0.0
    else:
        df["percent_frac"] = arr / arr.sum()
    return df


@st.cache_data(ttl=3600)
def load_etf_holdings(ticker):
    url = f"{FINNWORLDS_BASE}?key={FINNWORLDS_API_KEY}&ticker={ticker}"
    try:
        r = requests.get(url, timeout=20)
    except:
        return pd.DataFrame()

    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json().get("result", {})
    outputs = data.get("output") or []
    if isinstance(outputs, dict):
        outputs = [outputs]

    rows = []
    for out in outputs:
        holdings = out.get("holdings") or out.get("positions") or []
        for h in holdings:
            sec = h.get("investment_security", h)
            name = sec.get("name") or sec.get("title") or "Unknown"
            pv = float(sec.get("percent_value") or sec.get("percent") or 0)
            country = sec.get("invested_country") or sec.get("country") or None

            rows.append({
                "ticker": ticker,
                "name": name,
                "percent_raw": pv,
                "country_raw": country
            })

    return pd.DataFrame(rows)


# --------------------------
# Main Button
# --------------------------
st.markdown("## 🔄 Ejecutar análisis")

if st.button("Calcular"):
    with st.spinner("Cargando información..."):
        
        all_frames = []
        etf_metrics = []

        progress = st.empty()
        n = len(portfolio)
        i = 0

        for ticker, weight in portfolio.items():
            progress.text(f"Descargando {ticker} ({i+1}/{n})...")

            df = load_etf_holdings(ticker)
            if df.empty:
                st.warning(f"No se pudo obtener holdings para {ticker}")
                i += 1
                continue

            df = normalize_percent_column(df)
            df["weighted_frac"] = df["percent_frac"] * weight
            df["etf_weight"] = weight
            all_frames.append(df)

            # yfinance metrics
            t = yf.Ticker(ticker)
            info = t.info or {}
            name = info.get("shortName") or info.get("longName") or ticker
            price = info.get("regularMarketPrice") or np.nan
            if price is None:
                try:
                    price = t.history(period="1d")["Close"].iloc[-1]
                except:
                    price = np.nan

            yield_raw = info.get("yield") or info.get("dividendYield") or 0.0
            yield_pct = (yield_raw or 0.0) * 100

            # 5y CAGR
            try:
                hist = t.history(period="5y", auto_adjust=False)
                if len(hist) >= 2:
                    s = hist["Close"].iloc[0]
                    e = hist["Close"].iloc[-1]
                    years = (hist.index[-1] - hist.index[0]).days / 365.25
                    cagr = (e/s)**(1/years) - 1
                else:
                    cagr = np.nan
            except:
                cagr = np.nan

            etf_metrics.append({
                "Ticker": ticker,
                "Name": name,
                "Price": price,
                "Yield %": yield_pct,
                "5Y CAGR %": (cagr * 100) if pd.notna(cagr) else np.nan,
                "ETF Weight (frac)": weight
            })

            i += 1
            time.sleep(0.25)

        if not all_frames:
            st.error("Error: No se pudo descargar información de ningún ETF.")
            st.stop()

        combined = pd.concat(all_frames, ignore_index=True)
        combined["iso2"] = combined["country_raw"].apply(iso2_from_name_or_code)

        # Aggregations
        company_agg = (combined.groupby("name")
                       .agg(weighted_frac=("weighted_frac","sum"),
                            count_etfs=("ticker","nunique"))
                       .reset_index()
                       .sort_values("weighted_frac", ascending=False))
        company_agg["weighted_percent"] = company_agg["weighted_frac"] * 100

        country_agg = (combined.groupby("iso2")
                       .agg(weighted_frac=("weighted_frac","sum"),
                            count_holdings=("name","count"))
                       .reset_index()
                       .sort_values("weighted_frac", ascending=False))
        country_agg["weighted_percent"] = country_agg["weighted_frac"] * 100

        # Continents
        def continent_from_iso2(iso2):
            if not iso2: return "Otros"
            iso2 = iso2.upper()

            if iso2 == "US": return "USA"
            if iso2 == "CA": return "Canada"

            europe = {"ES","DE","FR","IT","UK","NL","PL","SE","FI","NO","BE","PT","CH","AT","CZ","DK","IE","RO","GR","HU"}
            asia = {"CN","JP","HK","TW","KR","IN","SG","TH","AE","QA"}
            rest_america = {"MX","AR","CL","CO","BR","PE","EC","UY","BO","PY","CR","PA","DO","SV","GT","HN","NI","VE"}
            africa = {"ZA","EG","NG","MA","KE"}
            oceania = {"AU","NZ"}

            if iso2 in europe: return "Europa"
            if iso2 in asia: return "ASIA"
            if iso2 in rest_america: return "Resto de América"
            if iso2 in africa: return "África"
            if iso2 in oceania: return "Oceanía"

            return "Otros"

        combined["region_group"] = combined["iso2"].apply(continent_from_iso2)

        region_agg = (combined.groupby("region_group")
                      .agg(weighted_frac=("weighted_frac","sum"))
                      .reset_index()
                      .sort_values("weighted_frac", ascending=False))
        region_agg["weighted_percent"] = region_agg["weighted_frac"] * 100

        etf_metrics_df = pd.DataFrame(etf_metrics).set_index("Ticker")

        # Weighted metrics
        def wavg(df, col):
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            w = df["ETF Weight (frac)"]
            return (vals*w).sum() / w.sum()

        w_yield = wavg(etf_metrics_df.reset_index(), "Yield %")
        w_cagr = wavg(etf_metrics_df.reset_index(), "5Y CAGR %")
        annual_income = portfolio_value * (w_yield/100)

        st.success("Análisis completo ✔")

        # --------------------------
        # Summary
        # --------------------------
        st.markdown("Portfolio Summary")
        c1,c2,c3 = st.columns(3)
        c1.metric("Dividend Yield", f"{w_yield:.2f}%")
        c2.metric("5Y CAGR", f"{w_cagr:.2f}%")
        c3.metric("Annual Dividend Income", f"${annual_income:,.2f}")

        # --------------------------
        # ETF Metrics
        # --------------------------
        st.markdown("ETF Metrics (per ETF)")
        df_disp = etf_metrics_df.reset_index()[["Ticker","Name","Price","Yield %","5Y CAGR %"]]
        df_disp["Yield %"] = df_disp["Yield %"].map(lambda x: f"{x:.2f}%")
        df_disp["5Y CAGR %"] = df_disp["5Y CAGR %"].map(lambda x: f"{x:.2f}%")
        df_disp["Price"] = df_disp["Price"].map(lambda x: f"${x:,.2f}")
        st.dataframe(df_disp, use_container_width=True)

        # --------------------------
        # Holdings
        # --------------------------
        st.markdown("Top Holdings (Top 100)")
        hd = company_agg.head(100)[["name","weighted_percent","count_etfs"]]
        hd["weighted_percent"] = hd["weighted_percent"].map(lambda x: f"{x:.2f}%")
        st.dataframe(hd, use_container_width=True)

        # --------------------------
        # Country exposures
        # --------------------------
        st.markdown("Exposición por País")
        cdf = country_agg.copy()
        cdf["weighted_percent"] = cdf["weighted_percent"].map(lambda x: f"{x:.2f}%")
        st.dataframe(cdf, use_container_width=True)

        # --------------------------
        # Region exposure bar
        # --------------------------
        st.markdown("Exposición por Continente")
        desired_order = ["USA","Canada","Europa","ASIA","Resto de América","África","Oceanía","Otros"]
        plot_df = region_agg.set_index("region_group").reindex(desired_order).fillna(0).reset_index()
        fig_region = px.bar(plot_df, x="region_group", y="weighted_percent",
                            color="region_group", text=plot_df["weighted_percent"].map(lambda x: f"{x:.2f}%"),
                            title="Exposición por Continente")
        fig_region.update_layout(showlegend=False)
        st.plotly_chart(fig_region, use_container_width=True)

        # --------------------------
        # Pie chart Holdings
        # --------------------------
        st.markdown("Top 10 Holdings + Others")
        top10 = company_agg.head(10)
        others = company_agg["weighted_percent"].iloc[10:].sum()
        labels = list(top10["name"]) + ["Otros"]
        values = list(top10["weighted_percent"]) + [others]
        fig_pie = px.pie(names=labels, values=values, hole=0.3)
        st.plotly_chart(fig_pie, use_container_width=True)

        # --------------------------
        # Pie: Income by ETF
        # --------------------------
        st.markdown("Distribución de dividendos por ETF")
        eif = etf_metrics_df.reset_index()
        eif["Income"] = portfolio_value * (eif["Yield %"]/100) * eif["ETF Weight (frac)"]
        fig_inc = px.pie(eif, names="Ticker", values="Income", hole=0.3)
        st.plotly_chart(fig_inc, use_container_width=True)

        # --------------------------
        # Price History & Growth
        # --------------------------
        st.markdown("5-Year Cumulative Growth")
        tickers_list = list(portfolio.keys())
        price_df = yf.download(tickers_list, period="5y", auto_adjust=True, threads=True)["Close"]
        price_df = price_df.dropna(how="all").ffill().bfill()
        normalized = (price_df / price_df.iloc[0]) * 100
        fig_growth = px.line(normalized)
        st.plotly_chart(fig_growth, use_container_width=True)

        # --------------------------
        # CORRELATION HEATMAP
        # --------------------------
        st.markdown("Correlation Heatmap")
        returns = price_df.pct_change().dropna()
        corr = returns.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_corr, use_container_width=True)

        # --------------------------
        # Rolling Volatility
        # --------------------------
        st.markdown("Volatilidad Anualizada (252d)")
        rolling = returns.rolling(252).std() * np.sqrt(252)
        fig_vol = px.line(rolling)
        st.plotly_chart(fig_vol, use_container_width=True)

        # --------------------------
        # Projection (20 years)
        # --------------------------
        st.markdown("Proyección a 20 años con reinversión de dividendos")

        def simulate(initial, g, y, years):
            rows = []
            cur = initial
            for yr in range(years+1):
                rows.append({"Year": yr, "Value": cur})
                cur = cur * (1 + g + y)
            return pd.DataFrame(rows)

        proj = simulate(portfolio_value, w_cagr/100, w_yield/100, 20)
        st.dataframe(proj, use_container_width=True)

        st.success("✓ Archivos generados y análisis completado")
