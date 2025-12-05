# app.py — Professional Streamlit ETF Portfolio Analyzer
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pycountry
import math
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import time

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="ETF Portfolio Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("📈 ETF Portfolio Analyzer — Professional")

# --------------------------
# Sidebar - Inputs
# --------------------------
st.sidebar.header("1) Configura tu portafolio")

st.sidebar.markdown("""
Ingrese los ETFs y pesos en formato:

Los pesos pueden sumar 100 (porcentajes) o cualquier suma — se normaliza.
""")

default_text = "SCHD: 25\nVOO: 75"
tickers_text = st.sidebar.text_area("Tickers y Pesos", value=default_text, height=140)
portfolio_value = st.sidebar.number_input("Valor total invertido (USD)", min_value=1.0, value=10000.0, step=100.0)

st.sidebar.markdown("---")
st.sidebar.markdown("API: FinnWorlds (holdings) — la clave está embebida (puedes reemplazarla).")
FINNWORLDS_API_KEY = "45f81fc8790e6e351032baab1a264a533f8ebe74"
FINNWORLDS_BASE = "https://api.finnworlds.com/api/v1/etfholdings"

# --------------------------
# Parse portfolio
# --------------------------
def parse_portfolio(text):
    out = {}
    for line in text.strip().splitlines():
        if ":" in line:
            t,w = line.split(":",1)
            try:
                out[t.strip().upper()] = float(w.strip())
            except:
                continue
    return out

raw_portfolio = parse_portfolio(tickers_text)
if not raw_portfolio:
    st.sidebar.error("No se detectaron tickers válidos. Revisa el formato.")
    st.stop()

# Normalize weights -> fractions summing 1.0
total_w = sum(raw_portfolio.values())
if abs(total_w - 100.0) < 1e-6:
    portfolio = {k: v/100.0 for k,v in raw_portfolio.items()}
else:
    # if they provided fractions or other sums, normalize to 1
    s = sum(raw_portfolio.values())
    portfolio = {k: (v/s) for k,v in raw_portfolio.items()}

st.sidebar.markdown("**Pesos normalizados (fractions)**")
st.sidebar.json({k: round(v,6) for k,v in portfolio.items()})

# --------------------------
# Helpers
# --------------------------
@st.cache_data(ttl=3600)
def iso2_from_name_or_code(raw):
    if raw is None: return None
    s = str(raw).strip()
    if not s: return None
    s_up = s.upper()
    if len(s_up) == 2 and s_up.isalpha(): return s_up
    if len(s_up) == 3 and s_up.isalpha():
        try:
            c = pycountry.countries.get(alpha_3=s_up)
            return c.alpha_2 if c else None
        except:
            return None
    try:
        c = pycountry.countries.get(name=s)
        if c: return c.alpha_2
    except:
        pass
    aliases = {"USA":"US","UNITED STATES":"US","UK":"GB","GREAT BRITAIN":"GB","HONG KONG":"HK","CHINA":"CN","TAIWAN":"TW"}
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
    except Exception:
        return pd.DataFrame()
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json().get("result", {})
    rows = []
    outputs = data.get("output") or []
    if isinstance(outputs, dict):
        outputs = [outputs]
    for out in outputs:
        holdings = out.get("holdings") or out.get("positions") or []
        for h in holdings:
            sec = h.get("investment_security", h) if isinstance(h, dict) else h
            name = sec.get("name") or sec.get("title") or "Unknown"
            pv = sec.get("percent_value") or sec.get("percent") or 0
            country = sec.get("invested_country") or sec.get("country") or None
            try:
                pv_f = float(pv)
            except:
                pv_f = 0.0
            rows.append({"ticker": ticker, "name": name, "percent_raw": pv_f, "country_raw": country})
    return pd.DataFrame(rows)

# --------------------------
# Download & process
# --------------------------
st.markdown("## 🔄 Ejecutar análisis")
if st.button("Calcular & Actualizar"):
    with st.spinner("Descargando holdings y métricas (tarda unos segundos)..."):
        all_frames = []
        etf_metrics = []
        progress_text = st.empty()
        n = len(portfolio)
        i = 0
        for ticker, weight in portfolio.items():
            progress_text.text(f"Descargando {ticker} ({i+1}/{n})...")
            df = load_etf_holdings(ticker)
            if df.empty:
                st.warning(f"No se pudieron obtener holdings para {ticker}")
                i += 1
                time.sleep(0.2)
                continue
            df = normalize_percent_column(df)
            df["weighted_frac"] = df["percent_frac"] * weight
            df["etf_weight"] = weight
            all_frames.append(df)

            # yfinance metrics (Name, Price, Yield, 5y CAGR)
            t = yf.Ticker(ticker)
            info = t.info or {}
            name = info.get("shortName") or info.get("longName") or ticker
            price = info.get("regularMarketPrice")
            if price is None:
                try:
                    price = t.history(period="1d")["Close"].iloc[-1]
                except Exception:
                    price = np.nan
            yield_raw = info.get("yield") or info.get("dividendYield") or 0.0
            yield_pct = (yield_raw or 0.0) * 100.0
            # 5y cagr
            try:
                hist = t.history(period="5y", auto_adjust=False)
                if len(hist) > 2:
                    s = hist["Close"].iloc[0]; e = hist["Close"].iloc[-1]
                    years = (hist.index[-1] - hist.index[0]).days / 365.25
                    cagr = (e/s)**(1/years) - 1
                else:
                    cagr = np.nan
            except Exception:
                cagr = np.nan

            etf_metrics.append({
                "Ticker": ticker,
                "Name": name,
                "Price": price,
                "Yield %": yield_pct,
                "5Y CAGR %": (cagr * 100.0) if pd.notna(cagr) else np.nan,
                "ETF Weight (frac)": weight  # keep for internal weighted calcs but not shown later
            })

            i += 1
            time.sleep(0.25)

        if not all_frames:
            st.error("No se descargaron holdings. Revisa ticks / API.")
            st.stop()

        combined = pd.concat(all_frames, ignore_index=True)
        combined["iso2"] = combined["country_raw"].apply(iso2_from_name_or_code)

        # Aggregations
        company_agg = (combined.groupby("name")
                       .agg(weighted_frac=("weighted_frac","sum"),
                            count_etfs=("ticker","nunique"))
                       .reset_index()
                       .sort_values("weighted_frac", ascending=False))
        company_agg["weighted_percent"] = company_agg["weighted_frac"] * 100.0

        country_agg = (combined.groupby("iso2")
                       .agg(weighted_frac=("weighted_frac","sum"),
                            count_holdings=("name","count"))
                       .reset_index()
                       .sort_values("weighted_frac", ascending=False))
        country_agg["weighted_percent"] = country_agg["weighted_frac"] * 100.0

        # Region grouping
        def continent_from_iso2(iso2):
            if not iso2 or not isinstance(iso2,str): return "Otros"
            iso2 = iso2.upper()
            if iso2 == "US": return "USA"
            if iso2 == "CA": return "Canada"
            europe = {"AL","AD","AM","AT","AZ","BY","BE","BA","BG","CH","CY","CZ","DE","DK","EE","ES","FI","FR","GE","GR","HR","HU","IE","IS","IT","KZ","LI","LT","LU","LV","MC","MD","ME","MK","MT","NL","NO","PL","PT","RO","RU","SE","SI","SK","SM","TR","UA","GB"}
            asia = {"AE","AF","AZ","BD","BH","BN","BT","CN","HK","ID","IL","IN","IQ","IR","JO","JP","KG","KH","KP","KR","KW","KZ","LA","LB","LK","MM","MN","MO","MY","NP","OM","PH","PK","QA","SA","SG","SY","TH","TJ","TL","TM","TR","TW","UZ","VN","YE"}
            rest_america = {"AR","BO","BR","CL","CO","CR","CU","DO","EC","GT","HN","JM","KY","KN","LC","MX","NI","PA","PE","PY","SR","TT","UY","VE","BZ","BS","BB","AG","AW","BM","VG"}
            africa = {"DZ","AO","BJ","BW","BF","BI","CM","CV","CF","TD","KM","CG","CD","DJ","EG","ER","SZ","ET","GA","GM","GH","GN","GQ","GW","KE","LS","LR","LY","MG","MW","ML","MR","MU","MA","MZ","NA","NE","NG","RW","ST","SN","SC","SL","SO","ZA","SS","SD","TZ","TG","TN","UG","ZM","ZW"}
            oceania = {"AU","NZ","FJ","PG","SB","TO","TV","VU"}
            if iso2 in europe: return "Europa"
            if iso2 in asia: return "ASIA"
            if iso2 in rest_america: return "Resto de América"
            if iso2 in africa: return "África"
            if iso2 in oceania: return "Oceanía"
            return "Otros"

        combined["region_group"] = combined["iso2"].apply(continent_from_iso2)
        region_agg = (combined.groupby("region_group")
                      .agg(weighted_frac=("weighted_frac","sum"),
                           count_holdings=("name","count"))
                      .reset_index()
                      .sort_values("weighted_frac", ascending=False))
        region_agg["weighted_percent"] = region_agg["weighted_frac"] * 100.0

        # ETF metrics DF
        etf_metrics_df = pd.DataFrame(etf_metrics).set_index("Ticker")

        # Weighted portfolio stats
        def safe_weighted_avg(df, col, weight_col="ETF Weight (frac)"):
            if col not in df.columns: return np.nan
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            w = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(vals) & np.isfinite(w)
            if mask.sum() == 0: return np.nan
            return (vals[mask] * w[mask]).sum() / w[mask].sum()

        weighted_yield = safe_weighted_avg(etf_metrics_df.reset_index(), "Yield %", "ETF Weight (frac)")
        weighted_cagr = safe_weighted_avg(etf_metrics_df.reset_index(), "5Y CAGR %", "ETF Weight (frac)")
        est_annual_div = portfolio_value * ( (weighted_yield or 0.0) / 100.0 )

        st.success("Análisis completo ✅")

        # --------------------------
        # LAYOUT: summary metrics and tables
        # --------------------------
        st.markdown("### 📌 Portfolio Summary")
        c1,c2,c3 = st.columns(3)
        c1.metric("Weighted Dividend Yield", f"{(weighted_yield or 0.0):.2f}%")
        c2.metric("Weighted 5Y CAGR", f"{(weighted_cagr or 0.0):.2f}%")
        c3.metric("Est. Annual Dividend Income", f"${est_annual_div:,.2f}")

        # --------------------------
        # ETF Metrics table (NAME + PRICE displayed; hide weight)
        # --------------------------
        st.markdown("### 📌 ETF Metrics (per ETF)")
        etf_display = etf_metrics_df.reset_index()[["Ticker","Name","Price","Yield %","5Y CAGR %"]].copy()
        # format percentages
        etf_display["Yield %"] = etf_display["Yield %"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        etf_display["5Y CAGR %"] = etf_display["5Y CAGR %"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        etf_display["Price"] = etf_display["Price"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
        st.dataframe(etf_display, use_container_width=True)

        # --------------------------
        # Holdings table top100
        # --------------------------
        st.markdown("### 📌 Holdings ponderados (Top 100)")
        holdings_display = company_agg.head(100)[["weighted_percent","count_etfs"]].copy()
        holdings_display = holdings_display.rename(columns={"weighted_percent":"Exposición (%)","count_etfs":"# ETFs"})
        holdings_display["Exposición (%)"] = holdings_display["Exposición (%)"].map(lambda x: f"{x:.2f}%")
        st.dataframe(holdings_display, use_container_width=True)

        # --------------------------
        # Country exposures
        # --------------------------
        st.markdown("### 📌 Exposición por país (Top 100)")
        country_display = country_agg.head(100)[["weighted_percent","count_holdings"]].copy()
        country_display = country_display.rename(columns={"weighted_percent":"Exposición (%)","count_holdings":"# Holdings"})
        country_display["Exposición (%)"] = country_display["Exposición (%)"].map(lambda x: f"{x:.2f}%")
        country_display = country_display.rename_axis("País (ISO2)").reset_index()
        st.dataframe(country_display, use_container_width=True)

        # --------------------------
        # Region bar chart (matplotlib -> plotly for nicer look)
        # --------------------------
        st.markdown("### 📊 Exposición por Región / Continente")
        desired_order = ["USA","Canada","Europa","ASIA","Resto de América","África","Oceanía","Otros"]
        plot_df = region_agg.set_index("region_group").reindex(desired_order).fillna(0).reset_index()
        fig_region = px.bar(plot_df, x="region_group", y="weighted_percent",
                            labels={"region_group":"Grupo", "weighted_percent":"Exposición (%)"},
                            color="region_group", text=plot_df["weighted_percent"].map(lambda x: f"{x:.2f}%"),
                            title="Exposición por Grupo (agrupado)")
        fig_region.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_region, use_container_width=True)

        # --------------------------
        # Pie chart top10 holdings + others
        # --------------------------
        st.markdown("### 🥧 Top 10 Holdings + Others")
        top10 = company_agg.head(10).copy()
        others = company_agg["weighted_percent"].iloc[10:].sum()
        labels = list(top10["name"]) if "name" in top10.columns else list(top10["name"])
        values = list(top10["weighted_percent"])
        if others > 0:
            labels = list(top10["name"]) + ["Otros"]
            values = list(top10["weighted_percent"]) + [others]
        fig_pie = px.pie(names=labels, values=values, hole=0.3)
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

        # --------------------------
        # Income contribution by ETF (pie)
        # --------------------------
        st.markdown("### 💵 Annual Dividend Income Contribution (por ETF)")
        etf_income_df = etf_metrics_df.reset_index().copy()
        etf_income_df["Income ($)"] = (pd.to_numeric(etf_income_df["Yield %"], errors="coerce").fillna(0)/100.0) * etf_income_df["ETF Weight (frac)"] * portfolio_value
        fig_income = px.pie(etf_income_df, names="Ticker", values="Income ($)", hole=0.3)
        fig_income.update_traces(texttemplate='%{label}: %{percent:.1%} <br>($%{value:,.2f})')
        st.plotly_chart(fig_income, use_container_width=True)

        # --------------------------
        # Price history / Cumulative growth (5y)
        # --------------------------
        st.markdown("### 📉 5-Year Cumulative Growth (normalized)")
        tickers_list = list(portfolio.keys())
        price_df = yf.download(tickers_list, period="5y", auto_adjust=True, threads=True)["Close"]
        if isinstance(price_df, pd.Series):
            price_df = price_df.to_frame(name=tickers_list[0])
        price_df = price_df.dropna(how="all").ffill().bfill()
        normalized = (price_df / price_df.iloc[0]) * 100.0
        fig_growth = px.line(normalized, x=normalized.index, y=normalized.columns, labels={"value":"Index (100)","variable":"Ticker"})
        st.plotly_chart(fig_growth, use_container_width=True)

        # --------------------------
        # Correlation heatmap (returns)
        # --------------------------
        st.markdown("### 🔗 Correlation Heatmap (returns)")
        returns = price_df.pct_change().dropna(how="all")
        corr = returns.corr().fillna(0)
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn", title="Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)

        # --------------------------
        # Extra chart: Rolling annualized volatility (1y window)
        # --------------------------
        st.markdown("### 📊 Rolling Volatility (annualized, 252 trading days window)")
        rolling = returns.rolling(window=252).std() * np.sqrt(252)
        fig_vol = px.line(rolling, x=rolling.index, y=rolling.columns, labels={"value":"Volatility","variable":"Ticker"})
        st.plotly_chart(fig_vol, use_container_width=True)

        # --------------------------
        # Projection (20 years sample)
        # --------------------------
        st.markdown("### 🔮 Projection (20 años - reinvirtiendo dividendos)")
        def simulate_total_growth(initial, annual_growth_decimal, dividend_yield_decimal, years):
            rows = []
            cur = initial
            for y in range(years+1):
                rows.append({"Year": y, "Portfolio Value": round(cur,2), "Estimated Annual Dividend": round(cur*dividend_yield_decimal,2)})
                growth = cur * annual_growth_decimal
                dividends = cur * dividend_yield_decimal
                cur = cur + growth + dividends
            return pd.DataFrame(rows)

        proj_df = simulate_total_growth(portfolio_value, (weighted_cagr or 0.0)/100.0, (weighted_yield or 0.0)/100.0, 20)
        st.dataframe(proj_df, use_container_width=True)

        # --------------------------
        # Save CSVs for download
        # --------------------------
        combined.to_csv("all_holdings_detailed.csv", index=False)
        company_agg.to_csv("holdings_weighted.csv", index=False)
        country_agg.to_csv("countries_weighted.csv", index=False)
        etf_metrics_df.reset_index().to_csv("etf_metrics.csv", index=False)
        region_agg.to_csv("regions_weighted.csv", index=False)

        st.success("CSV guardados: all_holdings_detailed.csv, holdings_weighted.csv, countries_weighted.csv, regions_weighted.csv, etf_metrics.csv")
        progress_text.empty()
