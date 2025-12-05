import streamlit as st
import requests, pycountry, math, time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from yfinance import Ticker, download
import json
from datetime import datetime

# =========================================================================
# 1. CONFIGURACIÓN Y HELPER FUNCTIONS
# =========================================================================

# Finworlds API configuration (using placeholder)
FINNWORLDS_BASE = "https://api.finnworlds.com/api/v1/etfholdings"

# Set Streamlit page config
st.set_page_config(
    page_title="Análisis Ponderado de Portfolio de ETFs",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robust normalization of percent_raw column per ETF:
def normalize_percent_column(df):
    """Detects and normalizes the raw percentage column to fractions."""
    if df is None or df.empty:
        return df, "empty", 0.0, 0.0
    arr = df["percent_raw"].fillna(0).astype(float).values
    candidates = {
        "as_fraction": lambda x: x,
        "percent/100": lambda x: x/100.0,
        "per_mille": lambda x: x/1000.0,
    }
    best = None
    best_diff = float("inf")
    best_frac = None
    
    for name, fn in candidates.items():
        try:
            frac = fn(arr)
            s = np.nansum(frac)
            diff = abs(s - 1.0)
            if math.isfinite(s) and diff < best_diff:
                best_diff = diff
                best = name
                best_frac = frac
        except Exception:
            continue
            
    if best_frac is None or (best_diff > 0.5 and arr.sum() > 0):
        # fallback: normalize by sum_raw
        sraw = arr.sum()
        if sraw == 0:
            norm = np.zeros_like(arr)
        else:
            norm = arr / sraw
        chosen = "normalized_by_sum"
    else:
        norm = best_frac
        chosen = best
        
    df2 = df.copy()
    df2["percent_frac"] = norm
    return df2, chosen, float(arr.sum()), float(np.nansum(norm))

# Country ISO helpers (copied from your script)
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
        except: return None
    try:
        c = pycountry.countries.get(name=s)
        if c: return c.alpha_2
    except: pass
    try:
        low = s.lower()
        for c in pycountry.countries:
            if low in (c.name or "").lower() or low in (getattr(c,"official_name","") or "").lower():
                return c.alpha_2
    except: pass
    aliases = {
        "USA":"US","UNITED STATES":"US","UNITED STATES OF AMERICA":"US",
        "UK":"GB","GREAT BRITAIN":"GB","ENGLAND":"GB",
        "HONG KONG":"HK","CHINA":"CN","TAIWAN":"TW","KOREA":"KR",
        "SOUTH KOREA":"KR","NETHERLANDS":"NL"
    }
    if s_up in aliases: return aliases[s_up]
    return None

def continent_from_iso2(iso2):
    """Applies your custom regional grouping logic."""
    if not iso2 or not isinstance(iso2, str):
        return "Otros"
    iso2 = iso2.upper()

    if iso2 == "US": return "USA"
    if iso2 == "CA": return "Canada"

    europe_iso2 = {"AL","AD","AM","AT","AZ","BY","BE","BA","BG","CH","CY","CZ","DE","DK","EE","ES","FI","FR","GE","GR","HR","HU","IE","IS","IT","KZ","LI","LT","LU","LV","MC","MD","ME","MK","MT","NL","NO","PL","PT","RO","RU","SE","SI","SK","SM","TR","UA","GB","VA","UK","RS","BG","XK"}
    if iso2 in europe_iso2: return "Europa"

    asia_iso2 = {"AE","AF","AZ","BD","BH","BN","BT","CN","GE","HK","ID","IL","IN","IQ","IR","JO","JP","KG","KH","KP","KR","KW","KZ","LA","LB","LK","MM","MN","MO","MY","NP","OM","PH","PK","QA","SA","SG","SY","TH","TJ","TL","TM","TR","TW","UZ","VN","YE"}
    if iso2 in asia_iso2: return "ASIA"

    rest_america_iso2 = {"AR","BO","BR","CL","CO","CR","CU","DO","EC","GT","HN","JM","KY","KN","LC","MX","NI","PA","PE","PY","SR","TT","UY","VE","BZ","BS","BB","AG","AW","BM","VG","TC","MQ","GP","GF"}
    if iso2 in rest_america_iso2: return "Resto de América"

    africa_iso2 = {"DZ","AO","BJ","BW","BF","BI","CM","CV","CF","TD","KM","CG","CD","DJ","EG","ER","SZ","ET","GA","GM","GH","GN","GQ","GW","KE","LS","LR","LY","MG","MW","ML","MR","MU","MA","MZ","NA","NE","NG","RW","ST","SN","SC","SL","SO","ZA","SS","SD","TZ","TG","TN","UG","ZM","ZW"}
    if iso2 in africa_iso2: return "África"

    oceania_iso2 = {"AU","NZ","FJ","PG","SB","TO","TV","VU"}
    if iso2 in oceania_iso2: return "Oceanía"

    return "Otros"

def safe_weighted_avg(df, col, weight_col="ETF Weight (frac)"):
    """Calculates the weighted average ignoring NaN."""
    if col not in df.columns: return np.nan
    vals = df[col].to_numpy(dtype=float)
    w = df[weight_col].to_numpy(dtype=float)
    mask = np.isfinite(vals) & np.isfinite(w)
    if mask.sum() == 0: return np.nan
    return (vals[mask] * w[mask]).sum() / w[mask].sum()

def simulate_total_growth(initial_investment, annual_growth_decimal, dividend_yield_decimal, years):
    rows=[]
    cur = initial_investment
    for y in range(years+1):
        if cur < 0: cur = 0.0
        
        rows.append({"Year": y, 
                     "Portfolio Value": round(cur,2), 
                     "Estimated Annual Dividend": round(cur*dividend_yield_decimal,2)})
        
        growth = cur * annual_growth_decimal
        dividends = cur * dividend_yield_decimal
        cur = cur + growth + dividends
    return pd.DataFrame(rows)

def load_etf_holdings(ticker, api_key):
    """Loads ETF holdings from FinnWorlds API."""
    url = f"{FINNWORLDS_BASE}?key={api_key}&ticker={ticker}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.warning(f"Error al solicitar {ticker}: {e}")
        return pd.DataFrame()
    
    data = r.json()
    rows = []
    outputs = data.get("result", {}).get("output") or data.get("holdings") or []

    if isinstance(outputs, dict): outputs = [outputs]

    for out in outputs:
        holdings = out.get("holdings") or out.get("positions") or []
        for h in holdings:
            sec = h.get("investment_security") or h.get("investmentSecurity") or h.get("name", h)
            if not isinstance(sec, dict): continue

            name = sec.get("name") or sec.get("title") or "Unknown"
            pv = sec.get("percent_value") or sec.get("percent") or 0
            country = sec.get("invested_country") or sec.get("country") or None
            
            try: pv_f = float(pv)
            except: pv_f = 0.0
            
            rows.append({"ticker": ticker, "name": name, "percent_raw": pv_f, "country_raw": country})
    
    return pd.DataFrame(rows)


# =========================================================================
# 2. CORE ANALYSIS FUNCTION (Cached)
# =========================================================================

@st.cache_data(show_spinner="Analizando y descargando datos (Puede tardar hasta 30s)...")
def run_analysis(portfolio_str, portfolio_value, api_key):
    """Executes the full portfolio analysis and returns results."""
    try:
        portfolio_raw = json.loads(portfolio_str)
        portfolio = {k.upper(): v for k, v in portfolio_raw.items()}
    except Exception as e:
        return {"error": f"Error al parsear el Portfolio. Asegúrate de que es un JSON válido: {e}"}

    portfolio_value = float(portfolio_value)
    if not api_key:
        return {"error": "Por favor, introduce tu clave API de FinnWorlds."}

    # 1. Normalize weights
    total_w = sum(portfolio.values())
    if total_w <= 0:
        return {"error": "Portfolio vacío o pesos inválidos."}
        
    if abs(total_w - 100.0) < 1e-6:
        portfolio = {k: v/100.0 for k, v in portfolio.items()}
    else:
        s = sum(portfolio.values())
        portfolio = {k: (v/s) for k, v in portfolio.items()}

    # 2. Download and combine holdings (FinnWorlds)
    all_frames = []
    etf_metrics_list = []
    tickers = list(portfolio.keys())
    
    for ticker, weight in portfolio.items():
        # --- Holdings ---
        df = load_etf_holdings(ticker, api_key)
        if df.empty:
            st.warning(f"ADVERTENCIA: No se pudieron obtener los holdings para {ticker}. Se omitirá en la exposición consolidada.")
            # Still try to get yfinance data even if holdings failed
        else:
            df_norm, _, _, _ = normalize_percent_column(df)
            df_norm["weighted_frac"] = df_norm["percent_frac"] * weight
            df_norm["etf_weight"] = weight
            all_frames.append(df_norm)

        # --- yfinance metrics ---
        try:
            t = Ticker(ticker)
            info = t.info or {}
            yield_raw = info.get('yield') or info.get('dividendYield')
            yield_pct = float(yield_raw) * 100.0 if yield_raw is not None else np.nan
            price = info.get('regularMarketPrice')
            
            hist = t.history(period="5y", auto_adjust=False)
            cagr5 = np.nan
            if hist is not None and len(hist) > 2:
                s = hist['Close'].iloc[0]; e = hist['Close'].iloc[-1]
                years = (hist.index[-1] - hist.index[0]).days / 365.25
                if s>0 and years>0:
                    cagr5 = ((e/s)**(1/years) - 1) * 100.0
            
            etf_metrics_list.append({
                "Ticker": ticker,
                "ETF Weight (frac)": weight,
                "Yield %": yield_pct,
                "Price": price,
                "5Y CAGR %": cagr5
            })
        except Exception:
            etf_metrics_list.append({"Ticker": ticker, "ETF Weight (frac)": weight, "Yield %": np.nan, "Price": np.nan, "5Y CAGR %": np.nan})
            
        time.sleep(0.3)

    if not all_frames:
        return {"error": "No se pudieron descargar los holdings para ningún ETF. Revise los tickers o la clave API."}

    combined = pd.concat(all_frames, ignore_index=True)

    # 3. Aggregations
    
    # 3.1 Company Aggregation
    company_agg = (combined.groupby("name", dropna=False)
                  .agg(weighted_frac = ("weighted_frac","sum"),
                       count_etfs = ("ticker","nunique"))
                  .reset_index()
                  .sort_values("weighted_frac", ascending=False))
    company_agg["weighted_percent"] = company_agg["weighted_frac"] * 100.0

    # 3.2 Country Aggregation
    combined["iso2"] = combined["country_raw"].apply(iso2_from_name_or_code)
    country_agg = (combined.groupby("iso2", dropna=False)
                  .agg(weighted_frac = ("weighted_frac","sum"),
                       count_holdings = ("name","count"))
                  .reset_index()
                  .sort_values("weighted_frac", ascending=False))
    country_agg["weighted_percent"] = country_agg["weighted_frac"] * 100.0
    
    # 3.3 Region Aggregation
    combined["region_group"] = combined["iso2"].apply(continent_from_iso2)
    region_agg = (combined.groupby("region_group")
                  .agg(weighted_frac=("weighted_frac","sum"),
                       count_holdings=("name","count"))
                  .reset_index()
                  .sort_values("weighted_frac", ascending=False))
    region_agg["weighted_percent"] = region_agg["weighted_frac"] * 100.0

    # 4. Portfolio-level Metrics
    etf_metrics_df = pd.DataFrame(etf_metrics_list)
    etf_metrics_df['Income ($)'] = (etf_metrics_df["Yield %"]/100.0) * etf_metrics_df["ETF Weight (frac)"] * portfolio_value
    
    weighted_yield_pct = safe_weighted_avg(etf_metrics_df, "Yield %")
    weighted_cagr_pct = safe_weighted_avg(etf_metrics_df, "5Y CAGR %")
    est_annual_dividend_income = portfolio_value * ( (weighted_yield_pct or 0.0) / 100.0 )
    
    # 5. Projection
    proj = simulate_total_growth(portfolio_value, (weighted_cagr_pct or 0.0)/100.0, (weighted_yield_pct or 0.0)/100.0, 20)
    
    # 6. Price history for Growth/Correlation
    price_df = pd.DataFrame()
    if tickers:
        try:
            price_df = download(tickers, period="5y", auto_adjust=True, threads=True)["Close"]
            if isinstance(price_df, pd.Series):
                price_df = price_df.to_frame(name=tickers[0])
            price_df = price_df.dropna(how="all").ffill().bfill()
        except Exception:
            pass

    return {
        "etf_metrics_df": etf_metrics_df,
        "company_agg": company_agg,
        "country_agg": country_agg,
        "region_agg": region_agg,
        "weighted_yield_pct": weighted_yield_pct,
        "weighted_cagr_pct": weighted_cagr_pct,
        "est_annual_dividend_income": est_annual_dividend_income,
        "proj": proj,
        "price_df": price_df,
        "portfolio_value": portfolio_value,
    }

# =========================================================================
# 3. STREAMLIT UI
# =========================================================================

st.title("📊 Análisis Ponderado de Portfolio de ETFs")
st.markdown("Herramienta para visualizar la exposición consolidada por holdings, país y región, utilizando datos de **FinnWorlds** y **yfinance**.")

# --- SIDEBAR FOR INPUTS ---
with st.sidebar:
    st.header("Configuración del Portfolio")
    
    api_key = st.text_input(
        "Clave API FinnWorlds:", 
        value="45f81fc8790e6e351032baab1a264a533f8ebe74", # Default value
        type="password",
        help="Necesaria para obtener los holdings de los ETFs."
    )
    
    portfolio_value = st.number_input(
        "Valor Total del Portfolio (USD):",
        value=10000.0,
        min_value=1.0,
        step=1000.0,
        help="Usado para calcular la proyección e ingresos estimados."
    )

    portfolio_str = st.text_area(
        "Tickers y Pesos (JSON):",
        value='{\n    "SCHD": 25,\n    "IDVO": 25,\n    "CGDG": 50\n}',
        height=150,
        help="Formato: {\"TICKER\": PESO}, donde PESO es un porcentaje (0-100) o una fracción (0-1)."
    )
    
    # Button to clear cache and force re-run
    if st.button("Limpiar Caché y Re-analizar"):
        st.cache_data.clear()
        st.experimental_rerun()
        
    st.markdown("---")
    st.caption("Los datos se cargan automáticamente al modificar los campos. Los datos de holdings y métricas se almacenan en caché para evitar llamadas API repetidas e innecesarias.")


# --- RUN ANALYSIS ---
results = run_analysis(portfolio_str, portfolio_value, api_key)

# --- ERROR HANDLING ---
if "error" in results:
    st.error(results["error"])
    st.stop()

# Desestructuración de resultados
etf_metrics_df = results["etf_metrics_df"]
company_agg = results["company_agg"]
country_agg = results["country_agg"]
region_agg = results["region_agg"]
weighted_yield_pct = results["weighted_yield_pct"]
weighted_cagr_pct = results["weighted_cagr_pct"]
est_annual_dividend_income = results["est_annual_dividend_income"]
proj = results["proj"]
price_df = results["price_df"]


# =========================================================================
# 4. DISPLAY RESULTS
# =========================================================================

st.header("Resumen del Portfolio")

# --- SUMMARY METRICS ---
col1, col2, col3 = st.columns(3)

col1.metric(
    "Rendimiento (Yield) Ponderado",
    f"{weighted_yield_pct:.2f}%" if math.isfinite(weighted_yield_pct) else "N/A"
)
col2.metric(
    "5Y CAGR Ponderado",
    f"{weighted_cagr_pct:.2f}%" if math.isfinite(weighted_cagr_pct) else "N/A"
)
col3.metric(
    f"Ingreso Anual Est. (@${portfolio_value:,.0f})",
    f"${est_annual_dividend_income:,.2f}"
)

# --- ETF METRICS TABLE ---
st.subheader("Métricas por ETF")
etf_metrics_display = etf_metrics_df.copy()
etf_metrics_display['ETF Weight (frac)'] = etf_metrics_display['ETF Weight (frac)'].apply(lambda x: f"{x:.4f}")
etf_metrics_display['Yield %'] = etf_metrics_display['Yield %'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
etf_metrics_display['Price'] = etf_metrics_display['Price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
etf_metrics_display['5Y CAGR %'] = etf_metrics_display['5Y CAGR %'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
etf_metrics_display['Income ($)'] = etf_metrics_display['Income ($)'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")

st.dataframe(
    etf_metrics_display, 
    use_container_width=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", help="Símbolo Bursátil"),
        "ETF Weight (frac)": st.column_config.TextColumn("Peso (fracción)", help="Peso de cada ETF en tu Portfolio"),
        "Yield %": st.column_config.TextColumn("Yield %"),
        "Price": st.column_config.TextColumn("Precio"),
        "5Y CAGR %": st.column_config.TextColumn("5Y CAGR %"),
        "Income ($)": st.column_config.TextColumn("Ingreso Anual Est.")
    },
    hide_index=True
)

st.markdown("---")
st.header("Exposición Consolidada y Holdings")

col4, col5 = st.columns(2)

with col4:
    # --- REGION BAR CHART ---
    st.subheader("Exposición Regional (Agrupado)")
    desired_order = ["USA", "Canada", "Europa", "ASIA", "Resto de América", "África", "Oceanía", "Otros"]
    plot_df = region_agg.set_index("region_group").reindex(desired_order).fillna(0).reset_index()
    fig_region = px.bar(
        plot_df, 
        x="region_group", 
        y="weighted_percent", 
        labels={"region_group": "Región", "weighted_percent": "Exposición (%)"},
        text="weighted_percent",
        color="region_group",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_region.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_region.update_layout(xaxis={'categoryorder':'array', 'categoryarray':desired_order}, uniformtext_minsize=8, uniformtext_mode='hide', yaxis_title="Exposición (%)", xaxis_title="")
    st.plotly_chart(fig_region, use_container_width=True)

    # --- COUNTRY TABLE ---
    st.subheader("Exposición por País (Top 100)")
    out_ctry = country_agg[["iso2","weighted_percent","count_holdings"]].copy().rename(columns={"iso2":"País (ISO2)","count_holdings":"# Holdings"})
    out_ctry["Exposición (%)"] = out_ctry["weighted_percent"].round(2)
    st.dataframe(
        out_ctry.head(100), 
        use_container_width=True,
        column_config={
            "weighted_percent": st.column_config.ProgressColumn(
                "Exposición (%)",
                format="%.2f%%",
                min_value=0,
                max_value=out_ctry["weighted_percent"].max()
            )
        },
        hide_index=True
    )

with col5:
    # --- HOLDINGS PIE CHART ---
    st.subheader("Top 10 Holdings Ponderados")
    top10 = company_agg.head(10).copy()
    others = company_agg["weighted_percent"].iloc[10:].sum()
    labels = list(top10["name"]) + (["Otros"] if others>0 else [])
    values = list(top10["weighted_percent"]) + ([others] if others>0 else [])
    fig_holdings = px.pie(
        names=labels, 
        values=values, 
        title="Exposición Consolidada por Empresa", 
        hole=0.4
    )
    fig_holdings.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_holdings, use_container_width=True)

    # --- INCOME PIE CHART ---
    st.subheader("Distribución de Ingresos por Dividendo")
    if etf_metrics_df['Income ($)'].sum() > 0:
        fig_income = px.pie(
            etf_metrics_df, 
            values="Income ($)", 
            names="Ticker",
            title="Contribución Anual al Ingreso", 
            hole=0.4
        )
        fig_income.update_traces(texttemplate='%{label}: %{percent:.1%}', textposition='inside')
        st.plotly_chart(fig_income, use_container_width=True)
    else:
        st.warning("Datos de Yield insuficientes o cero para calcular el ingreso por dividendo.")

st.markdown("---")
st.header("Análisis de Rendimiento y Proyección")

col6, col7 = st.columns(2)

with col6:
    # --- GROWTH LINE CHART ---
    st.subheader("Crecimiento Acumulado 5-Años")
    if not price_df.empty:
        normalized_growth = (price_df / price_df.iloc[0]) * 100
        fig_growth = px.line(
            normalized_growth, 
            title="Normalizado a 100 al Inicio",
            labels={"value": "Valor Normalizado (Inicio = 100)", "index": "Fecha", "variable": "Ticker"}
        )
        st.plotly_chart(fig_growth, use_container_width=True)
    else:
        st.warning("Datos de precios insuficientes para mostrar el gráfico de crecimiento.")

    # --- CORRELATION HEATMAP ---
    st.subheader("Heatmap de Correlación")
    if price_df.shape[1] > 1:
        corr = price_df.pct_change().corr().fillna(0)
        fig_corr = px.imshow(
            corr, 
            text_auto=".2f", 
            aspect="auto", 
            color_continuous_scale='RdYlGn', 
            title="Correlación del Cambio Diario",
            labels=dict(color="Correlación")
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Se necesitan al menos 2 ETFs para calcular la correlación.")

with col7:
    # --- PROJECTION TABLE ---
    st.subheader("Proyección de Crecimiento a 20 Años")
    st.info("Esta proyección asume una reinversión total (rendimiento + dividendo) y tasas de rendimiento (CAGR) y Yield constantes a las calculadas hoy. Es solo una ilustración.")
    
    proj_display = proj.copy()
    proj_display['Portfolio Value'] = proj_display['Portfolio Value'].apply(lambda x: f"${x:,.2f}")
    proj_display['Estimated Annual Dividend'] = proj_display['Estimated Annual Dividend'].apply(lambda x: f"${x:,.2f}")

    st.dataframe(
        proj_display, 
        use_container_width=True,
        column_config={
            "Year": "Año",
            "Portfolio Value": st.column_config.TextColumn("Valor del Portfolio"),
            "Estimated Annual Dividend": st.column_config.TextColumn("Dividendo Anual Estimado")
        },
        hide_index=True
    )
