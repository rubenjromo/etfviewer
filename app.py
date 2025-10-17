import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

# ============================================================
# 🔹 Funciones base de tu app (rendimientos, métricas, etc.)
# ============================================================

@st.cache_data(ttl=86400)
def get_data(tickers, start):
    data = yf.download(tickers, start=start, progress=False)["Adj Close"]
    data = data.dropna()
    return data

def portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility

def calculate_portfolio_metrics(data, weights):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    port_return, port_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = port_return / port_volatility if port_volatility != 0 else 0
    return {
        "Annual Return": port_return * 100,
        "Annual Volatility": port_volatility * 100,
        "Sharpe Ratio": sharpe_ratio,
    }

def calculate_cumulative_returns(data):
    returns = data.pct_change().dropna()
    cum_returns = (1 + returns).cumprod()
    return cum_returns

# ============================================================
# 🔹 Funciones nuevas: Holdings y sectores
# ============================================================

@st.cache_data(ttl=86400)
def get_etf_holdings(ticker_symbol):
    """Intenta obtener holdings detallados del ETF."""
    try:
        etf = yf.Ticker(ticker_symbol)
        holdings = etf.funds_holdings
        if holdings is None or holdings.empty:
            return pd.DataFrame()
        holdings = holdings[['symbol', 'holdingName', 'holdingPercent']]
        holdings.rename(columns={'symbol': 'Ticker', 'holdingName': 'Company', 'holdingPercent': 'Weight'}, inplace=True)
        holdings['ETF'] = ticker_symbol
        return holdings
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_etf_sector_data(ticker_symbol):
    """Obtiene los sectores del ETF con pesos porcentuales."""
    try:
        etf = yf.Ticker(ticker_symbol)
        sector_data = etf.fund_sector_weightings
        if sector_data:
            df = pd.DataFrame(sector_data.items(), columns=['Sector', 'Weight'])
            df['ETF'] = ticker_symbol
            df['Weight'] = df['Weight'] * 100  # convertir a porcentaje
            return df
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def combine_holdings(portfolio):
    """Combina holdings de todos los ETFs ponderados por su peso en el portafolio."""
    combined = pd.DataFrame()
    for etf, etf_weight in portfolio.items():
        holdings = get_etf_holdings(etf)
        if not holdings.empty:
            holdings['AdjustedWeight'] = holdings['Weight'] * (etf_weight / 100)
            combined = pd.concat([combined, holdings], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    final = combined.groupby(['Ticker', 'Company'], as_index=False)['AdjustedWeight'].sum()
    final.sort_values('AdjustedWeight', ascending=False, inplace=True)
    return final

def combine_sector_exposure(portfolio):
    """Combina exposición sectorial de los ETFs."""
    combined = pd.DataFrame()
    for etf, etf_weight in portfolio.items():
        df = get_etf_sector_data(etf)
        if not df.empty:
            df['AdjustedWeight'] = df['Weight'] * (etf_weight / 100)
            combined = pd.concat([combined, df], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    final = combined.groupby('Sector', as_index=False)['AdjustedWeight'].sum()
    final.sort_values('AdjustedWeight', ascending=False, inplace=True)
    return final

def plot_company_exposure(df):
    fig = px.bar(df.head(15), x='Company', y='AdjustedWeight',
                 title="Top 15 Holdings (Weighted by Portfolio)",
                 labels={'AdjustedWeight': 'Portfolio Exposure (%)'})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def plot_sector_pie(df):
    fig = px.pie(df, values='AdjustedWeight', names='Sector',
                 title='Portfolio Exposure by Sector', hole=0.3)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🔹 Interfaz principal Streamlit
# ============================================================

st.title("📈 Portfolio Analyzer")

with st.sidebar:
    st.header("Configuración del Portafolio")
    tickers = st.text_input("Ingrese los tickers separados por coma", "VOO, QQQ, SOXX").replace(" ", "").split(",")
    weights_input = st.text_input("Pesos (%) en el mismo orden", "40,40,20").replace(" ", "").split(",")
    weights = [float(w) for w in weights_input]
    start_date = st.date_input("Fecha de inicio", datetime(2015, 1, 1))

    portfolio = dict(zip(tickers, weights))
    st.write("**Portafolio actual:**", portfolio)

# --- Datos ---
data = get_data(tickers, start_date)
returns = data.pct_change().dropna()

# --- Cálculos ---
metrics = calculate_portfolio_metrics(data, np.array(weights) / 100)
cum_returns = calculate_cumulative_returns(data)

# --- Visualización principal ---
st.subheader("📊 Rendimiento Histórico")
fig = px.line(cum_returns, title="Cumulative Returns")
st.plotly_chart(fig, use_container_width=True)

# --- Métricas del Portafolio ---
st.subheader("📈 Métricas del Portafolio")
col1, col2, col3 = st.columns(3)
col1.metric("Rendimiento Anual (%)", f"{metrics['Annual Return']:.2f}")
col2.metric("Volatilidad Anual (%)", f"{metrics['Annual Volatility']:.2f}")
col3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")

# ============================================================
# 🔹 Sección adicional: Holdings y sectores
# ============================================================

st.markdown("---")
st.subheader("🔍 ETF Holdings & Sector Analysis")

combined_holdings = combine_holdings(portfolio)
if not combined_holdings.empty:
    st.markdown("### 📊 Top Holdings Combinados")
    st.dataframe(combined_holdings.head(20), use_container_width=True)
    plot_company_exposure(combined_holdings)
else:
    st.warning("⚠️ No se encontraron holdings detallados para los ETFs seleccionados.")

# --- Exposición sectorial ---
st.markdown("### 🏭 Exposición Sectorial Combinada")

sector_df = combine_sector_exposure(portfolio)
if not sector_df.empty:
    st.dataframe(sector_df, use_container_width=True)
    plot_sector_pie(sector_df)
else:
    st.warning("⚠️ No se encontró información sectorial para los ETFs seleccionados.")
