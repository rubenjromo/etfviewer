import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
import numpy as np
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="ETF Analysis Tool",
    page_icon="🛠️",
    layout="wide"
)

# --- Logic Functions ---

def get_cagr(ticker, years=5):
    """Calculates the Compound Annual Growth Rate for a ticker's price."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        hist = ticker.history(start=start_date, end=end_date, auto_adjust=False, back_adjust=False)
        if hist.empty or len(hist) < 2:
            return None
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        actual_years = (hist.index[-1] - hist.index[0]).days / 365.25
        if actual_years < 1:
            return None
        if start_price > 0 and end_price > 0:
            return ((end_price / start_price) ** (1 / actual_years)) - 1
        return None
    except Exception:
        return None

def get_dividend_frequency(dividends):
    """Analyzes the last 12 months of dividends to infer frequency."""
    if dividends.empty:
        return "N/A"
    twelve_months_ago = datetime.now(timezone.utc) - timedelta(days=365)
    recent_dividends = dividends[dividends.index > twelve_months_ago]
    count = len(recent_dividends)
    if count >= 10:
        return "Monthly"
    elif count >= 3:
        return "Quarterly"
    elif count >= 1:
        return "Annual"
    else:
        return "Irregular"

@st.cache_data(ttl=3600)
def get_etf_metrics(ticker_symbol):
    """Fetches key comparable metrics for an ETF."""
    try:
        etf_yf = yf.Ticker(ticker_symbol)
        info = etf_yf.info
        if not info:
            # Handle non-ETF assets like BTC-USD
            if '-' in ticker_symbol:
                return {'Ticker': ticker_symbol, 'Name': ticker_symbol}
            st.warning(f"Could not get valid ETF data for {ticker_symbol}.", icon="⚠️")
            return None

        dividends = etf_yf.dividends
        last_dividend = dividends.iloc[-1] if not dividends.empty else 0
        dividend_frequency = get_dividend_frequency(dividends)
        cagr_5y = get_cagr(etf_yf, years=5)

        metrics = {
            'Ticker': info.get('symbol', ticker_symbol),
            'Name': info.get('shortName', 'N/A'),
            'Category': info.get('category', 'N/A'),
            'Expense Ratio %': info.get('netExpenseRatio') or 0,
            'Yield %': info.get('dividendYield') or 0,
            'YTD Return %': info.get('ytdReturn'),
            '5Y CAGR %': (cagr_5y * 100) if cagr_5y is not None else np.nan,
            'Last Dividend': last_dividend,
            'Price': info.get('regularMarketPrice'),
            'Dividend Frequency': dividend_frequency,
        }
        return metrics
    except Exception:
        # Gracefully handle assets with no ETF-specific data
        if '-' in ticker_symbol:
            return {'Ticker': ticker_symbol, 'Name': ticker_symbol}
        return None

@st.cache_data(ttl=3600)
def get_historical_prices(tickers):
    """Fetches 5-year historical closing prices for a list of tickers."""
    try:
        data = yf.download(tickers, period="5y", auto_adjust=True)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        return data.round(2)
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_performance_stats(price_history):
    """Calculates annualized volatility and Sharpe ratio."""
    if price_history is None or price_history.empty:
        return None
    
    daily_returns = price_history.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    risk_free_rate = 0.02
    avg_daily_return = daily_returns.mean()
    sharpe_ratio = (avg_daily_return * 252 - risk_free_rate) / volatility
    
    stats_df = pd.DataFrame({
        'Annualized Volatility (%)': volatility * 100,
        'Sharpe Ratio': sharpe_ratio
    })
    return stats_df

def get_normalized_growth(price_history):
    """Converts a price history DataFrame to a normalized growth DataFrame."""
    if price_history is None or price_history.empty:
        return None
    normalized = (price_history / price_history.bfill().iloc
