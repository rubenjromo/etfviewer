import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
import numpy as np
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="ETF Analysis Tool",
    page_icon="🛠️",
    layout="wide"
)

# --- Logic Functions ---

def get_cagr(ticker, years=5):
    """Calculates the Compound Annual Growth Rate for a ticker."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        hist = ticker.history(start=start_date, end=end_date, auto_adjust=False, back_adjust=False)
        if hist.empty or len(hist) < 2: return None
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        actual_years = (hist.index[-1] - hist.index[0]).days / 365.25
        if actual_years < 1: return None
        if start_price > 0 and end_price > 0:
            return ((end_price / start_price) ** (1 / actual_years)) - 1
        return None
    except Exception:
        return None

def get_dividend_frequency(dividends):
    """Analyzes the last 12 months of dividends to infer frequency."""
    if dividends.empty: return "N/A"
    twelve_months_ago = datetime.now(timezone.utc) - timedelta(days=365)
    recent_dividends = dividends[dividends.index > twelve_months_ago]
    count = len(recent_dividends)
    if count >= 10: return "Monthly"
    elif count >= 3: return "Quarterly"
    elif count >= 1: return "Annual"
    else: return "Irregular"

@st.cache_data(ttl=3600)
def get_etf_metrics(ticker_symbol):
    """Fetches key comparable metrics for an ETF."""
    try:
        etf = yf.Ticker(ticker_symbol)
        info = etf.info
        if not info or info.get('quoteType') != 'ETF':
            st.warning(f"Could not get valid ETF data for {ticker_symbol}.", icon="⚠️")
            return None

        dividends = etf.dividends
        last_dividend = dividends.iloc[-1] if not dividends.empty else 0
        dividend_frequency = get_dividend_frequency(dividends)
        cagr_5y = get_cagr(etf, years=5)

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
            'Total Assets': info.get('totalAssets'),
        }
        return metrics
    except Exception:
        return None

# --- User Interface (UI) ---
st.title("🛠️ ETF Analysis Tool")
st.header("🆚 Key Metrics Comparator")
st.markdown("Enter multiple ETF tickers to compare their key financial metrics side-by-side.")
etf_input = st.text_area("Enter ETF tickers separated by commas or spaces", value="VOO, SCHD, QQQ, JEPI", help="Example: VOO SCHD QQQ JEPI")

if st.button("Compare ETFs"):
    tickers = [ticker.strip().upper() for ticker in etf_input.replace(',', ' ').split() if ticker.strip()]
    if tickers:
        with st.spinner("Fetching comparison data..."):
            all_metrics = [get_etf_metrics(ticker) for ticker in tickers if get_etf_metrics(ticker)]
        if all_metrics:
            st.success("Comparison data fetched successfully!")
            df_comp = pd.DataFrame(all_metrics).set_index('Ticker')
            st.dataframe(
                df_comp.style.format({
                    'Expense Ratio %': '{:.2f}%',
                    'Yield %': '{:.2f}%',
                    'YTD Return %': '{:.2f}%',
                    '5Y CAGR %': '{:.2f}%',
                    'Last Dividend': '${:,.4f}',
                    'Price': '${:,.2f}',
                    'Total Assets': '${:,.0f}'
                }, na_rep="N/A"),
                use_container_width=True
            )

st.header("📊 ETF Holdings Viewer")
st.markdown("Enter a single ETF ticker to view its top holdings. This uses an external data source and may take a moment.")
holdings_input = st.text_input("Enter a single ETF ticker for holdings analysis", value="SCHD")

if st.button("Get Holdings"):
    if holdings_input:
        ticker_str = holdings_input.strip().upper()
        with st.spinner(f"Fetching holdings for {ticker_str}..."):
            try:
                # CORRECTED: Removed the trailing slash from 'holdings/'. This is the definitive fix.
                url = f"https://finance.yahoo.com/quote/{ticker_str}/holdings?p={ticker_str}"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                
                tables = pd.read_html(response.text)
                
                if tables:
                    holdings_df = tables[0]
                    st.success(f"Top holdings for {ticker_str}:")
                    st.dataframe(holdings_df[['Name', 'Symbol', '% Assets']], use_container_width=True)
                else:
                    st.error(f"Could not find a holdings table for {ticker_str} on Yahoo Finance.")

            except Exception as e:
                st.error(f"Could not retrieve holdings for {ticker_str}. It may not be supported or the website structure changed. Error: {e}")

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info("This app provides tools for ETF analysis, including metric comparison and holdings data.")
bmac_link = "https://www.buymeacoffee.com/rubenjromo" 
st.sidebar.markdown(f"""<a href="{bmac_link}" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 50px !important;width: 200px !important;" ></a>""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.info("Created with ❤️ using Python and Streamlit.")
