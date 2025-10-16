import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
import numpy as np

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
        if hist.empty:
            return None
        
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        
        if start_price > 0 and end_price > 0:
            return ((end_price / start_price) ** (1 / years)) - 1
        return None
    except Exception:
        return None

def get_dividend_growth_cagr(dividends, years=5):
    """Calculates the CAGR of annual dividend sums."""
    try:
        if dividends.empty:
            return None
        
        # We need years+1 of data to calculate 'years' of growth
        start_year = datetime.now().year - (years + 1)
        
        annual_dividends = dividends[dividends.index.year > start_year].resample('YE').sum()
        
        # Need at least two data points to calculate growth
        if len(annual_dividends) < 2:
            return None
            
        # Drop years with zero dividends to avoid division errors
        annual_dividends = annual_dividends[annual_dividends > 0]
        if len(annual_dividends) < 2:
            return None

        start_value = annual_dividends.iloc[0]
        end_value = annual_dividends.iloc[-1]
        num_years = annual_dividends.index.year[-1] - annual_dividends.index.year[0]

        if num_years > 0:
            return ((end_value / start_value) ** (1 / num_years)) - 1
        return None
    except Exception:
        return None

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
        cagr_5y = get_cagr(etf, years=5)
        div_growth_5y = get_dividend_growth_cagr(dividends, years=5)

        metrics = {
            'Ticker': info.get('symbol', ticker_symbol),
            'Name': info.get('shortName', 'N/A'),
            'Category': info.get('category', 'N/A'),
            'Expense Ratio %': (info.get('netExpenseRatio') or 0) * 100,
            'Yield %': (info.get('yield') or 0) * 100,
            '5Y CAGR %': (cagr_5y * 100) if cagr_5y is not None else np.nan,
            '5Y Div. Growth %': (div_growth_5y * 100) if div_growth_5y is not None else np.nan,
            'Total Assets': info.get('totalAssets'),
        }
        return metrics
    except Exception:
        return None

# --- User Interface (UI) ---
st.title("🛠️ ETF Analysis Tool")

# --- Section 1: ETF Comparator ---
st.header("🆚 Key Metrics Comparator")
st.markdown("Enter multiple ETF tickers to compare their key financial metrics side-by-side.")

etf_input = st.text_area(
    "Enter ETF tickers separated by commas or spaces",
    value="VOO, SCHD, QQQ, JEPI",
    help="Example: VOO SCHD QQQ JEPI"
)

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
                    '5Y CAGR %': '{:.2f}%',
                    '5Y Div. Growth %': '{:.2f}%',
                    'Total Assets': '${:,.0f}'
                }, na_rep="N/A"),
                use_container_width=True
            )

# --- Section 2: Holdings Viewer ---
st.header("📊 ETF Holdings Viewer")
st.markdown("Enter a single ETF ticker to view its top 15 holdings. This uses a different data source and may take a moment.")

holdings_input = st.text_input("Enter a single ETF ticker for holdings analysis", value="SCHD")

if st.button("Get Holdings"):
    if holdings_input:
        ticker_str = holdings_input.strip().upper()
        with st.spinner(f"Fetching holdings for {ticker_str}..."):
            try:
                # This is a common workaround to get holdings data
                url = f"https://www.slickcharts.com/etf/{ticker_str}"
                df_holdings = pd.read_html(url, attrs = {'class': 'table table-hover table-borderless table-sm'})[0]
                st.success(f"Top holdings for {ticker_str}:")
                st.dataframe(df_holdings[['Company', 'Symbol', 'Weight']], use_container_width=True)
            except Exception as e:
                st.error(f"Could not retrieve holdings for {ticker_str}. It may not be supported by the data source. Error: {e}")

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info("This app provides tools for ETF analysis, including metric comparison and holdings data.")
bmac_link = "https://www.buymeacoffee.com/rubenjromo" 
st.sidebar.markdown(f"""
<a href="{bmac_link}" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 50px !important;width: 200px !important;" >
</a>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.info("Created with ❤️ using Python and Streamlit.")
