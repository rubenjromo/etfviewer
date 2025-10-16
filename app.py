import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone

# --- Page Configuration ---
st.set_page_config(
    page_title="ETF Comparator",
    page_icon="🆚",
    layout="wide"
)

# --- Logic Functions ---

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
    """Fetches key comparable metrics from the .info dictionary of an ETF."""
    try:
        etf = yf.Ticker(ticker_symbol)
        info = etf.info
        
        if not info or info.get('quoteType') != 'ETF':
            st.warning(f"Could not get valid ETF data for {ticker_symbol}.", icon="⚠️")
            return None

        dividends = etf.dividends
        last_dividend = dividends.iloc[-1] if not dividends.empty else 0
        dividend_frequency = get_dividend_frequency(dividends)

        metrics = {
            'Ticker': info.get('symbol', ticker_symbol),
            'Name': info.get('shortName', 'N/A'),
            'Category': info.get('category', 'N/A'),
            'Expense Ratio %': info.get('netExpenseRatio') or 0,
            'Yield %': info.get('dividendYield') or 0,
            'YTD Return %': info.get('ytdReturn'),
            'Last Dividend': last_dividend,
            'Price': info.get('regularMarketPrice'),
            'Dividend Frequency': dividend_frequency,
            'Beta (3Y)': info.get('beta3Year'),
            'Total Assets': info.get('totalAssets'),
        }
        return metrics

    except Exception as e:
        st.error(f"Failed to process {ticker_symbol}. Error: {e}", icon="🚨")
        return None

# --- User Interface (UI) ---
st.title("🆚 ETF Key Metrics Comparator")
st.markdown("Enter the tickers of the ETFs you want to compare. The app will fetch their key metrics and display them in a comparison table.")

st.sidebar.header("Enter ETFs to Compare")

bmac_link = "https://www.buymeacoffee.com/rubenjromo" 
st.sidebar.markdown(f"""
<a href="{bmac_link}" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 50px !important;width: 200px !important;" >
</a>
""", unsafe_allow_html=True)

etf_input = st.sidebar.text_area(
    "Enter ETF tickers separated by commas or spaces",
    value="VOO, SCHD, QQQ, JEPI",
    help="Example: VOO SCHD QQQ JEPI"
)

if st.sidebar.button("Compare ETFs"):
    tickers = [ticker.strip().upper() for ticker in etf_input.replace(',', ' ').split() if ticker.strip()]
    
    if tickers:
        with st.spinner("Fetching data for all ETFs..."):
            all_metrics = []
            progress_bar = st.progress(0)
            for i, ticker in enumerate(tickers):
                metrics = get_etf_metrics(ticker)
                if metrics:
                    all_metrics.append(metrics)
                progress_bar.progress((i + 1) / len(tickers))

        if all_metrics:
            st.success("Comparison data fetched successfully!")
            
            df = pd.DataFrame(all_metrics).set_index('Ticker')

            st.dataframe(
                df.style.format({
                    'Price': '${:,.2f}',
                    'Expense Ratio %': '{:,.2f}%',
                    'Yield %': '{:,.2f}%',
                    'YTD Return %': '{:,.2f}%',
                    'Last Dividend': '${:,.4f}',
                    'Beta (3Y)': '{:,.2f}',
                    'Total Assets': '{:,.0f}'
                }),
                use_container_width=True
            )
    else:
        st.warning("Please enter at least one ETF ticker.")

st.sidebar.markdown("---")
st.sidebar.info("Created with ❤️ using Python and Streamlit.")
