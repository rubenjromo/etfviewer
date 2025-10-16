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
        etf_yf = yf.Ticker(ticker_symbol)
        info = etf_yf.info
        if not info or info.get('quoteType') != 'ETF':
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
            'Total Assets': info.get('totalAssets'),
        }
        return metrics
    except Exception:
        return None

# --- User Interface (UI) ---
st.title("🛠️ ETF Analysis & Portfolio Tool")
st.header("💵 Portfolio Yield Calculator")
st.markdown("Enter your ETFs and their portfolio weight (%) to calculate your weighted average dividend yield.")

# NEW: Input area for portfolio weights
portfolio_input = st.text_area(
    "Enter each ETF on a new line followed by its weight. (e.g., VOO 50)",
    value="VOO 50\nSCHD 30\nQQQ 20",
    height=150
)

if st.button("Calculate & Compare ETFs"):
    lines = [line.strip() for line in portfolio_input.strip().split('\n') if line.strip()]
    portfolio = {}
    total_weight = 0
    
    # --- 1. Parse Input ---
    valid_input = True
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            st.error(f"Error in line: '{line}'. Please use the format 'TICKER WEIGHT'.")
            valid_input = False
            break
        try:
            ticker, weight = parts[0].upper(), float(parts[1])
            portfolio[ticker] = weight
            total_weight += weight
        except ValueError:
            st.error(f"Invalid weight in line: '{line}'. Please use a number for the weight.")
            valid_input = False
            break

    if total_weight != 100 and valid_input:
        st.warning(f"The sum of weights is {total_weight}%, not 100%. The calculation will be adjusted accordingly.", icon="⚠️")

    # --- 2. Fetch Data & Calculate ---
    if valid_input and portfolio:
        with st.spinner("Fetching data and calculating..."):
            all_metrics = []
            weighted_yield_sum = 0
            
            for ticker, weight in portfolio.items():
                metrics = get_etf_metrics(ticker)
                if metrics:
                    metrics['Portfolio Weight %'] = weight
                    all_metrics.append(metrics)
                    # Add to the weighted yield sum
                    weighted_yield_sum += (metrics.get('Yield %', 0) or 0) * (weight / 100.0)

        # --- 3. Display Results ---
        if all_metrics:
            st.success("Analysis complete!")
            
            # Display the main result: Weighted Average Yield
            st.metric(
                label="**Portfolio's Weighted Average Dividend Yield**",
                value=f"{weighted_yield_sum:.2f}%"
            )
            
            # Display the detailed comparison table
            st.markdown("---")
            st.subheader("Detailed Metrics Comparison")
            df_comp = pd.DataFrame(all_metrics).set_index('Ticker')
            
            # Reorder columns to show weight first
            cols = ['Portfolio Weight %'] + [col for col in df_comp.columns if col != 'Portfolio Weight %']
            df_comp = df_comp[cols]

            st.dataframe(
                df_comp.style.format({
                    'Portfolio Weight %': '{:.1f}%',
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

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info("This app provides tools for ETF analysis. The main feature is a portfolio yield calculator based on user-defined weights.")
bmac_link = "https://www.buymeacoffee.com/rubenjromo" 
st.sidebar.markdown(f"""<a href="{bmac_link}" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 50px !important;width: 200px !important;" ></a>""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.info("Created with ❤️ using Python and Streamlit.")
