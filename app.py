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
        }
        return metrics
    except Exception:
        return None

# NEW: A dedicated, reliable function for getting holdings via yfinance
@st.cache_data(ttl=3600)
def get_etf_holdings(ticker_symbol):
    """Gets top holdings for an ETF using the yfinance library's funds_data attribute."""
    try:
        etf = yf.Ticker(ticker_symbol)
        holdings_data = etf.funds_data.get('top_holdings')
        if holdings_data:
            df = pd.DataFrame(holdings_data)
            # Rename columns for clarity and consistency
            df = df.rename(columns={'holdingName': 'Company', 'holdingPercent': '% Assets'})
            return df
        else:
            return None
    except Exception:
        return None

# --- User Interface (UI) ---
st.title("🛠️ ETF Analysis & Portfolio Tool")

st.header("💵 Portfolio Dividend Calculator")
col1, col2 = st.columns(2)
with col1:
    portfolio_input = st.text_area(
        "**1. Enter your ETFs and weights** (one per line)",
        value="VOO 50\nSCHD 30\nQQQ 20",
        height=150, help="Use the format 'TICKER WEIGHT'. The sum of weights should be 100."
    )
with col2:
    portfolio_value = st.number_input(
        "**2. Enter total portfolio value ($)**",
        min_value=0.0, value=10000.0, step=1000.0, format="%f"
    )

if st.button("Calculate & Analyze Portfolio"):
    lines = [line.strip() for line in portfolio_input.strip().split('\n') if line.strip()]
    portfolio = {}
    total_weight = 0
    valid_input = True
    for line in lines:
        parts = line.split()
        if len(parts) != 2: st.error(f"Error in line: '{line}'. Use 'TICKER WEIGHT' format."); valid_input = False; break
        try:
            ticker, weight = parts[0].upper(), float(parts[1])
            portfolio[ticker] = weight
            total_weight += weight
        except ValueError: st.error(f"Invalid weight in line: '{line}'. Please use a number."); valid_input = False; break
    
    if abs(total_weight - 100.0) > 0.1 and valid_input:
        st.warning(f"The sum of weights is {total_weight}%. It is recommended that the sum be 100%.", icon="⚠️")

    if valid_input and portfolio:
        with st.spinner("Fetching data and calculating..."):
            all_metrics = []
            weighted_yield_sum = 0
            for ticker, weight in portfolio.items():
                metrics = get_etf_metrics(ticker)
                if metrics:
                    metrics['Portfolio Weight %'] = weight
                    all_metrics.append(metrics)
                    weighted_yield_sum += (metrics.get('Yield %', 0) or 0) * (weight / 100.0)
        
        if all_metrics:
            st.success("Analysis complete!")
            df_comp = pd.DataFrame(all_metrics).set_index('Ticker')
            cols = ['Portfolio Weight %'] + [col for col in df_comp.columns if col != 'Portfolio Weight %']
            df_comp = df_comp[cols]
            
            st.subheader("Detailed Metrics Comparison")
            st.dataframe(
                df_comp.style.format({
                    'Portfolio Weight %': '{:.1f}%', 'Expense Ratio %': '{:.2f}%',
                    'Yield %': '{:.2f}%', 'YTD Return %': '{:.2f}%', '5Y CAGR %': '{:.2f}%',
                    'Last Dividend': '${:,.4f}', 'Price': '${:,.2f}'
                }, na_rep="N/A"),
                use_container_width=True
            )
            
            st.markdown("---")
            st.subheader("Portfolio Summary")
            annual_income = portfolio_value * (weighted_yield_sum / 100.0)
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric(label="**Weighted Average Dividend Yield**", value=f"{weighted_yield_sum:.2f}%")
            with col_metric2:
                st.metric(label="**Estimated Annual Dividend Income**", value=f"${annual_income:,.2f}")
            
            st.markdown("---")
            st.subheader("Annual Dividend Income Contribution")
            df_comp['Income Contribution ($)'] = (df_comp['Yield %'] / 100) * (df_comp['Portfolio Weight %'] / 100) * portfolio_value
            fig = px.pie(df_comp, values='Income Contribution ($)', names=df_comp.index, title='Annual Dividend Projection by ETF', hole=.3)
            fig.update_traces(texttemplate='%{label}: %{percent:.1%} <br>($%{value:,.2f})', textposition='inside')
            st.plotly_chart(fig, use_container_width=True)

# --- Holdings Viewer Section ---
st.markdown("---")
st.header("📊 ETF Holdings Viewer")
st.markdown("Enter a single ETF ticker to view its top holdings directly from `yfinance`.")
holdings_input = st.text_input("Enter a single ETF ticker", value="SCHD")

if st.button("Get Holdings"):
    if holdings_input:
        ticker_str = holdings_input.strip().upper()
        with st.spinner(f"Fetching holdings for {ticker_str}..."):
            holdings_df = get_etf_holdings(ticker_str)
            if holdings_df is not None:
                st.success(f"Top holdings for {ticker_str}:")
                st.dataframe(holdings_df[['Company', 'symbol', '% Assets']], use_container_width=True)
            else:
                st.error(f"Could not retrieve holdings for {ticker_str}. The data may not be available for this ETF via the yfinance library.")

# --- Disclaimer and Sidebar ---
st.info("""
    **Disclaimer:** This tool is for informational purposes only and does not constitute financial advice. All calculations are based on publicly available data which may not be 100% accurate. Always do your own research before making any investment decisions.
""", icon="⚠️")

st.sidebar.header("About")
st.sidebar.info("This app provides tools for ETF analysis, including a portfolio dividend calculator and a holdings viewer.")
bmac_link = "https://www.buymeacoffee.com/rubenjromo" 
st.sidebar.markdown(f"""<a href="{bmac_link}" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 50px !important;width: 200px !important;" ></a>""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.info("Created with ❤️ using Python and Streamlit.")
