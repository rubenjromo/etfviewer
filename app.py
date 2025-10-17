import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
import numpy as np
import plotly.express as px
st.cache_resource.clear()

# --- Page Configuration ---
st.set_page_config(
    page_title="ETF Analysis Tool",
    page_icon="🛠️",
    layout="wide"
)

# --- Initialize Session State ---
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
    st.session_state.weighted_yield_decimal = 0.0
    st.session_state.weighted_cagr_decimal = 0.0
    st.session_state.portfolio_value = 10000.0

# --- Logic Functions ---

def normalize_yfinance_percent(value):
    """
    Robustly converts inconsistent percentage values from yfinance.
    Sometimes yfinance returns a ratio (0.03), sometimes a percentage (3.0).
    This function intelligently decides whether to multiply by 100.
    """
    if value is None or not isinstance(value, (int, float)):
        return np.nan
    val = float(value)
    # Heuristic: If the absolute value is greater than 1, it's likely already a percentage.
    if abs(val) > 1:
        return val
    # Otherwise, it's a ratio that needs to be converted.
    else:
        return val * 100.0

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
    try:
        if dividends is None or dividends.empty: return "N/A"
        twelve_months_ago = datetime.now(timezone.utc) - timedelta(days=365)
        recent_dividends = dividends[dividends.index > twelve_months_ago]
        count = len(recent_dividends)
        if count >= 10: return "Monthly"
        elif count >= 3: return "Quarterly"
        elif count >= 1: return "Annual"
        else: return "Irregular"
    except Exception:
        return "N/A"

@st.cache_data(ttl=3600)
def get_etf_metrics(ticker_symbol):
    """Fetches key comparable metrics for an ETF or asset with normalized values."""
    try:
        etf_yf = yf.Ticker(ticker_symbol)
        info = etf_yf.info
        if not info:
            if '-' in ticker_symbol: return {'Ticker': ticker_symbol, 'Name': ticker_symbol}
            st.warning(f"Could not get valid data for {ticker_symbol}.", icon="⚠️")
            return None

        dividends = etf_yf.dividends
        last_dividend = dividends.iloc[-1] if not dividends.empty else 0
        dividend_frequency = get_dividend_frequency(dividends)
        cagr_5y = get_cagr(etf_yf, years=5)

        metrics = {
            'Ticker': info.get('symbol', ticker_symbol),
            'Name': info.get('shortName', 'N/A'),
            'Category': info.get('category', 'N/A'),
            'Expense Ratio %': normalize_yfinance_percent(info.get('annualReportExpenseRatio')),
            'Yield %': normalize_yfinance_percent(info.get('yield')),
            'YTD Return %': normalize_yfinance_percent(info.get('ytdReturn')),
            '5Y CAGR %': (cagr_5y * 100) if cagr_5y is not None else np.nan,
            'Last Dividend': last_dividend,
            'Price': info.get('regularMarketPrice'),
            'Dividend Frequency': dividend_frequency,
        }
        return metrics
    except Exception:
        if '-' in ticker_symbol: return {'Ticker': ticker_symbol, 'Name': ticker_symbol}
        return None

def simulate_total_growth(initial_investment, annual_growth_decimal, dividend_yield_decimal, years):
    """Simulates total portfolio growth including price appreciation and reinvested dividends."""
    rows = []
    current_value = initial_investment
    for year in range(1, years + 2): # Loop one extra time to show the final value
        rows.append({'Year': year - 1, 'Projected Portfolio Value ($)': current_value})
        
        # Calculate growth and dividends based on the value at the START of the year
        price_appreciation = current_value * annual_growth_decimal
        dividends_earned = current_value * dividend_yield_decimal
        
        # Update value for the NEXT year (compounding)
        current_value += price_appreciation + dividends_earned
            
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600)
def get_historical_prices(tickers):
    """Fetches 5-year historical closing prices for a list of tickers."""
    try:
        if not tickers: return None
        data = yf.download(tickers, period="5y", auto_adjust=True, threads=True)['Close']
        if isinstance(data, pd.Series): data = data.to_frame(name=tickers[0])
        return data.round(2)
    except Exception:
        return None

# --- User Interface (UI) ---
st.title("🛠️ ETF & Asset Analysis Tool")
st.header("💵 Portfolio Analysis & Growth Projection")

col1, col2 = st.columns(2)
with col1:
    portfolio_input = st.text_area(
        "**1. Enter your assets and weights**",
        value="VOO 40\nSCHD 20\nQQQ 20\nCGDG 20",
        height=150, help="Use format 'TICKER WEIGHT'. The sum of weights should be 100."
    )
with col2:
    portfolio_value = st.number_input(
        "**2. Enter total portfolio value ($)**",
        min_value=0.0, value=10000.0, step=1000.0, format="%f"
    )

if st.button("Calculate & Analyze Portfolio"):
    lines = [line.strip() for line in portfolio_input.strip().split('\n') if line.strip()]
    portfolio = {}
    total_weight = 0.0
    valid_input = True

    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            st.error(f"Error in line: '{line}'. Use 'TICKER WEIGHT' format."); valid_input = False; break
        try:
            ticker, weight = parts[0].upper(), float(parts[1])
            portfolio[ticker] = weight
            total_weight += weight
        except ValueError:
            st.error(f"Invalid weight in line: '{line}'. Please use a number."); valid_input = False; break
    
    if abs(total_weight - 100.0) > 0.1 and valid_input:
        st.warning(f"The sum of weights is {total_weight}%. It is recommended that the sum be 100%.", icon="⚠️")

    if valid_input and portfolio:
        with st.spinner("Fetching data and calculating..."):
            all_metrics = []
            weighted_yield_sum_pct = 0.0
            weighted_expense_ratio_sum_pct = 0.0
            weighted_cagr_sum_pct = 0.0
            
            for ticker, weight in portfolio.items():
                metrics = get_etf_metrics(ticker)
                if metrics:
                    metrics['Portfolio Weight %'] = weight
                    all_metrics.append(metrics)
                    weighted_yield_sum_pct += np.nan_to_num(metrics.get('Yield %', 0)) * (weight / 100.0)
                    weighted_expense_ratio_sum_pct += np.nan_to_num(metrics.get('Expense Ratio %', 0)) * (weight / 100.0)
                    weighted_cagr_sum_pct += np.nan_to_num(metrics.get('5Y CAGR %', 0)) * (weight / 100.0)
            
            price_history = get_historical_prices(list(portfolio.keys()))
            
            # Save calculated weighted averages to session state for the projection
            st.session_state.analysis_run = True
            st.session_state.weighted_yield_decimal = weighted_yield_sum_pct / 100.0
            st.session_state.weighted_cagr_decimal = weighted_cagr_sum_pct / 100.0
            st.session_state.portfolio_value = portfolio_value

        if all_metrics:
            st.success("Analysis complete!")
            df_comp = pd.DataFrame(all_metrics).set_index('Ticker')
            
            st.subheader("Detailed Metrics Comparison")
            st.dataframe(
                df_comp.style.format({
                    'Portfolio Weight %': '{:.1f}%', 'Expense Ratio %': '{:.2f}%',
                    'Yield %': '{:.2f}%', 'YTD Return %': '{:.2f}%',
                    '5Y CAGR %': '{:.2f}%', 'Last Dividend': '${:,.4f}', 'Price': '${:,.2f}'
                }, na_rep="N/A"), use_container_width=True)
            
            st.markdown("---")
            st.subheader("Portfolio Summary")
            annual_income = portfolio_value * (weighted_yield_sum_pct / 100.0)
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric(label="**Weighted Avg Dividend Yield**", value=f"{weighted_yield_sum_pct:.2f}%")
            with col_metric2:
                st.metric(label="**Est. Annual Dividend Income**", value=f"${annual_income:,.2f}")
            with col_metric3:
                st.metric(label="**Weighted Avg Expense Ratio**", value=f"{weighted_expense_ratio_sum_pct:.2f}%")

            st.markdown("---")
            st.subheader("Annual Dividend Income Contribution")
            df_comp['Income Contribution ($)'] = (np.nan_to_num(df_comp['Yield %']) / 100.0) * (df_comp['Portfolio Weight %'] / 100.0) * portfolio_value
            fig_pie = px.pie(
                df_comp[df_comp['Income Contribution ($)'] > 0], 
                values='Income Contribution ($)', names=df_comp.index,
                title='Annual Dividend Projection by Asset', hole=.3
            )
            fig_pie.update_traces(texttemplate='%{label}: %{percent:.1%} <br>($%{value:,.2f})', textposition='inside')
            st.plotly_chart(fig_pie, use_container_width=True)

            if price_history is not None:
                st.markdown("---")
                st.subheader("5-Year Cumulative Growth (%)")
                growth_history = (price_history.resample('W').last().pct_change().cumsum() * 100).round(2)
                if not growth_history.empty:
                    st.line_chart(growth_history)
                else:
                    st.warning("Could not retrieve historical price data for growth chart.")

                if len(price_history.columns) > 1:
                    st.markdown("---")
                    st.subheader("Asset Correlation Heatmap")
                    st.info("Shows how similarly your assets move. **1.0** = move together. **0** = unrelated movement.", icon="💡")
                    corr = price_history.pct_change().corr()
                    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdYlGn', range_color=[-1,1])
                    st.plotly_chart(fig_corr, use_container_width=True)

# --- This section is now outside the button logic and uses session_state ---
if st.session_state.analysis_run:
    st.markdown("---")
    st.subheader("📈 Total Portfolio Growth Projection (with Reinvestment)")
    st.info("This simulation projects your **total portfolio value**. It assumes a constant annual growth rate (based on the weighted 5Y CAGR of your assets) and that all dividends are reinvested annually.", icon="💡")

    initial = st.number_input(
        "Initial portfolio value for projection ($)",
        value=float(st.session_state.portfolio_value),
        key='projection_value'
    )
    years_to_project = st.slider("Projection horizon (years)", 5, 40, 20, key='projection_years')

    if st.session_state.weighted_cagr_decimal > 0 or st.session_state.weighted_yield_decimal > 0:
        proj_df = simulate_total_growth(
            initial,
            st.session_state.weighted_cagr_decimal,
            st.session_state.weighted_yield_decimal,
            years_to_project
        )
        st.line_chart(proj_df.set_index('Year'))
    else:
        st.warning("No growth or dividend data available to create a projection.")

    st.info("""
        **Disclaimer:** This tool is for informational purposes only and does not constitute financial advice. All calculations are based on publicly available data which may not be 100% accurate. Always do your own research.
    """, icon="⚠️")

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info("This app provides portfolio analysis, performance metrics, and growth projections.")
bmac_link = "https://www.buymeacoffee.com/rubenjromo"
st.sidebar.markdown(
    f"""<a href="{bmac_link}" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 50px !important;width: 200px !important;" ></a>""",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
st.sidebar.info("Created with ❤️ using Python and Streamlit.")
