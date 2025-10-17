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
    st.session_state.weighted_div_growth_decimal = 0.0
    st.session_state.portfolio_value = 10000.0

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
    try:
        if dividends is None or dividends.empty:
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
    except Exception:
        return "N/A"

@st.cache_data(ttl=3600)
def get_etf_metrics(ticker_symbol):
    """Fetches key comparable metrics for an ETF or asset with normalized values."""
    try:
        etf_yf = yf.Ticker(ticker_symbol)
        info = etf_yf.info
        if not info:
            if '-' in ticker_symbol:
                return {'Ticker': ticker_symbol, 'Name': ticker_symbol}
            st.warning(f"Could not get valid data for {ticker_symbol}.", icon="⚠️")
            return None

        dividends = etf_yf.dividends
        last_dividend = dividends.iloc[-1] if not dividends.empty else 0
        dividend_frequency = get_dividend_frequency(dividends)
        cagr_5y = get_cagr(etf_yf, years=5)

        # --- CORRECTED LOGIC FOR PERCENTAGES ---
        # Convert ratio-based values (like yield and return) to percentages
        ytd_return_raw = info.get('ytdReturn')
        ytd_return_pct = ytd_return_raw * 100.0 if ytd_return_raw is not None else np.nan

        dividend_yield_raw = info.get('yield', info.get('dividendYield'))
        dividend_yield_pct = dividend_yield_raw * 100.0 if dividend_yield_raw is not None else np.nan
        
        # Handle expense ratio, which yfinance often provides as a percentage already
        expense_ratio_raw = info.get('annualReportExpenseRatio', info.get('netExpenseRatio'))
        expense_ratio_pct = expense_ratio_raw if expense_ratio_raw is not None else np.nan

        metrics = {
            'Ticker': info.get('symbol', ticker_symbol),
            'Name': info.get('shortName', 'N/A'),
            'Category': info.get('category', 'N/A'),
            'Expense Ratio %': expense_ratio_pct,
            'Yield %': dividend_yield_pct,
            'YTD Return %': ytd_return_pct,
            '5Y CAGR %': (cagr_5y * 100) if cagr_5y is not None else np.nan,
            'Last Dividend': last_dividend,
            'Price': info.get('regularMarketPrice'),
            'Dividend Frequency': dividend_frequency,
        }
        return metrics
    except Exception:
        if '-' in ticker_symbol:
            return {'Ticker': ticker_symbol, 'Name': ticker_symbol}
        return None

@st.cache_data(ttl=3600)
def get_historical_prices(tickers):
    """Fetches 5-year historical closing prices for a list of tickers."""
    try:
        if not tickers:
            return None
        data = yf.download(tickers, period="5y", auto_adjust=True, threads=True)['Close']
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
    if isinstance(volatility, pd.Series):
        avg_daily_return = daily_returns.mean()
        sharpe_ratio = (avg_daily_return * 252 - 0.02) / volatility
        stats_df = pd.DataFrame({
            'Annualized Volatility (%)': volatility * 100,
            'Sharpe Ratio': sharpe_ratio
        })
    else:
        avg_daily_return = daily_returns.mean()
        sharpe_ratio = (avg_daily_return * 252 - 0.02) / volatility if volatility != 0 else np.nan
        stats_df = pd.DataFrame({
            'Annualized Volatility (%)': [volatility * 100],
            'Sharpe Ratio': [sharpe_ratio]
        }, index=[price_history.columns[0] if len(price_history.columns)==1 else 0])
    return stats_df

def get_normalized_growth(price_history):
    """Converts a price history DataFrame to a normalized growth DataFrame."""
    if price_history is None or price_history.empty:
        return None
    sampled = price_history.resample('W').last()
    normalized = (sampled / sampled.bfill().iloc[0] - 1) * 100
    return normalized.round(2)

@st.cache_data(ttl=86400)
def get_dividend_series(symbol):
    """Return dividends Series for a symbol (cached)."""
    try:
        series = yf.Ticker(symbol).dividends
        if isinstance(series, pd.Series) and not series.empty:
            return series
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)

def compute_dividend_cagr(div_series, years=5):
    """Estimate CAGR of annual dividend sums over 'years'."""
    if div_series is None or div_series.empty:
        return None
    annual = div_series.resample('Y').sum()
    if len(annual) < 2:
        return None
    annual = annual.iloc[-years:]
    if len(annual.dropna()) < 2 or annual.iloc[0] <= 0:
        return None
    n = len(annual) - 1
    start = annual.iloc[0]
    end = annual.iloc[-1]
    if start <= 0:
        return None
    cagr = (end / start) ** (1 / n) - 1 if n > 0 else None
    return cagr

@st.cache_data(ttl=86400)
def get_dividend_metrics_for_projection(symbol):
    """Return only the yield and growth needed for the projection."""
    try:
        info = yf.Ticker(symbol).info or {}
        raw_yield = info.get('yield', info.get('dividendYield'))
        div_yield_pct = raw_yield * 100.0 if raw_yield is not None else np.nan
        
        div_series = get_dividend_series(symbol)
        div_cagr = compute_dividend_cagr(div_series, years=5)
        div_cagr_pct = div_cagr * 100 if div_cagr is not None else np.nan
        
        return {
            'Ticker': symbol,
            'Dividend Yield %': div_yield_pct,
            '5Y Dividend CAGR %': div_cagr_pct
        }
    except Exception:
        return {'Ticker': symbol, 'Dividend Yield %': np.nan, '5Y Dividend CAGR %': np.nan}

def simulate_dividend_growth(initial_investment, annual_yield_decimal, dividend_growth_decimal, years):
    """Simulate annual dividend income growth."""
    rows = []
    income = initial_investment * annual_yield_decimal
    for year in range(1, years + 1):
        rows.append({'Year': year, 'Annual Dividend Income ($)': income})
        income = income * (1 + dividend_growth_decimal)
    return pd.DataFrame(rows)

# --- User Interface (UI) ---
st.title("🛠️ ETF & Asset Analysis Tool")
st.header("💵 Portfolio Dividend Calculator")

col1, col2 = st.columns(2)
with col1:
    portfolio_input = st.text_area(
        "**1. Enter your assets and weights** (one per line)",
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
            st.error(f"Error in line: '{line}'. Use 'TICKER WEIGHT' format.")
            valid_input = False
            break
        try:
            ticker, weight = parts[0].upper(), float(parts[1])
            portfolio[ticker] = weight
            total_weight += weight
        except ValueError:
            st.error(f"Invalid weight in line: '{line}'. Please use a number.")
            valid_input = False
            break
    
    if abs(total_weight - 100.0) > 0.1 and valid_input:
        st.warning(f"The sum of weights is {total_weight}%. It is recommended that the sum be 100%.", icon="⚠️")

    if valid_input and portfolio:
        with st.spinner("Fetching data and calculating..."):
            all_metrics = []
            weighted_yield_sum_pct = 0.0
            weighted_expense_ratio_sum_pct = 0.0
            
            for ticker, weight in portfolio.items():
                metrics = get_etf_metrics(ticker)
                if metrics:
                    metrics['Portfolio Weight %'] = weight
                    all_metrics.append(metrics)
                    weighted_yield_sum_pct += np.nan_to_num(metrics.get('Yield %', 0)) * (weight / 100.0)
                    weighted_expense_ratio_sum_pct += np.nan_to_num(metrics.get('Expense Ratio %', 0)) * (weight / 100.0)
            
            price_history = get_historical_prices(list(portfolio.keys()))

            # --- Calculate projection data and save to session state ---
            div_metrics_proj = [get_dividend_metrics_for_projection(t) for t in portfolio.keys()]
            div_df_proj = pd.DataFrame(div_metrics_proj).set_index('Ticker')

            weighted_yield_decimal_proj = 0.0
            weighted_div_growth_decimal_proj = 0.0
            if not div_df_proj.empty:
                for t, w in portfolio.items():
                    if t in div_df_proj.index:
                        y_pct = div_df_proj.loc[t]['Dividend Yield %']
                        g_pct = div_df_proj.loc[t]['5Y Dividend CAGR %']
                        if not pd.isna(y_pct):
                            weighted_yield_decimal_proj += (y_pct / 100.0) * (w / 100.0)
                        if not pd.isna(g_pct):
                            weighted_div_growth_decimal_proj += (g_pct / 100.0) * (w / 100.0)
            
            st.session_state.analysis_run = True
            st.session_state.weighted_yield_decimal = weighted_yield_decimal_proj
            st.session_state.weighted_div_growth_decimal = weighted_div_growth_decimal_proj
            st.session_state.portfolio_value = portfolio_value

        if all_metrics:
            st.success("Analysis complete!")
            
            df_comp = pd.DataFrame(all_metrics).set_index('Ticker')
            
            st.subheader("Detailed Metrics Comparison")
            st.dataframe(
                df_comp.style.format({
                    'Portfolio Weight %': '{:.1f}%', 'Expense Ratio %': '{:.2f}%',
                    'Yield %': '{:.2f}%', 'YTD Return %': '{:.2f}%',
                    '5Y CAGR %': '{:.2f}%',
                    'Last Dividend': '${:,.4f}', 'Price': '${:,.2f}'
                }, na_rep="N/A"),
                use_container_width=True
            )
            
            st.markdown("---")
            st.subheader("🌟 Risk & Performance Analysis (5-Year)")
            performance_stats = get_performance_stats(price_history)
            if performance_stats is not None:
                st.dataframe(
                    performance_stats.style.format({
                        'Annualized Volatility (%)': '{:.2f}%',
                        'Sharpe Ratio': '{:.2f}'
                    }),
                    use_container_width=True
                )

            st.markdown("---")
            st.subheader("5-Year Cumulative Growth (%)")
            growth_history = get_normalized_growth(price_history)
            if growth_history is not None and not growth_history.empty:
                st.line_chart(growth_history)
            else:
                st.warning("Could not retrieve historical price data for the selected assets.")

            st.markdown("---")
            st.subheader("Portfolio Summary")
            annual_income = portfolio_value * (weighted_yield_sum_pct / 100.0)
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric(label="**Weighted Average Dividend Yield**", value=f"{weighted_yield_sum_pct:.2f}%")
            with col_metric2:
                st.metric(label="**Estimated Annual Dividend Income**", value=f"${annual_income:,.2f}")
            with col_metric3:
                st.metric(label="**Weighted Average Expense Ratio**", value=f"{weighted_expense_ratio_sum_pct:.2f}%")
            
            st.markdown("---")
            st.subheader("Annual Dividend Income Contribution")
            df_comp['Income Contribution ($)'] = (np.nan_to_num(df_comp['Yield %']) / 100.0) * (df_comp['Portfolio Weight %'] / 100.0) * portfolio_value
            fig_pie = px.pie(
                df_comp, values='Income Contribution ($)', names=df_comp.index,
                title='Annual Dividend Projection by ETF', hole=.3
            )
            fig_pie.update_traces(texttemplate='%{label}: %{percent:.1%} <br>($%{value:,.2f})', textposition='inside')
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")
            st.subheader("Asset Correlation Heatmap")
            st.info("This shows how similarly your assets move. A value near **1.0** means they move together (low diversification). A value near **0** means their movements are unrelated (high diversification).", icon="💡")

            if price_history is not None and len(price_history.columns) > 1:
                corr = price_history.pct_change().corr()
                fig_corr = px.imshow(
                    corr, text_auto=True, aspect="auto",
                    color_continuous_scale='RdYlGn', range_color=[-1,1],
                    title="Correlation of Daily Price Movements"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

# --- This section is now outside the button logic and uses session_state ---
if st.session_state.analysis_run:
    st.markdown("---")
    st.subheader("📈 Dividend Income Projection")
    st.info("This simulation projects your **annual dividend income** over time based on the portfolio's weighted average yield and dividend growth rate. It does not account for price changes or reinvestment.", icon="💡")

    initial = st.number_input(
        "Portfolio value for projection ($)",
        value=float(st.session_state.portfolio_value),
        key='projection_value'
    )
    years = st.slider("Projection horizon (years)", 5, 40, 20, key='projection_years')

    if st.session_state.weighted_yield_decimal > 0:
        proj_df = simulate_dividend_growth(
            initial,
            st.session_state.weighted_yield_decimal,
            st.session_state.weighted_div_growth_decimal,
            years
        )
        st.line_chart(proj_df.set_index('Year'))
    else:
        st.warning("No dividend data available in the portfolio to create a projection.")

    st.info("""
        **Disclaimer:** This tool is for informational purposes only and does not constitute financial advice. All calculations are based on publicly available data which may not be 100% accurate. Always do your own research before making any investment decisions.
    """, icon="⚠️")


# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info("This app provides a portfolio dividend, holdings, and exposure analysis tool.")
bmac_link = "https://www.buymeacoffee.com/rubenjromo"
st.sidebar.markdown(
    f"""<a href="{bmac_link}" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 50px !important;width: 200px !important;" ></a>""",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
st.sidebar.info("Created with ❤️ using Python and Streamlit.")
