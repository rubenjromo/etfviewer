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

def _percent_from_maybe_decimal(value):
    """
    Convert a numeric value that may be a decimal fraction (0.03) or a percent (3)
    into a percent number (e.g. 3.0).
    If value is None -> return 0.0
    """
    try:
        if value is None:
            return 0.0
        v = float(value)
        # if value is sensible fraction (<1 and >0) treat as decimal
        if abs(v) < 1:
            return v * 100.0
        return v

    except Exception:
        return 0.0

@st.cache_data(ttl=3600)
def get_etf_metrics(ticker_symbol):
    """Fetches key comparable metrics for an ETF."""
    try:
        etf_yf = yf.Ticker(ticker_symbol)
        info = etf_yf.info or {}
        if not info:
            # Handle non-ETF assets like BTC-USD
            if '-' in ticker_symbol:
                return {'Ticker': ticker_symbol, 'Name': ticker_symbol}
            st.warning(f"Could not get valid ETF data for {ticker_symbol}.", icon="⚠️")
            return None

        # Dividends series
        dividends = etf_yf.dividends if hasattr(etf_yf, "dividends") else pd.Series(dtype=float)
        last_dividend = dividends.iloc[-1] if (isinstance(dividends, pd.Series) and not dividends.empty) else 0
        dividend_frequency = get_dividend_frequency(dividends if isinstance(dividends, pd.Series) else pd.Series())
        cagr_5y = get_cagr(etf_yf, years=5)

        # Normalize yields / expense ratios to percent values
        raw_yield = info.get('dividendYield', None)
        yield_pct = _percent_from_maybe_decimal(raw_yield)

        raw_expense = info.get('netExpenseRatio', None)
        expense_pct = _percent_from_maybe_decimal(raw_expense)

        metrics = {
            'Ticker': info.get('symbol', ticker_symbol),
            'Name': info.get('shortName', 'N/A'),
            'Category': info.get('category', 'N/A') if info.get('category') else info.get('quoteType', 'N/A'),
            'Expense Ratio %': expense_pct,
            'Yield %': yield_pct,
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
        if not tickers:
            return None
        # threads=True for parallel fetch; cached so repeated runs are cheap
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
    # If volatility is a Series (multiple tickers), compute per-column
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
    # Downsample weekly to improve plotting performance
    sampled = price_history.resample('W').last()
    normalized = (sampled / sampled.bfill().iloc[0] - 1) * 100
    return normalized.round(2)

# --- ETF Holdings and Exposure Analysis ---

@st.cache_data(ttl=3600)
def get_etf_holdings(ticker_symbol):
    """Get ETF holdings (if available)."""
    try:
        etf = yf.Ticker(ticker_symbol)
        holdings = getattr(etf, "funds_holdings", None)
        # Some yfinance versions expose as etf.funds_holdings, sometimes None
        if holdings is None or (isinstance(holdings, (pd.DataFrame, pd.Series)) and getattr(holdings, "empty", True)):
            return pd.DataFrame()
        # Robust mapping for column names that may vary across versions
        df = holdings.copy()
        col_map = {}
        if 'symbol' in df.columns:
            col_map['symbol'] = 'Ticker'
        elif 'holdingSymbol' in df.columns:
            col_map['holdingSymbol'] = 'Ticker'
        if 'holdingName' in df.columns:
            col_map['holdingName'] = 'Company'
        elif 'name' in df.columns:
            col_map['name'] = 'Company'
        if 'holdingPercent' in df.columns:
            col_map['holdingPercent'] = 'Weight'
        elif 'weight' in df.columns:
            col_map['weight'] = 'Weight'
        available = [c for c in col_map.keys() if c in df.columns]
        if not available:
            return pd.DataFrame()
        df = df[available].rename(columns=col_map)
        df['ETF'] = ticker_symbol
        # Normalize weight to percent scale
        if df['Weight'].max() <= 1:
            df['Weight'] = df['Weight'] * 100
        return df[['Ticker', 'Company', 'Weight', 'ETF']]
    except Exception:
        return pd.DataFrame()

def combine_holdings(portfolio):
    """Combine holdings from all ETFs weighted by portfolio allocation."""
    combined = pd.DataFrame()
    for etf, etf_weight in portfolio.items():
        holdings = get_etf_holdings(etf)
        if not holdings.empty:
            h = holdings.copy()
            h['AdjustedWeight'] = h['Weight'] * (etf_weight / 100.0)
            combined = pd.concat([combined, h], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    final = combined.groupby(['Ticker', 'Company'], as_index=False)['AdjustedWeight'].sum()
    final['AdjustedWeight'] = final['AdjustedWeight'].round(6)
    final.sort_values('AdjustedWeight', ascending=False, inplace=True)
    return final

@st.cache_data(ttl=86400)
def get_company_info(symbol):
    """Get country and sector for a company via yfinance.info"""
    try:
        data = yf.Ticker(symbol).info or {}
        return {
            'Ticker': symbol,
            'Country': data.get('country', 'Unknown'),
            'Sector': data.get('sector', 'Unknown')
        }
    except Exception:
        return {'Ticker': symbol, 'Country': 'Unknown', 'Sector': 'Unknown'}

def enrich_holdings_with_info(df):
    if df is None or df.empty:
        return df
    unique = df['Ticker'].unique().tolist()
    info_list = [get_company_info(t) for t in unique]
    info_df = pd.DataFrame(info_list)
    df = df.merge(info_df, on='Ticker', how='left')
    return df

def plot_company_exposure(df):
    fig = px.bar(df.head(15), x='Company', y='AdjustedWeight',
                 title="Top 15 Holdings (Weighted by Portfolio)",
                 labels={'AdjustedWeight': 'Portfolio Exposure (%)'})
    fig.update_layout(xaxis_tickangle=-45, yaxis_title="Exposure (%)")
    st.plotly_chart(fig, use_container_width=True)

def plot_country_map(df):
    df2 = df[df['Country'].notna() & (df['Country'] != 'Unknown')]
    if df2.empty:
        st.info("No country-level data available to build a map.")
        return
    country_weights = df2.groupby('Country', as_index=False)['AdjustedWeight'].sum()
    fig = px.choropleth(country_weights, locations="Country",
                        locationmode="country names",
                        color="AdjustedWeight",
                        title="Portfolio Exposure by Country",
                        color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

def plot_sector_pie(df):
    sector_weights = df.groupby('Sector', as_index=False)['AdjustedWeight'].sum()
    if sector_weights.empty:
        st.info("No sector-level data available.")
        return
    fig = px.pie(sector_weights, values='AdjustedWeight', names='Sector',
                 title='Portfolio Exposure by Sector', hole=0.3)
    st.plotly_chart(fig, use_container_width=True)

# --- Dividend Growth (DGI) helpers ---

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
def get_dividend_metrics(symbol):
    """Return useful dividend metrics for an asset."""
    try:
        info = yf.Ticker(symbol).info or {}
        raw_yield = info.get('dividendYield', None)
        div_yield_pct = _percent_from_maybe_decimal(raw_yield)
        raw_payout = info.get('payoutRatio', None)
        payout_pct = _percent_from_maybe_decimal(raw_payout)
        div_series = get_dividend_series(symbol)
        div_cagr = compute_dividend_cagr(div_series, years=5)
        div_cagr_pct = div_cagr * 100 if div_cagr is not None else np.nan
        return {
            'Ticker': symbol,
            'Dividend Yield %': div_yield_pct,
            'Payout Ratio %': payout_pct,
            '5Y Dividend CAGR %': div_cagr_pct
        }
    except Exception:
        return {'Ticker': symbol, 'Dividend Yield %': np.nan, 'Payout Ratio %': np.nan, '5Y Dividend CAGR %': np.nan}

def simulate_dividend_growth(initial_investment, annual_yield_decimal, dividend_growth_decimal, years):
    """Simulate annual dividend income growth (not total portfolio growth).
       annual_yield_decimal must be in decimal (e.g. 0.03 for 3%)."""
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
        value="VOO 40\nSCHD 20\nQQQ 20\nBTC-USD 20",
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
            weighted_yield_sum_pct = 0.0  # stored as percent (e.g. 3.5)
            weighted_expense_ratio_sum_pct = 0.0
            
            # fetch metrics for each ticker (cached)
            for ticker, weight in portfolio.items():
                metrics = get_etf_metrics(ticker)
                if metrics:
                    metrics['Portfolio Weight %'] = weight
                    all_metrics.append(metrics)
                    weighted_yield_sum_pct += (metrics.get('Yield %', 0) or 0.0) * (weight / 100.0)
                    weighted_expense_ratio_sum_pct += (metrics.get('Expense Ratio %', 0) or 0.0) * (weight / 100.0)
            
            # Get price_history only once for all tickers — cached
            price_history = get_historical_prices(list(portfolio.keys()))

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
            # weighted_yield_sum_pct is percent (e.g. 3.5 for 3.5%)
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
            df_comp['Income Contribution ($)'] = (df_comp['Yield %'] / 100.0) * (df_comp['Portfolio Weight %'] / 100.0) * portfolio_value
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

            # --- NEW SECTION: ETF Holdings and Exposure ---
            st.markdown("---")
            st.subheader("🔍 ETF Holdings & Exposure Analysis")

            # Lazy load: user chooses to fetch holdings / sectors
            col_h1, col_h2 = st.columns([1, 2])
            with col_h1:
                fetch_holdings = st.checkbox("Fetch detailed holdings (may be slower)", value=False)
            with col_h2:
                fetch_sectors = st.checkbox("Show combined sector exposure", value=True)

            combined_holdings = pd.DataFrame()
            if fetch_holdings:
                combined_holdings = combine_holdings(portfolio)
                if not combined_holdings.empty:
                    enriched = enrich_holdings_with_info(combined_holdings)
                    st.markdown("### 📊 Combined Top Holdings")
                    enriched_display = enriched.copy()
                    enriched_display['AdjustedWeight (%)'] = (enriched_display['AdjustedWeight'] ).round(4)
                    st.dataframe(enriched_display.sort_values('AdjustedWeight', ascending=False).head(20), use_container_width=True)
                    plot_company_exposure(enriched_display)
                    plot_country_map(enriched_display)
                else:
                    st.warning("No holdings data available for the selected ETFs (via yfinance).")

            # Sector exposure aggregated from ETF-level sector breakdown
            if fetch_sectors:
                sector_combined = pd.DataFrame()
                for etf, w in portfolio.items():
                    try:
                        s = yf.Ticker(etf).fund_sector_weightings or {}
                    except Exception:
                        s = {}
                    if s:
                        df_s = pd.DataFrame(list(s.items()), columns=['Sector', 'Weight'])
                        df_s['Weight'] = df_s['Weight'] * 100  # convert to percent
                        df_s['AdjustedWeight'] = df_s['Weight'] * (w / 100.0)
                        sector_combined = pd.concat([sector_combined, df_s[['Sector','AdjustedWeight']]], ignore_index=True)
                if not sector_combined.empty:
                    sector_final = sector_combined.groupby('Sector', as_index=False)['AdjustedWeight'].sum()
                    sector_final.sort_values('AdjustedWeight', ascending=False, inplace=True)
                    sector_final['AdjustedWeight (%)'] = sector_final['AdjustedWeight'].round(4)
                    st.markdown("### 🏭 Combined Sector Exposure")
                    st.dataframe(sector_final[['Sector','AdjustedWeight (%)']], use_container_width=True)
                    plot_sector_pie(sector_final.rename(columns={'AdjustedWeight': 'AdjustedWeight'}))
                else:
                    st.warning("No sector weighting data available for the selected ETFs (via yfinance).")

            # --- Dividend Growth Features (optional) ---
            st.markdown("---")
            st.subheader("🧠 Dividend Growth (DGI) Analysis")
            with st.expander("Show Dividend Growth metrics & projections"):
                div_metrics = []
                for t in portfolio.keys():
                    dm = get_dividend_metrics(t)
                    div_metrics.append(dm)
                div_df = pd.DataFrame(div_metrics).set_index('Ticker')
                if not div_df.empty:
                    st.markdown("#### Dividend Metrics per Asset")
                    st.dataframe(div_df.style.format({
                        'Dividend Yield %': '{:.2f}%', 'Payout Ratio %': '{:.2f}%', '5Y Dividend CAGR %': '{:.2f}%'
                    }, na_rep="N/A"), use_container_width=True)

                    # Compute portfolio weighted metrics using decimals for simulation
                    weighted_yield_decimal = 0.0  # decimal (e.g. 0.03)
                    weighted_div_growth_decimal = 0.0
                    for t, w in portfolio.items():
                        if t in div_df.index:
                            y_pct = div_df.loc[t]['Dividend Yield %']
                            g_pct = div_df.loc[t]['5Y Dividend CAGR %']
                            if not pd.isna(y_pct):
                                weighted_yield_decimal += (y_pct / 100.0) * (w / 100.0)
                            if not pd.isna(g_pct):
                                weighted_div_growth_decimal += (g_pct / 100.0) * (w / 100.0)

                    st.markdown("**Portfolio (weighted) dividend metrics**")
                    st.metric("Weighted Avg Dividend Yield", f"{weighted_yield_decimal*100:.2f}%")
                    st.metric("Weighted Avg Dividend Growth (5y)", f"{weighted_div_growth_decimal*100:.2f}%")

                    # Projection simulator
                    st.markdown("#### Dividend income projection (uses weighted portfolio yield & growth)")
                    initial = st.number_input("Portfolio value for projection ($)", value=float(portfolio_value))
                    years = st.slider("Projection horizon (years)", 5, 40, 20)
                    proj_df = simulate_dividend_growth(initial, weighted_yield_decimal, weighted_div_growth_decimal, years)
                    st.line_chart(proj_df.set_index('Year'))
                else:
                    st.info("No dividend metrics available for the selected tickers.")

            # DGI philosophy (English)
            with st.expander("📜 Dividend Growth Philosophy (quick)"):
                st.markdown("""
                - Focus on **growing income**, not just price appreciation.
                - Reinvesting dividends (compounding) is the engine of long-term income growth.
                - Prefer companies or ETFs with **reasonable payout ratios**, consistent dividend growth, and durable advantages.
                - For ETFs, check that the fund's strategy aligns with income generation (e.g., dividend-focused funds).
                - Discipline matters: selling in panic breaks compounding.
                """)

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
