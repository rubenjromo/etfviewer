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

        dividends = etf_yf.dividends
        last_dividend = dividends.iloc[-1] if (isinstance(dividends, pd.Series) and not dividends.empty) else 0
        dividend_frequency = get_dividend_frequency(dividends if isinstance(dividends, pd.Series) else pd.Series())
        cagr_5y = get_cagr(etf_yf, years=5)

        metrics = {
            'Ticker': info.get('symbol', ticker_symbol),
            'Name': info.get('shortName', 'N/A'),
            'Category': info.get('category', 'N/A') if info.get('category') else info.get('quoteType', 'N/A'),
            'Expense Ratio %': (info.get('netExpenseRatio') or 0),
            'Yield %': (info.get('dividendYield') or 0) * 100 if info.get('dividendYield') is not None else 0,
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
        volatility = volatility
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
    # Downsample for plotting performance: weekly last 5 years
    sampled = price_history.resample('W').last()
    normalized = (sampled / sampled.bfill().iloc[0] - 1) * 100
    return normalized.round(2)

# --- NEW: ETF Holdings and Exposure Analysis ---

@st.cache_data(ttl=3600)
def get_etf_holdings(ticker_symbol):
    """Get ETF holdings (if available)."""
    try:
        etf = yf.Ticker(ticker_symbol)
        # yfinance exposes funds_holdings for some ETFs
        holdings = etf.funds_holdings
        if holdings is None or holdings.empty:
            return pd.DataFrame()
        # Standardize columns if possible
        # Some versions return slightly different names; we try to be robust
        cols = holdings.columns.str.lower()
        # try common column names
        mapping = {}
        if 'symbol' in holdings.columns:
            mapping['symbol'] = 'Ticker'
        elif 'holdingSymbol' in holdings.columns:
            mapping['holdingSymbol'] = 'Ticker'
        if 'holdingName' in holdings.columns:
            mapping['holdingName'] = 'Company'
        elif 'name' in holdings.columns:
            mapping['name'] = 'Company'
        if 'holdingPercent' in holdings.columns:
            mapping['holdingPercent'] = 'Weight'
        elif 'weight' in holdings.columns:
            mapping['weight'] = 'Weight'
        # select intersection
        available = [c for c in mapping.keys() if c in holdings.columns]
        if not available:
            return pd.DataFrame()
        df = holdings[available].rename(columns=mapping)
        df['ETF'] = ticker_symbol
        # Ensure numeric percent in 0..100 scale
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
            holdings = holdings.copy()
            holdings['AdjustedWeight'] = holdings['Weight'] * (etf_weight / 100.0)
            combined = pd.concat([combined, holdings], ignore_index=True)
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
    # if Country unknown, remove
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

# --- NEW: Dividend Growth (DGI) helpers ---

@st.cache_data(ttl=86400)
def get_dividend_series(symbol):
    """Return dividends Series for a symbol (cached)."""
    try:
        divs = yf.Ticker(symbol).dividends
        if isinstance(divs, pd.Series) and not divs.empty:
            return divs
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
    # consider last 'years' years
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
        div_yield = info.get('dividendYield', None)
        div_yield = div_yield * 100 if div_yield is not None else np.nan
        payout_ratio = info.get('payoutRatio', None)
        payout_ratio = payout_ratio * 100 if payout_ratio is not None else np.nan
        div_series = get_dividend_series(symbol)
        div_cagr = compute_dividend_cagr(div_series, years=5)
        div_cagr_pct = div_cagr * 100 if div_cagr is not None else np.nan
        return {
            'Ticker': symbol,
            'Dividend Yield %': div_yield,
            'Payout Ratio %': payout_ratio,
            '5Y Dividend CAGR %': div_cagr_pct
        }
    except Exception:
        return {'Ticker': symbol, 'Dividend Yield %': np.nan, 'Payout Ratio %': np.nan, '5Y Dividend CAGR %': np.nan}

def simulate_dividend_growth(initial_investment, annual_yield, dividend_growth, years):
    """Simulate annual dividend income growth (not total portfolio growth)."""
    rows = []
    income = initial_investment * annual_yield
    for year in range(1, years + 1):
        rows.append({'Year': year, 'Annual Dividend Income ($)': income})
        income = income * (1 + dividend_growth)
    return pd.DataFrame(rows)

# --- User Interface (UI) ---
st.title("🛠️ ETF & Asset Analysis Tool")
st.header("💵 Portfolio Dividend Calculator")

col1, col2 = st.columns(2)
with col1:
    portfolio_input = st.text_area(
        "**1. Enter your assets and weights** (one per line)",
        value="VOO 40\nSCHD 20\nQQQ 20\nBTC-USD 20",
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
            weighted_yield_sum = 0.0
            weighted_expense_ratio_sum = 0.0
            
            # fetch metrics for each ticker (cached)
            for ticker, weight in portfolio.items():
                metrics = get_etf_metrics(ticker)
                if metrics:
                    metrics['Portfolio Weight %'] = weight
                    all_metrics.append(metrics)
                    # Yield was converted to percent in get_etf_metrics
                    weighted_yield_sum += (metrics.get('Yield %', 0) or 0) * (weight / 100.0)
                    weighted_expense_ratio_sum += (metrics.get('Expense Ratio %', 0) or 0) * (weight / 100.0)
            
            # Get price_history only once for all tickers — may be heavy but cached
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
            annual_income = portfolio_value * (weighted_yield_sum / 100.0)
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric(label="**Weighted Average Dividend Yield**", value=f"{weighted_yield_sum:.2f}%")
            with col_metric2:
                st.metric(label="**Estimated Annual Dividend Income**", value=f"${annual_income:,.2f}")
            with col_metric3:
                st.metric(label="**Weighted Average Expense Ratio**", value=f"{weighted_expense_ratio_sum:.2f}%")
            
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

            # Lazy load: allow user to choose to fetch holdings / sectors
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
                    st.markdown("### 📊 Top Holdings Combinados")
                    # show top 20 with formatted percent
                    enriched_display = enriched.copy()
                    enriched_display['AdjustedWeight (%)'] = (enriched_display['AdjustedWeight'] ).round(4)
                    st.dataframe(enriched_display.sort_values('AdjustedWeight', ascending=False).head(20), use_container_width=True)
                    plot_company_exposure(enriched_display)
                    plot_country_map(enriched_display)
                else:
                    st.warning("No holdings data available for the selected ETFs (via yfinance).")

            # Sector exposure (combining ETF-level sector weights) — always available if user wants it
            if fetch_sectors:
                # combine sector exposure using ETF-level sector breakdown (safer / more available)
                sector_combined = pd.DataFrame()
                for etf, w in portfolio.items():
                    # fetch fund sector weightings from yfinance (cached)
                    try:
                        s = yf.Ticker(etf).fund_sector_weightings or {}
                    except Exception:
                        s = {}
                    if s:
                        df_s = pd.DataFrame(list(s.items()), columns=['Sector', 'Weight'])
                        # yfinance returns weights in 0..1; convert to %
                        df_s['Weight'] = df_s['Weight'] * 100
                        df_s['AdjustedWeight'] = df_s['Weight'] * (w / 100.0)
                        sector_combined = pd.concat([sector_combined, df_s[['Sector','AdjustedWeight']]], ignore_index=True)
                if not sector_combined.empty:
                    sector_final = sector_combined.groupby('Sector', as_index=False)['AdjustedWeight'].sum()
                    sector_final.sort_values('AdjustedWeight', ascending=False, inplace=True)
                    sector_final['AdjustedWeight (%)'] = sector_final['AdjustedWeight'].round(4)
                    st.markdown("### 🏭 Exposición Sectorial Combinada")
                    st.dataframe(sector_final[['Sector','AdjustedWeight (%)']], use_container_width=True)
                    plot_sector_pie(sector_final.rename(columns={'AdjustedWeight': 'AdjustedWeight'}))
                else:
                    st.warning("No sector weighting data available for the selected ETFs (via yfinance).")

            # --- Dividend Growth Features (optional) ---
            st.markdown("---")
            st.subheader("🧠 Dividend Growth (DGI) Analysis")
            with st.expander("Show Dividend Growth metrics & projections"):
                # lazy compute dividend metrics for each ticker
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

                    # Average yield and growth for portfolio (weighted)
                    # Use ETF portfolio weights to compute weighted average dividend yield & growth
                    weighted_yield = 0.0
                    weighted_div_growth = 0.0
                    for t, w in portfolio.items():
                        row = div_df.loc[t] if t in div_df.index else {}
                        y = row.get('Dividend Yield %', np.nan) if isinstance(row, dict) or 'Dividend Yield %' in row else row['Dividend Yield %']
                        g = row.get('5Y Dividend CAGR %', np.nan) if isinstance(row, dict) or '5Y Dividend CAGR %' in row else row['5Y Dividend CAGR %']
                        if not np.isnan(y):
                            weighted_yield += (y/100.0) * (w/100.0)
                        if not np.isnan(g):
                            weighted_div_growth += (g/100.0) * (w/100.0)
                    st.markdown("**Portfolio (weighted) dividend metrics**")
                    st.metric("Weighted Avg Dividend Yield", f"{weighted_yield*100:.2f}%")
                    st.metric("Weighted Avg Dividend Growth (5y)", f"{weighted_div_growth*100:.2f}%")

                    # Projection simulator
                    st.markdown("#### Dividend income projection (relying on weighted portfolio yield & growth)")
                    initial = st.number_input("Use this portfolio value for projection ($)", value=float(portfolio_value))
                    years = st.slider("Projection horizon (years)", 5, 40, 20)
                    proj_df = simulate_dividend_growth(initial, weighted_yield, weighted_div_growth, years)
                    st.line_chart(proj_df.set_index('Year'))
                else:
                    st.info("No dividend metrics available for the selected tickers.")

            # Philosophy panel (keeps UI original but adds DGI reminders)
            with st.expander("📜 Dividend Growth Philosophy (quick)"):
                st.markdown("""
                - Busca empresas o ETFs que **paguen y aumenten dividendos** a lo largo del tiempo.
                - La **reinvención de dividendos** (compounding) es la principal fuerza del DGI.
                - Preferir empresas con **payout razonable**, crecimiento de dividendos y ventajas competitivas.
                - En ETFs, revisa que la *estrategia* sea consistente con generación de flujo (p.ej. dividend-focused funds).
                - Mantén la disciplina: las ventas por pánico rompen el poder del interés compuesto.
                """)

            st.info("""
                **Disclaimer:** This tool is for informational purposes only and does not constitute financial advice. All calculations are based on publicly available data which may not be 100% accurate. Always do your own research before making any investment decisions.
            """, icon="⚠️")

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info("This app provides a portfolio dividend, holdings, and exposure analysis tool.")
bmac_link = "https://www.buymeacoffee.com/rubenjromo" 
st.sidebar.markdown(f"""<a href="{bmac_link}" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 50px !important;width: 200px !important;" ></a>""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.info("Created with ❤️ using Python and Streamlit.")
