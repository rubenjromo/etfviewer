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
    normalized = (price_history / price_history.bfill().iloc[0] - 1) * 100
    return normalized.round(2)

# --- NEW: ETF Holdings and Exposure Analysis ---

@st.cache_data(ttl=3600)
def get_etf_holdings(ticker_symbol):
    """Get ETF holdings (if available)."""
    try:
        etf = yf.Ticker(ticker_symbol)
        holdings = etf.funds_holdings
        if holdings is None or holdings.empty:
            return pd.DataFrame()
        holdings = holdings[['symbol', 'holdingName', 'holdingPercent']]
        holdings.rename(columns={'symbol': 'Ticker', 'holdingName': 'Company', 'holdingPercent': 'Weight'}, inplace=True)
        holdings['ETF'] = ticker_symbol
        return holdings
    except Exception:
        return pd.DataFrame()

def combine_holdings(portfolio):
    """Combine holdings from all ETFs weighted by portfolio allocation."""
    combined = pd.DataFrame()
    for etf, etf_weight in portfolio.items():
        holdings = get_etf_holdings(etf)
        if not holdings.empty:
            holdings['AdjustedWeight'] = holdings['Weight'] * (etf_weight / 100)
            combined = pd.concat([combined, holdings], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    final = combined.groupby(['Ticker', 'Company'], as_index=False)['AdjustedWeight'].sum()
    final.sort_values('AdjustedWeight', ascending=False, inplace=True)
    return final

@st.cache_data(ttl=86400)
def get_company_info(symbol):
    """Get country and sector for a company."""
    try:
        data = yf.Ticker(symbol).info
        return {
            'Ticker': symbol,
            'Country': data.get('country', 'Unknown'),
            'Sector': data.get('sector', 'Unknown')
        }
    except Exception:
        return {'Ticker': symbol, 'Country': 'Unknown', 'Sector': 'Unknown'}

def enrich_holdings_with_info(df):
    info_list = [get_company_info(t) for t in df['Ticker'].unique()]
    info_df = pd.DataFrame(info_list)
    df = df.merge(info_df, on='Ticker', how='left')
    return df

def plot_company_exposure(df):
    fig = px.bar(df.head(15), x='Company', y='AdjustedWeight',
                 title="Top 15 Holdings (Weighted by Portfolio)",
                 labels={'AdjustedWeight': 'Portfolio Exposure (%)'})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def plot_country_map(df):
    country_weights = df.groupby('Country', as_index=False)['AdjustedWeight'].sum()
    fig = px.choropleth(country_weights, locations="Country",
                        locationmode="country names",
                        color="AdjustedWeight",
                        title="Portfolio Exposure by Country",
                        color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

def plot_sector_pie(df):
    sector_weights = df.groupby('Sector', as_index=False)['AdjustedWeight'].sum()
    fig = px.pie(sector_weights, values='AdjustedWeight', names='Sector',
                 title='Portfolio Exposure by Sector', hole=0.3)
    st.plotly_chart(fig, use_container_width=True)

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
            weighted_yield_sum = 0
            weighted_expense_ratio_sum = 0
            
            for ticker, weight in portfolio.items():
                metrics = get_etf_metrics(ticker)
                if metrics:
                    metrics['Portfolio Weight %'] = weight
                    all_metrics.append(metrics)
                    weighted_yield_sum += (metrics.get('Yield %', 0) or 0) * (weight / 100.0)
                    weighted_expense_ratio_sum += (metrics.get('Expense Ratio %', 0) or 0) * (weight / 100.0)
        
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
            
            price_history = get_historical_prices(list(portfolio.keys()))

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
            df_comp['Income Contribution ($)'] = (df_comp['Yield %'] / 100) * (df_comp['Portfolio Weight %'] / 100) * portfolio_value
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

            combined_holdings = combine_holdings(portfolio)
            if not combined_holdings.empty:
                enriched = enrich_holdings_with_info(combined_holdings)

                st.dataframe(enriched.sort_values('AdjustedWeight', ascending=False).head(20),
                             use_container_width=True)

                plot_company_exposure(enriched)
                plot_country_map(enriched)
                plot_sector_pie(enriched)
            else:
                st.warning("No holdings data available for the selected ETFs.")

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
