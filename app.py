import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import requests_cache # MODIFIED: To make requests more reliable

# --- Page Configuration ---
st.set_page_config(
    page_title="ETF Portfolio Analyzer",
    page_icon="📈",
    layout="wide"
)

# MODIFIED: Setup a session with a browser header to avoid being blocked
session = requests_cache.CachedSession('yfinance.cache')
session.headers['User-agent'] = 'my-yf-app/1.0'


# --- Logic Functions ---

@st.cache_data(ttl=3600)
def get_etf_data(ticker_symbol):
    """Fetches key data for an ETF using yfinance."""
    try:
        # MODIFIED: Use the session for the request
        etf = yf.Ticker(ticker_symbol, session=session)
        info = etf.info
        
        # MODIFIED: More robust check to see if we got valid data
        if not info or len(info) < 5: 
             st.error(f"Could not retrieve valid data for {ticker_symbol}. It might be delisted or an incorrect ticker.")
             return None

        holdings = etf.holdings
        sector_weights = info.get('sectorWeightings', [])
        country_weights = info.get('countryWeightings', [])
        expense_ratio = info.get('annualReportExpenseRatio') # Get can return None
        
        if holdings is None or holdings.empty or not sector_weights or not country_weights:
            st.error(f"Data for {ticker_symbol} is incomplete (missing holdings, sector, or country data).")
            return None

        holdings_df = holdings
        sector_df = pd.DataFrame([{'sector': s['longName'], 'weight': s['value']} for s in sector_weights])
        country_df = pd.DataFrame(country_weights)
        
        return {
            'holdings': holdings_df,
            'sectors': sector_df,
            'countries': country_df,
            'expense_ratio': expense_ratio if expense_ratio is not None else 0
        }
    # MODIFIED: Catch the specific exception and print it for better debugging
    except Exception as e:
        st.error(f"Could not fetch data for {ticker_symbol}. The specific error is: {e}")
        return None

# The rest of the functions remain the same as they depend on the data fetching above
def analyze_portfolio(portfolio_str):
    """Analyzes the consolidated portfolio from user input."""
    lines = [line.strip() for line in portfolio_str.strip().split('\n') if line.strip()]
    portfolio = {}
    total_weight = 0
    for line in lines:
        try:
            ticker, weight_str = line.split()
            weight = float(weight_str)
            portfolio[ticker.upper()] = weight
            total_weight += weight
        except ValueError:
            st.error(f"Error in line: '{line}'. Format must be 'TICKER WEIGHT'. Ex: VOO 50")
            return None, None
    
    if round(total_weight) != 100:
        st.warning(f"The sum of weights is {total_weight}%. It should be 100%.")
    
    all_etf_data = {}
    for ticker in portfolio.keys():
        data = get_etf_data(ticker)
        if data:
            all_etf_data[ticker] = data
        else:
            return None, None 

    consolidated_holdings = pd.DataFrame()
    consolidated_sectors = pd.DataFrame()
    consolidated_countries = pd.DataFrame()
    weighted_expense_ratio = 0

    for ticker, weight in portfolio.items():
        etf_data = all_etf_data[ticker]
        portfolio_weight_fraction = weight / 100.0

        temp_holdings = etf_data['holdings'].copy()
        temp_holdings['weight'] = temp_holdings['% Assets'] * portfolio_weight_fraction
        consolidated_holdings = pd.concat([consolidated_holdings, temp_holdings[['Holding', 'weight']]])

        temp_sectors = etf_data['sectors'].copy()
        temp_sectors['weight'] *= portfolio_weight_fraction
        consolidated_sectors = pd.concat([consolidated_sectors, temp_sectors])

        temp_countries = etf_data['countries'].copy()
        temp_countries['weight'] *= portfolio_weight_fraction
        consolidated_countries = pd.concat([consolidated_countries, temp_countries])

        weighted_expense_ratio += (etf_data['expense_ratio'] or 0) * portfolio_weight_fraction

    final_holdings = consolidated_holdings.groupby('Holding')['weight'].sum().nlargest(15).reset_index()
    final_sectors = consolidated_sectors.groupby('sector')['weight'].sum().reset_index()
    final_countries = consolidated_countries.groupby('country')['weight'].sum().reset_index()

    return {
        'holdings': final_holdings,
        'sectors': final_sectors,
        'countries': final_countries,
        'expense_ratio': weighted_expense_ratio
    }, portfolio.keys()

# --- User Interface (UI) ---
st.title("📈 ETF Portfolio Analyzer")
st.markdown("Discover the true composition of your ETF portfolio. This tool shows you asset overlap, sector and country exposure, and the real cost of your portfolio.")

st.sidebar.header("Configure Your Portfolio")

bmac_link = "https://www.buymeacoffee.com/rubenjromo" 
st.sidebar.markdown(f"""
<a href="{bmac_link}" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 50px !important;width: 200px !important;" >
</a>
""", unsafe_allow_html=True)

portfolio_input = st.sidebar.text_area(
    "Enter your ETFs (one per line)", height=200, value="VOO 50\nQQQ 30\nSCHD 20",
    help="Enter the ETF ticker followed by its weight in your portfolio. Ex: VOO 50"
)

if st.sidebar.button("Analyze Portfolio"):
    if portfolio_input:
        with st.spinner("Fetching data and analyzing... please wait."):
            analysis_results, tickers = analyze_portfolio(portfolio_input)
        if analysis_results:
            st.success("Analysis Complete!")
            st.header(f"Consolidated Analysis for: {', '.join(tickers)}")
            st.metric(label="Weighted Expense Ratio (Real Cost)", value=f"{analysis_results['expense_ratio']:.4%}")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sector Exposure")
                fig_sectors = px.pie(analysis_results['sectors'], names='sector', values='weight', title='Consolidated Sector Distribution', hole=.3)
                fig_sectors.update_traces(textinfo='percent+label', showlegend=False)
                st.plotly_chart(fig_sectors, use_container_width=True)
            with col2:
                st.subheader("Country Exposure")
                fig_countries = px.pie(analysis_results['countries'], names='country', values='weight', title='Consolidated Geographic Distribution', hole=.3)
                fig_countries.update_traces(textinfo='percent+label', showlegend=False)
                st.plotly_chart(fig_countries, use_container_width=True)
            st.subheader("📊 Top 15 Underlying Assets (Overlap)")
            st.dataframe(analysis_results['holdings'].style.format({'weight': '{:.2%}'}), use_container_width=True)
    else:
        st.warning("Please enter your portfolio data in the sidebar.")

st.sidebar.markdown("---")
st.sidebar.info("Created with ❤️ using Python and Streamlit.")
