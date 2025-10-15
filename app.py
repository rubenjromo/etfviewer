@st.cache_data(ttl=3600)
def get_etf_metrics(ticker_symbol):
    """Fetches key comparable metrics from the .info dictionary of an ETF."""
    try:
        etf = yf.Ticker(ticker_symbol)
        info = etf.info
        
        if not info or info.get('quoteType') != 'ETF':
            st.warning(f"Could not get valid ETF data for {ticker_symbol}. It might be a stock or an invalid ticker.", icon="⚠️")
            return None

        # MODIFIED: Corrected Expense Ratio and removed Price-to-Book.
        metrics = {
            'Ticker': info.get('symbol', ticker_symbol),
            'Name': info.get('shortName', 'N/A'),
            'Family': info.get('fundFamily', 'N/A'),
            'Category': info.get('category', 'N/A'),
            'Expense Ratio %': info.get('netExpenseRatio'), # Corrected: Value is already the percentage
            'Yield %': info.get('dividendYield') or 0,
            'YTD Return %': info.get('ytdReturn'),
            'Beta (3Y)': info.get('beta3Year'),
            'Total Assets': info.get('totalAssets'),
        }
        return metrics

    except Exception as e:
        st.error(f"Failed to process {ticker_symbol}. Error: {e}", icon="🚨")
        return None
