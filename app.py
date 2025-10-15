def analyze_portfolio(portfolio_str):
    """Analyzes the consolidated portfolio from user input."""
    lines = [line.strip() for line in portfolio_str.strip().split('\n') if line.strip()]
    portfolio = {}
    total_weight = 0  # <--- CORREGIDO AQUÍ
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

        if not etf_data['sectors'].empty:
            temp_sectors = etf_data['sectors'].copy()
            temp_sectors['weight'] *= portfolio_weight_fraction
            consolidated_sectors = pd.concat([consolidated_sectors, temp_sectors])
        
        if not etf_data['countries'].empty:
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
