import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from io import BytesIO
import os
from SHARPR_backend import *  
from SHARPR_Vis import *
import matplotlib.pyplot as plt
from tiingo import TiingoClient
import logging
logging.getLogger("tiingo").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

plt.rcParams['figure.figsize'] = [20, 10]

tiingo_key = '5788d5bd795f6105fd8f4e5570a3bd6204dca3a6'
config = {'api_key': tiingo_key,'session': True}
client = TiingoClient(config)

selected_colors = [
    "#00008b",  # Dark Blue
    "#ff8c00",  # Dark Orange
    "#006400",  # Dark Green
    "#FF0000",  # Dark Red
    "#9400d3",  # Dark Violet
    "#a0522d",  # Saddle Brown
    "#00bfff",  # Deep Sky Blue
    "#00ff00",  # Lime
    "#ffd700",  # Gold
    "#ee82ee",  # Violet
    "#deb887",  # Burlywood
    "#C71585"]

additional_colors = plt.cm.tab20.colors[:13] 
color_cycle = selected_colors + list(additional_colors)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_cycle)
from SHARPR_backend import *


@st.cache_data
def compute_analysis(
    ticker_list,
    asset_returns,
    current_portfolio,
    method,
    current_wts,
    stock_numbers,
    current_values,
    benchmark_ticker,
    rfr_annual,
    tr_input,
    split_date,
    capital
):

    if asset_returns is None or asset_returns.empty:
        return None, None

    if current_portfolio:
        current_res = get_current_weights(
            ticker_list,
            current_wts=current_wts if method == "Current Weights" else None,
            stock_numbers=stock_numbers if method == "Stock Numbers" else None,
            current_values=current_values if method == "Current Values" else None
        )
    else:
        current_res = get_current_weights(ticker_list)
    
    rfr = rfr_annual / 252
    wts = random_wts(ticker_list)
    eq_wts = get_equal_weights(ticker_list)
    current_wts = current_res

    benchmark = get_tiingo_benchmark(benchmark_ticker)
    training_returns = asset_returns[asset_returns.index < split_date].dropna()
    test_returns = asset_returns[asset_returns.index >= split_date]

    ER = training_returns.mean()
    VCV = training_returns.cov()
    CORR = training_returns.corr()
    
    if tr_input == '':
        tr = ER.mean()
    else:
        tr = float(tr_input)

    ports_wts = get_portfolio_weights(w0=wts,rfr=rfr,target_return=tr,ER=ER,VCV=VCV,CORR=CORR,returns = training_returns)
    ports_wts = create_wts_df(ports_wts, current_res, current_portfolio=current_portfolio)
    
    portf_rets = calc_port_rets(test_returns, ports_wts)
    current_port_rets = calc_current_rets(test_returns, current_wts)

    wts_srs = multiple_pure_buy_and_hold(ports_wts, test_returns)
    avg_wts = calculate_average_weights(wts_srs)
    act_rets = get_multiple_actual_returns(wts_srs, test_returns)
    act_rets2 = get_multiple_actual_returns(wts_srs, test_returns)  
    
    rets_df = add_benchmark(act_rets, benchmark)
    ret_vol_df = calc_ret_vol_df(rets_df)
    current_stats = get_ret_vol(ret_vol_df, 'Current Portfolio')
    benchmark_stats = get_ret_vol(ret_vol_df, 'Benchmark')

    port_stats = calc_port_stats_2(act_rets, rfr, benchmark, tr, 252)
    factor_returns = load_daily_factor_returns()
    loadings = get_multiple_factor_loadings(act_rets, factor_returns)
    dd_alloc = discrete_and_dollar_allocation(capital, ticker_list, ports_wts)

    ER_test = test_returns.mean()
    VCV_test = test_returns.cov()
    CORR_test = test_returns.corr()

    portfolio_stats_df = get_portfolio_stats(ports_wts, VCV_test, ER_test, rfr)
    short_stats = calc_short_stats(act_rets,avg_wts,test_returns,ret_vol_df,rfr*252)

   
    analysis_results = {
        'fig0': process_and_plot_plotly(ret_vol_df),
        'fig0.5': plot_sharpe_ratios_plotly(ret_vol_df,rfr),
        'fig1': plot_indiv_assets_plotly(test_returns, benchmark=benchmark, plot_benchmark=True, current_portfolio=current_port_rets, plot_current_portfolio=current_portfolio),
        'fig2': plot_portfolios_plotly(rets_df),
        'fig3a': plot_portfolios_plotly(act_rets2, benchmark=benchmark, plot_benchmark=True),  
        'fig3b': plot_drawdowns_plotly(act_rets2, benchmark=benchmark, plot_benchmark=True),  
        'fig4': plot_portf_wts_db_plotly(ports_wts),
        'fig5': plot_risk_cont_db_plotly(ports_wts, VCV_test),
        'fig6': plot_sharpe_cont_plotly(ports_wts, ER_test, VCV_test, rfr),
        'fig7': plot_ef_w_assets_plotly(test_returns, ret_vol_df, rfr=rfr, n_points=100, cml=False, plot_benchmark=True, benchmark_stats=benchmark_stats, current_stats=current_stats, plot_current=current_portfolio),
        'fig8': plot_multiple_factor_loadings_plotly(act_rets, factor_returns, benchmark=benchmark),
        'portfolio_stats_df': portfolio_stats_df,
        'port_stats': port_stats,
        'dd_alloc': dd_alloc,
        'loadings': loadings,
        'rets_df': rets_df,
        'ticker_string': "_".join(ticker_list),
        'act_rets2': act_rets2,  
        'benchmark': benchmark,
        'short_stats':short_stats,
        'ports_wts':ports_wts

    }

    excel_data = generate_excel_data(analysis_results)    
    return analysis_results, excel_data


def save_to_downloads(analysis_results):
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    file_path = os.path.join(downloads_folder, f'SHARPR Results - {analysis_results["ticker_string"]}.xlsx')

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        analysis_results['portfolio_stats_df'].to_excel(writer, sheet_name='Portfolio Statistics')
        analysis_results['port_stats'].to_excel(writer, sheet_name='Performance Statistics')
        analysis_results['dd_alloc'].to_excel(writer, sheet_name='Allocations')
        analysis_results['loadings'].to_excel(writer, sheet_name='Factor Loadings')

    return file_path

def generate_excel_data(analysis_results):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        analysis_results['portfolio_stats_df'].to_excel(writer, sheet_name='Portfolio Statistics')
        analysis_results['port_stats'].to_excel(writer, sheet_name='Performance Statistics')
        analysis_results['dd_alloc'].to_excel(writer, sheet_name='Allocations')
        analysis_results['loadings'].to_excel(writer, sheet_name='Factor Loadings')
    buffer.seek(0)
    return buffer


@st.cache_data
def cached_get_returns(ticker_list):
    return get_tiingo_returns(ticker_list)


if 'portfolio_state' not in st.session_state:
    st.session_state.portfolio_state = {
        'ticker_list': [],
        'ask_portfolio': False,
        'asset_returns': None,
        'current_portfolio': False,
        'current_wts': None,
        'stock_numbers': None,
        'current_values': None,
        'rets_df': None,
        'port_stats': None,
        'dd_alloc': None,
        'loadings': None,
        'run_analysis': False,
        'run_simulation': False,  
        'simulation_fig': None, 
        'current_portfolio_submitted': False,
        'current_portfolio_option': None,
        'method': None,
    }

def reset_session_state():
    st.cache_data.clear() #### <- this is new
    st.session_state.portfolio_state = {
        'ticker_list': [],
        'ask_portfolio': False,
        'asset_returns': None,
        'current_portfolio': False,
        'current_wts': None,
        'stock_numbers': None,
        'current_values': None,
        'rets_df': None,
        'port_stats': None,
        'dd_alloc': None,
        'loadings': None,
        'run_analysis': False,
        'run_simulation': False,
        'simulation_fig': None,
        'current_portfolio_submitted': False,
        'current_portfolio_option': None,
        'method': None,
    }
    if 'analysis_results' in st.session_state:
        del st.session_state.analysis_results
    if 'excel_data' in st.session_state:
        del st.session_state.excel_data
    if 'asset_returns' in st.session_state.portfolio_state:
        del st.session_state.portfolio_state['asset_returns']

st.title("Welcome to the SHARPR-Portfolio-Optimization App!")
st.write("How does this work?")
st.write("Enter your the ticker symbolsfor the assets you hold(e.g. MSFT for Microsoft). Then, specify your Benchmark (e.g. SPY), the Risk-Free Rate and the Target Return")
st.write("The SHARPR-App will then calculate eleven Portfolios. Each portfolio optimizes your asset holdings following a investment strategy.")
st.write("The SHARPR-App (for now) shows you the gains of each stratey, the drawdowns (periods of losses), the composition of each portfolio, how much each asset contributes to your portfolio's risk, as well as to its risk-reward-ratio.")
st.write("Finally, the SHARPR-App also gives you performance statistics of each strategy. This helps you decide which strategy is most suitable for you")

if st.button("Reset Analysis"):
    reset_session_state()
    st.info("The analysis has been reset. Please adjust your inputs to start over.")


with st.sidebar:
    benchmark_ticker = st.text_input("Benchmark Ticker", value='SPY')
    rfr_annual = st.number_input("Annual Risk-Free Rate (as decimal)", min_value=0.0, max_value=1.0, value=0.025, step=0.001,format="%.3f")
    tr_input = st.text_input("Target Return (optional):", value='')
    split_date = st.text_input("Portfolio Formation Date (YYYY-MM-DD)", value='2023-01-01')
    capital = st.number_input("Initial Capital ($)", min_value=0.0, value=10000.00, step=100.0)
###########################################################################

#ticker_input = st.text_input("Enter tickers of the stocks in your portfolio below (comma-separated):")
#if st.button("Submit Tickers"):
    #if ticker_input.strip() != '':
        #st.session_state.portfolio_state['ticker_list'] = [ticker.strip() for ticker in ticker_input.split(',') if ticker.strip() != '']
        #st.session_state.portfolio_state['tickers_submitted'] = True
        #st.session_state.portfolio_state['ask_portfolio'] = True
        #st.session_state.portfolio_state['asset_returns'] = cached_get_returns(st.session_state.portfolio_state['ticker_list'])
        #st.session_state.portfolio_state['run_analysis'] = False
    #else:
        #st.warning("Please enter at least one ticker symbol.")
##################################################################################################################################

ticker_input = st.text_input("Enter tickers of the stocks in your portfolio below (comma-separated):")
if st.button("Submit Tickers"):
    if ticker_input.strip() != '':
        submitted_tickers = [ticker.strip().upper() for ticker in ticker_input.split(',') if ticker.strip()]
        ticker_validation = check_tickers_tiingo(submitted_tickers)  # Validate tickers
        valid_tickers = ticker_validation["valid tickers"]
        invalid_tickers = ticker_validation["invalid tickers"]

        validation_message = handle_invalid_tiingo_tickers(invalid_tickers)  # Handle invalid tickers
        st.write(validation_message)  # Display the message

        if valid_tickers:
            st.session_state.portfolio_state['ticker_list'] = valid_tickers
            st.session_state.portfolio_state['tickers_submitted'] = True
            st.session_state.portfolio_state['ask_portfolio'] = True
            st.session_state.portfolio_state['asset_returns'] = cached_get_returns(valid_tickers)
            st.session_state.portfolio_state['run_analysis'] = False
        else:
            st.warning("No valid tickers found. Please try again.")
    else:
        st.warning("Please enter at least one ticker symbol.")





#####################################################################################################################################
if st.session_state.portfolio_state.get('ask_portfolio', False):
    with st.sidebar:
        st.write("Do you have a current portfolio?")
        portfolio_option = st.radio("Select an option:", ["I don't have a Current Portfolio", "I have a Current Portfolio"], key='portfolio_option')
    
    if st.session_state.portfolio_state.get('current_portfolio_option') != portfolio_option:
        st.session_state.portfolio_state['current_portfolio_option'] = portfolio_option
        st.session_state.portfolio_state['run_analysis'] = False
        if 'analysis_results' in st.session_state:
            del st.session_state.analysis_results
        if 'excel_data' in st.session_state:
            del st.session_state.excel_data

    if portfolio_option == "I don't have a Current Portfolio":
        st.session_state.portfolio_state['current_wts'] = None
        st.session_state.portfolio_state['stock_numbers'] = None
        st.session_state.portfolio_state['current_values'] = None
        st.session_state.portfolio_state['current_portfolio'] = False

    elif portfolio_option == "I have a Current Portfolio":
        st.write("Please enter your current portfolio details.")
        method = st.radio("Select which argument to populate:", ["Current Weights", "Stock Numbers", "Current Values"], key='method_select')
        st.session_state.portfolio_state['method'] = method
        if method == "Current Weights":
            current_wts_input = {}
            for ticker in st.session_state.portfolio_state['ticker_list']:
                weight = st.number_input(f"Weight for {ticker} (%)", min_value=0.0, max_value=100.0, value=0.0, key=f"wt_{ticker}")
                current_wts_input[ticker] = weight / 100.0
            if st.button("Submit Current Portfolio"):
                st.session_state.portfolio_state['current_wts'] = pd.Series(current_wts_input)
                st.session_state.portfolio_state['stock_numbers'] = None
                st.session_state.portfolio_state['current_values'] = None
                st.session_state.portfolio_state['current_portfolio'] = True
                st.session_state.portfolio_state['current_portfolio_submitted'] = True
                st.session_state.portfolio_state['run_analysis'] = False
        elif method == "Stock Numbers":
            stock_numbers_input = {}
            for ticker in st.session_state.portfolio_state['ticker_list']:
                num_shares = st.number_input(f"Number of shares for {ticker}", min_value=0, value=0, key=f"num_{ticker}")
                stock_numbers_input[ticker] = num_shares
            if st.button("Submit Current Portfolio"):
                st.session_state.portfolio_state['current_wts'] = None
                st.session_state.portfolio_state['stock_numbers'] = pd.Series(stock_numbers_input)
                st.session_state.portfolio_state['current_values'] = None
                st.session_state.portfolio_state['current_portfolio'] = True
                st.session_state.portfolio_state['current_portfolio_submitted'] = True
                st.session_state.portfolio_state['run_analysis'] = False
        elif method == "Current Values":
            current_values_input = {}
            for ticker in st.session_state.portfolio_state['ticker_list']:
                value = st.number_input(f"Current value for {ticker} ($)", min_value=0.0, value=0.0, key=f"value_{ticker}")
                current_values_input[ticker] = value
            if st.button("Submit Current Portfolio"):
                st.session_state.portfolio_state['current_wts'] = None
                st.session_state.portfolio_state['stock_numbers'] = None
                st.session_state.portfolio_state['current_values'] = pd.Series(current_values_input)
                st.session_state.portfolio_state['current_portfolio'] = True
                st.session_state.portfolio_state['current_portfolio_submitted'] = True
                st.session_state.portfolio_state['run_analysis'] = False


if st.session_state.portfolio_state.get('tickers_submitted', False):
    if st.button("Run Analysis"):
        st.session_state.portfolio_state['run_analysis'] = True


#if st.session_state.portfolio_state.get('run_analysis', False):
    #ps = st.session_state.portfolio_state
    #analysis_results, excel_data = compute_analysis(
        #ticker_list=ps['ticker_list'],
        #asset_returns=ps['asset_returns'],
        #current_portfolio=ps['current_portfolio'],
        #method=ps['method'],
        #current_wts=ps['current_wts'],
        #stock_numbers=ps['stock_numbers'],
        #current_values=ps['current_values'],
        #benchmark_ticker=benchmark_ticker,
        #rfr_annual=rfr_annual,
        #tr_input=tr_input,
        #split_date=split_date,
        #capital=capital)

if st.session_state.portfolio_state.get('run_analysis', False):
    ps = st.session_state.portfolio_state
    analysis_results, excel_data = compute_analysis(
        ticker_list=ps['ticker_list'],
        asset_returns=ps['asset_returns'],
        current_portfolio=ps['current_portfolio'],
        method=ps['method'],
        current_wts=ps['current_wts'],
        stock_numbers=ps['stock_numbers'],
        current_values=ps['current_values'],
        benchmark_ticker=benchmark_ticker,
        rfr_annual=rfr_annual,
        tr_input=tr_input,
        split_date=split_date,
        capital=capital
    )

    portfolio_options = list(analysis_results['short_stats'].index)
    portfolio_selection = st.multiselect(
        "Select Portfolios to Analyze:",
        options=portfolio_options,
        default=None
    )

    if analysis_results is None:
        st.error("Analysis could not be performed due to missing or invalid data.")
    else:
        if portfolio_selection:
            try:
                chart = plot_change_wts(analysis_results['ports_wts'], portfolio_selection)
            except ValueError as e:
                st.error(f"Error generating chart: {e}")
                chart = None

            table = get_short_stats(analysis_results['short_stats'], portfolio_selection)
            col1, col2 = st.columns([2, 1])  
            with col1:
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            with col2:
                if not table.empty:
                    st.write(table)
                else:
                    st.warning("No data available for the selected portfolios.")
        else:
            st.info("Please select at least one portfolio to display the chart and table.")



    if analysis_results is None:
        st.error("Analysis could not be performed due to missing or invalid data.")
    else:
        st.session_state.analysis_results = analysis_results
        st.session_state.excel_data = excel_data

        st.header("Analysis Results")
        st.write("This chart shows how the Optimized Portfolios compare to your Current Portfolio and your Benchmark")
        st.plotly_chart(analysis_results['fig0'], use_container_width=True)
        st.plotly_chart(analysis_results['fig0.5'], use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(analysis_results['fig1'], use_container_width=False)
        with col2:
            st.plotly_chart(analysis_results['fig2'], use_container_width=False)

        st.sidebar.write("Select columns to display:")
        selected_columns = []
        for column in analysis_results['act_rets2'].columns:
            is_selected = st.sidebar.checkbox(column, value=True)
            if is_selected:
                selected_columns.append(column)

        if not selected_columns:
            st.warning("Please select at least one scenario to display.")
        else:
            act_rets2_filtered = analysis_results['act_rets2'][selected_columns]
            st.subheader('Portfolio Statistics')
            st.write("Here you can see how your Portfolio's are constructed, how Portfolio's Tickers contribute to each Portfolio's Risk and Sharpe Ratio")
            st.dataframe(analysis_results['portfolio_stats_df'])
            st.subheader('Cumulative Returns and Drawdowns')
            st.write("The following charts show Cumulative Returns and Drawdons of each Portfolio")
            st.write("You can check on the sidebar which Portfolios you want to display")
            st.plotly_chart(plot_portfolios_plotly(act_rets2_filtered))
            st.plotly_chart(plot_drawdowns_plotly(act_rets2_filtered, benchmark=analysis_results['benchmark'], plot_benchmark=True))
            st.subheader('Visualized Portfolio Statistics')
            st.write("The three tabs below visualize each Portfolio's Composition, how each ticker contributes to Risk and the Portfolio Sharpe Ratio")
        
        tab1, tab2, tab3 = st.tabs(["Portfolio Weights", "Risk Contribution", "Sharpe Ratio Contribution"])
        with tab1:
            st.plotly_chart(analysis_results['fig4'])
        with tab2:
            st.plotly_chart(analysis_results['fig5'])
        with tab3:
            st.plotly_chart(analysis_results['fig6'])
        
        st.subheader('Risk-Return Tradeoff & Risk Sources')
        st.write("This chart shows the efficient Frontier. The closer you are to the blue line, the better the Risk-Return Tradeoff")
        st.plotly_chart(analysis_results['fig7'])
        st.write("This chart shows how Portfolio Returns are exposed to common Risk Factors.")
        st.plotly_chart(analysis_results['fig8'])

        #st.subheader('Detailed Portfolio Statistics')
        #st.write("This table shows detailed Performance Statistics for each Portfolio")
        #st.dataframe(analysis_results['port_stats'])
        #st.subheader('Discrete and Dollar Allocation')
        #st.write("This table shows how many Stocks, or how much money, you have to hold of each Asset for the Portfolios")
        #st.dataframe(analysis_results['dd_alloc'])

        st.subheader('Detailed Portfolio Statistics')
        st.write("Thse tables shows detailed Performance Statistics for each Portfolio and how many Stocks, or how much money, you have to hold of each Asset for the Portfolios")
        tab4, tab5 = st.tabs(["Portfolio Statistics", "Allocation"])
        with tab4:
            st.dataframe(analysis_results['port_stats'])
        with tab5:
            st.dataframe(analysis_results['dd_alloc'])

        st.subheader('Portfolio Simulation')
        st.write("Use the Sidebar to pick Portfolios whose Returns you want to simulate (click the 'Run Simulation'-button)")

        with st.sidebar:
            plot_sims = st.multiselect("Select scenarios for simulation", analysis_results['rets_df'].columns.tolist(), default=None)
            run_simulation_button = st.button("Run Simulation")

        if run_simulation_button and plot_sims:
            st.session_state.portfolio_state['run_simulation'] = True
            st.session_state.portfolio_state['simulation_fig'] = plot_gbm_and_cum_rets_plotly(
                analysis_results['rets_df'],
                plot_last=252,
                plot_sims=plot_sims
            )

            st.session_state.portfolio_state['density_fig'] = plot_return_density(rets_df=analysis_results['rets_df'], 
                        plot_sims=plot_sims,n_years=1,n_scenarios=1000,steps_per_year=252,plot_last=252) 

        if st.session_state.portfolio_state.get('simulation_fig') and plot_sims:
            st.subheader("Cumulative Portfolio Returns with Simulations")
            st.plotly_chart(st.session_state.portfolio_state['simulation_fig'])
            st.subheader("Simulated Return Density") #
            st.plotly_chart(st.session_state.portfolio_state['density_fig']) 


    st.write("Click here to download the results of the Analysis as Excel-File")
    st.download_button(
        label='Download Results as Excel',
        data=excel_data,  
        file_name=f'SHARPR Results - {analysis_results["ticker_string"]}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        # Optional: Save the file to Downloads folder (local environment use)
        # file_path = save_to_downloads(analysis_results)
        # st.write(f"File saved to: {file_path}")
