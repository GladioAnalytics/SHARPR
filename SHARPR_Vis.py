import yfinance as yf 
from yahoo_fin.stock_info import *
## Standard Python Data Science stack
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import datetime as dt
from scipy.stats import (norm as norm, linregress as linregress)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde

plt.rcParams['figure.figsize'] = [20, 10]
#plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)

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


def plot_sharpe_ratios_plotly(df, rfr_annual, By="Sharpe Ratio"):
    # Prepare the data
    df_copy = df.copy()
    df_copy['Return'] = df_copy['Return'] * 252
    df_copy['Volatility'] = df_copy['Volatility'] * np.sqrt(252)
    df_copy['Sharpe Ratio'] = (df_copy['Return'] - rfr_annual) / df_copy['Volatility']
    df_copy['Ret+Vol'] = df_copy['Return'] + df_copy['Volatility']
    df_copy['Ret-Vol'] = df_copy['Return'] - df_copy['Volatility']
    
    df_copy = df_copy.sort_values(by=By, ascending=True)

    # Define color mapping
    color_mapping = {
        'Markowitz': "#00008b", 'Global Minimum Variance': "#ff8c00", 'Maximum Sharpe Ratio Portfolio': "#006400",
        'Equal Weights': "#FF0000", 'Risk Parity': "#9400d3", 'Inverse Variance': "#a0522d",
        'Inverse Volatility': "#00bfff", 'Maximum Diversification': "#00ff00", 
        'Maximum De-Correlation': "#ffd700", 'Minimum Correlation': "#ee82ee",
        'Equal Sharpe Ratio': "#deb887", 'Benchmark': "grey", 'Current Portfolio': "black",
        'Minimum Expected Shortfall': "#8B0000",
        'Minimum Max Drawdown': "#4682B4"}

    fig = go.Figure()
    
    # Add data to the figure
    for i, (idx, row) in enumerate(df_copy.iterrows()):
        portfolio_type = idx  
        color = color_mapping.get(portfolio_type, 'blue')  
        x_value = i + 1
        
        # Dynamically adjust text position to prevent overlapping
        if i % 2 == 0:
            text_position = "top center"
        else:
            text_position = "bottom center"
        
        # Main return point
        fig.add_trace(go.Scatter(
            x=[x_value],
            y=[row['Return']],
            mode='markers+text',  # Enables both markers and text
            marker=dict(color=color, size=15),  # Sets marker color and size
            text=portfolio_type,  # The text label for the point
            textposition=text_position,  # Dynamic text positioning (top/bottom center)
            textfont=dict(color='black', size=10),  # Explicitly set the text font color to black and size to 10
            name=portfolio_type,  # Legend entry name
            showlegend=portfolio_type not in fig.data))
        
        # Ret-Vol and Ret+Vol points
        fig.add_trace(go.Scatter(
            x=[x_value],
            y=[row['Ret-Vol']],
            mode='markers',
            marker=dict(color=color, size=6),
            name=f"{portfolio_type} Ret-Vol",
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[x_value],
            y=[row['Ret+Vol']],
            mode='markers',
            marker=dict(color=color, size=6),
            name=f"{portfolio_type} Ret+Vol",
            showlegend=False
        ))

        # Vertical line between Ret-Vol and Ret+Vol
        fig.add_trace(go.Scatter(
            x=[x_value, x_value],
            y=[row['Ret-Vol'], row['Ret+Vol']],
            mode='lines',
            line=dict(color=color, dash='dash'),
            showlegend=False
        ))

    # Update layout to add padding for the labels
    fig.update_layout(
        title="Return and Volatility across Portfolios, ordered by Sharpe Ratio",
        xaxis=dict(
            title="Portfolios (ordered by Sharpe Ratio)",
            range=[0, len(df_copy) + 1],  # Add padding on the x-axis
            showgrid=False
        ),
        yaxis=dict(
            title="Return & Volatility",
            range=[
                df_copy['Ret-Vol'].min() - 0.05,  # Add padding below the min
                df_copy['Ret+Vol'].max() + 0.05   # Add padding above the max
            ],
            showgrid=True
        ),
        plot_bgcolor="white",
        showlegend=True,
        margin=dict(t=80, b=60, l=60, r=60)  # Add margins to prevent clipping
    )

    return fig



def plot_portf_wts_db_plotly(ports_wts):
    """
    Plots asset weights per portfolio using Plotly in a grouped bar chart.
    
    Parameters:
    - ports_wts: pandas DataFrame where rows are portfolios and columns are assets. Values are weights.
    """
    # Convert DataFrame to long format for Plotly
    ports_wts_long = ports_wts.reset_index().melt(id_vars='index', var_name='Asset', value_name='Weight')
    ports_wts_long.rename(columns={'index': 'Portfolio'}, inplace=True)

    # Create a grouped bar chart
    fig = px.bar(
        ports_wts_long,
        x="Portfolio",
        y="Weight",
        color="Asset",
        title="Asset Weights per Portfolio",
        labels={"Portfolio": "Portfolio", "Weight": "Asset Weight", "Asset": "Asset"},
        barmode="group"  # Use grouped bar mode
    )

    # Customize layout
    fig.update_layout(
        xaxis_title="Portfolio",
        yaxis_title="Weight",
        plot_bgcolor="white",
        height=600,
        width=900
    )

    return fig

def plot_risk_cont_db_plotly(portf_wts, VCV):
    """
    Plots risk contributions per ticker using Plotly.
    
    Parameters:
    - portf_wts: pandas DataFrame of portfolio weights (rows: portfolios, columns: tickers).
    - VCV: Variance-Covariance matrix (pandas DataFrame or numpy array).
    """
    # Compute the risk contributions
    res_df = risk_cont_ports(portf_wts, VCV)  # Assumes `risk_cont_ports` is defined elsewhere

    # Transpose for easier visualization (tickers as rows, portfolios as columns)
    res_df = res_df.T

    # Convert DataFrame to long format for Plotly
    res_df_long = res_df.reset_index().melt(id_vars='index', var_name='Asset', value_name='Risk Contribution')
    res_df_long.rename(columns={'index': 'Ticker'}, inplace=True)

    # Create a grouped bar chart
    fig = px.bar(
        res_df_long,
        x="Ticker",
        y="Risk Contribution",
        color="Asset",
        title="Risk Contribution per Ticker",
        labels={"Ticker": "Ticker", "Risk Contribution": "Risk Contribution", "Asset": "Asset"},
        barmode="group"  # Use grouped bar mode
    )

    # Customize layout
    fig.update_layout(
        xaxis_title="Ticker",
        yaxis_title="Risk Contribution",
        plot_bgcolor="white",
        height=600,
        width=900
    )

    return fig

def plot_sharpe_cont_plotly(portf_wts, ER, VCV, rfr):
    """
    Plots Sharpe contributions per ticker using Plotly.
    
    Parameters:
    - portf_wts: pandas DataFrame of portfolio weights (rows: portfolios, columns: tickers).
    - ER: Expected returns (pandas Series or DataFrame).
    - VCV: Variance-Covariance matrix (pandas DataFrame or numpy array).
    - rfr: Risk-free rate (float or pandas Series).
    """
    # Compute the Sharpe contributions
    red_df = sharpe_cont_ports(portf_wts, ER, VCV, rfr)  # Assumes `sharpe_cont_ports` is defined elsewhere

    # Transpose for easier visualization (tickers as rows, portfolios as columns)
    red_df = red_df.T

    # Convert DataFrame to long format for Plotly
    red_df_long = red_df.reset_index().melt(id_vars='index', var_name='Asset', value_name='Sharpe Contribution')
    red_df_long.rename(columns={'index': 'Ticker'}, inplace=True)

    # Create a grouped bar chart
    fig = px.bar(
        red_df_long,
        x="Ticker",
        y="Sharpe Contribution",
        color="Asset",
        title="Sharpe Contribution per Ticker",
        labels={"Ticker": "Ticker", "Sharpe Contribution": "Sharpe Contribution", "Asset": "Asset"},
        barmode="group"  # Use grouped bar mode
    )

    # Customize layout
    fig.update_layout(
        xaxis_title="Ticker",
        yaxis_title="Sharpe Contribution",
        plot_bgcolor="white",
        height=600,
        width=900
    )

    return fig

def plot_ef_w_assets_plotly_old(rets, plot_df, rfr=0, n_points=1000, cml=True,
                            plot_benchmark=False, benchmark_stats=None,
                            plot_current=False, current_stats=None, custom_portfolio=None, custom_value=None):
    # Compute Efficient Frontier
    er, asset_vols, cov = rets.mean(), rets.std(), rets.cov()
    targets = np.linspace(er.min(), er.max(), n_points)
    wts = [minimize_vol(tr, er, cov) for tr in targets]
    ef = pd.DataFrame({
        "Returns": [portfolio_return(w, er) for w in wts],
        "Volatility": [portfolio_vol(w, cov) for w in wts]
    })

    # Create Plotly figure
    fig = go.Figure()

    # Plot Efficient Frontier
    fig.add_trace(go.Scatter(
        x=ef["Volatility"], y=ef["Returns"],
        mode='lines', name="Efficient Frontier",
        line=dict(color="blue", width=2)
    ))

    # Plot individual assets
    fig.add_trace(go.Scatter(
        x=asset_vols, y=er, mode='markers+text',
        text=rets.columns, name="Assets",
        textposition="top center", marker=dict(color="blue", size=10)
    ))

    # Plot additional portfolios
    fig.add_trace(go.Scatter(
        x=plot_df['Volatility'], y=plot_df['Return'],
        mode='markers+text', text=plot_df.index,
        name="Portfolios", textposition="top center",
        marker=dict(color="red", size=10)
    ))

    # Plot CML
    if cml:
        w_msr = msr(rfr, er, cov)
        r_msr, vol_msr = portfolio_return(w_msr, er), portfolio_vol(w_msr, cov)
        fig.add_trace(go.Scatter(
            x=[0, vol_msr], y=[rfr, r_msr], mode='lines+markers',
            name="CML", line=dict(color="black", dash="dash"),
            marker=dict(size=7, color="black")
        ))

    # Plot Benchmark
    if plot_benchmark and benchmark_stats is not None:
        if isinstance(benchmark_stats, dict):
            benchmark_vol = benchmark_stats.get('Volatility')
            benchmark_ret = benchmark_stats.get('Return')
        elif isinstance(benchmark_stats, pd.DataFrame):
            benchmark_vol = benchmark_stats['Volatility'].iloc[0]
            benchmark_ret = benchmark_stats['Return'].iloc[0]
        else:
            raise ValueError("benchmark_stats must be a dictionary or DataFrame.")

        if benchmark_vol is not None and benchmark_ret is not None:
            fig.add_trace(go.Scatter(
                x=[benchmark_vol], y=[benchmark_ret],
                mode='markers+text',
                text=["Benchmark"],
                name="Benchmark",
                textposition="top center",
                marker=dict(color="black", size=12, symbol="diamond"),
                showlegend=True
            ))

    # Plot Current Portfolio
    if plot_current and current_stats is not None:
        if isinstance(current_stats, pd.DataFrame):
            current_vol = current_stats['Volatility'].iloc[0]
            current_ret = current_stats['Return'].iloc[0]
        else:
            raise ValueError("current_stats must be a DataFrame.")

        if current_vol is not None and current_ret is not None:
            fig.add_trace(go.Scatter(
                x=[current_vol], y=[current_ret],
                mode='markers+text',
                text=["Current Portfolio"],
                name="Current Portfolio",
                textposition="top center",
                marker=dict(color="black", size=12, symbol="x"),
                showlegend=True
            ))
            fig.add_shape(
                type="line", 
                x0=current_vol, x1=current_vol, y0=0, y1=current_ret,
                line=dict(color="black", dash="dot")
            )
            fig.add_shape(
                type="line", 
                x0=0, x1=current_vol, y0=current_ret, y1=current_ret,
                line=dict(color="black", dash="dot")
            )

    # Plot Custom Portfolio
    if custom_portfolio and custom_value:
        custom_df = ef.loc[ef['Volatility'].idxmin():]
        if custom_portfolio == 'return':
            closest_idx = np.abs(custom_df['Returns'] - custom_value).idxmin()
        else:  # custom_portfolio == 'volatility'
            closest_idx = np.abs(custom_df['Volatility'] - custom_value).idxmin()
        fig.add_trace(go.Scatter(
            x=[ef.loc[closest_idx, 'Volatility']], y=[ef.loc[closest_idx, 'Returns']],
            mode='markers+text', text=["Custom Portfolio"],
            name="Custom Portfolio", textposition="top center",
            marker=dict(color="purple", size=12, symbol="star")
        ))

    # Update layout
    fig.update_layout(
        title="Efficient Frontier with Assets",
        xaxis_title="Volatility", yaxis_title="Returns",
        legend_title="Legend", plot_bgcolor="white",
        height=700, width=1000
    )

    return fig

def plot_multiple_factor_loadings_plotly(portf_rets, factor_returns, benchmark=None):
    """
    Plots factor loadings across multiple portfolios using Plotly.
    
    Parameters:
    - portf_rets: pandas DataFrame containing portfolio returns (columns as portfolios).
    - factor_returns: pandas DataFrame containing factor returns (columns as factors).
    - benchmark: Optional pandas Series or DataFrame for benchmark returns.
    """
    # Include benchmark if provided
    if benchmark is not None:
        portf_rets = pd.concat([portf_rets, benchmark], axis=1)

    # Compute factor loadings
    res = get_multiple_factor_loadings(portf_rets, factor_returns)  # Assumes this function is defined elsewhere

    # Transpose result for correct structure (Factors as rows, Portfolios as columns)
    res = res.T

    # Convert result to long format for Plotly
    res_long = res.reset_index().melt(id_vars='index', var_name='Factor', value_name='Factor Loading')
    res_long.rename(columns={'index': 'Portfolio'}, inplace=True)

    # Create grouped bar chart
    fig = px.bar(
        res_long,
        x='Portfolio',
        y='Factor Loading',
        color='Factor',
        title="Factor Loadings across Portfolios",
        barmode="group",
        labels={"Portfolio": "Portfolio", "Factor Loading": "Factor Loading", "Factor": "Factor"}
    )

    # Customize layout
    fig.update_layout(
        xaxis_title="Portfolio",
        yaxis_title="Factor Loading",
        plot_bgcolor="white",
        height=600,
        width=900
    )

    return fig

def plot_gbm_and_cum_rets_plotly_old(asset_returns, n_years=1, n_scenarios=100, steps_per_year=252, plot_last=252*2, plot_sims=None):
    """
    Plots cumulative portfolio returns with optional GBM simulations using Plotly.
    
    Parameters:
    - asset_returns: pandas DataFrame of asset returns (columns are assets, index is time).
    - n_years: Number of years for GBM simulations.
    - n_scenarios: Number of GBM simulation scenarios.
    - steps_per_year: Number of steps per year for GBM simulations.
    - plot_last: Number of last data points to include.
    - plot_sims: List of asset names to include GBM simulations for.
    """
    # Filter the last `plot_last` rows if specified
    if plot_last:
        asset_returns = asset_returns.tail(plot_last)

    # Calculate cumulative returns
    cum_rets = (asset_returns + 1).cumprod()

    # Generate GBM simulations
    gbm_simulations = generate_gbm_from_asset_returns(
        asset_returns, n_years=n_years, n_scenarios=n_scenarios, steps_per_year=steps_per_year
    )

    # Initialize Plotly figure
    fig = go.Figure()

    # Define a consistent color mapping for assets
    asset_colors = {col: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)] for i, col in enumerate(cum_rets.columns)}

    # Plot cumulative returns for each asset
    for col in cum_rets.columns:
        fig.add_trace(go.Scatter(
            x=cum_rets.index,
            y=cum_rets[col],
            mode='lines',
            name=col,
            line=dict(
                dash='dash' if col in ['Benchmark', 'Current Portfolio'] else 'solid',
                color=asset_colors[col]  # Use consistent colors
            )
        ))

    # Overlay GBM simulations if specified
    if plot_sims:
        for sim in plot_sims:
            if sim in gbm_simulations:
                sim_color = asset_colors.get(sim, "lightgrey")  # Match simulation color to the asset
                for scenario in gbm_simulations[sim].columns:
                    fig.add_trace(go.Scatter(
                        x=gbm_simulations[sim].index,
                        y=gbm_simulations[sim][scenario],
                        mode='lines',
                        name=f"{sim} Simulation",
                        line=dict(color=sim_color, width=1),
                        opacity=0.2,
                        showlegend=False
                    ))

    # Update layout
    fig.update_layout(
        title="Cumulative Portfolio Returns with Simulations" if plot_sims else "Cumulative Portfolio Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        legend_title="Assets",
        plot_bgcolor="white",
        height=700,
        width=1000
    )

    return fig




def plot_portfolios_plotly(asset_returns, benchmark=None, plot_benchmark=False, current_portfolio=None, plot_current_portfolio=False):
    # Add benchmark data if requested
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")
        if isinstance(benchmark, pd.DataFrame):
            benchmark.columns = ["Benchmark"]  # Rename column if it's a DataFrame
        elif isinstance(benchmark, pd.Series):
            benchmark = benchmark.rename("Benchmark")  # Rename if it's a Series
        else:
            raise ValueError("Benchmark must be a pandas DataFrame or Series.")
        asset_returns = pd.concat([asset_returns, benchmark], axis=1)

    # Add current portfolio data if requested
    if plot_current_portfolio:
        if current_portfolio is not None:
            if isinstance(current_portfolio, pd.DataFrame):
                current_portfolio.columns = ["Current Portfolio"]  # Rename column if it's a DataFrame
            elif isinstance(current_portfolio, pd.Series):
                current_portfolio = current_portfolio.rename("Current Portfolio")  # Rename if it's a Series
            else:
                raise ValueError("Current Portfolio must be a pandas DataFrame or Series.")
            asset_returns = pd.concat([current_portfolio, asset_returns], axis=1)

    # Drop missing data
    asset_returns = asset_returns.dropna(axis=0)

    # Calculate cumulative returns
    asset_cum_returns = (asset_returns + 1).cumprod()

    # Define styles
    #line_styles = {"Current Portfolio": "dash","Benchmark": "dash"}
    #line_colors = {"Current Portfolio": "black","Benchmark": "grey"}

    line_colors = {
        'Markowitz': "#00008b", 'Global Minimum Variance': "#ff8c00", 'Maximum Sharpe Ratio Portfolio': "#006400",
        'Equal Weights': "#FF0000", 'Risk Parity': "#9400d3", 'Inverse Variance': "#a0522d",
        'Inverse Volatility': "#00bfff", 'Maximum Diversification': "#00ff00", 
        'Maximum De-Correlation': "#ffd700", 'Minimum Correlation': "#ee82ee",
        'Equal Sharpe Ratio': "#deb887", 'Benchmark': "grey", 'Current Portfolio': "black",
        'Minimum Expected Shortfall': "#8B0000",
        'Minimum Max Drawdown': "#4682B4"}

    line_styles = {
    "Benchmark": "dash",
    "Current Portfolio": "dash",
    "Markowitz": "solid",
    "Global Minimum Variance": "solid",
    "Maximum Sharpe Ratio Portfolio": "solid",
    "Equal Weights": "solid",
    "Risk Parity": "solid",
    "Inverse Variance": "solid",
    "Inverse Volatility": "solid","Maximum Diversification": "solid","Maximum De-Correlation": "solid",
    "Minimum Correlation": "solid","Equal Sharpe Ratio": "solid",
    "Minimum Expected Shortfall": "solid","Minimum Max Drawdown'": "solid"}


    # Create a Plotly figure
    fig = go.Figure()

    for column in asset_cum_returns.columns:
        fig.add_trace(go.Scatter(
            x=asset_cum_returns.index,
            y=asset_cum_returns[column],
            mode='lines',
            line=dict(
                dash=line_styles.get(column, "solid"),
                color=line_colors.get(column, None)
            ),
            name=column
        ))

    # Update layout
    fig.update_layout(
        title="Cumulative Portfolio Returns over Time",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        plot_bgcolor="white",
        legend_title="Portfolios",
        height=600,
        width=900
    )

    # Show the plot
    return fig


def plot_indiv_assets_plotly(asset_returns, benchmark=None, plot_benchmark=False, current_portfolio=None, plot_current_portfolio=False):
    # Add benchmark data if requested
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")
        if isinstance(benchmark, pd.DataFrame):
            benchmark.columns = ["Benchmark"]  # Rename column if it's a DataFrame
        elif isinstance(benchmark, pd.Series):
            benchmark = benchmark.rename("Benchmark")  # Rename if it's a Series
        else:
            raise ValueError("Benchmark must be a pandas DataFrame or Series.")
        asset_returns = pd.concat([asset_returns, benchmark], axis=1)

    # Add current portfolio data if requested
    if plot_current_portfolio:
        if current_portfolio is not None:
            if isinstance(current_portfolio, pd.DataFrame):
                current_portfolio.columns = ["Current Portfolio"]  # Rename column if it's a DataFrame
            elif isinstance(current_portfolio, pd.Series):
                current_portfolio = current_portfolio.rename("Current Portfolio")  # Rename if it's a Series
            else:
                raise ValueError("Current Portfolio must be a pandas DataFrame or Series.")
            asset_returns = pd.concat([current_portfolio, asset_returns], axis=1)

    # Drop missing data
    asset_returns = asset_returns.dropna(axis=0)

    # Calculate cumulative returns
    asset_cum_returns = (asset_returns + 1).cumprod()

    # Define styles
    line_styles = {"Current Portfolio": "dash","Benchmark": "dash"}
    line_colors = {"Current Portfolio": "black","Benchmark": "grey"}

    # Create a Plotly figure
    fig = go.Figure()

    for column in asset_cum_returns.columns:
        fig.add_trace(go.Scatter(
            x=asset_cum_returns.index,
            y=asset_cum_returns[column],
            mode='lines',
            line=dict(
                dash=line_styles.get(column, "solid"),
                color=line_colors.get(column, None)
            ),
            name=column
        ))

    # Update layout
    fig.update_layout(
        title="Cumulative Ticker Returns over Time",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        plot_bgcolor="white",
        legend_title="Assets",
        height=600,
        width=900
    )

    return fig  # Return the Plotly figure object


def plot_drawdowns_plotly(rets, benchmark=None, plot_benchmark=False):
    # Add benchmark data if requested
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")
        if isinstance(benchmark, pd.DataFrame):
            benchmark.columns = ["Benchmark"]  # Rename column if it's a DataFrame
        elif isinstance(benchmark, pd.Series):
            benchmark = benchmark.rename("Benchmark")  # Rename if it's a Series
        else:
            raise ValueError("Benchmark must be a pandas DataFrame or Series.")
        rets = pd.concat([rets, benchmark], axis=1)

    # Drop missing data
    rets = rets.dropna(axis=0)

    # Calculate cumulative returns and drawdowns
    cum_rets = (rets + 1).cumprod()
    cum_max = cum_rets.cummax()
    drawdown = (cum_rets - cum_max) / cum_max

    # Define styles
    line_styles = {"Benchmark": "dash","Current Portfolio": "dash","Custom Portfolio": "dot"}
    line_colors = {"Benchmark": "grey","Current Portfolio": "black","Custom Portfolio": "black"}

    line_colors = {
        'Markowitz': "#00008b", 'Global Minimum Variance': "#ff8c00", 'Maximum Sharpe Ratio Portfolio': "#006400",
        'Equal Weights': "#FF0000", 'Risk Parity': "#9400d3", 'Inverse Variance': "#a0522d",
        'Inverse Volatility': "#00bfff", 'Maximum Diversification': "#00ff00", 
        'Maximum De-Correlation': "#ffd700", 'Minimum Correlation': "#ee82ee",
        'Equal Sharpe Ratio': "#deb887", 'Benchmark': "grey", 'Current Portfolio': "black",
        'Minimum Expected Shortfall': "#8B0000",
        'Minimum Max Drawdown': "#4682B4"}

    line_styles = {
    "Benchmark": "dash",
    "Current Portfolio": "dash",
    "Markowitz": "solid",
    "Global Minimum Variance": "solid",
    "Maximum Sharpe Ratio Portfolio": "solid",
    "Equal Weights": "solid",
    "Risk Parity": "solid",
    "Inverse Variance": "solid",
    "Inverse Volatility": "solid","Maximum Diversification": "solid","Maximum De-Correlation": "solid",
    "Minimum Correlation": "solid","Equal Sharpe Ratio": "solid",
    "Minimum Expected Shortfall": "solid","Minimum Max Drawdown'": "solid"}


    
    # Create a Plotly figure
    fig = go.Figure()

    for column in drawdown.columns:
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown[column],
            mode='lines',
            line=dict(
                dash=line_styles.get(column, "solid"),
                color=line_colors.get(column, None)
            ),
            name=column
        ))

    # Update layout
    fig.update_layout(
        title="Portfolio Drawdowns over Time",
        xaxis_title="Date",
        yaxis_title="Drawdowns",
        plot_bgcolor="white",
        legend_title="Portfolios",
        height=600,
        width=900
    )

    return fig  # Return the Plotly figure object


def plot_return_density_old(rets_df, n_years=1, plot_last=90, n_scenarios=1000, steps_per_year=252, plot_sims=None):
    """
    Plots an overlapping density plot of simulated returns for specified portfolios.

    Parameters:
    - rets_df: pandas DataFrame of asset returns (columns are assets, index is time).
    - n_years: Number of years for GBM simulations.
    - plot_last: Number of last data points to include in the simulation.
    - n_scenarios: Number of GBM simulation scenarios.
    - steps_per_year: Number of steps per year for GBM simulations.
    - plot_sims: List of column names in `rets_df` to include in the density plot.

    Returns:
    - fig: A Plotly figure object.
    """
    if plot_sims is None:
        raise ValueError("The `plot_sims` argument cannot be None. Provide a list of column names to include.")

    # Generate simulations using the provided GBM function
    sims = calc_gbm(
        rets_df,
        n_years=n_years,
        n_scenarios=n_scenarios,
        steps_per_year=steps_per_year,
        plot_last=plot_last
    )

    # Initialize Plotly figure
    fig = go.Figure()

    # Loop through the specified simulations in `plot_sims`
    for sim in plot_sims:
        if sim not in sims:
            raise ValueError(f"'{sim}' is not a valid column name in the simulated data.")

        # Extract the last simulation values for the current simulation
        data = sims[sim].tail(1).to_numpy().flatten()

        # Perform Kernel Density Estimation (KDE) for smoothing
        kde = gaussian_kde(data)

        # Define x-axis range for the density plot
        x_range = np.linspace(data.min(), data.max(), 500)

        # Evaluate density
        density = kde(x_range)

        # Add KDE trace for the current simulation
        fig.add_trace(go.Scatter(
            x=x_range,
            y=density,
            mode='lines',
            name=sim,
            line=dict(width=2),
            fill='tozeroy',  # Fill the area under the curve
            opacity=0.5
        ))

    # Update layout for better visualization
    fig.update_layout(
        title="Distribution of simulated Cumulative Returns across Portfolios",
        xaxis_title="Simulated Returns",
        yaxis_title="Density",
        plot_bgcolor='white',
        legend=dict(title="Portfolio"),
        height=600,
        width=900)

    return fig



def plot_gbm_and_cum_rets_plotly(asset_returns, n_years=1, n_scenarios=100, steps_per_year=252, plot_last=252*2, plot_sims=None, color_mapping=None):
    """
    Plots cumulative portfolio returns with optional GBM simulations using Plotly.
    
    Parameters:
    - asset_returns: pandas DataFrame of asset returns (columns are assets, index is time).
    - n_years: Number of years for GBM simulations.
    - n_scenarios: Number of GBM simulation scenarios.
    - steps_per_year: Number of steps per year for GBM simulations.
    - plot_last: Number of last data points to include.
    - plot_sims: List of asset names to include GBM simulations for.
    - color_mapping: Dictionary mapping asset names to specific colors.
    """
    # Default color mapping if not provided
    if color_mapping is None:
        color_mapping = {
        'Markowitz': "#00008b", 'Global Minimum Variance': "#ff8c00", 'Maximum Sharpe Ratio Portfolio': "#006400",
        'Equal Weights': "#FF0000", 'Risk Parity': "#9400d3", 'Inverse Variance': "#a0522d",
        'Inverse Volatility': "#00bfff", 'Maximum Diversification': "#00ff00", 
        'Maximum De-Correlation': "#ffd700", 'Minimum Correlation': "#ee82ee",
        'Equal Sharpe Ratio': "#deb887", 'Benchmark': "grey", 'Current Portfolio': "black",
        'Minimum Expected Shortfall': "#8B0000",
        'Minimum Max Drawdown': "#4682B4"}

    # Filter the last `plot_last` rows if specified
    if plot_last:
        asset_returns = asset_returns.tail(plot_last)

    # Calculate cumulative returns
    cum_rets = (asset_returns + 1).cumprod()

    # Generate GBM simulations
    gbm_simulations = generate_gbm_from_asset_returns(
        asset_returns, n_years=n_years, n_scenarios=n_scenarios, steps_per_year=steps_per_year
    )

    # Initialize Plotly figure
    fig = go.Figure()

    # Plot cumulative returns for each asset
    for col in cum_rets.columns:
        fig.add_trace(go.Scatter(
            x=cum_rets.index,
            y=cum_rets[col],
            mode='lines',
            name=col,
            line=dict(
                dash='dash' if col in ['Benchmark', 'Current Portfolio'] else 'solid',
                color=color_mapping.get(col, "lightgrey")  # Use color from mapping, default to grey
            )
        ))

    # Overlay GBM simulations if specified
    if plot_sims:
        for sim in plot_sims:
            if sim in gbm_simulations:
                sim_color = color_mapping.get(sim, "lightgrey")  # Match simulation color to the asset
                for scenario in gbm_simulations[sim].columns:
                    fig.add_trace(go.Scatter(
                        x=gbm_simulations[sim].index,
                        y=gbm_simulations[sim][scenario],
                        mode='lines',
                        name=f"{sim} Simulation",
                        line=dict(color=sim_color, width=1),
                        opacity=0.2,
                        showlegend=False
                    ))

    # Update layout
    fig.update_layout(
        title="Cumulative Portfolio Returns with Simulations" if plot_sims else "Cumulative Portfolio Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        legend_title="Assets",
        plot_bgcolor="white",
        height=700,
        width=1000
    )

    return fig





def plot_return_density(rets_df, n_years=1, plot_last=90, n_scenarios=1000, steps_per_year=252, plot_sims=None):
    """
    Plots an overlapping density plot of simulated returns for specified portfolios.

    Parameters:
    - rets_df: pandas DataFrame of asset returns (columns are assets, index is time).
    - n_years: Number of years for GBM simulations.
    - plot_last: Number of last data points to include in the simulation.
    - n_scenarios: Number of GBM simulation scenarios.
    - steps_per_year: Number of steps per year for GBM simulations.
    - plot_sims: List of column names in `rets_df` to include in the density plot.

    Returns:
    - fig: A Plotly figure object.
    """
    if plot_sims is None:
        raise ValueError("The `plot_sims` argument cannot be None. Provide a list of column names to include.")

    # Predefined color mapping
    color_mapping = {
        'Markowitz': "#00008b", 'Global Minimum Variance': "#ff8c00", 'Maximum Sharpe Ratio Portfolio': "#006400",
        'Equal Weights': "#FF0000", 'Risk Parity': "#9400d3", 'Inverse Variance': "#a0522d",
        'Inverse Volatility': "#00bfff", 'Maximum Diversification': "#00ff00", 
        'Maximum De-Correlation': "#ffd700", 'Minimum Correlation': "#ee82ee",
        'Equal Sharpe Ratio': "#deb887", 'Benchmark': "grey", 'Current Portfolio': "black",
        'Minimum Expected Shortfall': "#8B0000",
        'Minimum Max Drawdown': "#4682B4"}

    # Generate simulations using the provided GBM function
    sims = calc_gbm(
        rets_df,
        n_years=n_years,
        n_scenarios=n_scenarios,
        steps_per_year=steps_per_year,
        plot_last=plot_last
    )

    # Initialize Plotly figure
    fig = go.Figure()

    # Loop through the specified simulations in `plot_sims`
    for sim in plot_sims:
        if sim not in sims:
            raise ValueError(f"'{sim}' is not a valid column name in the simulated data.")

        # Extract the last simulation values for the current simulation
        data = sims[sim].tail(1).to_numpy().flatten()

        # Perform Kernel Density Estimation (KDE) for smoothing
        kde = gaussian_kde(data)

        # Define x-axis range for the density plot
        x_range = np.linspace(data.min(), data.max(), 500)

        # Evaluate density
        density = kde(x_range)

        # Add KDE trace for the current simulation
        fig.add_trace(go.Scatter(
            x=x_range,
            y=density,
            mode='lines',
            name=sim,
            line=dict(
                color=color_mapping.get(sim, "lightgrey"),  # Use predefined color mapping
                width=2
            ),
            fill='tozeroy',  # Fill the area under the curve
            opacity=0.5
        ))

    # Update layout for better visualization
    fig.update_layout(
        title="Distribution of Simulated Cumulative Returns Across Portfolios",
        xaxis_title="Simulated Returns",
        yaxis_title="Density",
        plot_bgcolor='white',
        legend=dict(title="Portfolio"),
        height=600,
        width=900
    )

    return fig




def process_and_plot_plotly(df):
    df_copy = df.copy()
    df_copy['Return'] = df_copy['Return'] * 252
    df_copy['Volatility'] = df_copy['Volatility'] * np.sqrt(252)
    df_sorted = df_copy.sort_values(by="Volatility").reset_index()

    color_mapping = {'Markowitz': "#00008b",'Global Minimum Variance': "#ff8c00",
        'Maximum Sharpe Ratio Portfolio': "#006400",'Equal Weights': "#FF0000",'Risk Parity': "#9400d3",
        'Inverse Variance': "#a0522d",'Inverse Volatility': "#00bfff",'Maximum Diversification': "#00ff00",'Maximum De-Correlation': "#ffd700",'Minimum Correlation': "#ee82ee",'Equal Sharpe Ratio': "#deb887",'Benchmark': "grey",'Current Portfolio': "black"}

    fig = go.Figure()

    for i, row in df_sorted.iterrows():
        index_name = row[df_sorted.columns[0]]  # Assume the first column contains names
        color = color_mapping.get(index_name, "#C71585")  # Default color if not found
        
        fig.add_trace(go.Scatter(
            x=[row['Volatility']],
            y=[row['Return']],
            mode='markers+text',
            name=index_name,
            text=[index_name],
            textposition="top center",
            textfont=dict(color="black"),  # Force black text
            hovertemplate=(
                f"<b>{index_name}</b><br>"
                f"Volatility: {row['Volatility']:.2%}<br>"
                f"Return: {row['Return']:.2%}<extra></extra>"
            ),
            marker=dict(color=color, size=12)
        ))

    if 'Current Portfolio' in df_sorted[df_sorted.columns[0]].values:
        current_portfolio_row = df_sorted[df_sorted[df_sorted.columns[0]] == 'Current Portfolio']
        current_volatility = current_portfolio_row['Volatility'].values[0]
        current_return = current_portfolio_row['Return'].values[0]

        fig.add_shape(
            type="line",
            x0=current_volatility,
            x1=current_volatility,
            y0=0,
            y1=current_return,
            line=dict(color="black", dash="dash")
        )
        fig.add_shape(
            type="line",
            x0=0,
            x1=current_volatility,
            y0=current_return,
            y1=current_return,
            line=dict(color="black", dash="dash")
        )

    fig.update_layout(
        title="Return vs Volatility Scatter Plot",
        xaxis_title="Volatility",
        yaxis_title="Return",
        plot_bgcolor="white",  # White plot background
        paper_bgcolor="white",  # White paper background
        height=600,
        width=900,
        showlegend=False)
    return fig





def plot_ef_w_assets_plotly(rets, plot_df, rfr=0, n_points=1000, cml=True,
                            plot_benchmark=False, benchmark_stats=None,
                            plot_current=False, current_stats=None, custom_portfolio=None, custom_value=None):
    # Compute Efficient Frontier
    er, asset_vols, cov = rets.mean(), rets.std(), rets.cov()
    targets = np.linspace(er.min(), er.max(), n_points)
    wts = [minimize_vol(tr, er, cov) for tr in targets]
    ef = pd.DataFrame({
        "Returns": [portfolio_return(w, er) for w in wts],
        "Volatility": [portfolio_vol(w, cov) for w in wts]
    })

    # Create Plotly figure
    fig = go.Figure()

    # Plot Efficient Frontier
    fig.add_trace(go.Scatter(
        x=ef["Volatility"], y=ef["Returns"],
        mode='lines', name="Efficient Frontier",
        line=dict(color="blue", width=2)
    ))

    # Plot individual assets
    fig.add_trace(go.Scatter(
        x=asset_vols, y=er, mode='markers+text',
        text=rets.columns, name="Assets",
        textposition="top center", textfont=dict(color="black"),  # Ensure black font
        marker=dict(color="blue", size=10)
    ))

    # Plot additional portfolios
    fig.add_trace(go.Scatter(
        x=plot_df['Volatility'], y=plot_df['Return'],
        mode='markers+text', text=plot_df.index,
        name="Portfolios", textposition="top center",
        textfont=dict(color="black"),  # Ensure black font
        marker=dict(color="red", size=10)
    ))

    # Plot CML
    if cml:
        w_msr = msr(rfr, er, cov)
        r_msr, vol_msr = portfolio_return(w_msr, er), portfolio_vol(w_msr, cov)
        fig.add_trace(go.Scatter(
            x=[0, vol_msr], y=[rfr, r_msr], mode='lines+markers',
            name="CML", line=dict(color="black", dash="dash"),
            marker=dict(size=7, color="black")
        ))

    # Plot Benchmark
    if plot_benchmark and benchmark_stats is not None:
        if isinstance(benchmark_stats, dict):
            benchmark_vol = benchmark_stats.get('Volatility')
            benchmark_ret = benchmark_stats.get('Return')
        elif isinstance(benchmark_stats, pd.DataFrame):
            benchmark_vol = benchmark_stats['Volatility'].iloc[0]
            benchmark_ret = benchmark_stats['Return'].iloc[0]
        else:
            raise ValueError("benchmark_stats must be a dictionary or DataFrame.")

        if benchmark_vol is not None and benchmark_ret is not None:
            fig.add_trace(go.Scatter(
                x=[benchmark_vol], y=[benchmark_ret],
                mode='markers+text',
                text=["Benchmark"],
                name="Benchmark",
                textposition="top center",
                textfont=dict(color="black"),  # Ensure black font
                marker=dict(color="black", size=12, symbol="diamond"),
                showlegend=True
            ))

    # Plot Current Portfolio
    if plot_current and current_stats is not None:
        if isinstance(current_stats, pd.DataFrame):
            current_vol = current_stats['Volatility'].iloc[0]
            current_ret = current_stats['Return'].iloc[0]
        else:
            raise ValueError("current_stats must be a DataFrame.")

        if current_vol is not None and current_ret is not None:
            fig.add_trace(go.Scatter(
                x=[current_vol], y=[current_ret],
                mode='markers+text',
                text=["Current Portfolio"],
                name="Current Portfolio",
                textposition="top center",
                textfont=dict(color="black"),  # Ensure black font
                marker=dict(color="black", size=12, symbol="x"),
                showlegend=True
            ))
            fig.add_shape(
                type="line", 
                x0=current_vol, x1=current_vol, y0=0, y1=current_ret,
                line=dict(color="black", dash="dot")
            )
            fig.add_shape(
                type="line", 
                x0=0, x1=current_vol, y0=current_ret, y1=current_ret,
                line=dict(color="black", dash="dot")
            )

    # Plot Custom Portfolio
    if custom_portfolio and custom_value:
        custom_df = ef.loc[ef['Volatility'].idxmin():]
        if custom_portfolio == 'return':
            closest_idx = np.abs(custom_df['Returns'] - custom_value).idxmin()
        else:  # custom_portfolio == 'volatility'
            closest_idx = np.abs(custom_df['Volatility'] - custom_value).idxmin()
        fig.add_trace(go.Scatter(
            x=[ef.loc[closest_idx, 'Volatility']], y=[ef.loc[closest_idx, 'Returns']],
            mode='markers+text', text=["Custom Portfolio"],
            name="Custom Portfolio", textposition="top center",
            textfont=dict(color="black"),  # Ensure black font
            marker=dict(color="purple", size=12, symbol="star")
        ))

    # Update layout
    fig.update_layout(
        title="Efficient Frontier with Assets",
        xaxis_title="Volatility", yaxis_title="Returns",
        legend_title="Legend", plot_bgcolor="white",
        height=700, width=1000
    )

    return fig


def plot_change_wts(ports_wts, portfolio_selection=None):
    change_wts = calc_change(ports_wts)

    color_mapping = {
        'Markowitz': "#00008b", 'Global Minimum Variance': "#ff8c00", 'Maximum Sharpe Ratio Portfolio': "#006400",
        'Equal Weights': "#FF0000", 'Risk Parity': "#9400d3", 'Inverse Variance': "#a0522d",
        'Inverse Volatility': "#00bfff", 'Maximum Diversification': "#00ff00", 
        'Maximum De-Correlation': "#ffd700", 'Minimum Correlation': "#ee82ee",
        'Equal Sharpe Ratio': "#deb887", 'Benchmark': "grey", 'Current Portfolio': "black",
        'Minimum Expected Shortfall': "#8B0000",
        'Minimum Max Drawdown': "#4682B4"}

    if portfolio_selection is None:
        return None

    if not isinstance(portfolio_selection, list):
        portfolio_selection = [portfolio_selection]
    missing_portfolios = [p for p in portfolio_selection if p not in change_wts.index]
    if missing_portfolios:
        raise ValueError(f"Portfolio(s) '{missing_portfolios}' not found in DataFrame index.")
    change_wts = change_wts.loc[portfolio_selection]
    title_prefix = "Change in Portfolio Weights" if "Current Portfolio" in ports_wts.index else "Portfolio Weights"
    fig = make_subplots(rows=len(change_wts), cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=[f"{title_prefix}: {portfolio}" for portfolio in change_wts.index])
    for i, (portfolio_name, weights) in enumerate(change_wts.iterrows(), start=1):
        x_values = weights.index
        y_values = weights.values
        #formatted_values = [f"{round(value * 100, 2)}%" for value in y_values]
        formatted_values = [f"{round(value * 100, 2)}%" if pd.notna(value) else "0%" for value in y_values]

        portfolio_color = color_mapping.get(portfolio_name, 'blue')  
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=y_values,
                name=portfolio_name,
                marker_color=portfolio_color
            ),
            row=i,
            col=1
        )

        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(weights) - 0.5,
            y0=0,
            y1=0,
            line=dict(color="black", width=2),
            row=i,
            col=1
        )

        for x, y, text in zip(x_values, y_values, formatted_values):
            fig.add_annotation(
                x=x,
                y=y + 0.02 if y >= 0 else y - 0.02,
                text=text,
                showarrow=False,
                font=dict(color="black"),
                align="center",
                yanchor="bottom" if y >= 0 else "top",
                row=i,
                col=1
            )

    fig.update_layout(
        paper_bgcolor="white",  # Outer background. This is new!
        plot_bgcolor="white",  # Inner plot background. This is new!
        height=400 * len(change_wts),  
        title={
            'text': title_prefix,
            'font': dict(color="black"),
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color="black"),  
            titlefont=dict(color="black")  
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            tickfont=dict(color="black"),  
            titlefont=dict(color="black")  
        ),
        legend=dict(
            font=dict(color="black") 
        ),
        template="plotly_white"
    )

    for annotation in fig.layout.annotations:
        annotation.font.color = "black"

    return fig