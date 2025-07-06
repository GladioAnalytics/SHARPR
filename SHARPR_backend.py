import yfinance as yf 
from yahoo_fin.stock_info import *
## Standard Python Data Science stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statsmodels.api as sm
import datetime as dt
import math
import os
from tqdm import tqdm
from tqdm.notebook import tqdm
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
import logging
from scipy.stats import gaussian_kde
from scipy.stats import (norm as norm, linregress as linregress)
from scipy.optimize import linprog
from plotly.subplots import make_subplots
from tiingo import TiingoClient
import logging
logging.getLogger("tiingo").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

tiingo_key = '5788d5bd795f6105fd8f4e5570a3bd6204dca3a6'
config = {'api_key': tiingo_key,'session': True}
client = TiingoClient(config)

plt.rcParams['figure.figsize'] = [20, 10]

######################################################################################################################


def check_tickers_tiingo(tickers):
    tickers = [t.upper() for t in tickers]
    valid = []
    invalid = []

    for t in tickers:
        try:
            data = client.get_ticker_price(t, fmt='json')
            (valid if data else invalid).append(t)
        except Exception:
            invalid.append(t)

    return {
        "valid tickers": valid,
        "invalid tickers": invalid
    }

def handle_invalid_tiingo_tickers(invalid_tickers):
    if not invalid_tickers:
        return "All tickers were found in the database."
    else:
        return f"The following tickers were not found in the database: {', '.join(invalid_tickers)}."


def get_tiingo_returns(ticker_list):
    if 'client' not in globals():
        return "Error: 'client' is not defined. Ensure TiingoClient is initialized globally."

    if not hasattr(client, 'get_dataframe'):
        return "Error: 'client' is not a valid TiingoClient instance."

    ticker_list = [t.upper() for t in ticker_list]
    price_list = []
    failed_tickers = []

    for ticker in ticker_list:
        try:
            stock_df = client.get_dataframe(ticker, startDate="1900-01-01", frequency="daily")
            if not stock_df.empty:
                adj_price = stock_df['adjClose'].rename(ticker)
                price_list.append(adj_price)
            else:
                failed_tickers.append(ticker)
        except Exception:
            failed_tickers.append(ticker)

    if not price_list:
        return f"Error: None of the tickers were successfully retrieved. Failed tickers: {', '.join(failed_tickers)}"

    prices_df = pd.concat(price_list, axis=1)
    returns_df = prices_df.pct_change()
    returns_df.index = pd.to_datetime(returns_df.index.date)
    returns_df.index.name = 'Date'

    return returns_df


def get_tiingo_benchmark(benchmark_ticker):
    if 'client' not in globals():
        return "Error: 'client' is not defined. Ensure TiingoClient is initialized globally."

    if not hasattr(client, 'get_dataframe'):
        return "Error: 'client' is not a valid TiingoClient instance."

    ticker = benchmark_ticker.upper()

    try:
        stock_df = client.get_dataframe(ticker, startDate="1900-01-01", frequency="daily")
        if stock_df.empty:
            return f"Error: No data found for ticker '{ticker}'."
        benchmark_series = stock_df['adjClose'].pct_change().dropna().rename('Benchmark')
        benchmark_series.index = pd.to_datetime(benchmark_series.index.date)
        benchmark_series.index.name = 'Date'
        return benchmark_series

    except Exception as e:
        return f"Error retrieving benchmark ticker '{ticker}': {e}"


def get_latest_prices_tiingo(ticker_list):
    latest_prices = {}

    for ticker in ticker_list:
        try:
            df = client.get_dataframe(ticker, startDate="2020-01-01", frequency="daily")
            if not df.empty and 'adjClose' in df.columns:
                latest_prices[ticker] = df['adjClose'].iloc[-1]
            else:
                latest_prices[ticker] = float('nan')
        except Exception:
            latest_prices[ticker] = float('nan')

    return pd.Series(latest_prices).dropna()

######################################################################################################################













def check_tickers(tickers):
    logger = logging.getLogger("yfinance")
    logger.setLevel(logging.CRITICAL)

    tickers = [t.upper() for t in tickers]
    valid = []
    invalid = []

    for t in tickers:
        try:
            df = yf.download(t, period='1d', progress=False)
            (valid if not df.empty else invalid).append(t)
        except Exception:
            invalid.append(t)

    return {'valid': valid, 'invalid': invalid}


def handle_invalid_tickers(invalid_tickers):
    if not invalid_tickers:
        ret_string = "All Tickers were found in the database"
    else:
        ret_string = f"The tickers {', '.join(invalid_tickers)} were not found in the database."
    return ret_string

def get_returns(ticker_list):
    if not ticker_list:
        return "No Tickers submitted"
    ticker_list = [ticker.upper() for ticker in ticker_list]
    ticker_list = check_tickers(ticker_list)['valid tickers']
    hist = yf.download(ticker_list,auto_adjust = False)
    data = hist.loc[:, 'Adj Close']
    data = ((data / data.shift(1)) - 1)
    returns = data.dropna()
    return returns


def get_returns_old(ticker_list):
    hist = yf.download(ticker_list)
    data = hist.loc[:,'Adj Close']
    data = ((data/data.shift(1))-1)
    returns = data.dropna()
    return returns

def get_benchmark_old(benchmark_ticker,returns):
    benchmark = yf.download(benchmark_ticker).loc[:,'Adj Close']
    benchmark = ((benchmark/benchmark.shift(1))-1)
    benchmark = benchmark.loc[returns.index[0]:]
    benchmark = benchmark.to_frame(name='Benchmark')
    return benchmark

def get_benchmark(benchmark_ticker, returns):
    if benchmark_ticker is None or benchmark_ticker == '':
        return None
    benchmark = yf.download(benchmark_ticker,auto_adjust = False)['Adj Close']
    if benchmark.empty:
        return None
    else:
        benchmark = ((benchmark / benchmark.shift(1)) - 1).dropna()
        benchmark = benchmark.loc[returns.index[0]:]
        benchmark.columns=['Benchmark']
    return benchmark


def add_benchmark(rets,benchmark_rets):
    if benchmark_rets is not None:
        res = pd.concat([rets, benchmark_rets], axis=1, join='inner')
    else:
        res = rets
    return res

def equal_weights(returns):
    weights = [1/len(returns.columns)]*len(returns.columns)
    return weights

def get_equal_weights(ticker_list):
    wts = [1/len(ticker_list)]*len(ticker_list)
    wts = pd.Series(wts, index=ticker_list)
    return wts

def random_wts_old(tickers):
    weights = np.random.random(len(tickers))
    weights /= weights.sum()
    return pd.Series(np.round(weights,4),index=tickers)

def random_wts(tickers):
    weights = np.random.random(len(tickers))
    weights /= weights.sum()
    rounded_weights = np.round(weights, 4)
    difference = 1 - rounded_weights.sum()
    adjustment = difference / len(tickers)
    adjusted_weights = rounded_weights + adjustment

    return pd.Series(adjusted_weights, index=tickers)

def create_current_wts(tickers, wts):
    if len(tickers) != len(wts):
        raise ValueError("The lists tickers and weights must have the same length.")
    return pd.Series(wts, index=tickers)

def calc_current_rets(returns, current_wts):
    if current_wts is None:
        current_rets = None
    else:
        current_wts = current_wts.T
        current_rets = returns @ current_wts.T
    return current_rets

def calc_current_port_rets(returns, ports_wts):
    ports_wts.squeeze()
    portf_rets = returns @ ports_wts.T
    portf_rets_df = pd.DataFrame(portf_rets, index=returns.index, columns=["Current Portfolio"])
    return portf_rets_df

def create_current_portfolio(ticker_list,wts):
    current_wts_df = pd.DataFrame([wts], columns=ticker_list, index=['Current Portfolio'])
    return current_wts_df

def calc_current_stats(current_port_rets):
    Returns = current_port_rets.mean().iloc[0]
    Volatility = current_port_rets.std().iloc[0]
    
    current_stats = pd.DataFrame({
        'Returns': [Returns],
        'Volatility': [Volatility]
    }, index=["Current Portfolio"])
    
    return current_stats

def calc_ret_vol_df(df):
    mean_returns = df.mean()
    std_volatility = df.std()
    result_df = pd.DataFrame({
        'Return': mean_returns,
        'Volatility': std_volatility})
    return result_df


def get_current_weights(ticker_list, current_values=None, stock_numbers=None, current_wts=None):
    if current_wts is not None:
        current_wts = pd.Series(current_wts, index=ticker_list)
        res = pd.DataFrame(current_wts.values, index=ticker_list, columns=["Current Portfolio"])

    elif stock_numbers is not None:
        stock_prices = yf.download(ticker_list)['Adj Close'].tail(1).squeeze()
        capital = (stock_prices * stock_numbers).sum()
        current_wts = (stock_prices * stock_numbers) / capital
        res = pd.DataFrame(current_wts.values, index=ticker_list, columns=["Current Portfolio"])

    elif current_values is not None:
        current_values = pd.Series(current_values, index=ticker_list)
        current_wts = current_values / current_values.sum()
        res = pd.DataFrame(current_wts.values, index=ticker_list, columns=["Current Portfolio"])

    else:
        res = None
    
    return res

def get_ret_vol(ret_vol_df, Portfolio):
    if Portfolio in ret_vol_df.index:
        res = pd.DataFrame(ret_vol_df.loc[[Portfolio]])
        return res
    else:
        return None


def ann_ret(r, ppy):
    comp_growth = (1+r).prod()
    periods = r.shape[0]
    return comp_growth**(ppy/periods)-1

def ann_vol(r, ppy):
    return r.std()*(ppy**0.5)

def portfolio_return(weights, returns):
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    vol = (weights.T @ covmat @ weights)**0.5
    return vol

def calc_port_rets(returns, ports_wts):
    portf_rets = returns @ ports_wts.T
    return portf_rets

def sharpe_ratio(r, rfr, ppy):
    rf_pp = (1+rfr)**(1/ppy)-1
    excess = r - rf_pp
    ann_ex = ann_ret(excess, ppy)
    ann_volat = ann_vol(r, ppy)
    return ann_ex/ann_volat

def portfolio_sharpe_ratio(w0,ER,VCV,rfr):
    pvol = portfolio_vol(w0,VCV)
    p_ex_ret = portfolio_return(w0,ER)-rfr
    p_sharpe = p_ex_ret/pvol
    return p_sharpe

def risk_contribution(w,cov):
    total_portfolio_var = portfolio_vol(w,cov)**2
    marginal_contrib = cov@w
    risk_contrib = np.multiply(marginal_contrib,w.T)/total_portfolio_var
    return risk_contrib

def sharpe_contribution(w0,ER,VCV,rfr):
    indiv_ex_ret = w0*(ER-rfr)
    port_vol = portfolio_vol(w0,VCV)
    indiv_sharpe = indiv_ex_ret/port_vol
    port_sharpe = (portfolio_sharpe_ratio(w0,ER,VCV,rfr))
    sharpe_cont = indiv_sharpe/port_sharpe
    return sharpe_cont

def div_ratio(w0,VCV):
    div_measure = np.dot(w0.T,np.diag(VCV))/portfolio_vol(w0,VCV)
    return - div_measure

def target_risk_contributions(target_risk, cov):
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def msd_risk(weights, target_risk, cov):
        w_contribs = risk_contribution(weights, cov)
        return ((w_contribs-target_risk)**2).sum()
    
    weights = minimize(msd_risk, init_guess,
                       args=(target_risk, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return pd.Series(np.round(weights.x,6),index=cov.columns)

def target_sharpe_contribution(sharpe_target,ER,VCV,rfr):
    n = VCV.shape[0]
    w0 = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    weights_sum_to_1 = {'type': 'eq','fun': lambda w0: np.sum(w0)-1} 

    def msd_sharpe(w0=w0, sharpe_target=sharpe_target, VCV=VCV):
        actual_sharpe_cont = sharpe_contribution(w0,ER=ER,VCV=VCV,rfr=rfr)
        sharpe_msd = ((sharpe_target - actual_sharpe_cont)**2).sum()
        return sharpe_msd
    
    weights = minimize(msd_sharpe,w0,args=(sharpe_target), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return pd.Series(np.round(weights.x,6),index=VCV.columns)

def prep_std_mean(srs):
    srs = srs.squeeze()
    ret = srs.mean()
    vol = srs.std()
    res = pd.Series([ret, vol],index=["Returns", "Volatility"])
    return res

####Markowitz
def minimize_vol(target_return, er, cov):
    from scipy.optimize import minimize
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!

    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return pd.Series(np.round(weights.x,6),index=cov.columns)

####
def maximize_tr(target_vol, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n 
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
                        }

    def neg_portfolio_return(weights, returns):
        return -np.sum(weights * returns)

    vol_is_target = {'type': 'eq',
                     'args': (cov,),
                     'fun': lambda weights, cov: target_vol - portfolio_vol(weights, cov)
                     }

    weights = minimize(neg_portfolio_return, init_guess,
                       args=(er,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1, vol_is_target),
                       bounds=bounds)

    return pd.Series(np.round(weights.x, 6), index=cov.columns)

#####Max Sharpe Ratio
def msr(rfr, er, cov):
    from scipy.optimize import minimize
    n = er.shape[0]
    guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n 
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1}
    def neg_sharpe(weights, rfr, er, cov):

        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - rfr)/vol
    
    weights = minimize(neg_sharpe, guess,
                       args=(rfr, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return pd.Series(np.round(weights.x,6),index=cov.columns)

#####GMV
def gmv(cov):
    n = cov.shape[0]
    return msr(0, np.repeat(1,n), cov)

####Risk-Parity
def equal_risk_contributions(cov):
    n = cov.shape[0]
    return target_risk_contributions(target_risk=np.repeat(1/n,n), cov=cov)

####Equal Weighted
def eq_weight_portf(VCV):
    wts = 1/len(VCV.columns)
    return pd.Series(np.round(wts,6),index=VCV.columns)

####Inverse Volatility
def inv_vol_portf(VCV):
    a = (1/np.sqrt(np.diag(VCV)))
    b = sum((1/np.sqrt(np.diag(VCV))))
    res = a/b
    return pd.Series(np.round(res,6),index=VCV.columns)

####Inverse Variance
def inv_var_portf(VCV):
    a = (1/np.diag(VCV))
    b = sum((1/np.diag(VCV)))
    res = a/b
    return pd.Series(np.round(res,6),index=VCV.columns)

####Max Diversification 
def max_div_port(w0,VCV): 
    pdm = div_ratio(w0,VCV)
    bounds = ((0.0, 1.0),)*VCV.shape[0]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    res = minimize(div_ratio, w0, args=(VCV,), constraints=constraints,bounds = bounds)
    return pd.Series(np.round(res.x, 6), index=VCV.columns)

####Max Decorrelation
def max_decorr_portf(w0,CORR):
    def synth_corr(w0,CORR):
        corr_matrix = np.dot(np.dot(w0.T, CORR),w0)
        return -(1-corr_matrix)
    cons = ({'type': 'eq', 'fun': lambda w0: np.sum(w0) - 1})
    bounds = ((0.0, 1.0),) * len(w0) 
    res = minimize(synth_corr, w0, args=(CORR,), method='SLSQP', constraints=cons,bounds=bounds)
    return pd.Series(np.round(res.x,6),index=CORR.columns)

####Minimum Correlation 
def min_corr_portf(CORR,COV):
    if len(COV.columns) == 2:
        wts = inv_vol_portf(COV)
        return pd.Series(np.round(wts,6),index=CORR.columns)
    else:
        indiv_stddevs = np.sqrt(np.diag(COV.values))
        rCORR = CORR.values
        np.fill_diagonal(rCORR, 0)
        avg_corr_cont = np.sum(rCORR, axis=0) / (rCORR.shape[0] - 1)
        perc_cont = (np.argsort(np.argsort(avg_corr_cont))+1)/np.sum(np.argsort(np.argsort(avg_corr_cont))+1)
        avg_corr = np.mean(rCORR[np.nonzero(np.triu(rCORR))])
        std_corr = std_corr = np.std(rCORR[np.nonzero(np.triu(rCORR))],ddof=1)
        norm_dist_matrix = (1-np.round(norm.cdf(rCORR,loc=avg_corr,scale=std_corr),6))
        np.fill_diagonal(norm_dist_matrix, 0)
        rmat = (norm_dist_matrix*perc_cont).T
        rmat_perc = np.sum(rmat, axis=0) / np.sum(np.sum(rmat, axis=0))

        inv_vols_wts = 1/indiv_stddevs / (np.sum(1/indiv_stddevs))

        raw_mat = inv_vols_wts * rmat_perc
        wts = raw_mat / np.sum(raw_mat)
        return pd.Series(np.round(wts,6),index=CORR.columns)

######Equal Sharpe Ratio Portfolio
def target_sharpe_contribution(sharpe_target,ER,VCV,rfr):
    n = VCV.shape[0]
    w0 = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    weights_sum_to_1 = {'type': 'eq','fun': lambda w0: np.sum(w0)-1} 

    def msd_sharpe(w0, sharpe_target,VCV):
        actual_sharpe_cont = sharpe_contribution(w0,ER,VCV,rfr)
        sharpe_msd = ((actual_sharpe_cont - sharpe_target)**2).sum()
        return sharpe_msd
    
    weights = minimize(msd_sharpe,w0,args=(sharpe_target,VCV,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return pd.Series(np.round(weights.x,6),index=VCV.columns)

def plot_asset_returns(asset_returns, benchmark=None, plot_benchmark=False):
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")

        asset_returns = pd.concat([asset_returns,benchmark],axis=1)
    asset_cum_returns = (asset_returns + 1).cumprod()
    asset_cum_returns.plot()

    plt.title('Cumulative Ticker Returns over Time')
    plt.show()

def plot_indiv_assets_old(asset_returns, benchmark=None, plot_benchmark=False, current_portfolio=None, plot_current_portfolio=False):
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")
        asset_returns = pd.concat([asset_returns, benchmark], axis=1)

    if plot_current_portfolio:
        if current_portfolio is None:
            raise ValueError("No current portfolio data provided.")
        asset_returns = pd.concat([asset_returns, current_portfolio], axis=1)

    asset_returns = asset_returns.dropna(axis=0)
    asset_cum_returns = (asset_returns + 1).cumprod()

    line_styles = {}
    line_colors = {}
    
    if "Current Portfolio" in asset_cum_returns.columns:
        line_styles["Current Portfolio"] = 'dashed'
        line_colors["Current Portfolio"] = 'black'
    
    if "Benchmark" in asset_cum_returns.columns:
        line_styles["Benchmark"] = 'dashed'
        line_colors["Benchmark"] = 'grey'

    plt.figure(figsize=(12, 8.5))
    plt.title('Cumulative Ticker Returns over Time')
    for column in asset_cum_returns.columns:
        plt.plot(asset_cum_returns.index, asset_cum_returns[column], label=column, 
                 linestyle=line_styles.get(column, '-'), color=line_colors.get(column, None))

    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(False)
    plt.show()

def plot_indiv_assets(asset_returns, benchmark=None, plot_benchmark=False, current_portfolio=None, plot_current_portfolio=False):
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")
        asset_returns = pd.concat([asset_returns, benchmark], axis=1)

    if plot_current_portfolio:
        if current_portfolio is None:
            asset_returns = asset_returns
        if current_portfolio is not None:
            asset_returns = pd.concat([current_portfolio, asset_returns], axis=1)

    asset_returns = asset_returns.dropna(axis=0)
    asset_cum_returns = (asset_returns + 1).cumprod()

    line_styles = {}
    line_colors = {}
    
    if "Current Portfolio" in asset_cum_returns.columns:
        line_styles["Current Portfolio"] = 'dashed'
        line_colors["Current Portfolio"] = 'black'
    
    if "Benchmark" in asset_cum_returns.columns:
        line_styles["Benchmark"] = 'dashed'
        line_colors["Benchmark"] = 'grey'

    plt.figure(figsize=(12, 8.5))
    plt.title('Cumulative Ticker Returns over Time')
    for column in asset_cum_returns.columns:
        plt.plot(asset_cum_returns.index, asset_cum_returns[column], label=column, 
                 linestyle=line_styles.get(column, '-'), color=line_colors.get(column, None))

    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(False)
    plt.show()

def plot_portfolios(asset_returns, benchmark=None, plot_benchmark=False, current_portfolio=None, plot_current_portfolio=False):
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")
        asset_returns = pd.concat([asset_returns, benchmark], axis=1)

    if plot_current_portfolio:
        if current_portfolio is None:
            raise ValueError("No current portfolio data provided.")
        asset_returns = pd.concat([asset_returns, current_portfolio], axis=1)

    asset_returns = asset_returns.dropna(axis=0)
    asset_cum_returns = (asset_returns + 1).cumprod()

    line_styles = {}
    line_colors = {}
    
    if "Current Portfolio" in asset_cum_returns.columns:
        line_styles["Current Portfolio"] = 'dashed'
        line_colors["Current Portfolio"] = 'black'
    
    if "Benchmark" in asset_cum_returns.columns:
        line_styles["Benchmark"] = 'dashed'
        line_colors["Benchmark"] = 'grey'

    plt.figure(figsize=(12, 8.5))
    plt.title('Cumulative Portfolio Returns over Time')
    for column in asset_cum_returns.columns:
        plt.plot(asset_cum_returns.index, asset_cum_returns[column], label=column, 
                 linestyle=line_styles.get(column, '-'), color=line_colors.get(column, None))

    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(False)
    plt.show()


def portfolio_weights(w0,rfr,target_return,ER,VCV,CORR):
    markowitz_wts = minimize_vol(target_return, ER, VCV)
    glob_min_var_wts = gmv(VCV)
    max_sharpe_wts = msr(rfr, ER, VCV)
    equal_wts = eq_weight_portf(VCV)
    risk_parity_wts = equal_risk_contributions(VCV)
    inv_var_wts = inv_vol_portf(VCV)
    inv_vol_wts = inv_var_portf(VCV)
    max_div_wts = max_div_port(w0,VCV)
    max_decorr_wts = max_decorr_portf(w0,CORR)
    min_corr_wts = min_corr_portf(CORR,VCV)
    equal_sharpe_wts = target_sharpe_contribution((1/len(VCV.columns)),ER,VCV,rfr)
    res = pd.concat([markowitz_wts,glob_min_var_wts,max_sharpe_wts,equal_wts,risk_parity_wts,inv_var_wts,inv_vol_wts,max_div_wts,max_decorr_wts,min_corr_wts,equal_sharpe_wts],axis=1)
    df = pd.DataFrame(res.T)
    df.index = ["Markowitz","Global Minimum Variance","Maximum Sharpe Ratio Portfolio","Equal Weights","Risk Parity","Inverse Variance","Inverse Volatility","Maximum Diversification","Maximum De-Correlation","Minimum Correlation","Equal Sharpe Ratio"]
    return df

def create_wts_df(ports_wts,current_wts=None,current_portfolio=False):
    if current_portfolio and current_wts is not None:
        ports_wts = pd.concat([current_wts.T,ports_wts], ignore_index=False)
    else:
        ports_wts = ports_wts
    return ports_wts

def plot_portf_wts(ports_wts):
    ports_wts.plot(kind='bar').set_title('Asset Weights per Portfolio')

def risk_cont_ports(portf_wts,VCV):
    res_df = portf_wts.T.apply((lambda x: risk_contribution(x,VCV)))
    return res_df

def plot_risk_cont(portf_wts,VCV):
    res_df = risk_cont_ports(portf_wts,VCV)
    res_df.T.plot(kind='bar').set_title('Risk-Contribution per Ticker')

def sharpe_cont_ports(portf_wts,ER,VCV,rfr):
    '''ER and COV are  Expected Return, Variance-Covariance-Matrix of the CONSTITUENT ASSETS, NOT the Portfolios!'''
    res_df = portf_wts.T.apply(lambda x: sharpe_contribution(x,ER,VCV,rfr))
    return res_df

def plot_sharpe_cont(portf_wts,ER,VCV,rfr):
    red_df = sharpe_cont_ports(portf_wts,ER,VCV,rfr)
    red_df.T.plot(kind='bar').set_title('Sharpe-Contribution per Ticker')

def process_and_plot(df):
    df_copy = df.copy()
    df_copy['Return'] = df_copy['Return'] * 252
    df_copy['Volatility'] = df_copy['Volatility'] * np.sqrt(252)
    df_sorted = df_copy.sort_values(by="Volatility").reset_index()

    color_mapping = {
        'Markowitz': "#00008b",  
        'Global Minimum Variance': "#ff8c00",  
        'Maximum Sharpe Ratio Portfolio': "#006400",  
        'Equal Weights': "#FF0000",  
        'Risk Parity': "#9400d3",  
        'Inverse Variance': "#a0522d",  #
        'Inverse Volatility': "#00bfff",  
        'Maximum Diversification': "#00ff00",  
        'Maximum De-Correlation': "#ffd700",  
        'Minimum Correlation': "#ee82ee",  
        'Equal Sharpe Ratio': "#deb887", 
        'Benchmark': "grey",  
        'Current Portfolio': "black"  
    }

    plt.figure(figsize=(10, 6))
    for i, row in df_sorted.iterrows():
        index_name = row[df_sorted.columns[0]]  
        color = color_mapping.get(index_name, "#C71585")  
        
        plt.scatter(row['Volatility'], row['Return'], color=color, s=100)
        plt.text(row['Volatility'], row['Return'], index_name, fontsize=9, ha='left', va='bottom')

    current_portfolio_row = df_sorted[df_sorted[df_sorted.columns[0]] == 'Current Portfolio']

    if not current_portfolio_row.empty:
        current_volatility = current_portfolio_row['Volatility'].values[0]
        current_return = current_portfolio_row['Return'].values[0]
        
        plt.axvline(x=current_volatility, color='black', linestyle='--')
        plt.axhline(y=current_return, color='black', linestyle='--')

    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Return vs Volatility Scatter Plot')
    plt.tight_layout()
    plt.show()

def plot_ef_w_assets(rets, plot_df, rfr=0, n_points=1000, cml=True,
                       plot_benchmark=False, benchmark_stats=None,
                       plot_current=False, current_stats=None, custom_portfolio=None, custom_value=None):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    er = rets.mean()
    asset_vols = rets.std()
    cov = rets.cov()
    targets = np.linspace(er.min(), er.max(), n_points)
    wts = [minimize_vol(tr, er, cov) for tr in targets]
    ptf_rts = [portfolio_return(w, er) for w in wts]
    ptf_vls = [portfolio_vol(w, cov) for w in wts]
    
    ef = pd.DataFrame({
        "Returns": ptf_rts, 
        "Volatility": ptf_vls
    })
    
    ax = ef.plot.line(x="Volatility", y="Returns")
    
    # Plot individual assets
    x = asset_vols
    y = er
    labels = rets.columns
    ax.scatter(x, y, color="blue", s=25, alpha=0.5, linewidths=1)
    for x_pos, y_pos, label in zip(x, y, labels):
        ax.annotate(label, xy=(x_pos, y_pos), xytext=(7, 0),
                    textcoords='offset points', ha='left', va='center')

    # Plot additional points (e.g., portfolio)
    x2 = plot_df['Volatility']
    y2 = plot_df['Return']
    labels2 = plot_df.index
    ax.scatter(x2, y2, color="red", s=25, alpha=0.5, linewidths=1)
    for x_pos, y_pos, label in zip(x2, y2, labels2):
        ax.annotate(label, xy=(x_pos, y_pos), xytext=(7, 0),
                    textcoords='offset points', ha='left', va='center')
    
    # Plot CML (Capital Market Line)
    if cml:
        ax.set_xlim(left=0)
        w_msr = msr(rfr, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        cml_x = [0, vol_msr]
        cml_y = [rfr, r_msr]
        ax.plot(cml_x, cml_y, color='black', marker='o', linestyle='dashed', linewidth=1, markersize=5)
    
    # Plot Benchmark
    if plot_benchmark and benchmark_stats is not None:
        benchmark_return = benchmark_stats['Return'].item()
        benchmark_vol = benchmark_stats['Volatility'].item()
        ax.scatter(benchmark_vol, benchmark_return, color='black', marker='o', s=50, label='Benchmark')
        ax.annotate('Benchmark', xy=(benchmark_vol, benchmark_return), xytext=(7, 0),
                    textcoords='offset points', ha='left', va='center')
    
    # Plot Current Portfolio and draw dotted lines
    if plot_current and current_stats is not None:
        current_return = current_stats['Return'].item()
        current_vol = current_stats['Volatility'].item()
        
        # Scatter for Current Portfolio
        ax.scatter(current_vol, current_return, color='black', marker='o', s=50, label='Current Portfolio')
        ax.annotate('Current Portfolio', xy=(current_vol, current_return), xytext=(7, 0),
                    textcoords='offset points', ha='left', va='center')
        
        # Draw vertical and horizontal dotted lines
        ax.axvline(x=current_vol, color='black', linestyle=':', linewidth=1)  # Vertical line
        ax.axhline(y=current_return, color='black', linestyle=':', linewidth=1)  # Horizontal line

    # Custom Portfolio logic remains unchanged
    if custom_portfolio == 'return':
        custom_df = ef.loc[ef['Volatility'].idxmin():]
        custom_val = custom_value
        abs_diff = np.abs(custom_df['Returns'] - custom_val)
        closest_indices = abs_diff.nsmallest(2).index
        closest_return_values = custom_df.loc[closest_indices, 'Returns']
        weights = (custom_val - closest_return_values) / np.diff(abs(closest_return_values))
        closest_volatility = custom_df.loc[closest_indices, 'Volatility']
        custom_vol = (abs(weights) * closest_volatility).sum()
        custom_ret = custom_val
        ax.scatter(custom_vol, custom_ret, color='black', marker='o', s=50, label='Custom Portfolio')
        ax.annotate('Custom Portfolio', xy=(custom_vol, custom_ret), xytext=(7, 0),
                    textcoords='offset points', ha='left', va='center')
    
    if custom_portfolio == 'volatility':
        custom_df = ef.loc[ef['Volatility'].idxmin():]
        custom_val = custom_value
        abs_diff = np.abs(custom_df['Volatility'] - custom_val)
        closest_indices = abs_diff.nsmallest(2).index
        closest_volatility_values = custom_df.loc[closest_indices, 'Volatility']
        weights = (custom_val - closest_volatility_values) / np.diff(abs(closest_volatility_values))
        closest_returns = custom_df.loc[closest_indices, 'Returns']
        custom_ret = (abs(weights) * closest_returns).sum()
        custom_vol = custom_val
        ax.scatter(custom_vol, custom_ret, color='black', marker='o', s=50, label='Custom Portfolio')
        ax.annotate('Custom Portfolio', xy=(custom_vol, custom_ret), xytext=(7, 0),
                    textcoords='offset points', ha='left', va='center')
    
    plt.show()

def drawdown(returns,capital=1):
    wealth = capital*(1+returns).cumprod()
    peaks = wealth.cummax()
    drawdowns = (wealth - peaks)/peaks
    return drawdowns, wealth, peaks

def semidev(r):
    return r[r<0].std().to_frame(name='Semi-Deviation')

def var_hist(r,level=5):
    if isinstance(r,pd.Series):
        res = np.round(np.percentile(r,level),6)
        res = pd.Series(res, name="Historical VaR")
    elif isinstance(r,pd.DataFrame):
        res = r.apply(lambda x: np.percentile(x,q=level)).to_frame(name = "Historical VaR")
    return res.T

def cvar_hist(r,level=5):
    if isinstance(r,pd.Series):
        is_beyond = r<=var_hist(r,level=5).squeeze()
        res = np.round(r[is_beyond].mean(),6)
        df = pd.Series(res,name="Conditional VaR")
    if isinstance(r,pd.DataFrame):
        df = r.apply(lambda x: cvar_hist(x,level=level))
        df.index = ['Conditional VaR']
    return df

def var_gauss(r, level=5, modified=False):
    from scipy.stats import norm
    from scipy.stats import skew
    from scipy.stats import kurtosis
    z = norm.ppf(level/100)
    if isinstance(r, pd.Series):
        if modified:
            s = skew(r)
            k = kurtosis(r)
            z = norm.ppf(level/100) + (z**2 - 1)*s/6 + (z**3 -3*z)*(k-3)/24 - (2*z**3 - 5*z)*(s**2)/36
        else:
            z = norm.ppf(level/100)
        gauss_var = pd.Series((r.mean() + z * r.std(ddof=0)), name='Gaussian VaR')
    if isinstance(r, pd.DataFrame):
        level = level
        modified = modified
        gauss_var = r.apply(lambda x: var_gauss(x, level=level, modified=modified),axis=0)
        gauss_var.index = ['Gaussian VaR']
    return gauss_var

def comp(returns):
    return returns.add(1).prod() - 1 

def group_returns(returns, groupby, compounded=False):
    #group_returns(df, df.index.year)
    #group_returns(df, [df.index.year, df.index.month])
    if compounded:
        return returns.groupby(groupby).apply(comp)
    return returns.groupby(groupby).sum()

def aggregate_returns(returns, period=None, compounded=True):
    if period is None or 'day' in period:
        return returns
    index = returns.index
    if 'month' in period:
        return group_returns(returns, index.month, compounded=compounded)
    if 'quarter' in period:
        return group_returns(returns, index.quarter, compounded=compounded)
    if period == "A" or any(x in period for x in ['year', 'eoy', 'yoy']):
        return group_returns(returns, index.year, compounded=compounded)
    if 'week' in period:
        return group_returns(returns, index.week, compounded=compounded)
    if 'eow' in period or period == "W":
        return group_returns(returns, [index.year, index.week],
                             compounded=compounded)
    if 'eom' in period or period == "M":
        return group_returns(returns, [index.year, index.month],
                             compounded=compounded)
    if 'eoq' in period or period == "Q":
        return group_returns(returns, [index.year, index.quarter],
                             compounded=compounded)
    if not isinstance(period, str):
        return group_returns(returns, period, compounded)
    return returns
def count_consecutive(data):
    def _count(data):
        return data * (data.groupby(
            (data != data.shift(1)).cumsum()).cumcount() + 1)
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            data[col] = _count(data[col])
        return data
    return _count(data)

def best(returns, aggregate=None, compounded=True):
    return aggregate_returns(returns, aggregate, compounded).max().to_frame(name='Best Return')
def worst(returns, aggregate=None, compounded=True):
    return aggregate_returns(returns, aggregate, compounded).min().to_frame(name='Worst Return')
def consecutive_wins(returns, aggregate=None, compounded=True):
    returns = aggregate_returns(returns, aggregate, compounded) > 0
    return count_consecutive(returns).max().to_frame(name='Consecutive Winds')
def consecutive_losses(returns, aggregate=None, compounded=True):
    returns = aggregate_returns(returns, aggregate, compounded) < 0
    return count_consecutive(returns).max().to_frame(name='Consecutive Losses')

def win_rate(returns):
    def win_rate(series):
        try:
            return (len(series[series > 0]) / len(series[series != 0]))
        except Exception:
            return 0.
    if isinstance(returns, pd.DataFrame):
        df = {}
        for col in returns.columns:
            df[col] = win_rate(returns[col])
        return pd.Series(df).to_frame(name='Win Rate')
    return win_rate(returns).to_frame(name='Win Rate')

def avg_return(returns, aggregate=None, compounded=True):
    if aggregate:
        returns = aggregate_returns(returns, aggregate, compounded)
    return (returns[returns != 0].dropna().mean()).to_frame(name='Average Return')

def avg_win(returns, aggregate=None, compounded=True):
    if aggregate:
        returns = aggregate_returns(returns, aggregate, compounded)
    return (returns[returns > 0].dropna().mean()).to_frame(name='Average Win')

def avg_loss(returns, aggregate=None, compounded=True):
    if aggregate:
        returns = aggregate_returns(returns, aggregate, compounded)
    return (returns[returns < 0].dropna().mean()).to_frame(name='Average Loss')

def avg_drawdow(returns):
    dd = - drawdown_series(returns)
    res = dd.mean().to_frame(name='Average Drawdown')
    return res

def sharpe(returns, rfr):
    res = ((returns.mean() - rfr)/returns.std()).to_frame(name='Sharpe Ratio')
    return res

def treynor(returns, benchmark,rfr):
    temp = returns.apply(lambda x: get_alpha_and_beta(x,benchmark))
    temp.index = ['Alpha','Beta']
    beta = temp.loc['Beta']
    res = ((returns.mean() - rfr)/beta).to_frame(name='Treynor Ratio')
    return res

def sortino(returns, rf=0, periods=252, annualize=True,):
    if rf != 0 and periods is None:
        raise Exception('Must provide periods if rf != 0')
    returns = returns
    downside = np.sqrt((returns[returns < 0] ** 2).sum() / len(returns))
    res = (returns.mean() / downside).to_frame(name='Sortino Ratio')
    if annualize:
        return res * np.sqrt(1 if periods is None else periods)
    return res.to_frame(name='Sortino Ratio')

def omega(returns,required_returns,periods=252):
    return_threshold = (1 + required_returns) ** (1. / periods) - 1
    win = returns[returns>0] - return_threshold
    lose = return_threshold  - returns[returns<0]
    omega = win.sum()/lose.sum()
    return omega.to_frame(name='Omega Ratio')

def gain_to_pain_ratio(returns, rf=0, resolution="D"):
    downside = abs(returns[returns < 0].sum())
    return (returns.sum() / downside).to_frame(name='Gain-To-Pain-Ratio')
    
def cagr(returns, rf=0., compounded=True):
    total = returns
    if compounded:
        total = comp(total)
    else:
        total = np.sum(total)
    years = (returns.index[-1] - returns.index[0]).days / 365.
    res = abs(total + 1.0) ** (1.0 / years) - 1
    if isinstance(returns, pd.DataFrame):
        res = pd.Series(res)
        res.index = returns.columns

    return res.to_frame(name='CAGR')

def get_rsquare(returns, benchmark):
    df = pd.concat([returns,benchmark],axis=1)
    df = df.dropna(axis=0)
    _, _, r_val, _, _= linregress(df)
    res = r_val**2
    return res

def get_alpha_and_beta(returns, benchmark):
    df = pd.concat([returns,benchmark],axis=1)
    df = df.dropna(axis=0)
    beta, alpha, _, _, _= linregress(df)
    return alpha, beta
#res = rets.apply(lambda x: get_alpha_and_beta(x,benchmark))
#res.index = ['Alpha','Beta']

def info_ratio(rets,benchmark):
    data=pd.concat([benchmark,rets],axis=1)
    data = data.iloc[:,1:].subtract(data['Benchmark'], axis=0)
    res = pd.DataFrame(pd.Series(data.mean()/data.std(),name = 'Information Ratio'))
    return res

def kelly_criterion(returns):
    win_loss_ratio = payoff_ratio(returns).squeeze()
    win_prob = win_rate(returns).squeeze()
    lose_prob = 1 - win_prob
    return (((win_loss_ratio * win_prob) - lose_prob) / win_loss_ratio).to_frame(name='Kelly Criterion')

def profit_ratio(returns):
    wins = returns[returns >= 0]
    loss = returns[returns < 0]

    win_ratio = abs(wins.mean() / wins.count())
    loss_ratio = abs(loss.mean() / loss.count())
    try:
        return (win_ratio / loss_ratio).to_frame(name='Profit Ratio')
    except Exception:
        return 0.

def profit_factor(returns):
    return (abs(returns[returns >= 0].sum() / returns[returns < 0].sum())).to_frame(name='Profit Factor')

def payoff_ratio(returns):
    return (avg_win(returns).squeeze() / abs(avg_loss(returns)).squeeze()).to_frame(name='Payoff Ratio')

def cpc_index(returns):
    return (profit_factor(returns).squeeze() * win_rate(returns).squeeze() * \
         payoff_ratio(returns).squeeze()).to_frame(name='CPC Index')

def tail_ratio(returns, cutoff=0.95):
    return (abs(returns.quantile(cutoff) / returns.quantile(1-cutoff))).to_frame(name='Tail Ratio')

def common_sense_ratio(returns):
    a = profit_factor(returns).squeeze()
    b = tail_ratio(returns).squeeze()
    res = a*b
    return res.to_frame(name='Common Sense Ratio')

def outlier_win_ratio(returns, quantile=.99, prepare_returns=True):
    return (returns.quantile(quantile).mean() / returns[returns >= 0].mean()).to_frame(name='Outlier-to-Win Ratio')

def outlier_loss_ratio(returns, quantile=.01):
    return (returns.quantile(quantile).mean() / returns[returns < 0].mean()).to_frame(name='Outlier-to-Loss Ratio')

def max_drawdown(returns):
    wealth = (returns+1).cumprod()
    peaks = wealth.cummax()
    drawdown = (wealth - peaks)/peaks
    return (drawdown.min()).to_frame(name='Max. Drawdown')

def calmar(returns):
    cagr_ratio = cagr(returns).squeeze()
    max_dd = max_drawdown(returns).squeeze()
    return (cagr_ratio / abs(max_dd)).to_frame(name='Calmar Ratio')

def drawdown_series(returns):
    wealth = (returns+1).cumprod()
    peaks = wealth.cummax()
    drawdown = (wealth - peaks)/peaks
    return drawdown

def ulcer_index(returns):
    dd = drawdown_series(returns)
    return (np.sqrt(np.divide((dd**2).sum(), returns.shape[0] - 1))).to_frame(name='Ulcer Index')

def ulcer_performance_index(returns, rf=0.):
    return ((comp(returns)-rf) / ulcer_index(returns).squeeze()).to_frame(name='Ulcer Performance Index')

def serenity_index(returns, rf=0):
    dd = drawdown_series(returns)
    pitfall = cvar_hist(dd).squeeze() / returns.std()
    res = (comp(returns)-rf) / (ulcer_index(returns).squeeze() * pitfall)
    return res.to_frame(name='Serenity Index')

def recovery_factor(returns):
    total_returns = comp(returns).squeeze()
    max_dd = max_drawdown(returns).squeeze()
    return (total_returns / abs(max_dd)).to_frame(name='Recovery Factor')

def risk_of_ruin(returns, trials=None):
    winrate = win_rate(returns).squeeze()
    if trials is None:
        trials = len(returns)
    return ((1 - (winrate- (1-winrate))) / (1 + (winrate- (1-winrate)))**trials).to_frame(name='Risk of Ruin')

def drawdown_details(drawdown):
    def _drawdown_details(drawdown):
        no_dd = drawdown == 0
        starts = ~no_dd & no_dd.shift(1)
        starts = list(starts[starts].index)
        ends = no_dd & (~no_dd).shift(1)
        ends = list(ends[ends].index)
        if not starts:
            return pd.DataFrame(
                index=[], columns=('Start', 'Valley', 'End', 'Days',
                                   'Max. Drawdown', '99% Max. Drawdown'))
        if ends and starts[0] > ends[0]:
            starts.insert(0, drawdown.index[0])

        if not ends or starts[-1] > ends[-1]:
            ends.append(drawdown.index[-1])
        data = []
        for i, _ in enumerate(starts):
            dd = drawdown[starts[i]:ends[i]]
            clean_dd = dd
            data.append((starts[i], dd.idxmin(), ends[i],
                         (ends[i] - starts[i]).days,
                         dd.min() * 100, clean_dd.min() * 100))

        df = pd.DataFrame(data=data,
                           columns=('Start', 'Valley', 'End', 'Days',
                                    'Max. Drawdown',
                                    '99% Max. Drawdown'))
        df['Days'] = df['Days'].astype(int)
        df['Max. Drawdown'] = df['Max. Drawdown'].astype(float).round(2)
        df['99% Max. Drawdown'] = df['99% Max. Drawdown'].astype(float).round(2)

        df['Start'] = df['Start'].dt.strftime('%Y-%m-%d')
        df['End'] = df['End'].dt.strftime('%Y-%m-%d')
        df['Valley'] = df['Valley'].dt.strftime('%Y-%m-%d')
        return df
    if isinstance(drawdown, pd.DataFrame):
        _dfs = {}
        for col in drawdown.columns:
            _dfs[col] = _drawdown_details(drawdown[col])
        return pd.concat(_dfs, axis=1)
    return _drawdown_details(drawdown)

def drawdown_info(rets):
    drd = drawdown_series(rets) 
    df = drawdown_details(drd)
    srs = df.loc[:,(slice(None),'Days')].idxmax().droplevel(level=1)
    df2 = df.iloc[srs]
    groups = df.groupby(level=0, axis=1)
    group_dict = groups.groups
    result = {}
    for level in group_dict:
        days_column = df.xs(level, level=0, axis=1)['Days']
        max_row = days_column.idxmax()
        row = df.loc[max_row]
        result[level] = row[level]
    result_df = pd.DataFrame.from_dict(result)
    return result_df

def get_portfolio_stats(ports_wts,VCV,ER,rfr):
    ret_cont = (ports_wts.T * 100).round(2).astype(str) + "%"
    risk_cont = (ports_wts.T.apply((lambda x: risk_contribution(x,VCV)))*100).round(2).astype(str) + "%"
    sharpe_cont = (ports_wts.T.apply(lambda x: sharpe_contribution(x,ER,VCV,rfr))*100).round(2).astype(str) + "%"


    ret_cont.index = pd.MultiIndex.from_product([['Portfolio Composition'], ret_cont.index])
    risk_cont.index = pd.MultiIndex.from_product([['Risk Contribution'], risk_cont.index])
    sharpe_cont.index = pd.MultiIndex.from_product([['Sharpe-Ratio Contribution'], sharpe_cont.index])
    portfolio_data = pd.concat([ret_cont,risk_cont,sharpe_cont],axis=0)

    return portfolio_data

def calc_port_stats(returns, rfr, benchmark, target_return,periods):
    rf = rfr/periods
    rets = returns

    var = var_hist(rets,level=5).T
    cvar = cvar_hist(rets, level=5).T
    gvar =  var_gauss(rets, level=5).T
    semdev = semidev(rets)
    dd = drawdown(rets)
    avgr = avg_return(rets)
    avgw = avg_win(rets)
    avgl = avg_loss(rets)
    bst = best(rets)
    wrst = worst(rets)

    sort = sortino(rets, rf=rf, periods=periods, annualize=True)
    om = omega(rets,target_return,periods)
    gtpr = gain_to_pain_ratio(rets)
    rsquares = pd.Series(rets.apply(lambda x: get_rsquare(x,benchmark))).to_frame(name='R-Squared')
    alphasbetas = rets.apply(lambda x: get_alpha_and_beta(x,benchmark)).T
    alphasbetas.columns = ['Alpha','Beta']
    inforatio = info_ratio(rets,benchmark)
    kelly = kelly_criterion(rets)

    pr = profit_ratio(rets)
    pf = profit_factor(rets)
    payoffr = payoff_ratio(rets)
    cpc = cpc_index(rets)
    tailrat = tail_ratio(rets)
    csr = common_sense_ratio(rets)
    owr = outlier_win_ratio(rets)
    owl = outlier_loss_ratio(rets)

    maxdd = max_drawdown(rets)
    calm = calmar(rets)
    ui = ulcer_index(rets)
    upi = ulcer_performance_index(rets,rf=rfr)
    si = serenity_index(returns,rf=rfr)
    ror = risk_of_ruin(rets,trials=15)
    recfac = recovery_factor(rets)
    dd_info = drawdown_info(rets).T

    res_list = [var,cvar,gvar,semdev,avgr,avgw,avgl,bst,wrst,sort,om,gtpr,rsquares,alphasbetas,inforatio,kelly,pr,pf,payoffr,cpc,tailrat,csr,owr,owl,maxdd,calm,ui,upi,si,ror,recfac,dd_info]

    res = pd.concat(res_list, axis=1).T
    #res.index = ['VaR','CVaR','Semideviation','Average Return','Average Win','Average Loss','Best Return','Worst Return',
    #'Sortino Ratio','Omega Ratio','Gain-to-Pain-Ratio','R-Squared','Alpha','Beta','Information Ratio','Kelly Criterion',
    #'Profit Ratio','Profit Factor','Payoff Ratio','CPC-Index','Tail Ratio','Common Sense Ratio','Outlier-to-Win-Ration',
    #'Outlier-to-Loss-Ratio','Maximum Drawdown','Calmar Ratio','Ulcer Index','Ulcer Performance Index','Serenity Index','Risk of Ruin','Recovery Factor']
    return res

def plot_optimal_weights(wts_srs, portfolio):
    wts_srs[portfolio].plot(title=f'Optimal {portfolio} Portfolio Weights Over Time')

def prep_trade_days(df, freq=None):
    dates = df.index
    df2 = pd.DataFrame(dates, index=dates)

    if freq is not None:
        df2[f'is_new_{freq}'] = df2['Date'] == df2['Date'].groupby(pd.Grouper(freq=freq)).transform('first')
        df2[f'is_end_{freq}'] = df2['Date'] == df2['Date'].groupby(pd.Grouper(freq=freq)).transform('last')
        df2 = df2[[f'is_new_{freq}', f'is_end_{freq}']]

    if freq is None:
        freqs = ['D','W', 'M', 'Q', 'Y']
        for freq in freqs:
            df2[f'is_new_{freq}'] = df2['Date'] == df2['Date'].groupby(pd.Grouper(freq=freq)).transform('first')
            df2[f'is_end_{freq}'] = df2['Date'] == df2['Date'].groupby(pd.Grouper(freq=freq)).transform('last')
        df2['is_new_N'] = df2.index == df2.index[0]
        df2['is_end_N'] = df2.index == df2.index[-1]

        df2 = df2[['is_new_N','is_end_N','is_new_D', 'is_end_D', 'is_new_W', 'is_end_W', 'is_new_M', 'is_end_M', 'is_new_Q','is_end_Q', 'is_new_Y', 'is_end_Y']]
    return df2

def get_trade_days(weight_series):
    trade_days = {key: prep_trade_days(value) for key, value in weight_series.items()}
    return trade_days

def pure_buy_and_hold(init_w,rets):
    gross_rets = rets+1
    weights=pd.DataFrame(columns=init_w.index,index = rets.index)
    weights.iloc[0] = init_w

    for t in (range(len(rets.index)-1)):
        pr = weights.iloc[t].T @ gross_rets.iloc[t]
        weights.iloc[t+1] = (weights.iloc[t].T * gross_rets.iloc[t]) / pr
    return weights

def pure_buy_and_hold2(init_w,rets):
    first_date = init_w.index[0]
    rets = rets.loc[first_date:]
    gross_rets = rets+1
    init_w = init_w.iloc[0]
    weights=pd.DataFrame(columns=init_w.index,index = rets.index)
    weights.iloc[0] = init_w
    for t in (range(len(rets.index)-1)):
        pr = weights.iloc[t].T @ gross_rets.iloc[t]
        weights.iloc[t+1] = (weights.iloc[t].T * gross_rets.iloc[t]) / pr
    return weights

def rebalance_buy_and_hold(opt_wts,rets,freq = 'W'):
    gross_rets = rets.loc[opt_wts.index]+1 
    rebalance_dates = prep_trade_days(opt_wts,freq=freq)[f'is_new_{freq}'] 
    rebalance_w = pd.DataFrame(zip(*[rebalance_dates]*len(opt_wts.columns)),index = opt_wts.index,columns = opt_wts.columns)
    actual_weights = rebalance_w*opt_wts
    indices = rebalance_dates.reset_index(drop=True).index[rebalance_dates.isin([False])]
    for t in indices:
        pr = actual_weights.iloc[t-1].T @ gross_rets.iloc[t-1]
        actual_weights.iloc[t] = (actual_weights.iloc[t-1].T * gross_rets.iloc[t-1]) / pr
    return actual_weights

def plot_weights(wts_srs,asset_returns,port_method,freq):
    opt_wts = wts_srs[port_method]
    wts = rebalance_buy_and_hold(opt_wts,asset_returns,freq)
    wts.plot()

def get_actual_weights(wts_srs,returns,NoRebalanceWts=False):
    freq_values = ['D','W', 'M', 'Q', 'Y']
    actual_weights = {freq: {key: rebalance_buy_and_hold(value, returns, freq=freq) for key, value in wts_srs.items()} for freq in freq_values}
    if NoRebalanceWts is True:
        start_weights  = {key: df.head(1) for key, df in wts_srs.items()}
        buy_and_hold_weights = {key: pure_buy_and_hold2(value,returns) for key, value in start_weights.items()}
        actual_weights['N'] = buy_and_hold_weights
    return actual_weights

def plot_actual_weights(actual_weights,portfolio,freq):
    actual_weights[freq][portfolio].plot()

def get_actual_returns(asset_wts,asset_returns):
    asset_returns = asset_returns.loc[asset_wts.index]
    actual_returns = asset_wts * asset_returns
    return actual_returns.T.sum()

def get_all_results(asset_returns,actual_weights):
    actual_returns = {key1: {key2: get_actual_returns(value2, asset_returns) for key2, value2 in inner_dict.items()} for key1, inner_dict in actual_weights.items()}
    results_df = {key: pd.DataFrame.from_dict(value, orient='index') for key, value in actual_returns.items()}
    result_df = pd.concat({key: pd.DataFrame.from_dict(value, orient='index').T for key, value in actual_returns.items()},axis=1)
    return result_df

def plot_results_by_method(result_df,portfolio,benchmark=None,plot_benchmark=False):
    df_by_method = result_df.loc[:,(slice(None),portfolio)].droplevel(axis=1,level=1)
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")

        df_by_method = pd.concat([df_by_method, benchmark], axis=1).dropna()

    df_by_method=(df_by_method+1).cumprod()
    if "Benchmark" in df_by_method.columns:
        other_columns = df_by_method.drop("Benchmark", axis=1)
        other_columns.plot()
        df_by_method["Benchmark"].plot(color="black", legend=True)
    
    else:
        df_by_method.plot()

    plt.title(f'Cumulative Portfolio Returns over Time using the {portfolio}-Method across Rebalancing Frequencies')
    plt.show()

def plot_result_by_freq(result_df,freq,benchmark=None,plot_benchmark=False):
    df_by_rebal_freq = result_df[freq]
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")

        df_by_rebal_freq = pd.concat([df_by_rebal_freq, benchmark], axis=1).dropna()
    df_by_rebal_freq=(df_by_rebal_freq+1).cumprod()
    if "Benchmark" in df_by_rebal_freq.columns:
        other_columns = df_by_rebal_freq.drop("Benchmark", axis=1)
        other_columns.plot()
        df_by_rebal_freq["Benchmark"].plot(color="black", legend=True)
    
    else:
        df_by_rebal_freq.plot()

    plt.title(f'Cumulative Portfolio Returns over Time using {freq}-Rebalancing across various Optimization Methods')
    plt.show()

#def regress(dep_var, regressors, alpha=True):
    #import statsmodels.api as sm
    #if alpha:
        #regressors = regressors.copy()
        #regressors["Alpha"] = 1

    #lm = sm.OLS(dep_var, regressors).fit()
    #return lm

def regress(dep_var, regressors, alpha=True):
    import statsmodels.api as sm
    if alpha:
        regressors = regressors.copy()
        regressors["Alpha"] = 1

    data = pd.concat([dep_var,regressors],axis=1).dropna()
    regressors = data.drop(dep_var.name,axis=1)
    y = data[dep_var.name]

    lm = sm.OLS(y, regressors).fit()
    return lm

def te(r_a, r_b):
    import numpy as np
    return np.sqrt(((r_a - r_b)**2).sum())

def portf_te(weights, ref_r, bb_r):
    return te(ref_r, (weights*bb_r).sum(axis=1))

def style_analysis(dep_var, regressors):
    from scipy.optimize import minimize
    n = regressors.shape[1]
    guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n 

    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    solution = minimize(portf_te, guess,
                       args=(dep_var, regressors,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    weights = pd.Series(solution.x, index=regressors.columns)
    return weights

def load_monthly_factor_returns():
    data = pd.read_csv('monthly_factor_returns.csv',index_col='Date')
    data = data[['Mom   -FF', 'ST_Rev-FF', 'LT_Rev-FF', 'Mkt-RF-FF', 'SMB-FF','HML-FF', 'RMW-FF', 'CMA-FF', 'QMJ-AQR','Traded Liq (LIQ_V)-STAMB']]
    data.columns = ['MOM','ST_Rev','LT_Rev','MKT','SMB','HML','RMW','CMA','QMJ','LIQ']
    data.index = pd.to_datetime(data.index)
    return data

def load_daily_factor_returns():
    data = pd.read_csv('daily_factor_returns.csv',index_col='Date')
    data = data[['Mom   -FF','ST_Rev-FF', 'LT_Rev-FF', 'Mkt-RF-FF', 'SMB-FF','HML-FF', 'RMW-FF', 'CMA-FF', 'QMJ','UMD','R_ME-Q', 'R_IA-Q', 'R_ROE-Q', 'R_EG-Q']]
    data.columns = ['MOM','ST_Rev','LT_Rev','MKT_EXC','SMB','HML','RMW','CMA','QMJ','UMD','R_ME','INV','ROE','EXPGRO']
    data.index = pd.to_datetime(data.index)
    return data

#def get_factor_loadigns(strategy,regressors):
    #start = max(strategy.index[0],regressors.index[0])
    #end = min(strategy.index[-1],regressors.index[-1])
    #strategy = strategy.loc[start:end,]
    #regressors = regressors.loc[start:end,]
    #model = regress(strategy, regressors, alpha=True)
    #res = pd.DataFrame(np.round(model.params,4))
    #return res

def get_factor_loadigns(strategy,regressors):
    model = regress(strategy, regressors, alpha=True)
    res = pd.DataFrame(np.round(model.params,4))
    return res

#def get_multiple_factor_loadings(strategies,regressors):
    #start = max(strategies.index[0],regressors.index[0])
    #end = min(strategies.index[-1],regressors.index[-1])
    #strategies = strategies.loc[start:end,]
    #regressors = regressors.loc[start:end,]
    #res = strategies.apply(lambda x:regress(x, regressors, alpha=True).params)
    #return res

def plot_multiple_factor_loadings(portf_rets, factor_returns, benchmark=None):
    if benchmark is not None:
        portf_rets = pd.concat([portf_rets, benchmark], axis=1)
    res = get_multiple_factor_loadings(portf_rets, factor_returns)
    res.T.plot(kind='bar').set_title('Factor Loadings across Portfolios')
    plt.show()


def get_multiple_factor_loadings(strategies,regressors):
    res = strategies.apply(lambda x:regress(x, regressors, alpha=True).params)
    return res

def plot_factor_loadings_by_strat(result_df,freq,factor_returns):
    get_multiple_factor_loadings(result_df[freq],factor_returns).T.plot(kind='bar')

def plot_underwater_mult(results_df, freq=None, portfolio=None,benchmark=None, plot_benchmark=False):
    if freq is None and portfolio is None:
        ret = results_df
    elif freq is None and portfolio is not None:
        ret = results_df.loc[:, (slice(None), portfolio)].droplevel(1,axis=1)
    elif freq is not None and portfolio is None:
        ret = results_df.loc[:, (freq, slice(None))].droplevel(0,axis=1)
    else:
        ret = results_df.loc[:, (freq, portfolio)]
    
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")

        ret = pd.concat([ret, benchmark], axis=1) 

    cum_ret = (ret+1).cumprod()
    cum_max = cum_ret.cummax()
    drawdown = (cum_ret-cum_max)/cum_max

    if "Benchmark" in drawdown.columns:
        other_columns = drawdown.drop("Benchmark", axis=1)
        other_columns.plot()
        drawdown["Benchmark"].plot(color="black", legend=True)
    else:
        drawdown.plot()

    plt.title('Underwater Plot: Drawdowns over Time')
    plt.show()

def plot_underwater(ret,benchmark=None, plot_benchmark=False):
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")

        ret = pd.concat([ret, benchmark], axis=1) 

    cum_ret = (ret+1).cumprod()
    cum_max = cum_ret.cummax()
    drawdown = (cum_ret - cum_max)/cum_max
    
    if "Benchmark" in drawdown.columns:
        other_columns = drawdown.drop("Benchmark", axis=1)
        other_columns.plot()
        drawdown["Benchmark"].plot(color="grey", legend=True)
    else:
        drawdown.plot()

    plt.title('Underwater Plot: Drawdowns over Time')
    plt.show()

def expanding_portfolio_weights(returns,init_len=252, target_return = None, rfr = None,w0=None):
    markowitz_wts = []
    gmv_wts = []
    max_sharpe_wts = []
    equal_wts = []
    risk_parity_wts = []
    inv_var_wts = []
    inv_vol_wts = []
    max_div_wts = []
    max_decorr_wts = []
    min_corr_wts = []
    equal_sharpe_wts = []
    
    dates = []
    if w0 is None:
        w0 = [1/len(returns.columns)]*len(returns.columns)
    w0 = pd.Series(w0,index = returns.columns)
    pbar = tqdm(range(len(returns.index) - init_len))
    for i in pbar:
        start = returns.index[0]
        end = returns.index[init_len+i]

        df = returns.loc[start:end,]
        COV = df.cov()
        ER = df.mean()
        CORR = df.corr()

        if target_return is None:
            target_return = ER.mean()
        else:
            target_return = target_return
        if rfr is None:
            rfr=0.0
        else:
            rfr = rfr
            
        markowitz_weights = minimize_vol(target_return=target_return,er=ER,cov=COV)
        markowitz_wts.append(markowitz_weights)
        
        gmv_weights = wglob_min_var_wts = gmv(COV)
        gmv_wts.append(gmv_weights)

        max_sharpe_weights = msr(rfr, ER, COV)
        max_sharpe_wts.append(max_sharpe_weights)

        equal_weights = eq_weight_portf(COV)
        equal_wts.append(equal_weights)

        risk_parity_weights = equal_risk_contributions(COV)
        risk_parity_wts.append(risk_parity_weights)

        inv_var_weights = inv_vol_portf(COV)
        inv_var_wts.append(inv_var_weights)

        inv_vol_weights = inv_vol_portf(COV)
        inv_vol_wts.append(inv_vol_weights)

        max_div_weights = max_div_port(w0,COV)
        max_div_wts.append(max_div_weights)

        max_decorr_weights = max_decorr_portf(w0,CORR)
        max_decorr_wts.append(max_decorr_weights)

        min_corr_weights = min_corr_portf(CORR,COV)
        min_corr_wts.append(min_corr_weights)

        equal_sharpe_weights = target_sharpe_contribution((1/len(COV.columns)),ER,COV,rfr)
        equal_sharpe_wts.append(equal_sharpe_weights)
        
        dates.append(end)
        pbar.set_description("Optimizing Portfolio for %s" % returns.index[i+init_len].strftime('%d-%m-%Y'))

    res = [markowitz_wts,gmv_wts,max_sharpe_wts,equal_wts,risk_parity_wts,inv_var_wts,inv_vol_wts,max_div_wts,max_decorr_wts,min_corr_wts,equal_sharpe_wts]
    df_list = [pd.DataFrame(lst,index=dates).rename_axis('Date') for lst in res]
    keys = ["Markowitz","Global Minimum Variance","Maximum Sharpe Ratio","Equal Weights","Risk Parity","Inverse Variance","Inverse Volatility","Maximum Diversification","Maximum De-Correlation","Minimum Correlation","Equal Sharpe Ratio"]       
    res = dict(zip(keys, df_list))
    return res

def plot_portf_rets(returns, ports_wts, benchmark=None, plot_benchmark=False):
    portf_rets = calc_port_rets(returns, ports_wts)

    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")
        portf_rets = pd.concat([portf_rets, benchmark], axis=1) 

    portf_rets = portf_rets.dropna(axis=0)
    portf_cum_returns = (portf_rets + 1).cumprod()

    line_styles = {}
    line_colors = {}
    
    if "Benchmark" in portf_cum_returns.columns:
        line_styles["Benchmark"] = 'dashed'
        line_colors["Benchmark"] = 'grey'
    
    if "Current Portfolio" in portf_cum_returns.columns:
        line_styles["Current Portfolio"] = 'dashed'
        line_colors["Current Portfolio"] = 'black'

    plt.figure(figsize=(12, 8.5))
    plt.title('Cumulative Returns per Portfolio over Time')
    for column in portf_cum_returns.columns:
        plt.plot(portf_cum_returns.index, portf_cum_returns[column], label=column, 
                 linestyle=line_styles.get(column, '-'), color=line_colors.get(column, None))

    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(False)
    plt.show()

def exp_var_hist(srs, level=5, init_len=252):
    if isinstance(srs, pd.Series):
        calc = []
        for i in range(len(srs) - init_len):  
            res = var_hist(srs[0:(init_len+i)], level=level).values[0]  
            calc.append(res)
        var_series = pd.Series([arr[0] for arr in calc], index=srs.index[init_len:], name="Historical VaR-Series")
        return var_series
    elif isinstance(srs, pd.DataFrame):
        var_series = srs.apply(lambda x: exp_var_hist(x), axis=0)
        return var_series

#def exp_var_hist(srs, level=5, init_len=252):
    #calc = []
    #for i in range(len(srs) - init_len):  
        #res = var_hist(srs[0:(init_len+i)], level=level).values[0]  # Extract the first element as a single value
        #calc.append(res)
    #var_series = pd.Series([arr[0] for arr in calc], index=srs.index[init_len:], name="Historical VaR-Series")
    #return var_series

#vars_srs = asset_returns.apply(lambda x: exp_var_hist(x), axis=0)

def exp_cvar_hist(srs, level=5, init_len=252):
    if isinstance(srs, pd.Series):
        calc = []
        for i in range(len(srs) - init_len):  
            res = cvar_hist(srs[0:(init_len+i)], level=level) 
            calc.append(res)
        cvar_series = cvar_series = pd.Series([arr[0] for arr in calc], index=srs.index[init_len:], name="Historical CVaR-Series")
        return cvar_series
    elif isinstance(srs, pd.DataFrame):
        cvar_series = srs.apply(lambda x: exp_cvar_hist(x), axis=0)
        return cvar_series
    
def exp_var_gauss(r, level=5, modified=True, init_len=252):
    if isinstance(r,pd.Series):
        calc = []
        for i in range(len(r) - init_len):
            res = var_gauss(r[0:(init_len+i)], level=level, modified=modified)
            calc.append(res)
        var_gauss_series = pd.Series([arr[0] for arr in calc], index=r.index[init_len:], name="Gaussian VaR-Series")
    if isinstance(r,pd.DataFrame):
        var_gauss_series = r.apply(lambda x: exp_var_gauss(x,level=level,modified=modified))
    return var_gauss_series

def generate_random_weights(asset_returns):
    k = len(asset_returns.columns)
    weights = np.random.random(k)
    weights /= weights.sum()
    return weights

def drawdown_info2(rets):
    drd = drawdown_series(rets)
    df = drawdown_details(drd)
    
    if df.empty:
        return pd.DataFrame()  # Return an empty DataFrame if df is empty
    
    if 'Days' not in df.columns:
        return pd.DataFrame()  # Return an empty DataFrame if 'Days' column is missing
    
    srs = df.loc[:, (slice(None), 'Days')].idxmax().droplevel(level=1)
    df2 = df.iloc[srs]
    
    groups = df.groupby(level=0, axis=1)
    group_dict = groups.groups
    result = {}
    
    for level in group_dict:
        if level not in df.columns:
            continue  # Skip if the level column is missing
        
        days_column = df[level]['Days']
        max_row = days_column.idxmax()
        row = df.loc[max_row]
        result[level] = row[level]
    
    result_df = pd.DataFrame.from_dict(result)
    return result_df

def format_numeric(val):
    if isinstance(val, float):
        return round(val, 4)
    return val

def discrete_and_dollar_allocation_old(wts,ticker_list,capital):
    latest_prices = yf.download(ticker_list).loc[:,'Adj Close'].tail(1)
    dollar_allocation = capital*wts
    discrete_allocation = dollar_allocation/latest_prices
    discrete_allocation = discrete_allocation.apply(round).astype(int)
    res = pd.concat([discrete_allocation.T,dollar_allocation.T],axis=1)
    res = res.T
    res.index = ['Discrete Asset Allocation','Dollar Allocation']
    return res

def calc_leftover(wts,ticker_list,capital):
    res = discrete_and_dollar_allocation(wts,ticker_list,capital)
    calc = res.loc['Dollar Allocation'].sum()
    res = capital - calc
    return res

def discrete_and_dollar_allocation(capital, ticker_list, wts):
    latest_prices = get_latest_prices_tiingo(ticker_list)

    target_dollar_allocation = (capital * wts).round(2)
    discrete_asset_allocation = (target_dollar_allocation / latest_prices).round(0)
    actual_dollar_allocation = (discrete_asset_allocation * latest_prices).round(2)
    distance = (target_dollar_allocation - actual_dollar_allocation).round(2)

    actual_perc_allocation = (actual_dollar_allocation / float(capital)).round(4)
    distance_perc_allocation = (wts - actual_perc_allocation).round(4)

    discrete_asset_allocation = discrete_asset_allocation.T.round(0)
    actual_dollar_allocation = actual_dollar_allocation.T.round(2)
    target_dollar_allocation = target_dollar_allocation.T.round(2)
    distance = distance.T.round(2)

    actual_perc_allocation = (actual_perc_allocation.T * 100).round(2)
    distance_perc_allocation = (distance_perc_allocation.T * 100).round(2)
    optimal_portf_allocation = (wts.T * 100).round(2)

    discrete_asset_allocation.index = pd.MultiIndex.from_product([['Discrete Asset Allocation'], discrete_asset_allocation.index])
    actual_dollar_allocation.index = pd.MultiIndex.from_product([['Actual Dollar Allocation'], actual_dollar_allocation.index])
    target_dollar_allocation.index = pd.MultiIndex.from_product([['Target Dollar Allocation'], target_dollar_allocation.index])
    distance.index = pd.MultiIndex.from_product([['Distance between Target- and Dollar Allocation'], distance.index])

    total_money_invested = pd.Series(actual_dollar_allocation.sum()).to_frame().rename(columns={0: 'Total Funds Invested'}).round(2)
    leftover_funds = (capital - total_money_invested).rename(columns={'Total Funds Invested': 'Leftover Funds'}).round(2)

    funds = pd.concat([total_money_invested, leftover_funds], axis=1).T
    funds.index = pd.MultiIndex.from_product([['Funds invested and left over'], funds.index])

    optimal_portf_allocation.index = pd.MultiIndex.from_product([['Optimal Portfolio Allocation (%)'], optimal_portf_allocation.index])
    actual_perc_allocation.index = pd.MultiIndex.from_product([['Actual Portfolio Allocation (%)'], actual_perc_allocation.index])
    distance_perc_allocation.index = pd.MultiIndex.from_product([['Difference between Optimal- and Actual Portfolio Allocation'], distance_perc_allocation.index])

    df = pd.concat([
        discrete_asset_allocation,
        actual_dollar_allocation,
        target_dollar_allocation,
        distance,
        funds,
        optimal_portf_allocation,
        actual_perc_allocation,
        distance_perc_allocation
    ], axis=0)

    return df




def plot_port_cum_rets(rets):
    cum_rets = (rets + 1).cumprod()
    line_styles = {}
    line_colors = {}
    if "Benchmark" in cum_rets.columns:
        line_styles["Benchmark"] = 'solid'
        line_colors["Benchmark"] = 'black'
    if "Current Portfolio" in cum_rets.columns:
        line_styles["Current Portfolio"] = 'dashed'
        line_colors["Current Portfolio"] = 'black'
    if "Custom Portfolio" in cum_rets.columns:
        line_styles["Custom Portfolio"] = 'dotted'
        line_colors["Custom Portfolio"] = 'black'

    plt.figure(figsize=(12, 8.5))
    plt.title("Cumulative Portfolio Returns over time")
    for column in cum_rets.columns:
        plt.plot(cum_rets.index, cum_rets[column], label=column, linestyle=line_styles.get(column, '-'), color=line_colors.get(column, None))

    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid()
    plt.show()  

def plot_asset_cum_rets(rets):
    cum_rets = (rets + 1).cumprod()
    line_styles = {}
    line_colors = {}
    if "Benchmark" in cum_rets.columns:
        line_styles["Benchmark"] = 'solid'
        line_colors["Benchmark"] = 'black'
    if "Current Portfolio" in cum_rets.columns:
        line_styles["Current Portfolio"] = 'dashed'
        line_colors["Current Portfolio"] = 'black'
    if "Custom Portfolio" in cum_rets.columns:
        line_styles["Custom Portfolio"] = 'dotted'
        line_colors["Custom Portfolio"] = 'black'

    plt.figure(figsize=(12, 8.5))
    plt.title("Cumulative Asset Returns over time")
    for column in cum_rets.columns:
        plt.plot(cum_rets.index, cum_rets[column], label=column, linestyle=line_styles.get(column, '-'), color=line_colors.get(column, None))

    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid()
    plt.show() 

def plot_drawdowns(rets,benchmark=None,plot_benchmark=False):
    
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")
        rets = pd.concat([rets, benchmark], axis=1) 

    rets = rets.dropna(axis=0)
    cum_rets = (rets + 1).cumprod()

    cum_max = cum_rets.cummax()
    drawdown = (cum_rets - cum_max)/cum_max
    line_styles = {}
    line_colors = {}
    if "Benchmark" in drawdown.columns:
        line_styles["Benchmark"] = 'dashed'
        line_colors["Benchmark"] = 'grey'
    if "Current Portfolio" in drawdown.columns:
        line_styles["Current Portfolio"] = 'dashed'
        line_colors["Current Portfolio"] = 'black'
    if "Custom Portfolio" in drawdown.columns:
        line_styles["Custom Portfolio"] = 'dotted'
        line_colors["Custom Portfolio"] = 'black'

    plt.figure(figsize=(12, 8.5))
    plt.title("Portfolio Drawdowns over Time")
    for column in drawdown.columns:
        plt.plot(drawdown.index, drawdown[column], label=column, linestyle=line_styles.get(column, '-'), color=line_colors.get(column, None))

    plt.xlabel("Date")
    plt.ylabel("Drawdowns")
    plt.legend()
    plt.grid(False)
    plt.show()

def plot_assets(asset_returns, benchmark=None, plot_benchmark=False):
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")
        
        asset_returns = pd.concat([asset_returns, benchmark], axis=1)

    asset_cum_returns = (asset_returns + 1).cumprod()

    ax = asset_cum_returns.plot()

    if 'Benchmark' in asset_cum_returns.columns:
        lines = ax.get_lines()
        for line in lines:
            if line.get_label() == 'Benchmark':
                line.set_color('black')

    plt.title('Cumulative Ticker Returns over Time')
    plt.show()

def gbm(n_years = 1, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val

def generate_gbm_from_asset_returns(asset_returns, n_years=1, n_scenarios=1000, steps_per_year=252, prices=True):
    cum_rets = (asset_returns + 1).cumprod()
    last_cum_ret = cum_rets.iloc[-1]
    mu = asset_returns.mean() * 252
    sigma = asset_returns.std() * np.sqrt(252)
    last_date = asset_returns.index[-1]
    future_dates = pd.date_range(start=last_date, periods=int(n_years * steps_per_year) + 1, freq='B')
    def gbm(n_years, n_scenarios, mu, sigma, steps_per_year, s_0, prices, future_dates):
        dt = 1 / steps_per_year
        n_steps = len(future_dates)
        rets_plus_1 = np.random.normal(loc=(1 + mu)**dt, scale=(sigma * np.sqrt(dt)), size=(n_steps, n_scenarios))
        rets_plus_1[0] = 1
        ret_val = s_0 * pd.DataFrame(rets_plus_1, index=future_dates).cumprod() if prices else rets_plus_1 - 1
        return ret_val

    gbm_results = {}
    
    for asset in asset_returns.columns:
        s_0 = last_cum_ret[asset]
        mu_asset = mu[asset]
        sigma_asset = sigma[asset]
        
        gbm_results[asset] = gbm(n_years, n_scenarios, mu_asset, sigma_asset, steps_per_year, s_0, prices, future_dates)
    
    return gbm_results


def plot_gbm_and_cum_rets_old_and_broken(asset_returns, n_years=1, n_scenarios=100, steps_per_year=252, plot_last=252*2, plot_sims=None, ax=None):
    
    if plot_last is not None:
        asset_returns = asset_returns.tail(plot_last)
        
    cum_rets = (asset_returns + 1).cumprod()
    gbm_simulations = generate_gbm_from_asset_returns(asset_returns, n_years=n_years, n_scenarios=n_scenarios, steps_per_year=steps_per_year)

    line_styles = {}
    line_colors = {}
    
    if "Benchmark" in cum_rets.columns:
        line_styles["Benchmark"] = 'dashed'
        line_colors["Benchmark"] = 'grey'
    if "Current Portfolio" in cum_rets.columns:
        line_styles["Current Portfolio"] = 'dashed'
        line_colors["Current Portfolio"] = 'black'
    
    cum_rets.plot(ax=ax, figsize=(12, 5))
    lines = ax.get_lines()
    colors = [line.get_color() for line in lines]
    handles, labels = ax.get_legend_handles_labels()
    
    for line in lines:
        label = line.get_label()
        if label in line_styles:
            line.set_linestyle(line_styles[label])
            line.set_color(line_colors[label])
    
    if plot_sims:
        if isinstance(plot_sims, list):
            for sim in plot_sims:
                if sim in gbm_simulations:
                    sim_color = line_colors.get(sim, colors[asset_returns.columns.get_loc(sim) % len(colors)])
                    gbm_simulations[sim].plot(ax=ax, color=sim_color, alpha=0.25, linewidth=2, legend=False)
    
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title("Cumulative Portfolio Returns with Simulations" if plot_sims else "Cumulative Portfolio Returns")
    
    plt.show()


def plot_gbm_and_cum_rets(asset_returns, n_years=1, n_scenarios=100, steps_per_year=252, plot_last=252*2, plot_sims=None, ax=None):
    if plot_last is not None:
        asset_returns = asset_returns.tail(plot_last)
        
    cum_rets = (asset_returns + 1).cumprod()
    gbm_simulations = generate_gbm_from_asset_returns(asset_returns, n_years=n_years, n_scenarios=n_scenarios, steps_per_year=steps_per_year)

    line_styles = {}
    line_colors = {}
    
    if "Benchmark" in cum_rets.columns:
        line_styles["Benchmark"] = 'dashed'
        line_colors["Benchmark"] = 'grey'
    if "Current Portfolio" in cum_rets.columns:
        line_styles["Current Portfolio"] = 'dashed'
        line_colors["Current Portfolio"] = 'black'
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    cum_rets.plot(ax=ax, figsize=(12, 5))
    lines = ax.get_lines()
    colors = [line.get_color() for line in lines]
    handles, labels = ax.get_legend_handles_labels()
    
    for line in lines:
        label = line.get_label()
        if label in line_styles:
            line.set_linestyle(line_styles[label])
            line.set_color(line_colors[label])
    
    if plot_sims:
        if isinstance(plot_sims, list):
            for sim in plot_sims:
                if sim in gbm_simulations:
                    sim_color = line_colors.get(sim, colors[asset_returns.columns.get_loc(sim) % len(colors)])
                    gbm_simulations[sim].plot(ax=ax, color=sim_color, alpha=0.25, linewidth=2, legend=False)
    
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title("Cumulative Portfolio Returns with Simulations" if plot_sims else "Cumulative Portfolio Returns")
    
    plt.show()


def multiple_pure_buy_and_hold(wts_df, asset_returns):
    return {row: pure_buy_and_hold(wts_df.loc[row], asset_returns) for row in wts_df.index}


def get_multiple_actual_returns_old(wts_srs, asset_returns):
    return pd.DataFrame({
        name: get_actual_returns(wts, asset_returns)
        for name, wts in wts_srs.items()})

def get_multiple_actual_returns(wts_srs, asset_returns):
    result_df = pd.DataFrame({
        name: get_actual_returns(wts, asset_returns)
        for name, wts in wts_srs.items()
    })
    result_df = result_df.apply(pd.to_numeric, errors='coerce')
    return result_df

def plot_act_rets(act_rets, benchmark=None, plot_benchmark=True):
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")
        act_rets = pd.concat([act_rets, benchmark], axis=1)
    act_rets = act_rets.dropna(axis=0)

    line_styles = {}
    line_colors = {}
    
    if "Benchmark" in act_rets.columns:
        line_styles["Benchmark"] = 'dashed'
        line_colors["Benchmark"] = 'grey'
    
    if "Current Portfolio" in act_rets.columns:
        line_styles["Current Portfolio"] = 'dashed'
        line_colors["Current Portfolio"] = 'black'
    
    asset_cum_returns = (act_rets + 1).cumprod()

    for column in asset_cum_returns.columns:
        plt.plot(asset_cum_returns.index, asset_cum_returns[column], 
                 label=column, linestyle=line_styles.get(column, '-'), 
                 color=line_colors.get(column, None))
    
    plt.title('Cumulative Portfolio Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns (without rebalancing)')
    plt.legend()
    plt.show()



def download_analysis_as_xlsx(portfolio_file_name, download_results=False):
    global portfolio_stats_df, dd_alloc, port_stats, loadings
    
    if download_results:
        with pd.ExcelWriter(portfolio_file_name, engine="openpyxl") as excel_file:
            portfolio_stats_df.to_excel(excel_file, sheet_name="Portfolio Info", index=True)
            dd_alloc.to_excel(excel_file, sheet_name="Allocation Info", index=True)
            port_stats.to_excel(excel_file, sheet_name="Performance Info", index=True)
            loadings.to_excel(excel_file, sheet_name="Factor Info", index=True)
        
        print(f"File saved to: {portfolio_file_name}")
    else:
        print("Download results set to False. No file saved.")




########################################################################################################################################################################################################
######## PLOTTING FOR STREAMLIT
########################################################################################################################################################################################################


def plot_indiv_assets_db(asset_returns, benchmark=None, plot_benchmark=False, current_portfolio=None, plot_current_portfolio=False):
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")
        asset_returns = pd.concat([asset_returns, benchmark], axis=1)

    if plot_current_portfolio:
        if current_portfolio is None:
            asset_returns = asset_returns
        if current_portfolio is not None:
            asset_returns = pd.concat([current_portfolio, asset_returns], axis=1)

    asset_returns = asset_returns.dropna(axis=0)
    asset_cum_returns = (asset_returns + 1).cumprod()

    line_styles = {}
    line_colors = {}
    
    if "Current Portfolio" in asset_cum_returns.columns:
        line_styles["Current Portfolio"] = 'dashed'
        line_colors["Current Portfolio"] = 'black'
    
    if "Benchmark" in asset_cum_returns.columns:
        line_styles["Benchmark"] = 'dashed'
        line_colors["Benchmark"] = 'grey'

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.set_title('Cumulative Ticker Returns over Time')

    # Plot data on the axis
    for column in asset_cum_returns.columns:
        ax.plot(asset_cum_returns.index, asset_cum_returns[column], label=column, 
                linestyle=line_styles.get(column, '-'), color=line_colors.get(column, None))

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    ax.legend()
    ax.grid(False)

    # Return the figure
    return fig



def plot_portfolios_db(asset_returns, benchmark=None, plot_benchmark=False, current_portfolio=None, plot_current_portfolio=False):
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")
        asset_returns = pd.concat([asset_returns, benchmark], axis=1)

    if plot_current_portfolio:
        if current_portfolio is None:
            raise ValueError("No current portfolio data provided.")
        asset_returns = pd.concat([asset_returns, current_portfolio], axis=1)

    asset_returns = asset_returns.dropna(axis=0)
    asset_cum_returns = (asset_returns + 1).cumprod()

    line_styles = {}
    line_colors = {}
    
    if "Current Portfolio" in asset_cum_returns.columns:
        line_styles["Current Portfolio"] = 'dashed'
        line_colors["Current Portfolio"] = 'black'
    
    if "Benchmark" in asset_cum_returns.columns:
        line_styles["Benchmark"] = 'dashed'
        line_colors["Benchmark"] = 'grey'

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.set_title('Cumulative Portfolio Returns over Time')
    
    # Plot each column
    for column in asset_cum_returns.columns:
        ax.plot(asset_cum_returns.index, asset_cum_returns[column], label=column, 
                 linestyle=line_styles.get(column, '-'), color=line_colors.get(column, None))

    # Set labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    ax.legend()
    ax.grid(False)

    # Return the figure object
    return fig


def plot_drawdowns_db(rets, benchmark=None, plot_benchmark=False):
    
    if plot_benchmark:
        if benchmark is None:
            raise ValueError("No benchmark data provided.")
        rets = pd.concat([rets, benchmark], axis=1) 

    rets = rets.dropna(axis=0)
    cum_rets = (rets + 1).cumprod()

    cum_max = cum_rets.cummax()
    drawdown = (cum_rets - cum_max) / cum_max

    line_styles = {}
    line_colors = {}
    
    if "Benchmark" in drawdown.columns:
        line_styles["Benchmark"] = 'dashed'
        line_colors["Benchmark"] = 'grey'
    
    if "Current Portfolio" in drawdown.columns:
        line_styles["Current Portfolio"] = 'dashed'
        line_colors["Current Portfolio"] = 'black'
    
    if "Custom Portfolio" in drawdown.columns:
        line_styles["Custom Portfolio"] = 'dotted'
        line_colors["Custom Portfolio"] = 'black'

 
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.set_title("Portfolio Drawdowns over Time")

    for column in drawdown.columns:
        ax.plot(drawdown.index, drawdown[column], label=column, 
                linestyle=line_styles.get(column, '-'), color=line_colors.get(column, None))

    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdowns")
    ax.legend()
    ax.grid(False)

    return fig  

def plot_portf_wts_db(ports_wts):
    fig, ax = plt.subplots()
    ports_wts.plot(kind='bar', ax=ax)
    ax.set_title('Asset Weights per Portfolio')
    return fig  

def plot_risk_cont_db(portf_wts, VCV):
    res_df = risk_cont_ports(portf_wts, VCV)
    
    fig, ax = plt.subplots()
    res_df.T.plot(kind='bar', ax=ax)
    ax.set_title('Risk-Contribution per Ticker')
    return fig  

def plot_sharpe_cont_db(portf_wts, ER, VCV, rfr):
    res_df = sharpe_cont_ports(portf_wts, ER, VCV, rfr)
    
    fig, ax = plt.subplots()
    res_df.T.plot(kind='bar', ax=ax)
    ax.set_title('Sharpe-Contribution per Ticker')
    return fig  

def plot_ef_w_assets_db(rets, plot_df, rfr=0, n_points=1000, cml=True,
                        plot_benchmark=False, benchmark_stats=None,
                        plot_current=False, current_stats=None, custom_portfolio=None, custom_value=None):
    
    er = rets.mean()
    asset_vols = rets.std()
    cov = rets.cov()
    targets = np.linspace(er.min(), er.max(), n_points)
    wts = [minimize_vol(tr, er, cov) for tr in targets]
    ptf_rts = [portfolio_return(w, er) for w in wts]
    ptf_vls = [portfolio_vol(w, cov) for w in wts]
    
    ef = pd.DataFrame({
        "Returns": ptf_rts, 
        "Volatility": ptf_vls
    })

    fig, ax = plt.subplots(figsize=(12, 8.5))
    
    # Plot efficient frontier
    ax.plot(ef["Volatility"], ef["Returns"], label="Efficient Frontier")
    
    # Plot individual assets
    x = asset_vols
    y = er
    labels = rets.columns
    ax.scatter(x, y, color="blue", s=25, alpha=0.5, linewidths=1, label="Assets")
    for x_pos, y_pos, label in zip(x, y, labels):
        ax.annotate(label, xy=(x_pos, y_pos), xytext=(7, 0),
                    textcoords='offset points', ha='left', va='center')

    # Plot additional portfolios from plot_df
    x2 = plot_df['Volatility']
    y2 = plot_df['Return']
    labels2 = plot_df.index
    ax.scatter(x2, y2, color="red", s=25, alpha=0.5, linewidths=1)
    for x_pos, y_pos, label in zip(x2, y2, labels2):
        ax.annotate(label, xy=(x_pos, y_pos), xytext=(7, 0),
                    textcoords='offset points', ha='left', va='center')

    # Plot Capital Market Line (CML)
    if cml:
        w_msr = msr(rfr, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        cml_x = [0, vol_msr]
        cml_y = [rfr, r_msr]
        ax.plot(cml_x, cml_y, color='black', marker='o', linestyle='dashed', linewidth=1, markersize=5, label='CML')

    # Plot Benchmark
    if plot_benchmark and benchmark_stats is not None:
        benchmark_return = benchmark_stats['Return'].item()
        benchmark_vol = benchmark_stats['Volatility'].item()
        ax.scatter(benchmark_vol, benchmark_return, color='black', marker='o', s=50, label='Benchmark')
        ax.annotate('Benchmark', xy=(benchmark_vol, benchmark_return), xytext=(7, 0),
                    textcoords='offset points', ha='left', va='center')

    # Plot Current Portfolio with dotted lines
    if plot_current and current_stats is not None:
        current_return = current_stats['Return'].item()
        current_vol = current_stats['Volatility'].item()
        
        ax.scatter(current_vol, current_return, color='black', marker='o', s=50, label='Current Portfolio')
        ax.annotate('Current Portfolio', xy=(current_vol, current_return), xytext=(7, 0),
                    textcoords='offset points', ha='left', va='center')
        
        ax.axvline(x=current_vol, color='black', linestyle=':', linewidth=1)
        ax.axhline(y=current_return, color='black', linestyle=':', linewidth=1)

    # Plot Custom Portfolio logic
    if custom_portfolio == 'return':
        custom_df = ef.loc[ef['Volatility'].idxmin():]
        custom_val = custom_value
        abs_diff = np.abs(custom_df['Returns'] - custom_val)
        closest_indices = abs_diff.nsmallest(2).index
        closest_return_values = custom_df.loc[closest_indices, 'Returns']
        weights = (custom_val - closest_return_values) / np.diff(abs(closest_return_values))
        closest_volatility = custom_df.loc[closest_indices, 'Volatility']
        custom_vol = (abs(weights) * closest_volatility).sum()
        custom_ret = custom_val
        ax.scatter(custom_vol, custom_ret, color='black', marker='o', s=50, label='Custom Portfolio')
        ax.annotate('Custom Portfolio', xy=(custom_vol, custom_ret), xytext=(7, 0),
                    textcoords='offset points', ha='left', va='center')
    
    if custom_portfolio == 'volatility':
        custom_df = ef.loc[ef['Volatility'].idxmin():]
        custom_val = custom_value
        abs_diff = np.abs(custom_df['Volatility'] - custom_val)
        closest_indices = abs_diff.nsmallest(2).index
        closest_volatility_values = custom_df.loc[closest_indices, 'Volatility']
        weights = (custom_val - closest_volatility_values) / np.diff(abs(closest_volatility_values))
        closest_returns = custom_df.loc[closest_indices, 'Returns']
        custom_ret = (abs(weights) * closest_returns).sum()
        custom_vol = custom_val
        ax.scatter(custom_vol, custom_ret, color='black', marker='o', s=50, label='Custom Portfolio')
        ax.annotate('Custom Portfolio', xy=(custom_vol, custom_ret), xytext=(7, 0),
                    textcoords='offset points', ha='left', va='center')

    ax.set_xlabel("Volatility")
    ax.set_ylabel("Returns")
    ax.legend()
    ax.grid(False)
    
    return fig

def plot_multiple_factor_loadings_db(portf_rets, factor_returns, benchmark=None):
    if benchmark is not None:
        portf_rets = pd.concat([portf_rets, benchmark], axis=1)
    
    res = get_multiple_factor_loadings(portf_rets, factor_returns)
    
    fig, ax = plt.subplots(figsize=(12, 8.5))
    #fig, ax = plt.subplots()

    bars = res.T.plot(kind='bar', ax=ax)
    ax.set_title('Factor Loadings across Portfolios')
    ax.set_xlabel('Portfolios')
    ax.set_ylabel('Factor Loadings')
    ax.grid(False)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Factors')
    
    plt.tight_layout()

    return fig

def plot_gbm_and_cum_rets_db(asset_returns, n_years=1, n_scenarios=100, steps_per_year=252, plot_last=252*2, plot_sims=None):
    if plot_last is not None:
        asset_returns = asset_returns.tail(plot_last)
    
    cum_rets = (asset_returns + 1).cumprod()
    
    gbm_simulations = generate_gbm_from_asset_returns(asset_returns, n_years=n_years, n_scenarios=n_scenarios, steps_per_year=steps_per_year)
    
    line_styles = {}
    line_colors = {}
    
    if "Benchmark" in cum_rets.columns:
        line_styles["Benchmark"] = 'dashed'
        line_colors["Benchmark"] = 'grey'
    if "Current Portfolio" in cum_rets.columns:
        line_styles["Current Portfolio"] = 'dashed'
        line_colors["Current Portfolio"] = 'black'
    
    fig, ax = plt.subplots(figsize=(12, 8.5))
    
    cum_rets.plot(ax=ax, figsize=(12, 8.5))
    
    lines = ax.get_lines()
    colors = [line.get_color() for line in lines]
    handles, labels = ax.get_legend_handles_labels()
    
    for line in lines:
        label = line.get_label()
        if label in line_styles:
            line.set_linestyle(line_styles[label])
            line.set_color(line_colors[label])
    
    if plot_sims:
        if isinstance(plot_sims, list):
            for sim in plot_sims:
                if sim in gbm_simulations:
                    sim_color = line_colors.get(sim, colors[asset_returns.columns.get_loc(sim) % len(colors)])
                    gbm_simulations[sim].plot(ax=ax, color=sim_color, alpha=0.25, linewidth=2, legend=False)
    
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title("Cumulative Portfolio Returns with Simulations" if plot_sims else "Cumulative Portfolio Returns")
    
    return fig


def process_and_plot_db(df):
    df_copy = df.copy()
    df_copy['Return'] = df_copy['Return'] * 252
    df_copy['Volatility'] = df_copy['Volatility'] * np.sqrt(252)
    df_sorted = df_copy.sort_values(by="Volatility").reset_index()

    color_mapping = {
        'Markowitz': "#00008b",  
        'Global Minimum Variance': "#ff8c00",  
        'Maximum Sharpe Ratio Portfolio': "#006400",  
        'Equal Weights': "#FF0000",  
        'Risk Parity': "#9400d3",  
        'Inverse Variance': "#a0522d",  
        'Inverse Volatility': "#00bfff",  
        'Maximum Diversification': "#00ff00",  
        'Maximum De-Correlation': "#ffd700",  
        'Minimum Correlation': "#ee82ee",  
        'Equal Sharpe Ratio': "#deb887", 
        'Benchmark': "grey",  
        'Current Portfolio': "black"  
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, row in df_sorted.iterrows():
        index_name = row[df_sorted.columns[0]]  
        color = color_mapping.get(index_name, "#C71585")  
        
        ax.scatter(row['Volatility'], row['Return'], color=color, s=100)
        ax.text(row['Volatility'], row['Return'], index_name, fontsize=9, ha='left', va='bottom')

    current_portfolio_row = df_sorted[df_sorted[df_sorted.columns[0]] == 'Current Portfolio']

    if not current_portfolio_row.empty:
        current_volatility = current_portfolio_row['Volatility'].values[0]
        current_return = current_portfolio_row['Return'].values[0]
        
        ax.axvline(x=current_volatility, color='black', linestyle='--')
        ax.axhline(y=current_return, color='black', linestyle='--')

    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.set_title('Return vs Volatility Scatter Plot')
    plt.tight_layout()
    
    return fig

def plot_sharpe_ratios(df, rfr_annual, By="Sharpe Ratio"):
    df_copy = df.copy()
    df_copy['Return'] = df_copy['Return'] * 252
    df_copy['Volatility'] = df_copy['Volatility'] * np.sqrt(252)
    df_copy['Sharpe Ratio'] = (df_copy['Return'] - rfr_annual) / df_copy['Volatility']
    df_copy['Ret+Vol'] = df_copy['Return'] + df_copy['Volatility'] 
    df_copy['Ret-Vol'] = df_copy['Return'] - df_copy['Volatility']
    
    df_copy = df_copy.sort_values(by=By, ascending=True)

    color_mapping = {
        'Markowitz': "#00008b", 'Global Minimum Variance': "#ff8c00", 'Maximum Sharpe Ratio Portfolio': "#006400", 'Equal Weights': "#FF0000",'Risk Parity': "#9400d3", 'Inverse Variance': "#a0522d", 'Inverse Volatility': "#00bfff", 
        'Maximum Diversification': "#00ff00", 'Maximum De-Correlation': "#ffd700", 'Minimum Correlation': "#ee82ee",'Equal Sharpe Ratio': "#deb887", 'Benchmark': "grey", 'Current Portfolio': "black"}

    fig, ax = plt.subplots()
    added_labels = set()
    for i, (idx, row) in enumerate(df_copy.iterrows()):
        portfolio_type = idx  
        color = color_mapping.get(portfolio_type, 'blue')  
        x_value = i + 1
        
        if portfolio_type not in added_labels:
            ax.plot(x_value, row['Return'], marker='o', color=color, label=portfolio_type, markersize=15)
            added_labels.add(portfolio_type)  
        else:
            ax.plot(x_value, row['Return'], marker='o', color=color, markersize=20)
        
        ax.plot(x_value, row['Ret-Vol'], marker='o', color=color, markersize=6)
        ax.plot(x_value, row['Ret+Vol'], marker='o', color=color, markersize=6)
        ax.plot([x_value, x_value], [row['Ret-Vol'], row['Ret+Vol']], linestyle='--', color=color)
        ax.text(x_value, row['Return'] + 0.01, portfolio_type, color='black', ha='left', rotation=45)

    ax.set_xlabel("")
    ax.set_ylabel("Return & Volatility")
    ax.set_title("Return and Volatility across Portfolios, ordered by Sharpe Ratio")
    ax.grid(False)

    plt.tight_layout()
    plt.close(fig)

    return fig

def ports_wts_to_table(ports_wts):
    ports_wts = (ports_wts*100).round(2).astype(str) + "%"
    table = ports_wts.astype(str) + "%"
    return table


def calc_port_stats_2(returns, rfr, benchmark, target_return,periods):
    rf = rfr/periods
    rets = returns

    var = (var_hist(rets,level=5).T*100).round(2).astype(str) + "%"
    cvar = (cvar_hist(rets, level=5).T*100).round(2).astype(str) + "%"
    gvar =  (var_gauss(rets, level=5)*100).T.round(2).astype(str) + "%"
    semdev = (semidev(rets)*100).round(2).astype(str) + "%"
    dd = drawdown(rets)
    avgr = (avg_return(rets)*100).round(2).astype(str) + "%"
    avgw = (avg_win(rets)*100).round(2).astype(str) + "%"
    avgl = (avg_loss(rets)*100).round(2).astype(str) + "%"
    bst = (best(rets)*100).round(2).astype(str) + "%"
    wrst = (worst(rets)*100).round(2).astype(str) + "%"

    sort = sortino(rets, rf=rf, periods=periods, annualize=True).round(2).astype(str) + "%"
    om = omega(rets,target_return,periods).round(2).astype(str) + "%"
    gtpr = gain_to_pain_ratio(rets).round(2).astype(str) + "%"
    rsquares = pd.Series(rets.apply(lambda x: get_rsquare(x,benchmark))).to_frame(name='R-Squared').round(2)
    alphasbetas = rets.apply(lambda x: get_alpha_and_beta(x,benchmark)).T.round(4)
    alphasbetas.columns = ['Alpha','Beta']
    inforatio = info_ratio(rets,benchmark).round(2).astype(str) + "%"
    kelly = kelly_criterion(rets).round(2).astype(str) + "%"

    pr = profit_ratio(rets).round(2).astype(str) + "%"
    pf = profit_factor(rets).round(2)
    payoffr = payoff_ratio(rets).round(2).astype(str) + "%"
    cpc = cpc_index(rets).round(2)
    tailrat = tail_ratio(rets).round(2).astype(str) + "%"
    csr = common_sense_ratio(rets).round(2).astype(str) + "%"
    owr = outlier_win_ratio(rets).round(2).astype(str) + "%"
    owl = outlier_loss_ratio(rets).round(2).astype(str) + "%"

    maxdd = max_drawdown(rets).round(2).astype(str) + "%"
    calm = calmar(rets).round(2)
    ui = ulcer_index(rets).round(2)
    upi = ulcer_performance_index(rets,rf=rfr).round(2)
    si = serenity_index(returns,rf=rfr).round(2)
    ror = risk_of_ruin(rets,trials=15).round(2)
    recfac = recovery_factor(rets).round(2)
    dd_info = drawdown_info(rets).T

    res_list = [var,cvar,gvar,semdev,avgr,avgw,avgl,bst,wrst,sort,om,gtpr,rsquares,alphasbetas,inforatio,kelly,pr,pf,payoffr,cpc,tailrat,csr,owr,owl,maxdd,calm,ui,upi,si,ror,recfac,dd_info]

    res = pd.concat(res_list, axis=1).T
    #res.index = ['VaR','CVaR','Semideviation','Average Return','Average Win','Average Loss','Best Return','Worst Return',
    #'Sortino Ratio','Omega Ratio','Gain-to-Pain-Ratio','R-Squared','Alpha','Beta','Information Ratio','Kelly Criterion',
    #'Profit Ratio','Profit Factor','Payoff Ratio','CPC-Index','Tail Ratio','Common Sense Ratio','Outlier-to-Win-Ration',
    #'Outlier-to-Loss-Ratio','Maximum Drawdown','Calmar Ratio','Ulcer Index','Ulcer Performance Index','Serenity Index','Risk of Ruin','Recovery Factor']
    return res


def calc_gbm(asset_returns, n_years=1, n_scenarios=100, steps_per_year=252, plot_last=252*2):
    if plot_last:
        asset_returns = asset_returns.tail(plot_last)
    gbm_simulations = generate_gbm_from_asset_returns(asset_returns, n_years=n_years, n_scenarios=n_scenarios, steps_per_year=steps_per_year)
    return gbm_simulations




################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################


def cvar_optimize(ret_df, beta, bound=1.0):
    """
    Optimize portfolio weights to minimize Conditional Value at Risk (CVaR).
    
    Parameters:
        ret_df (pd.DataFrame): A DataFrame of historical asset returns (rows: observations, columns: assets).
        beta (float): Confidence level for CVaR (e.g., 0.95 for 5% CVaR).
        bound (float, optional): Upper bound on portfolio weights for each asset (default: 1.0).
    
    Returns:
        dict: Dictionary containing the optimized weights, minimized CVaR, and the success status.

    USE THIS FUNTION LATER WHEN YOU NEED TO CONSIDER TRANSACTION COST ETC
    """
    # Validate input
    if not isinstance(ret_df, pd.DataFrame):
        raise ValueError("ret_df must be a pandas DataFrame.")
    
    returns = ret_df.values
    q, n = returns.shape  # q = number of observations, n = number of assets
    m = 1 / (q * (1 - beta))
    
    # Objective function: Minimize [CVaR + mean slack variables + portfolio weights]
    c = np.array([1] + [m for _ in range(q)] + [0 for _ in range(n)])
    
    # Inequality constraints (Gx <= h)
    # 1. Non-negativity of CVaR, slack variables, and portfolio weights
    A1 = np.eye(1 + q + n)
    
    # 2. CVaR + slack_i - returns_i.dot(weights) >= 0
    A2 = np.hstack([
        np.ones((q, 1)),  # CVaR term
        np.eye(q),        # Slack variables
        -returns          # Returns matrix for portfolio
    ])
    
    A_ub = -np.vstack([A1, A2])  # Combine constraints and negate for <=
    b_ub = np.zeros(1 + 2 * q + n)
    
    # Equality constraints (Ax = b): Sum of weights equals 1
    A_eq = np.zeros((1, 1 + q + n))
    A_eq[0, (1 + q):] = 1
    b_eq = np.array([1])
    
    # Bounds: CVaR >= 0, slack variables >= 0, 0 <= weights <= bound
    bounds = [(0, None)] * (1 + q) + [(0, bound)] * n
    
    # Solve the linear programming problem
    result = linprog(
        c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs'
    )
    
    # Process the result
    if result.success:
        weights = result.x[(1 + q):]  # Extract portfolio weights
        cvar = result.fun  # CVaR is the minimized objective
        return {"weights": weights, "cvar": cvar, "success": True}
    else:
        return {"weights": None, "cvar": None, "success": False, "message": result.message}



def cvar_find(ret, beta, bound=1.0):
    'this function finds weights which construct the Minimum CVaR portfolio'
    q, n = ret.shape 
    m = 1 / (q * (1 - beta)) 

    c = np.array([1] + [m] * q + [0] * n)
    A1 = np.eye(1 + q + n)  
    A2 = np.hstack([
        np.ones((q, 1)),  
        np.eye(q),       
        ret.values        ])
    A_ub = -np.vstack([A1, A2])
    b_ub = np.zeros(1 + 2 * q + n)  

    A_eq = np.hstack([
        np.zeros(1 + q), 
        np.ones(n)        
    ]).reshape(1, -1)
    b_eq = np.array([1])  

    bnds = [(0, None)] * (1 + q) + [(0, bound)] * n

    linpro = linprog(
        c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bnds,
        method='highs', options={'disp': True})

    if linpro.success:
        optimal_weights = pd.Series(
            linpro.x[(1 + q):],
            index=ret.columns,  
            name="Optimal Weights")
        cvar = linpro.fun  
        return {"Optimal Weights": optimal_weights,"CVaR": -cvar}
    else:
        raise ValueError("Optimization failed: " + linpro.message)
    

def calculate_cvar_for_portfolios(weights_df, asset_returns, beta=0.95):
    """
    Calculates the Conditional Value at Risk (CVaR) for multiple portfolios.

    Parameters:
        weights_df (pd.DataFrame): A DataFrame where rows represent portfolios,
                                   columns represent assets, and entries are weights.
        asset_returns (pd.DataFrame): A DataFrame of asset return series
                                       (rows: observations, columns: assets).
        beta (float, optional): Confidence level for CVaR (e.g., 0.95 for 5% CVaR). Default is 0.95.

    Returns:
        pd.Series: A pandas Series with the CVaR for each portfolio.
    """
    if not isinstance(weights_df, pd.DataFrame):
        raise ValueError("weights_df must be a pandas DataFrame.")
    if not isinstance(asset_returns, pd.DataFrame):
        raise ValueError("asset_returns must be a pandas DataFrame.")
    if weights_df.shape[1] != asset_returns.shape[1]:
        raise ValueError("Number of assets in weights_df must match number of columns in asset_returns.")
    if not np.allclose(weights_df.sum(axis=1), 1):
        raise ValueError("Each portfolio's weights must sum to 1.")

    portfolio_returns = weights_df.values @ asset_returns.values.T
    cvars = {}
    for portfolio_name, returns in zip(weights_df.index, portfolio_returns):
        sorted_returns = np.sort(returns)
        cutoff_index = int(np.floor(len(sorted_returns) * (1 - beta)))
        cvar = sorted_returns[:cutoff_index].mean()
        cvars[portfolio_name] = cvar
    return pd.DataFrame({"Expected Shortfall": cvars})




def maximum_decorrelation_portfolio(returns):
    'This is the new function'
    cov_matrix = returns.cov().values
    std_devs = np.sqrt(np.diag(cov_matrix))
    corr_matrix = returns.corr().values
    n_assets = corr_matrix.shape[0]
    def objective(weights):
        weighted_std = weights / std_devs 
        return np.dot(weighted_std, np.dot(corr_matrix, weighted_std))
    constraints = ({'type': 'eq','fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * n_assets
    init_guess = np.ones(n_assets) / n_assets
    result = minimize(objective,x0=init_guess,bounds=bounds,constraints=constraints,method='SLSQP')
    if result.success:
        return pd.Series(result.x.round(6), index=returns.columns, name="Maximum Decorrelation Portfolio Weights")
    else:
        raise ValueError("Optimization failed: " + result.message)

def minimum_correlation_portfolio(returns):
    'This is the new function'
    corr_matrix = returns.corr().values
    n_assets = corr_matrix.shape[0]
    def objective(weights):
        portfolio_corr = np.dot(weights, np.dot(corr_matrix, weights)) / (np.sum(weights) ** 2)
        return portfolio_corr
    constraints = ({'type': 'eq','fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * n_assets
    init_guess = np.ones(n_assets) / n_assets
    result = minimize(objective,x0=init_guess,bounds=bounds,constraints=constraints,method='SLSQP')
    if result.success:
        return pd.Series(result.x, index=returns.columns, name="Minimum Correlation Portfolio Weights")
    else:
        raise ValueError("Optimization failed: " + result.message)



def calculate_diversification_ratio(returns, weights): #This is function used in the optimization process
    volatilities = returns.std().values  
    covariance_matrix = returns.cov().values
    portfolio_volatility = np.sqrt(weights.T @ covariance_matrix @ weights)
    weighted_volatility_sum = np.sum(weights * volatilities)
    diversification_ratio = weighted_volatility_sum / portfolio_volatility
    return diversification_ratio

def maximize_diversification_ratio(returns): # This function maximizes the Diversification Ratio given a pandas dataframe of asset returns
    n_assets = returns.shape[1]
    initial_weights = np.ones(n_assets) / n_assets
    bounds = [(0, 1) for _ in range(n_assets)]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    def objective(weights):
        return -calculate_diversification_ratio(returns, weights)
    result = minimize(
        objective,
        initial_weights,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP')
    optimal_weights = result.x
    max_diversification_ratio = -result.fun
    optimal_weights_df = pd.DataFrame(
        [optimal_weights],
        columns=returns.columns,
        index=["Maximum Diversification"])
    return {"Optimal Weights": optimal_weights_df,"Max Diversification Ratio": max_diversification_ratio}

def div_ratio(weights, VCV):
    'this calculates the Diversification Ratio for multiple Portfolios'
    if isinstance(weights, np.ndarray):
        weights = pd.DataFrame(weights)
    volatilities = np.sqrt(np.diag(VCV))
    diversification_ratios = []
    for _, w in weights.iterrows():
        w = w.values  
        weighted_volatility = np.dot(w, volatilities)
        portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(VCV, w)))
        div_measure = weighted_volatility / portfolio_volatility
        diversification_ratios.append(div_measure)
    return pd.DataFrame(diversification_ratios, index=weights.index, columns=["Diversification Ratio"])

def calc_change(ports_wts,portfolio_selection = None):
    if "Current Portfolio" in ports_wts.index:
        change_wts = (ports_wts - ports_wts.loc["Current Portfolio"]).drop("Current Portfolio")
    else:
        change_wts = ports_wts
    if portfolio_selection is not None:
        change_wts = change_wts.loc[[portfolio_selection]]
    else:
        change_wts = change_wts
    return change_wts



def calculate_max_drawdown(ports_wts, training_returns):
    """
    Calculates the Maximum Drawdown (MDD) for multiple portfolios.

    Parameters:
        ports_wts (pd.DataFrame): DataFrame where rows are portfolios, columns are assets, and values are weights.
        training_returns (pd.DataFrame): DataFrame where rows are dates, columns are assets, and values are returns.

    Returns:
        pd.Series: A Series with Maximum Drawdown for each portfolio.
    """
    # Validate input
    if not isinstance(ports_wts, pd.DataFrame):
        raise ValueError("ports_wts must be a pandas DataFrame.")
    if not isinstance(training_returns, pd.DataFrame):
        raise ValueError("training_returns must be a pandas DataFrame.")
    if not np.allclose(ports_wts.sum(axis=1), 1):
        raise ValueError("Each portfolio's weights must sum to 1.")

    portfolio_returns = ports_wts.values @ training_returns.T.values
    cumulative_returns = np.cumsum(portfolio_returns, axis=1)
    max_drawdowns = []
    for cum_returns in cumulative_returns:
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns - running_max
        max_drawdowns.append(np.min(drawdowns))

    return pd.Series(max_drawdowns, index=ports_wts.index, name="Max Drawdown")

def calculate_max_drawdown_with_series(ports_wts, training_returns):
    """
    Calculates the Maximum Drawdown (MDD) and the drawdown series for multiple portfolios.

    Parameters:
        ports_wts (pd.DataFrame): DataFrame where rows are portfolios, columns are assets, and values are weights.
        training_returns (pd.DataFrame): DataFrame where rows are dates, columns are assets, and values are returns.

    Returns:
        dict: A dictionary with:
            - "Max Drawdown": pd.Series with Maximum Drawdown for each portfolio.
            - "Drawdown Series": pd.DataFrame where each column is the drawdown series for a portfolio.
    """
    # Validate input
    if not isinstance(ports_wts, pd.DataFrame):
        raise ValueError("ports_wts must be a pandas DataFrame.")
    if not isinstance(training_returns, pd.DataFrame):
        raise ValueError("training_returns must be a pandas DataFrame.")
    if not np.allclose(ports_wts.sum(axis=1), 1):
        raise ValueError("Each portfolio's weights must sum to 1.")

    portfolio_returns = ports_wts.values @ training_returns.T.values
    cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1)
    drawdown_series = pd.DataFrame(index=training_returns.index, columns=ports_wts.index)
    max_drawdowns = []
    for i, (portfolio_name, cum_returns) in enumerate(zip(ports_wts.index, cumulative_returns)):
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        drawdown_series[portfolio_name] = drawdowns
        max_drawdowns.append(np.min(drawdowns))
    max_drawdowns = pd.Series(max_drawdowns, index=ports_wts.index, name="Max Drawdown")

    return {"Max Drawdown": max_drawdowns,"Drawdown Series": drawdown_series}


def calculate_average_weights(weight_dict):
    avg_weights_list = []
    for portfolio, weights_df in weight_dict.items():
        if not isinstance(weights_df, pd.DataFrame):
            raise ValueError(f"Value for portfolio '{portfolio}' must be a pandas DataFrame.")
        avg_weights = weights_df.mean(axis=0)
        avg_weights.name = portfolio  
        avg_weights_list.append(avg_weights)
    avg_weights_df = pd.DataFrame(avg_weights_list)
    return avg_weights_df

def get_portfolio_weights(w0,rfr,target_return,ER,VCV,CORR,returns):
    markowitz_wts = minimize_vol(target_return, ER, VCV)
    glob_min_var_wts = gmv(VCV)
    max_sharpe_wts = msr(rfr, ER, VCV)
    equal_wts = eq_weight_portf(VCV)
    risk_parity_wts = equal_risk_contributions(VCV)
    inv_var_wts = inv_vol_portf(VCV)
    inv_vol_wts = inv_var_portf(VCV)
    max_div_wts = maximize_diversification_ratio(returns)['Optimal Weights'].loc['Maximum Diversification']
    max_decorr_wts = maximum_decorrelation_portfolio(returns)
    min_corr_wts = minimum_correlation_portfolio(returns)
    equal_sharpe_wts = target_sharpe_contribution((1/len(VCV.columns)),ER,VCV,rfr)
    cvar_wts = cvar_find(returns,beta=0.95)['Optimal Weights']
    max_dd_wts = minimize_max_drawdown(returns)['Optimal Weights']
    res = pd.concat([markowitz_wts,glob_min_var_wts,max_sharpe_wts,equal_wts,risk_parity_wts,inv_var_wts,inv_vol_wts,
                     max_div_wts,max_decorr_wts,min_corr_wts,equal_sharpe_wts,cvar_wts,max_dd_wts],axis=1)
    df = pd.DataFrame(res.T)
    df.index = ["Markowitz","Global Minimum Variance","Maximum Sharpe Ratio Portfolio","Equal Weights","Risk Parity","Inverse Variance","Inverse Volatility",
                "Maximum Diversification","Maximum De-Correlation","Minimum Correlation","Equal Sharpe Ratio","Minimum Expected Shortfall","Minimum Max Drawdown"]
    return df

def calculate_cumulative_returns(weights, asset_returns):
    portfolio_returns = asset_returns @ weights
    cumulative_returns = np.cumsum(portfolio_returns)
    return cumulative_returns

def max_dd(cumulative_returns):
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = cumulative_returns - running_max
    return np.min(drawdowns)

def minimize_max_drawdown(asset_returns):
    n_assets = asset_returns.shape[1]
    def objective(weights):
        cumulative_returns = calculate_cumulative_returns(weights, asset_returns)
        return -max_dd(cumulative_returns)  
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_assets)]
    initial_weights = np.ones(n_assets) / n_assets
    result = minimize(objective,initial_weights,bounds=bounds,constraints=constraints,method='SLSQP')
    if result.success:
        return {"Optimal Weights": pd.Series(result.x.round(6),index=asset_returns.columns),"Minimized MDD": -result.fun}
    else:
        raise ValueError("Optimization failed: " + result.message)




def calculate_cvar(portfolio_returns, confidence_level=0.95):
    alpha = 1 - confidence_level
    var = portfolio_returns.quantile(alpha, axis=0)
    cvar = portfolio_returns.apply(lambda x: x[x <= var[x.name]].mean(), axis=0)
    return cvar

def calculate_diversification_ratio_2(ports_wts, returns):
    if not np.allclose(ports_wts.sum(axis=1), 1):
        raise ValueError("Each portfolio's weights must sum to 1.")
    asset_volatilities = returns.std(axis=0)
    covariance_matrix = returns.cov()
    diversification_ratios = {}
    for portfolio_name, weights in ports_wts.iterrows():
        weights_array = weights.values
        portfolio_volatility = np.sqrt(weights_array @ covariance_matrix.values @ weights_array.T)
        weighted_volatilities = (weights_array * asset_volatilities.values).sum()
        diversification_ratio = weighted_volatilities / portfolio_volatility
        diversification_ratios[portfolio_name] = diversification_ratio
    return pd.Series(diversification_ratios, name="Diversification Ratio")

def calculate_cumulative_return(returns):
    gross_returns = (returns+1)
    cum_returns = gross_returns.cumprod()
    final_cum_returns = cum_returns.iloc[-1] - 1
    return final_cum_returns

def calculate_max_drawdown(returns):
    cum_returns = (returns + 1).cumprod()
    cum_max = cum_returns.cummax()
    drawdown = (cum_returns - cum_max) / cum_max
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_ex_post_diversification_ratio(ports_wts, returns):
    if not isinstance(ports_wts, pd.DataFrame):
        raise ValueError("ports_wts must be a pandas DataFrame.")
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a pandas DataFrame.")
    covariance_matrix = returns.cov()
    asset_vols = np.sqrt(np.diag(covariance_matrix))
    asset_vols = pd.Series(asset_vols, index=returns.columns)
    diversification_ratios = []
    for _, weights in ports_wts.iterrows():
        weighted_vol = np.dot(weights, asset_vols)
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_variance)
        diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else np.nan
        diversification_ratios.append(diversification_ratio)
    return pd.Series(diversification_ratios, index=ports_wts.index, name="Ex-Post Diversification Ratio")

def calc_short_stats(act_rets,avg_wts,returns,ret_vol_df,rfr_annual):
    cum_rets = (calculate_cumulative_return(act_rets)*100).round(2).astype(str) + "%"
    sharpe = (ret_vol_df['Return']*252-rfr_annual)/(ret_vol_df['Volatility']*np.sqrt(252))
    max_dd = (calculate_max_drawdown(act_rets)*100).round(2).astype(str) + "%"
    div_ratio = calculate_ex_post_diversification_ratio(avg_wts,returns).round(2)
    cvar = (calculate_cvar(act_rets)*100).round(2).astype(str) + "%"
    res = pd.concat([cum_rets,sharpe.round(2),div_ratio,max_dd,cvar],axis=1)
    df = pd.DataFrame(res)
    df.columns = ['Cumulative Return','Sharpe Ratio','Diversification Ratio','Max Drawdown','Expected Shortfall']
    return df

def get_short_stats(short_stats, portfolio_selection=None):
    if portfolio_selection is None:
        return short_stats.T
    else:
        if not isinstance(portfolio_selection, list):
            portfolio_selection = [portfolio_selection]
        if 'Current Portfolio' in short_stats.index and 'Current Portfolio' not in portfolio_selection:
            portfolio_selection.append('Current Portfolio')
        valid_selection = [p for p in portfolio_selection if p in short_stats.index]
        res = short_stats.loc[valid_selection].T
        return res

def calculate_cvar_for_assets(returns_df, beta=0.95):
    'this function calculates the "realized CVaR" of a dataframe of Returns'
    if not isinstance(returns_df, pd.DataFrame):
        raise ValueError("returns_df must be a pandas DataFrame.")
    cvar_dict = {}
    for asset in returns_df.columns:
        sorted_returns = np.sort(returns_df[asset])
        cutoff_index = int(np.floor(len(sorted_returns) * (1 - beta)))
        cvar = sorted_returns[:cutoff_index].mean()
        cvar_dict[asset] = cvar
    cvar_df = pd.DataFrame({"Expected Shortfall": cvar_dict})
    return cvar_df
