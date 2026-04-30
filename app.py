import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
import plotly.express as px

# --- APP CONFIG ---
st.set_page_config(page_title="Portfolio AI Pro", layout="wide")
st.title("Advanced ML Portfolio Optimizer")

# --- UTILITY FUNCTIONS ---
def calculate_metrics(returns, weights_df=None):
    """Comprehensive financial metric suite."""
    # Annualized Stats
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    
    # Sharpe & Sortino
    downside_rets = returns[returns < 0]
    sortino = ann_ret / (downside_rets.std() * np.sqrt(252)) if len(downside_rets) > 0 else 0
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    
    # Drawdown & Calmar
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    mdd = drawdown.min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0
    
    # Hit Ratio (Percentage of positive days)
    hit_ratio = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
    
    # Turnover (Avg daily change in weights)
    turnover = 0
    if weights_df is not None:
        turnover = weights_df.diff().abs().sum(axis=1).mean()

    return {
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max DD": mdd,
        "Calmar": calmar,
        "Hit Ratio": hit_ratio,
        "Avg Turnover": turnover
    }

# --- DATA FETCHING ---
@st.cache_data
def get_data(tickers, cash, start, end):
    assets = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    all_tickers = list(set(assets + [cash, "^GSPC"]))
    df = yf.download(all_tickers, start=start, end=end)['Close']
    if df.empty: return pd.DataFrame(), [], None
    valid_assets = df.columns[df.notna().sum() > (len(df) * 0.5)]
    sp500 = df["^GSPC"].pct_change().dropna()
    return df[valid_assets].dropna(), [a for a in assets if a in valid_assets], sp500

# --- BACKTEST ENGINE ---
@st.cache_data
def run_backtest(tickers, cash_ticker, start, end, turn_limit):
    data, assets, sp500_all = get_data(tickers, cash_ticker, start, end)
    returns = data[assets].pct_change().dropna()
    cash_rets = data[cash_ticker].pct_change().dropna()
    
    # Feature Engineering
    m_ret = returns.mean(axis=1).rolling(5).mean()
    m_vol = returns.mean(axis=1).rolling(20).std()
    features = pd.concat([m_ret, m_vol], axis=1).dropna()
    
    total_len = len(features)
    window = min(252, int(total_len * 0.4))
    step = max(1, int(total_len * 0.05))
    
    strat_rets, weight_history, ml_metrics = [], [], []
    cur_weights = np.array([1.0/len(assets)] * len(assets))
    
    for i in range(window, total_len - 1, step):
        train_idx = features.index[max(0, i-window) : i]
        test_idx = features.index[i : min(i + step, total_len)]
        
        # Models
        scaler = RobustScaler().fit(features.loc[train_idx].values)
        hmm = GaussianHMM(n_components=3, random_state=42).fit(scaler.transform(features.loc[train_idx].values))
        
        y_train = returns.loc[train_idx].shift(-5).rolling(5).mean().dropna()
        mlp = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
        mlp.fit(features.loc[y_train.index].values, y_train.values)
        
        # Execution
        test_feat = features.loc[test_idx].values
        regimes = hmm.predict(scaler.transform(test_feat))
        pred_rets = mlp.predict(test_feat)
        
        for j, date in enumerate(test_idx):
            target = np.exp(pred_rets[j]) / np.sum(np.exp(pred_rets[j]))
            risk_adj = 0.6 if regimes[j] == 0 else 1.0 # De-risk in regime 0
            
            cur_weights = (1 - turn_limit) * cur_weights + (turn_limit * target)
            day_ret = (cur_weights * returns.loc[date] * risk_adj).sum() + ((1 - risk_adj) * cash_rets.loc[date])
            
            strat_rets.append(day_ret)
            weight_history.append(list(cur_weights * risk_adj) + [1 - risk_adj])
            ml_metrics.append({"Date": date, "Regime": regimes[j]})

    final_idx = features.index[window : window + len(strat_rets)]
    w_df = pd.DataFrame(weight_history, index=final_idx, columns=assets + [cash_ticker])
    return pd.Series(strat_rets, index=final_idx), sp500_all.loc[final_idx], w_df, pd.DataFrame(ml_metrics).set_index("Date")

# --- UI LOGIC ---
with st.sidebar:
    st.header("Settings")
    t_input = st.text_input("Tickers", "AAPL, MSFT, NVDA, TSLA, SPY")
    c_proxy = st.selectbox("Cash Proxy", ["BIL", "SHV"])
    start = st.date_input("Start", pd.to_datetime("2023-01-01"))
    end = st.date_input("End", pd.to_datetime("2024-05-01"))
    smooth = st.slider("Smoothing (Turnover Control)", 0.01, 0.50, 0.15)
    run = st.button("Execute Model")

if run:
    s_ret, sp_ret, weights, ml_diag = run_backtest(t_input, c_proxy, start, end, smooth)
    s_m = calculate_metrics(s_ret, weights)
    sp_m = calculate_metrics(sp_ret)

    # Metric Dashboard
    cols = st.columns(4)
    cols[0].metric("Sortino Ratio", f"{s_m['Sortino']:.2f}", f"{s_m['Sortino']-sp_m['Sortino']:.2f} vs SPY")
    cols[1].metric("Calmar Ratio", f"{s_m['Calmar']:.2f}")
    cols[2].metric("Hit Ratio", f"{s_m['Hit Ratio']:.1%}")
    cols[3].metric("Daily Turnover", f"{s_m['Avg Turnover']:.2%}")

    st.divider()

    # Detailed Comparison Table
    st.subheader("Strategy vs Benchmark")
    comparison = pd.DataFrame({"Strategy": s_m, "S&P 500": sp_m}).T
    st.table(comparison.style.format({
        "Ann. Return": "{:.2%}", "Ann. Vol": "{:.2%}", "Max DD": "{:.2%}", 
        "Hit Ratio": "{:.2%}", "Avg Turnover": "{:.2%}", 
        "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Calmar": "{:.2f}"
    }))

    # Charts
    tab1, tab2 = st.tabs(["Performance", "Model Context"])
    with tab1:
        st.line_chart(pd.DataFrame({"Strategy": (1+s_ret).cumprod(), "S&P 500": (1+sp_ret).cumprod()}))
    with tab2:
        st.area_chart(weights)
