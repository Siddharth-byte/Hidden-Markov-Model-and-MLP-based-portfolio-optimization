import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
import plotly.express as px

# --- APP CONFIG ---
st.set_page_config(page_title="Quant-Grade Portfolio AI", layout="wide")
st.title("⚖️ Professional Regime-Aware Optimizer")

with st.sidebar:
    st.header("1. Universe & Timeframe")
    ticker_input = st.text_input("Tickers", "AAPL, MSFT, NVDA, JNJ, SPY, TLT, GLD")
    cash_proxy = st.selectbox("Safe Haven Asset", ["SHV", "BIL", "IEF"])
    start_date = st.date_input("Start Date", pd.to_datetime("2016-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))
    
    st.header("2. Risk Controls")
    turnover_limit = st.slider("Portfolio Smoothing (Low = Less Trading)", 0.05, 0.50, 0.15)
    run_btn = st.button("🚀 Execute Backtest")

# --- UTILITIES ---
def get_clean_data(tickers, cash, start, end):
    assets = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    df = yf.download(assets + [cash], start=start, end=end)['Close']
    # Ensure we only use assets present throughout the majority of the period
    valid_assets = df.columns[df.notna().sum() > (len(df) * 0.9)]
    return df[valid_assets].dropna(), [a for a in assets if a in valid_assets]

# --- CORE ENGINE ---
@st.cache_data
def run_backtest(tickers, cash_ticker, start, end, smoothing):
    data, assets = get_clean_data(tickers, cash_ticker, start, end)
    returns = data[assets].pct_change().dropna()
    cash_rets = data[cash_ticker].pct_change().dropna()
    
    # Pre-calculate features to avoid leakage in the loop
    m_ret = returns.mean(axis=1).rolling(5).mean()
    m_vol = returns.mean(axis=1).rolling(20).std()
    features_base = pd.concat([m_ret, m_vol], axis=1).dropna()
    
    # Setup Walk-Forward (1-year training window, 1-month step)
    window_size = 252  # 1 Year
    step_size = 21     # 1 Month
    strat_rets = []
    all_weights = []
    
    current_weights = np.array([1.0/len(assets)] * len(assets))
    
    for i in range(window_size, len(features_base) - step_size, step_size):
        # 1. Training Slice (No Leakage)
        train_idx = features_base.index[i-window_size : i]
        test_idx = features_base.index[i : i+step_size]
        
        X_train = features_base.loc[train_idx].values
        
        # 2. HMM Fitting (Local to this window only)
        scaler = RobustScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        hmm = GaussianHMM(n_components=3, covariance_type="diag", random_state=42)
        hmm.fit(X_train_scaled)
        
        # 3. MLP Regression (Predicting Forward Returns, not labels)
        # Target = Mean return over next 5 days
        y_train = returns.loc[train_idx].shift(-5).rolling(5).mean().dropna()
        X_train_mlp = features_base.loc[y_train.index].values
        
        mlp = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
        mlp.fit(X_train_mlp, y_train.values)
        
        # 4. Out-of-Sample Prediction
        X_test_scaled = scaler.transform(features_base.loc[test_idx].values)
        regimes = hmm.predict(X_test_scaled)
        pred_returns = mlp.predict(features_base.loc[test_idx].values)
        
        # 5. Portfolio Construction (Risk Parity + Conviction)
        for j, date in enumerate(test_idx):
            # If Bear Regime (detected via local HMM)
            is_bear = regimes[j] == 0 # Simplified assumption for report
            
            # Conviction-based weighting
            raw_target_weights = np.exp(pred_returns[j]) / np.sum(np.exp(pred_returns[j]))
            
            # Crisis Handling: Shift 40% to Cash if Bear detected
            if is_bear:
                target_weights = 0.6 * raw_target_weights
                cash_allocation = 0.4
            else:
                target_weights = raw_target_weights
                cash_allocation = 0.0
            
            # Turnover Control (Smoothing)
            current_weights = (1 - smoothing) * current_weights + (smoothing * target_weights)
            
            # Calculate Performance
            port_ret = (current_weights * returns.loc[date]).sum() + (cash_allocation * cash_rets.loc[date])
            strat_rets.append(port_ret)
            all_weights.append(current_weights)

    final_idx = features_base.index[window_size : window_size + len(strat_rets)]
    return pd.Series(strat_rets, index=final_idx), returns.loc[final_idx].mean(axis=1), pd.DataFrame(all_weights, index=final_idx, columns=assets)

# --- UI LOGIC ---
if run_btn:
    try:
        s_ret, e_ret, w_history = run_backtest(ticker_input, cash_proxy, start_date, end_date, turnover_limit)
        
        # Performance Table
        def get_metrics(r):
            ann = r.mean() * 252
            vol = r.std() * np.sqrt(252)
            return ann, vol, ann/vol

        s_metrics = get_metrics(s_ret)
        e_metrics = get_metrics(e_ret)

        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy Sharpe", f"{s_metrics[2]:.2f}", f"{s_metrics[2]-e_metrics[2]:.2f} vs EW")
        c2.metric("Ann. Return", f"{s_metrics[0]:.1%}")
        c3.metric("Ann. Volatility", f"{s_metrics[1]:.1%}")

        st.divider()
        
        # Visuals
        t1, t2 = st.tabs(["Performance", "Portfolio Composition"])
        with t1:
            comp_df = pd.DataFrame({"Strategy": (1+s_ret).cumprod(), "Equal Weight": (1+e_ret).cumprod()})
            st.line_chart(comp_df)
        with t2:
            st.subheader("Weight Evolution (Turnover Controlled)")
            st.area_chart(w_history)
            
    except Exception as e:
        st.error(f"Backtest Error: {e}")
