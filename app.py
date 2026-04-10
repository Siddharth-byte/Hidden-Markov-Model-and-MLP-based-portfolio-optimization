import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
import plotly.express as px

# --- APP CONFIG ---
st.set_page_config(page_title="Pro-Grade Portfolio Optimizer", layout="wide")
st.title("🛡️ Risk-Managed HMM-MLP Strategy")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Strategy Settings")
    ticker_input = st.text_input("Tickers", "AAPL, MSFT, NVDA, JNJ, SPY, TLT, GLD, QQQ")
    cash_proxy = st.selectbox("Cash Proxy (Defensive)", ["SHV", "BIL", "IEF"])
    
    # Re-adding the HMM Slider
    n_regimes = st.slider("Number of Market Regimes", 2, 5, 3)
    
    start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))
    run_btn = st.button("Run Improved Optimization")

# --- TECHNICAL INDICATORS ---
def add_indicators(df):
    returns = df.pct_change()
    change = returns.mean(axis=1)
    gain = (change.where(change > 0, 0)).rolling(window=14).mean()
    loss = (-change.where(change < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9) # Added epsilon to prevent div by zero
    rsi = 100 - (100 / (1 + rs))
    sma200 = df.mean(axis=1).rolling(200).mean()
    dist_sma = (df.mean(axis=1) / (sma200 + 1e-9)) - 1
    return rsi, dist_sma

# --- CORE STRATEGY ---
@st.cache_data
def run_improved_strategy(tickers, cash_ticker, start, end, n_states):
    assets = [t.strip().upper() for t in tickers.split(",")]
    all_tickers = assets + [cash_ticker]
    data = yf.download(all_tickers, start=start, end=end)['Close'].dropna()
    
    returns = data[assets].pct_change().dropna()
    cash_returns = data[cash_ticker].pct_change().dropna()
    
    # 1. HMM Input Alignment FIX
    m_ret = returns.mean(axis=1).rolling(5).mean()
    m_vol = returns.mean(axis=1).rolling(20).std()
    
    # The fix: Aligning by dropping all NaNs across both series
    hmm_features = pd.concat([m_ret, m_vol], axis=1).dropna()
    hmm_features.columns = ['ret', 'vol']
    
    scaler_hmm = RobustScaler() 
    hmm_input_scaled = scaler_hmm.fit_transform(hmm_features.values)
    
    hmm = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000, random_state=42)
    hmm.fit(hmm_input_scaled)
    regime_probs = hmm.predict_proba(hmm_input_scaled)
    
    # 2. Advanced Feature Engineering Alignment
    rsi, dist_sma = add_indicators(data[assets])
    
    # Create a feature dataframe and align everything to the same index
    features = pd.DataFrame(regime_probs, index=hmm_features.index, columns=[f'P_Regime_{i}' for i in range(n_states)])
    features['RSI'] = rsi
    features['SMA_Dist'] = dist_sma
    features = features.dropna()
    
    # Target Alignment
    # We find the winning asset over the next 5 days
    y_raw = returns.shift(-5).rolling(5).mean().idxmax(axis=1)
    common_index = features.index.intersection(y_raw.dropna().index)
    
    X = features.loc[common_index]
    y = y_raw.loc[common_index]
    
    # 3. Walk-Forward Logic
    tscv = TimeSeriesSplit(n_splits=5)
    strat_rets = []
    
    for train_ix, test_ix in tscv.split(X):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train = y.iloc[train_ix]
        
        mlp = MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.1, max_iter=2000, random_state=42)
        mlp.fit(X_train, y_train)
        
        probs = mlp.predict_proba(X_test)
        weights_df = pd.DataFrame(probs, columns=mlp.classes_, index=X_test.index).reindex(columns=assets, fill_value=0)
        
        # Risk Switch: If the first regime probability (usually the high-vol one) is > 40%
        bear_filter = X_test.iloc[:, 0] > 0.4
        
        test_returns = returns.loc[X_test.index]
        test_cash = cash_returns.loc[X_test.index]
        
        for date in X_test.index:
            w, r, c = weights_df.loc[date].values, test_returns.loc[date].values, test_cash.loc[date]
            if bear_filter.loc[date]:
                daily_perf = (0.5 * (w * r).sum()) + (0.5 * c)
            else:
                daily_perf = (w * r).sum()
            strat_rets.append(daily_perf)

    test_period_index = X.index[-len(strat_rets):]
    individual_cum_rets = (1 + returns.loc[test_period_index]).cumprod()

    return (pd.Series(strat_rets, index=test_period_index), 
            returns.loc[test_period_index].mean(axis=1), 
            weights_df.iloc[-1], 
            individual_cum_rets)

# --- UI EXECUTION ---
if run_btn:
    with st.spinner("Aligning Data and Running Optimization..."):
        try:
            strat, ew, last_w, asset_cum_rets = run_improved_strategy(ticker_input, cash_proxy, start_date, end_date, n_regimes)
            
            # Statistics
            def stats(r):
                return r.mean()*252, r.std()*np.sqrt(252), (r.mean()*252)/(r.std()*np.sqrt(252))
            
            s_ret, s_vol, s_sh = stats(strat)
            e_ret, e_vol, e_sh = stats(ew)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Strategy Sharpe", f"{s_sh:.2f}", f"{s_sh - e_sh:.2f} vs EW")
            c2.metric("Annualized Volatility", f"{s_vol:.2%}")
            c3.metric("Annualized Return", f"{s_ret:.2%}")
            
            st.divider()

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Strategy vs. Benchmark")
                perf_comp = pd.DataFrame({"AI Strategy": (1+strat).cumprod(), "Equal Weight": (1+ew).cumprod()})
                st.line_chart(perf_comp)
            with col_b:
                st.subheader("Individual Asset Returns")
                fig_assets = px.line(asset_cum_rets, labels={"value": "Cumulative Return", "index": "Date"})
                st.plotly_chart(fig_assets, use_container_width=True)

            st.subheader("Current Regime-Based Weights")
            st.bar_chart(last_w)
            
        except Exception as e:
            st.error(f"Error encountered: {e}")
