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
    n_regimes = st.slider("Number of Market Regimes", 2, 5, 3)
    
    st.subheader("Timeframe")
    st.info("Note: Use at least 1 year of data for the 200-day indicators to work.")
    start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))
    run_btn = st.button("Run Improved Optimization")

# --- TECHNICAL INDICATORS ---
def add_indicators(df):
    returns = df.pct_change()
    change = returns.mean(axis=1)
    gain = (change.where(change > 0, 0)).rolling(window=14).mean()
    loss = (-change.where(change < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    
    # Check if we have enough data for 200-day SMA
    if len(df) > 200:
        sma_window = 200
    else:
        sma_window = len(df) // 2 # Fallback to half the data length
        
    sma = df.mean(axis=1).rolling(window=sma_window).mean()
    dist_sma = (df.mean(axis=1) / (sma + 1e-9)) - 1
    return rsi, dist_sma

# --- CORE STRATEGY ---
@st.cache_data
def run_improved_strategy(tickers, cash_ticker, start, end, n_states):
    assets = [t.strip().upper() for t in tickers.split(",")]
    all_tickers = assets + [cash_ticker]
    
    # 1. Robust Data Fetching
    raw_data = yf.download(all_tickers, start=start, end=end)['Close']
    
    if raw_data.empty:
        raise ValueError("No data found for these tickers/dates.")
    
    # Drop columns that are entirely NaN, then drop rows with any NaN
    data = raw_data.dropna(axis=1, how='all').dropna()
    
    if len(data) < 50:
        raise ValueError(f"Insufficient data points ({len(data)}). Increase your date range.")

    # Re-identify available assets after cleaning
    available_assets = [a for a in assets if a in data.columns]
    returns = data[available_assets].pct_change().dropna()
    cash_returns = data[cash_ticker].pct_change().dropna()
    
    # 2. HMM Logic
    m_ret = returns.mean(axis=1).rolling(5).mean()
    m_vol = returns.mean(axis=1).rolling(20).std()
    hmm_features = pd.concat([m_ret, m_vol], axis=1).dropna()
    
    scaler_hmm = RobustScaler() 
    hmm_input_scaled = scaler_hmm.fit_transform(hmm_features.values)
    
    hmm = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000, random_state=42)
    hmm.fit(hmm_input_scaled)
    regime_probs = hmm.predict_proba(hmm_input_scaled)
    
    # 3. Indicators & Alignment
    rsi, dist_sma = add_indicators(data[available_assets])
    features = pd.DataFrame(regime_probs, index=hmm_features.index, columns=[f'P_Regime_{i}' for i in range(n_states)])
    features['RSI'] = rsi
    features['SMA_Dist'] = dist_sma
    features = features.dropna()
    
    # Shift forward to predict the "Winner" of the next week
    y_raw = returns.shift(-5).rolling(5).mean().idxmax(axis=1)
    common_index = features.index.intersection(y_raw.dropna().index)
    
    if len(common_index) < 20:
        raise ValueError("Not enough overlapping data for training. Try a longer date range.")
        
    X = features.loc[common_index]
    y = y_raw.loc[common_index]
    
    # 4. Walk-Forward
    tscv = TimeSeriesSplit(n_splits=3 if len(X) < 500 else 5)
    strat_rets = []
    
    for train_ix, test_ix in tscv.split(X):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train = y.iloc[train_ix]
        
        mlp = MLPClassifier(hidden_layer_sizes=(16, 8), alpha=0.5, max_iter=1000, random_state=42)
        mlp.fit(X_train, y_train)
        
        probs = mlp.predict_proba(X_test)
        weights_df = pd.DataFrame(probs, columns=mlp.classes_, index=X_test.index).reindex(columns=available_assets, fill_value=0)
        
        # Risk Switch (Regime 0 is typically the high-vol/low-return state)
        bear_filter = X_test.iloc[:, 0] > 0.4
        
        test_returns = returns.loc[X_test.index]
        test_cash = cash_returns.loc[X_test.index]
        
        for date in X_test.index:
            w, r, c = weights_df.loc[date].values, test_returns.loc[date].values, test_cash.loc[date]
            strat_rets.append((0.5 * (w * r).sum() + 0.5 * c) if bear_filter.loc[date] else (w * r).sum())

    test_idx = X.index[-len(strat_rets):]
    return (pd.Series(strat_rets, index=test_idx), 
            returns.loc[test_idx].mean(axis=1), 
            weights_df.iloc[-1], 
            (1 + returns.loc[test_idx]).cumprod())

# --- UI EXECUTION ---
if run_btn:
    with st.spinner("Crunching data..."):
        try:
            strat, ew, last_w, asset_cum_rets = run_improved_strategy(ticker_input, cash_proxy, start_date, end_date, n_regimes)
            
            # Summary Metrics
            s_ret = strat.mean() * 252
            s_vol = strat.std() * np.sqrt(252)
            s_sharpe = s_ret / s_vol if s_vol != 0 else 0
            
            e_ret = ew.mean() * 252
            e_vol = ew.std() * np.sqrt(252)
            e_sharpe = e_ret / e_vol if e_vol != 0 else 0
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Strategy Sharpe", f"{s_sharpe:.2f}", f"{s_sharpe - e_sharpe:.2f}")
            m2.metric("Annual Volatility", f"{s_vol:.2%}")
            m3.metric("Annual Return", f"{s_ret:.2%}")
            
            st.divider()
            
            c_left, c_right = st.columns(2)
            with c_left:
                st.subheader("Cumulative Growth")
                st.line_chart(pd.DataFrame({"Strategy": (1+strat).cumprod(), "Benchmark": (1+ew).cumprod()}))
            with c_right:
                st.subheader("Asset Performance")
                st.plotly_chart(px.line(asset_cum_rets), use_container_width=True)
                
            st.subheader("Current Weights")
            st.bar_chart(last_w)
            
        except Exception as e:
            st.error(f"Error: {e}")
