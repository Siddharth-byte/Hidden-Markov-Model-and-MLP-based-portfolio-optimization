import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
import plotly.express as px

# --- APP CONFIG ---
st.set_page_config(page_title="Portfolio AI Pro", layout="wide")
st.title("⚖️ Institutional-Grade Portfolio Optimizer")

with st.sidebar:
    st.header("1. Universe & Timeframe")
    ticker_input = st.text_input("Tickers", "AAPL, MSFT, NVDA, JNJ, SPY, TLT, GLD")
    cash_proxy = st.selectbox("Safe Haven Asset (Cash)", ["SHV", "BIL", "IEF"])
    start_date = st.date_input("Start Date", pd.to_datetime("2017-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))
    
    st.header("2. Optimization Controls")
    smoothing = st.slider("Turnover Control (Smoothing)", 0.01, 0.30, 0.10)
    st.info("Lower values reduce trading costs by preventing sudden weight flips.")
    run_btn = st.button("🚀 Run Backtest")

# --- DATA FETCHING ---
@st.cache_data
def get_data(tickers, cash, start, end):
    assets = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    df = yf.download(assets + [cash], start=start, end=end)['Close']
    valid_assets = df.columns[df.notna().sum() > (len(df) * 0.9)]
    return df[valid_assets].dropna(), [a for a in assets if a in valid_assets]

# --- BACKTEST ENGINE ---
@st.cache_data
def run_backtest(tickers, cash_ticker, start, end, turn_limit):
    data, assets = get_data(tickers, cash_ticker, start, end)
    returns = data[assets].pct_change().dropna()
    cash_rets = data[cash_ticker].pct_change().dropna()
    
    # Feature Engineering (Return momentum and volatility)
    m_ret = returns.mean(axis=1).rolling(5).mean()
    m_vol = returns.mean(axis=1).rolling(20).std()
    features = pd.concat([m_ret, m_vol], axis=1).dropna()
    
    window = 252 # 1 Year Training
    step = 21    # 1 Month Rebalance
    
    strat_rets = []
    weight_history = []
    cur_weights = np.array([1.0/len(assets)] * len(assets))
    
    # Walk-Forward Loop
    for i in range(window, len(features) - step, step):
        train_idx = features.index[i-window : i]
        test_idx = features.index[i : i+step]
        
        # 1. Local HMM (Fixes Regime Leakage)
        X_train = features.loc[train_idx].values
        scaler = RobustScaler().fit(X_train)
        hmm = GaussianHMM(n_components=3, covariance_type="diag", random_state=42)
        hmm.fit(scaler.transform(X_train))
        
        # 2. MLP Regression (Fixes Unstable Labels)
        y_train = returns.loc[train_idx].shift(-5).rolling(5).mean().dropna()
        X_mlp = features.loc[y_train.index].values
        mlp = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
        mlp.fit(X_mlp, y_train.values)
        
        # 3. Predict & Allocate
        test_feat = features.loc[test_idx].values
        regimes = hmm.predict(scaler.transform(test_feat))
        pred_rets = mlp.predict(test_feat)
        
        for j, date in enumerate(test_idx):
            # Softmax to get target weights from predicted returns
            target = np.exp(pred_rets[j]) / np.sum(np.exp(pred_rets[j]))
            
            # Regime-aware shift (Regime 0 often maps to high volatility)
            is_crisis = regimes[j] == 0
            allocation_to_assets = 0.6 if is_crisis else 1.0
            
            # Apply Smoothing (Turnover Control)
            cur_weights = (1 - turn_limit) * cur_weights + (turn_limit * target)
            
            # Execute
            day_ret = (cur_weights * returns.loc[date] * allocation_to_assets).sum() + \
                      ((1 - allocation_to_assets) * cash_rets.loc[date])
            
            strat_rets.append(day_ret)
            weight_history.append(cur_weights * allocation_to_assets)

    final_idx = features.index[window : window + len(strat_rets)]
    return pd.Series(strat_rets, index=final_idx), \
           returns.loc[final_idx].mean(axis=1), \
           pd.DataFrame(weight_history, index=final_idx, columns=assets), \
           (1 + returns.loc[final_idx]).cumprod()

# --- DASHBOARD UI ---
if run_btn:
    try:
        s_ret, e_ret, weights, asset_ts = run_backtest(ticker_input, cash_proxy, start_date, end_date, smoothing)
        
        # 1. KEY METRICS
        def get_stats(r):
            return r.mean()*252, r.std()*np.sqrt(252), (r.mean()*252)/(r.std()*np.sqrt(252))
        
        s_an, s_v, s_sh = get_stats(s_ret)
        e_an, e_v, e_sh = get_stats(e_ret)

        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy Sharpe", f"{s_sh:.2f}", f"{s_sh-e_sh:.2f} vs EW")
        c2.metric("Annualized Return", f"{s_an:.1%}")
        c3.metric("Max Daily Vol", f"{s_ret.std():.2%}")

        st.divider()

        # 2. PERFORMANCE CHARTS
        tab1, tab2, tab3 = st.tabs(["💰 Strategy Performance", "📊 Asset Time-Series", "⚖️ Weight Allocation"])
        
        with tab1:
            st.subheader("Cumulative Strategy Returns")
            main_df = pd.DataFrame({"AI Strategy": (1+s_ret).cumprod(), "Equal Weight": (1+e_ret).cumprod()})
            st.line_chart(main_df)

        with tab2:
            st.subheader("Individual Asset Performance (Test Period)")
            st.plotly_chart(px.line(asset_ts, labels={"value": "Growth", "index": "Date"}), use_container_width=True)

        with tab3:
            st.subheader("Evolution of Portfolio Weights")
            st.info("The smooth transitions below demonstrate effective Turnover Control.")
            st.area_chart(weights)
            
            st.subheader("Current Portfolio Snapshot")
            st.bar_chart(weights.iloc[-1])

    except Exception as e:
        st.error(f"Error during analysis: {e}")
