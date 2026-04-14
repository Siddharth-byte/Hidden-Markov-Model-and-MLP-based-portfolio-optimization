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
st.title("MLP based Portfolio Optimizer")

with st.sidebar:
    st.header("1. Universe & Timeframe")
    ticker_input = st.text_input("Tickers", "AAPL, MSFT, NVDA, JNJ, SPY, TLT, GLD")
    cash_proxy = st.selectbox("Safe Haven Asset (Cash)", ["SHV", "BIL", "IEF"])
    
    # Date selection
    start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))
    
    st.header("2. Optimization Controls")
    smoothing = st.slider("Turnover Control (Smoothing)", 0.01, 0.30, 0.10)
    run_btn = st.button("Run Backtest")

# --- DATA FETCHING ---
@st.cache_data
def get_data(tickers, cash, start, end):
    assets = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    # Changed from 'Adj Close' to 'Close' per your requirement
    df = yf.download(assets + [cash], start=start, end=end)['Close']
    
    if df.empty:
        return pd.DataFrame(), []
        
    # Validation: Ensure we have enough rows for calculations
    valid_assets = df.columns[df.notna().sum() > (len(df) * 0.5)]
    return df[valid_assets].dropna(), [a for a in assets if a in valid_assets]

# --- BACKTEST ENGINE ---
@st.cache_data
def run_backtest(tickers, cash_ticker, start, end, turn_limit):
    data, assets = get_data(tickers, cash_ticker, start, end)
    if data.empty:
        raise ValueError("No data found for the selected range.")

    returns = data[assets].pct_change().dropna()
    cash_rets = data[cash_ticker].pct_change().dropna()
    
    m_ret = returns.mean(axis=1).rolling(5).mean()
    m_vol = returns.mean(axis=1).rolling(20).std()
    features = pd.concat([m_ret, m_vol], axis=1).dropna()
    
    # DYNAMIC WINDOW FIX: Use 40% of data for training if timeframe is short
    total_len = len(features)
    window = min(252, int(total_len * 0.4)) 
    step = max(1, int(total_len * 0.05)) # Small steps for short periods
    
    if window < 20:
        raise ValueError("Time period is too short for model training. Please select at least 3-4 months.")
    
    strat_rets = []
    weight_history = []
    cur_weights = np.array([1.0/len(assets)] * len(assets))
    
    for i in range(window, total_len - 1, step):
        train_idx = features.index[max(0, i-window) : i]
        # Predict the next 'step' of days
        test_end = min(i + step, total_len)
        test_idx = features.index[i : test_end]
        
        # 1. Local HMM
        X_train = features.loc[train_idx].values
        scaler = RobustScaler().fit(X_train)
        hmm = GaussianHMM(n_components=3, covariance_type="diag", random_state=42)
        hmm.fit(scaler.transform(X_train))
        
        # 2. MLP Regression
        y_train = returns.loc[train_idx].shift(-5).rolling(5).mean().dropna()
        X_mlp = features.loc[y_train.index].values
        mlp = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
        mlp.fit(X_mlp, y_train.values)
        
        # 3. Predict & Allocate
        test_feat = features.loc[test_idx].values
        regimes = hmm.predict(scaler.transform(test_feat))
        pred_rets = mlp.predict(test_feat)
        
        for j, date in enumerate(test_idx):
            target = np.exp(pred_rets[j]) / np.sum(np.exp(pred_rets[j]))
            is_crisis = regimes[j] == 0
            allocation_to_assets = 0.6 if is_crisis else 1.0
            
            # Turnover Control
            cur_weights = (1 - turn_limit) * cur_weights + (turn_limit * target)
            
            day_ret = (cur_weights * returns.loc[date] * allocation_to_assets).sum() + \
                      ((1 - allocation_to_assets) * cash_rets.loc[date])
            
            strat_rets.append(day_ret)
            weight_history.append(list(cur_weights * allocation_to_assets) + [1 - allocation_to_assets])

    final_idx = features.index[window : window + len(strat_rets)]
    cols = assets + [cash_ticker]
    return pd.Series(strat_rets, index=final_idx), \
           returns.loc[final_idx].mean(axis=1), \
           pd.DataFrame(weight_history, index=final_idx, columns=cols), \
           (1 + data[assets].loc[final_idx].pct_change().dropna()).cumprod()

# --- DASHBOARD UI ---
if run_btn:
    try:
        s_ret, e_ret, weights, asset_ts = run_backtest(ticker_input, cash_proxy, start_date, end_date, smoothing)
        
        # Metrics
        ann_ret = s_ret.mean() * 252
        ann_vol = s_ret.std() * np.sqrt(252)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy Sharpe", f"{(ann_ret/ann_vol if ann_vol !=0 else 0):.2f}")
        c2.metric("Annualized Return", f"{ann_ret:.1%}")
        c3.metric("Annualized Volatility", f"{ann_vol:.1%}")

        st.divider()

        tab1, tab2, tab3 = st.tabs(["Performance", "Asset Time-Series", "Asset Weights"])
        
        with tab1:
            st.subheader("Strategy Cumulative Growth")
            st.line_chart(pd.DataFrame({"Strategy": (1+s_ret).cumprod(), "Equal Weight": (1+e_ret).cumprod()}))

        with tab2:
            st.subheader("Asset Price Growth (Test Period)")
            st.plotly_chart(px.line(asset_ts), use_container_width=True)

        with tab3:
            st.subheader("Latest Calculated Asset Weights")
            # Presenting the Weight Table as requested
            latest_weights = weights.iloc[-1].to_frame().T
            latest_weights.index = ["Current Allocation (%)"]
            st.table(latest_weights.style.format("{:.2%}"))
            
            st.subheader("Weight Evolution Over Time")
            st.area_chart(weights)
            
    except Exception as e:
        st.error(f"Error: {e}")
