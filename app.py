import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Regime-Aware Portfolio AI", layout="wide", initial_sidebar_state="expanded")

st.title("🤖 HMM-MLP Portfolio Optimizer")
st.markdown("""
    This application uses a **Hidden Markov Model (HMM)** to identify market regimes and a 
    **Neural Network (MLP)** to allocate weights to the assets most likely to outperform in those regimes.
""")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("1. Strategy Settings")
    ticker_input = st.text_input("Enter Tickers (comma separated)", "AAPL, MSFT, NVDA, JNJ, SPY, TLT, GLD, QQQ")
    
    st.header("2. Date Range")
    start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))
    
    st.header("3. Model Parameters")
    n_regimes = st.slider("HMM Regimes", 2, 4, 3)
    run_optimization = st.button("🚀 Run AI Optimization")

# --- BACKEND LOGIC ---
def calculate_metrics(rets):
    """Calculates annualized metrics."""
    ann_return = rets.mean() * 252
    ann_risk = rets.std() * np.sqrt(252)
    sharpe = ann_return / ann_risk if ann_risk != 0 else 0
    return ann_return, ann_risk, sharpe

@st.cache_data
def fetch_and_optimize(tickers, start, end, n_states):
    # Data Fetching
    asset_list = [t.strip().upper() for t in tickers.split(",")]
    df = yf.download(asset_list, start=start, end=end)['Close']
    returns = df.pct_change().dropna()
    
    # HMM Regime Detection
    market_signal = returns.mean(axis=1).rolling(5).mean().dropna().values.reshape(-1, 1)
    scaler_hmm = StandardScaler()
    market_signal_scaled = scaler_hmm.fit_transform(market_signal)
    
    hmm = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000, random_state=42)
    hmm.fit(market_signal_scaled)
    regimes = hmm.predict(market_signal_scaled)
    
    # Feature Engineering for MLP
    features = pd.DataFrame({'regime': regimes}, index=returns.index[-len(regimes):])
    X = pd.get_dummies(features, columns=['regime']).astype(float)
    y = returns.iloc[-len(regimes):].shift(-1).dropna().idxmax(axis=1)
    X = X.loc[y.index]
    
    # Train/Test Split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # MLP Training
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), alpha=0.5, max_iter=2000, random_state=42)
    mlp.fit(X_train, y_train)
    
    # Weight Generation
    probs = mlp.predict_proba(X_test)
    weights_df = pd.DataFrame(probs, columns=mlp.classes_, index=X_test.index).reindex(columns=asset_list, fill_value=0)
    
    # Backtest Execution
    test_rets = returns.loc[X_test.index]
    strat_rets = (weights_df * test_rets).sum(axis=1)
    eq_rets = test_rets.mean(axis=1)
    
    return strat_rets, eq_rets, weights_df.iloc[-1], asset_list

# --- UI DASHBOARD ---
if run_optimization:
    try:
        with st.spinner("Analyzing Market States and Training Neural Network..."):
            strat_rets, eq_rets, final_weights, asset_list = fetch_and_optimize(ticker_input, start_date, end_date, n_regimes)
            
            # 1. Statistics Summary
            s_ret, s_risk, s_sharpe = calculate_metrics(strat_rets)
            e_ret, e_risk, e_sharpe = calculate_metrics(eq_rets)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Strategy Sharpe", f"{s_sharpe:.2f}", f"{s_sharpe - e_sharpe:.2f} vs Benchmark")
            col2.metric("EW Sharpe", f"{e_sharpe:.2f}")
            col3.metric("Expected Return", f"{s_ret:.2%}")
            col4.metric("Risk (Annual Vol)", f"{s_risk:.2%}")
            
            st.divider()
            
            # 2. Visualization Row
            vcol1, vcol2 = st.columns([1, 1.5])
            
            with vcol1:
                st.subheader("Final Optimized Weights")
                fig_pie = go.Figure(data=[go.Pie(labels=final_weights.index, values=final_weights.values, hole=.4)])
                fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # CSV Download
                csv = final_weights.to_csv().encode('utf-8')
                st.download_button("📥 Download Weights as CSV", data=csv, file_name="portfolio_weights.csv", mime='text/csv')

            with vcol2:
                st.subheader("Performance Comparison")
                perf_df = pd.DataFrame({
                    "AI Strategy": (1 + strat_rets).cumprod(),
                    "Equal Weight": (1 + eq_rets).cumprod()
                })
                st.line_chart(perf_df)
                
            # 3. Allocation Table
            st.subheader("Current Asset Allocation")
            weight_display = final_weights.to_frame(name="Weighting").sort_values(by="Weighting", ascending=False)
            st.table(weight_display.style.format("{:.2%}"))

    except Exception as e:
        st.error(f"Analysis Failed: {str(e)}")
        st.info("Ensure all tickers are valid Yahoo Finance symbols and there is enough data for the selected date range.")
