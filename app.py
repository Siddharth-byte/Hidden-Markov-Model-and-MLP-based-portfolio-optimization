import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score

# --- APP CONFIG ---
st.set_page_config(page_title="ML Portfolio Optimizer", layout="wide")
st.title("MLP and HMM based Portfolio Optimiser")
st.caption("Active weighting based on MLP return predictions with HMM regime-switching risk overlay.")

# --- METRICS ---
def calculate_metrics(returns, weights_df=None):
    if returns.empty:
        return {k: 0 for k in ["CAGR","Ann. Return","Ann. Vol","Sharpe","Sortino","Max DD","Calmar","Hit Ratio","Avg Turnover"]}

    cum_ret_series = (1 + returns).cumprod()
    final_cum = cum_ret_series.iloc[-1]
    n_years = len(returns) / 252
    cagr = (final_cum ** (1 / n_years)) - 1 if n_years > 0 and final_cum > 0 else 0

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)

    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252)
    sortino = (ann_ret - 0.02) / downside_std if downside_std > 0 else 0

    peak = cum_ret_series.cummax()
    dd = (cum_ret_series - peak) / peak
    mdd = dd.min()

    calmar = ann_ret / abs(mdd) if mdd != 0 else 0
    hit = (returns > 0).mean()

    turnover = 0
    if weights_df is not None:
        turnover = weights_df.diff().abs().sum(axis=1).mean()

    return {
        "CAGR": cagr, "Ann. Return": ann_ret, "Ann. Vol": ann_vol,
        "Sharpe": sharpe, "Sortino": sortino, "Max DD": mdd,
        "Calmar": calmar, "Hit Ratio": hit, "Avg Turnover": turnover
    }

# --- DATA ---
@st.cache_data
def get_data(tickers, cash, start, end):
    assets = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    all_tickers = list(set(assets + [cash, "^GSPC"]))
    df = yf.download(all_tickers, start=start, end=end)['Close']

    if df.empty:
        return pd.DataFrame(), [], None

    valid = df.columns[df.notna().sum() > (len(df) * 0.5)]
    sp500 = df["^GSPC"].pct_change().dropna()

    return df[valid].dropna(), [a for a in assets if a in valid], sp500

# --- BACKTEST ---
@st.cache_data
def run_backtest(tickers, cash_ticker, start, end, turn_limit):
    data, assets, sp500_all = get_data(tickers, cash_ticker, start, end)

    if data.empty or len(assets) == 0:
        raise ValueError("No valid data available")

    returns = data[assets].pct_change().dropna()
    cash_rets = data[cash_ticker].pct_change().dropna()

    # Features: Rolling mean and volatility of the equal-weighted universe as a proxy for market state
    m_ret = returns.mean(axis=1).rolling(5).mean()
    m_vol = returns.mean(axis=1).rolling(20).std()
    features = pd.concat([m_ret, m_vol], axis=1).dropna()
    features.columns = ['Mean_Ret', 'Vol']

    total_len = len(features)
    window = min(252, int(total_len * 0.4))
    step = max(5, int(total_len * 0.05)) # Retrain every 5 days or 5% of data

    strat_rets, weights_hist, ml_logs = [], [], []
    cur_weights = np.array([1.0 / len(assets)] * len(assets))

    for i in range(window, total_len - 1, step):
        train_idx = features.index[max(0, i-window):i]
        test_idx = features.index[i:min(i + step, total_len)]

        if len(train_idx) < 40:
            continue

        # --- HMM (Regime Detection) ---
        X_train_df = features.loc[train_idx]
        scaler = RobustScaler().fit(X_train_df.values)
        X_train_scaled = scaler.transform(X_train_df.values)

        hmm = GaussianHMM(n_components=3, random_state=42)
        hmm.fit(X_train_scaled)

        # Crisis detection via variance
        covars = hmm.covars_
        vol_measure = [np.trace(c) if c.ndim == 2 else c for c in covars]
        crisis_state = np.argmax(vol_measure)

        # --- MLP (Predicting Individual Asset Returns) ---
        # Shift Y to predict NEXT day returns for each asset
        Y_train_assets = returns.loc[train_idx].shift(-1).dropna()
        X_mlp_train = features.loc[Y_train_assets.index]

        mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
        mlp.fit(X_mlp_train.values, Y_train_assets.values)

        # --- TESTING ON OUT-OF-SAMPLE WINDOW ---
        test_feat = features.loc[test_idx].dropna()
        if len(test_feat) == 0:
            continue

        # Get regime predictions and return predictions
        test_scaled = scaler.transform(test_feat.values)
        regimes = hmm.predict(test_scaled)
        pred_rets_all = mlp.predict(test_feat.values) 

        for j, date in enumerate(test_feat.index):
            preds = pred_rets_all[j]
            
            # --- SOFTMAX OPTIMIZATION ---
            # Instead of equal weight, we allocate based on predicted return magnitude
            # Using a temperature-scaled softmax to determine weights
            exp_preds = np.exp(preds / (np.std(preds) + 1e-6))
            target_weights = exp_preds / exp_preds.sum()

            # Regime Filter: If in crisis, reduce risky exposure significantly
            is_crisis = regimes[j] == crisis_state
            risk_multiplier = 0.15 if is_crisis else 1.0 

            # Apply Smoothing (Turnover/Inertia control)
            cur_weights = (1 - turn_limit) * cur_weights + (turn_limit * target_weights)

            # Portfolio Return = (Weighted Asset Returns * Risk Multiplier) + (Cash * Remaining)
            day_ret = (cur_weights * returns.loc[date] * risk_multiplier).sum() + \
                      ((1 - risk_multiplier) * cash_rets.loc[date])

            strat_rets.append(day_ret)
            
            # Record historical weights (including the cash component)
            weight_entry = list(cur_weights * risk_multiplier) + [1 - risk_multiplier]
            weights_hist.append(weight_entry)

            ml_logs.append({
                "Date": date,
                "Avg_Pred": np.mean(preds),
                "Regime": regimes[j]
            })

    final_idx = features.index[window:window + len(strat_rets)]
    weights_df = pd.DataFrame(weights_hist, index=final_idx, columns=assets + [cash_ticker])

    return (
        pd.Series(strat_rets, index=final_idx),
        returns.loc[final_idx].mean(axis=1),
        weights_df,
        sp500_all.loc[final_idx],
        pd.DataFrame(ml_logs).set_index("Date")
    )

# --- UI ---
with st.sidebar:
    st.header("Parameters")
    t_in = st.text_input("Tickers (Comma Separated)", "AAPL, MSFT, NVDA, SPY, GLD, TLT")
    c_in = st.selectbox("Cash Asset (Defense)", ["BIL", "SHV"])
    start = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end = st.date_input("End Date", pd.to_datetime("2024-01-01"))
    smooth = st.slider("Weight Adaptability (High = High Turnover)", 0.01, 0.5, 0.10)
    run = st.button("Run Optimizer")

if run:
    with st.spinner("Training Models and Backtesting..."):
        try:
            s_ret, e_ret, weights, sp_ret, ml_diag = run_backtest(t_in, c_in, start, end, smooth)
            s_m = calculate_metrics(s_ret, weights)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CAGR", f"{s_m['CAGR']:.2%}")
            c2.metric("Sharpe Ratio", f"{s_m['Sharpe']:.2f}")
            c3.metric("Sortino Ratio", f"{s_m['Sortino']:.2f}")
            c4.metric("Max Drawdown", f"{s_m['Max DD']:.2%}")

            st.divider()

            tab1, tab2, tab3 = st.tabs(["Performance Analysis", "ML Insights", "Allocation History"])

            with tab1:
                comparison_df = pd.DataFrame({
                    "AI Strategy": (1+s_ret).cumprod(),
                    "S&P 500 (Benchmark)": (1+sp_ret).cumprod(),
                    "Equal Weight Universe": (1+e_ret).cumprod()
                })
                st.line_chart(comparison_df)
                
                st.subheader("Current Allocation")
                latest = weights.tail(1).T
                latest.columns = ["Weight"]
                st.bar_chart(latest)

            with tab2:
                st.subheader("Model Indicators")
                col1, col2 = st.columns(2)
                col1.metric("Avg Predicted Return", f"{ml_diag['Avg_Pred'].mean():.4%}")
                
                st.write("Regime History (0=Low Vol, 2=High Vol/Crisis)")
                st.line_chart(ml_diag["Regime"])

            with tab3:
                st.subheader("Portfolio Composition Over Time")
                st.area_chart(weights)
                st.dataframe(weights.tail(10).style.format("{:.2%}"))

        except Exception as e:
            st.error(f"Error during execution: {e}")
