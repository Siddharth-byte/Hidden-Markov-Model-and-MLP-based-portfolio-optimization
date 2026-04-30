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
st.set_page_config(page_title="Portfolio Opt", layout="wide")
st.title(" MLP and HMM based Portfolio Optimiser")

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
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252)
    sortino = ann_ret / downside_std if downside_std > 0 else 0

    peak = cum_ret_series.cummax()
    dd = (cum_ret_series - peak) / peak
    mdd = dd.min()

    calmar = ann_ret / abs(mdd) if mdd != 0 else 0
    hit = (returns > 0).mean()

    turnover = 0
    if weights_df is not None:
        turnover = weights_df.diff().abs().sum(axis=1).mean()

    return {
        "CAGR": cagr,
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max DD": mdd,
        "Calmar": calmar,
        "Hit Ratio": hit,
        "Avg Turnover": turnover
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
    if data.empty:
        raise ValueError("No data found")

    returns = data[assets].pct_change().dropna()
    cash_rets = data[cash_ticker].pct_change().dropna()

    # --- FEATURES ---
    m_ret = returns.mean(axis=1).rolling(5).mean()
    m_vol = returns.mean(axis=1).rolling(20).std()
    downside_vol = returns.clip(upper=0).mean(axis=1).rolling(20).std()
    trend = returns.mean(axis=1).rolling(50).mean()

    features = pd.concat([m_ret, m_vol, downside_vol], axis=1).dropna()
    features.columns = ['Mean_Ret', 'Vol', 'Downside_Vol']

    total_len = len(features)
    window = min(252, int(total_len * 0.4))
    step = max(1, int(total_len * 0.05))

    strat_rets, weights_hist, ml_metrics = [], [], []
    cur_weights = np.array([1.0 / len(assets)] * len(assets))

    for i in range(window, total_len - 1, step):
        train_idx = features.index[max(0, i-window):i]
        test_idx = features.index[i:min(i + step, total_len)]

        # --- HMM REGIME DETECTION ---
        scaler = RobustScaler().fit(features.loc[train_idx])
        X_train = scaler.transform(features.loc[train_idx])

        hmm = GaussianHMM(n_components=3, covariance_type="diag", random_state=42)
        hmm.fit(X_train)

        # Identify crisis state (lowest mean return in training sample)
        state_means = hmm.means_[:, 0]
        crisis_state = np.argmin(state_means)

        # --- MLP RETURN PREDICTION ---
        y_train = returns.loc[train_idx].shift(-2).rolling(2).mean().dropna()
        X_mlp = features.loc[y_train.index]

        mlp = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
        mlp.fit(X_mlp, y_train.values)

        # --- INFERENCE ---
        test_feat = features.loc[test_idx]
        regimes = hmm.predict(scaler.transform(test_feat))
        pred_rets = mlp.predict(test_feat)

        for j, date in enumerate(test_idx):
            pred = pred_rets[j]
            # Penalize negative return predictions to favor stability
            pred = np.where(pred < 0, pred * 2.0, pred)

            # Softmax with stability
            shift_pred = pred - np.max(pred)
            target = np.exp(shift_pred) / (np.sum(np.exp(shift_pred)) + 1e-9)

            # Risk Overlay
            is_crisis = regimes[j] == crisis_state
            allocation = 0.3 if is_crisis else 1.0

            # Trend Filter
            if date in trend.index and trend.loc[date] < 0:
                allocation *= 0.5

            # Apply Smoothing
            cur_weights = (1 - turn_limit) * cur_weights + (turn_limit * target)

            # Compute Day Return
            day_ret = (cur_weights * returns.loc[date] * allocation).sum() + \
                      ((1 - allocation) * cash_rets.loc[date])

            strat_rets.append(day_ret)
            weights_hist.append(list(cur_weights * allocation) + [1 - allocation])

            ml_metrics.append({
                "Date": date,
                "Regime_ID": int(regimes[j]),
                "Is_Crisis": 1 if is_crisis else 0
            })

    final_idx = features.index[window:window + len(strat_rets)]

    return (
        pd.Series(strat_rets, index=final_idx),
        returns.loc[final_idx].mean(axis=1),
        pd.DataFrame(weights_hist, index=final_idx, columns=assets + [cash_ticker]),
        sp500_all.loc[final_idx],
        pd.DataFrame(ml_metrics).set_index("Date")
    )

# --- UI ---
with st.sidebar:
    st.header("Configuration")
    ticker_input = st.text_input("Tickers", "AAPL, MSFT, NVDA, SPY, TLT")
    cash_proxy = st.selectbox("Cash Asset", ["BIL", "SHV"])
    start = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end = st.date_input("End Date", pd.to_datetime("2024-01-01"))
    smooth = st.slider("Turnover Limit", 0.01, 0.5, 0.15)
    run = st.button("Run Analysis")

if run:
    try:
        s_ret, e_ret, weights, sp_ret, ml_diag = run_backtest(
            ticker_input, cash_proxy, start, end, smooth
        )

        s_m = calculate_metrics(s_ret, weights)
        e_m = calculate_metrics(e_ret)
        sp_m = calculate_metrics(sp_ret)

        st.header("Results")
        
        # Display Metrics Table
        metrics_df = pd.DataFrame({
            "Strategy": s_m,
            "Equal Weight": e_m,
            "S&P 500": sp_m
        }).T
        st.table(metrics_df.style.format("{:.4f}"))

        tab1, tab2, tab3 = st.tabs(["Performance Charts", "ML Diagnostics", "Asset Weights"])

        with tab1:
            st.subheader("Cumulative Returns")
            comp_df = pd.DataFrame({
                "Strategy": (1+s_ret).cumprod(),
                "Benchmark S&P": (1+sp_ret).cumprod()
            })
            st.line_chart(comp_df)

        with tab2:
            st.subheader("Machine Learning Regime Detection")
            st.write("Current Regime ID tracking (Crisis state is identified per training window)")
            st.line_chart(ml_diag["Regime_ID"])
            
            st.subheader("Crisis State Boolean")
            st.bar_chart(ml_diag["Is_Crisis"])

        with tab3:
            st.subheader("Portfolio Allocation Evolution")
            st.area_chart(weights)
            
    except Exception as e:
        st.error(f"Error encountered: {e}")
