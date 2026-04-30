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
st.title("Portfolio AI Pro: Improved HMM + MLP Optimizer")

# --- METRICS ---
def calculate_metrics(returns, weights_df=None):
    if returns.empty:
        return {k: 0 for k in ["CAGR","Ann. Return","Ann. Vol","Sharpe","Sortino","Max DD","Calmar","Hit Ratio","Avg Turnover"]}

    cum_ret = (1 + returns).cumprod().iloc[-1]
    n_years = len(returns) / 252
    cagr = (cum_ret ** (1 / n_years)) - 1 if n_years > 0 else 0

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252)
    sortino = ann_ret / downside_std if downside_std > 0 else 0

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
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
    downside = returns.clip(upper=0).rolling(20).std()
    trend = returns.mean(axis=1).rolling(50).mean()

    features = pd.concat([m_ret, m_vol, downside], axis=1).dropna()

    total_len = len(features)
    window = min(252, int(total_len * 0.4))
    step = max(1, int(total_len * 0.05))

    strat_rets, weights_hist, ml_metrics = [], [], []
    cur_weights = np.array([1.0 / len(assets)] * len(assets))

    for i in range(window, total_len - 1, step):
        train_idx = features.index[max(0, i-window):i]
        test_idx = features.index[i:min(i + step, total_len)]

        # --- HMM ---
        scaler = RobustScaler().fit(features.loc[train_idx])
        X_train = scaler.transform(features.loc[train_idx])

        hmm = GaussianHMM(n_components=3, covariance_type="diag", random_state=42)
        hmm.fit(X_train)

        # Identify crisis state (lowest return mean)
        state_means = hmm.means_[:, 0]
        crisis_state = np.argmin(state_means)

        # --- MLP ---
        y_train = returns.loc[train_idx].shift(-2).rolling(2).mean().dropna()
        X_mlp = features.loc[y_train.index]

        mlp = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
        mlp.fit(X_mlp, y_train.values)

        # --- TEST ---
        test_feat = features.loc[test_idx]
        regimes = hmm.predict(scaler.transform(test_feat))
        pred_rets = mlp.predict(test_feat)

        for j, date in enumerate(test_idx):

            pred = pred_rets[j]

            # --- Penalize negative predictions ---
            pred = np.where(pred < 0, pred * 2, pred)

            # --- Softmax allocation ---
            target = np.exp(pred) / np.sum(np.exp(pred))

            # --- Regime-based risk ---
            is_crisis = regimes[j] == crisis_state
            allocation = 0.3 if is_crisis else 1.0

            # --- Trend filter ---
            if trend.loc[date] < 0:
                allocation *= 0.5

            # --- Smooth weights ---
            cur_weights = (1 - turn_limit) * cur_weights + (turn_limit * target)

            # --- Daily return ---
            day_ret = (cur_weights * returns.loc[date] * allocation).sum() + \
                      ((1 - allocation) * cash_rets.loc[date])

            strat_rets.append(day_ret)
            weights_hist.append(list(cur_weights * allocation) + [1 - allocation])

            ml_metrics.append({
                "Date": date,
                "Regime": regimes[j]
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
    ticker_input = st.text_input("Tickers", "AAPL, MSFT, NVDA, SPY, TLT")
    cash_proxy = st.selectbox("Cash", ["BIL", "SHV"])
    start = st.date_input("Start", pd.to_datetime("2020-01-01"))
    end = st.date_input("End", pd.to_datetime("2024-01-01"))
    smooth = st.slider("Turnover", 0.01, 0.5, 0.15)

    run = st.button("Run")

if run:
    s_ret, e_ret, weights, sp_ret, ml_diag = run_backtest(
        ticker_input, cash_proxy, start, end, smooth
    )

    s_m = calculate_metrics(s_ret, weights)
    e_m = calculate_metrics(e_ret)
    sp_m = calculate_metrics(sp_ret)

    st.subheader("Performance")
    st.write(pd.DataFrame({
        "Strategy": s_m,
        "Equal Weight": e_m,
        "S&P 500": sp_m
    }))

    st.subheader("Equity Curve")
    st.line_chart(pd.DataFrame({
        "Strategy": (1+s_ret).cumprod(),
        "S&P": (1+sp_ret).cumprod()
    }))

    st.subheader("Weights")
    st.area_chart(weights)
