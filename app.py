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
st.set_page_config(page_title="Portfolio Optimiser with HMM and MLP", layout="wide")
st.title(" MLP & HMM based Portfolio Optimizer")

RISK_FREE_RATE = 0.02  # 2% annual

# --- UTILITY FUNCTIONS ---
def calculate_metrics(returns, weights_df=None):
    if returns.empty:
        return {k: 0 for k in ["CAGR","Ann. Return","Ann. Vol","Sharpe","Sortino","Max DD","Calmar","Hit Ratio","Avg Turnover"]}

    # CAGR
    cum_ret = (1 + returns).cumprod().iloc[-1]
    years = len(returns) / 252
    cagr = (cum_ret ** (1 / years)) - 1 if years > 0 else 0

    # Annual stats
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)

    # Sharpe
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252)
    sortino = (ann_ret - RISK_FREE_RATE) / downside_std if downside_std > 0 else 0

    # Drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = dd.min()

    calmar = cagr / abs(mdd) if mdd != 0 else 0

    # Hit ratio
    hit = (returns > 0).mean()

    # Turnover
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

    valid = df.columns[df.notna().sum() > len(df)*0.5]
    sp = df["^GSPC"].pct_change().dropna()

    return df[valid].dropna(), [a for a in assets if a in valid], sp

# --- BACKTEST ---
@st.cache_data
def run_backtest(tickers, cash_ticker, start, end, turn_limit):
    data, assets, sp_all = get_data(tickers, cash_ticker, start, end)
    if data.empty:
        raise ValueError("No data")

    returns = data[assets].pct_change().dropna()
    cash_rets = data[cash_ticker].pct_change().dropna()

    # Features
    m_ret = returns.mean(axis=1).rolling(5).mean()
    m_vol = returns.mean(axis=1).rolling(20).std()
    features = pd.concat([m_ret, m_vol], axis=1).dropna()

    total_len = len(features)
    window = min(252, int(total_len * 0.4))
    step = max(1, int(total_len * 0.05))

    strat_rets, weight_hist, ml_diag = [], [], []
    cur_w = np.array([1/len(assets)] * len(assets))

    for i in range(window, total_len-1, step):
        train_idx = features.index[max(0, i-window):i]
        test_idx = features.index[i:min(i+step, total_len)]

        # HMM
        scaler = RobustScaler().fit(features.loc[train_idx])
        X_train = scaler.transform(features.loc[train_idx])

        hmm = GaussianHMM(n_components=3, covariance_type="diag", random_state=42)
        hmm.fit(X_train)

        # Determine crisis state via highest variance
        state_vars = [np.trace(cov) for cov in hmm.covars_]
        crisis_state = np.argmax(state_vars)

        # MLP
        y_train = returns.loc[train_idx].shift(-5).rolling(5).mean().dropna()
        X_mlp = features.loc[y_train.index]

        mlp = MLPRegressor(hidden_layer_sizes=(16,8), max_iter=1000, random_state=42)
        mlp.fit(X_mlp, y_train.values)

        # Predict
        X_test = scaler.transform(features.loc[test_idx])
        regimes = hmm.predict(X_test)
        pred = mlp.predict(features.loc[test_idx])

        actual = returns.loc[test_idx].mean(axis=1).values
        pred_mean = pred.mean(axis=1)
        mae = mean_absolute_error(actual, pred_mean)

        for j, date in enumerate(test_idx):
            # Stable softmax
            exp_vals = np.exp(pred[j] - np.max(pred[j]))
            target = exp_vals / np.sum(exp_vals)

            is_crisis = regimes[j] == crisis_state
            alloc = 0.6 if is_crisis else 1.0

            cur_w = (1-turn_limit)*cur_w + turn_limit*target

            day_ret = (cur_w * returns.loc[date] * alloc).sum() + \
                      ((1-alloc) * cash_rets.loc[date])

            strat_rets.append(day_ret)
            weight_hist.append(list(cur_w * alloc) + [1-alloc])
            ml_diag.append({"Date": date, "MAE": mae, "Regime": regimes[j]})

    final_idx = features.index[window:window+len(strat_rets)]

    # Equal Weight (FIXED)
    ew_w = np.array([1/len(assets)] * len(assets))
    ew_ret = (returns.loc[final_idx] * ew_w).sum(axis=1)

    w_df = pd.DataFrame(weight_hist, index=final_idx, columns=assets+[cash_ticker])

    return (
        pd.Series(strat_rets, index=final_idx),
        ew_ret,
        w_df,
        (1 + data[assets].loc[final_idx].pct_change().dropna()).cumprod(),
        sp_all.loc[final_idx],
        pd.DataFrame(ml_diag).set_index("Date")
    )

# --- UI ---
with st.sidebar:
    st.header("Settings")
    tickers = st.text_input("Tickers", "AAPL, MSFT, NVDA, SPY")
    cash = st.selectbox("Cash", ["BIL","SHV","IEF"])
    start = st.date_input("Start", pd.to_datetime("2023-01-01"))
    end = st.date_input("End", pd.to_datetime("2024-05-01"))
    smooth = st.slider("Turnover", 0.01, 0.5, 0.15)
    run = st.button("Run")

if run:
    s_ret, ew_ret, weights, asset_ts, sp_ret, diag = run_backtest(tickers, cash, start, end, smooth)

    s_m = calculate_metrics(s_ret, weights)
    ew_m = calculate_metrics(ew_ret)
    sp_m = calculate_metrics(sp_ret)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("CAGR", f"{s_m['CAGR']:.2%}")
    c2.metric("Sortino", f"{s_m['Sortino']:.2f}")
    c3.metric("Max DD", f"{s_m['Max DD']:.2%}")
    c4.metric("Turnover", f"{s_m['Avg Turnover']:.2%}")

    st.divider()

    comp = pd.DataFrame({"Strategy": s_m, "Equal Weight": ew_m, "S&P 500": sp_m})
    st.table(comp.style.format({
        "CAGR":"{:.2%}",
        "Ann. Return":"{:.2%}",
        "Ann. Vol":"{:.2%}",
        "Max DD":"{:.2%}",
        "Hit Ratio":"{:.2%}",
        "Avg Turnover":"{:.2%}",
        "Sharpe":"{:.2f}",
        "Sortino":"{:.2f}",
        "Calmar":"{:.2f}"
    }))

    tab1,tab2,tab3 = st.tabs(["Performance","ML","Weights"])

    with tab1:
        st.line_chart(pd.DataFrame({
            "Strategy": (1+s_ret).cumprod(),
            "Equal Weight": (1+ew_ret).cumprod(),
            "S&P 500": (1+sp_ret).cumprod()
        }))

    with tab2:
        st.line_chart(diag["MAE"])
        st.scatter_chart(diag["Regime"])

    with tab3:
        st.area_chart(weights)
        st.table(weights.tail(1).style.format("{:.2%}"))
