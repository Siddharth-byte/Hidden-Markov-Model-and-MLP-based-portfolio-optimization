import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score

# --- APP CONFIG ---
st.set_page_config(page_title="Portfolio Opt", layout="wide")
st.title("MLP and HMM based Portfolio Optimiser")

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

    # Features
    m_ret = returns.mean(axis=1).rolling(5).mean()
    m_vol = returns.mean(axis=1).rolling(20).std()
    features = pd.concat([m_ret, m_vol], axis=1).dropna()
    features.columns = ['Mean_Ret', 'Vol']

    total_len = len(features)
    window = min(252, int(total_len * 0.4))
    step = max(1, int(total_len * 0.05))

    strat_rets, weights_hist, ml_logs = [], [], []
    cur_weights = np.array([1.0 / len(assets)] * len(assets))

    for i in range(window, total_len - 1, step):
        train_idx = features.index[max(0, i-window):i]
        test_idx = features.index[i:min(i + step, total_len)]

        if len(train_idx) < 20:
            continue

        # --- HMM ---
        X_train_df = features.loc[train_idx].dropna()
        if len(X_train_df) < 20:
            continue

        scaler = RobustScaler().fit(X_train_df.values)
        X_train = scaler.transform(X_train_df.values)

        hmm = GaussianHMM(n_components=3, random_state=42)
        hmm.fit(X_train)

        # Detect crisis regime (highest variance)
        covars = hmm.covars_
        if covars.ndim == 3:
            vol_measure = [np.trace(c) for c in covars]
        else:
            vol_measure = covars

        crisis_state = np.argmax(vol_measure)
        log_likelihood = hmm.score(X_train)

        # --- MLP ---
        y_train = returns.loc[train_idx].shift(-1).mean(axis=1).dropna()
        X_mlp = features.loc[y_train.index]

        if len(X_mlp) < 20:
            continue

        mlp = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
        mlp.fit(X_mlp.values, y_train.values)

        # --- TEST ---
        test_feat = features.loc[test_idx].dropna()
        if len(test_feat) == 0:
            continue

        test_array = test_feat.values
        regimes = hmm.predict(scaler.transform(test_array))
        pred_rets = mlp.predict(test_array)

        actual_test_rets = returns.loc[test_feat.index].mean(axis=1).values
        mse = mean_squared_error(actual_test_rets, pred_rets)
        r2 = r2_score(actual_test_rets, pred_rets) if len(actual_test_rets) > 1 else 0
        dir_acc = np.mean(np.sign(actual_test_rets) == np.sign(pred_rets))

        for j, date in enumerate(test_feat.index):
            pred_scalar = pred_rets[j]

            # Convert scalar → portfolio weights
            target = np.ones(len(assets)) * (1 + pred_scalar)
            target = target / target.sum()

            is_crisis = regimes[j] == crisis_state
            allocation = 0.4 if is_crisis else 1.0

            cur_weights = (1 - turn_limit) * cur_weights + (turn_limit * target)

            day_ret = (cur_weights * returns.loc[date] * allocation).sum() + \
                      ((1 - allocation) * cash_rets.loc[date])

            strat_rets.append(day_ret)
            weights_hist.append(list(cur_weights * allocation) + [1 - allocation])

            ml_logs.append({
                "Date": date,
                "MSE": mse,
                "R2": r2,
                "Dir_Accuracy": dir_acc,
                "Log_Likelihood": log_likelihood
            })

    final_idx = features.index[window:window + len(strat_rets)]

    return (
        pd.Series(strat_rets, index=final_idx),
        returns.loc[final_idx].mean(axis=1),
        pd.DataFrame(weights_hist, index=final_idx, columns=assets + [cash_ticker]),
        sp500_all.loc[final_idx],
        pd.DataFrame(ml_logs).set_index("Date")
    )

# --- UI ---
with st.sidebar:
    st.header("Parameters")
    t_in = st.text_input("Tickers", "AAPL, MSFT, NVDA, SPY, GLD")
    c_in = st.selectbox("Cash", ["BIL", "SHV"])
    start = st.date_input("Start", pd.to_datetime("2021-01-01"))
    end = st.date_input("End", pd.to_datetime("2024-01-01"))
    smooth = st.slider("Smoothing", 0.01, 0.5, 0.15)
    run = st.button("Execute")

if run:
    s_ret, e_ret, weights, sp_ret, ml_diag = run_backtest(t_in, c_in, start, end, smooth)
    s_m = calculate_metrics(s_ret, weights)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"### CAGR\n## {s_m['CAGR']:.2%}")
    c2.markdown(f"### Sharpe\n## {s_m['Sharpe']:.2f}")
    c3.markdown(f"### Sortino\n## {s_m['Sortino']:.2f}")
    c4.markdown(f"### Max DD\n## {s_m['Max DD']:.2%}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Performance", "ML Diagnostics", "Portfolio Weights"])

    with tab1:
        st.line_chart(pd.DataFrame({
            "Strategy": (1+s_ret).cumprod(),
            "S&P 500": (1+sp_ret).cumprod()
        }))
        st.subheader("Latest Allocations")
        st.table(weights.tail(1).style.format("{:.2%}"))

    with tab2:
        st.subheader("Diagnostics")
        col1, col2 = st.columns(2)
        col1.metric("Avg MSE", f"{ml_diag['MSE'].mean():.6f}")
        col2.metric("Directional Accuracy", f"{ml_diag['Dir_Accuracy'].mean():.2%}")

        st.line_chart(ml_diag[["Log_Likelihood", "R2"]])

    with tab3:
        st.area_chart(weights)
