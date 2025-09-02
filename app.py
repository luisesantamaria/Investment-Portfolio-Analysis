from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Iterable, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

st.set_page_config(page_title="Efficient Frontier — Portfolio Optimization", layout="wide")
alt.data_transformers.disable_max_rows()

BASE = Path(".").resolve()
DATA_DIR = BASE / "data"
CANDIDATES = [DATA_DIR / "ConstituentsWithYahoo.csv", DATA_DIR / "constituents_with_yahoo.csv"]
UNIVERSE_PATH = next((p for p in CANDIDATES if p.exists()), None)

@st.cache_data(show_spinner=False)
def load_universe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["index","yahoo","name","status"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {path}")
    df = df.astype({"index":"string","yahoo":"string","name":"string","status":"string"})
    ok_counts = df[df["status"]=="ok"].groupby("index")["yahoo"].nunique()
    valid_indices = ok_counts[ok_counts>=2].index.tolist()
    return df[df["index"].isin(valid_indices)].copy()

const_df = load_universe(UNIVERSE_PATH) if UNIVERSE_PATH else pd.DataFrame()
TICKER2NAME = (const_df.dropna(subset=["yahoo","name"])
               .drop_duplicates(subset=["yahoo"])
               .set_index("yahoo")["name"].to_dict()) if not const_df.empty else {}

st.sidebar.header("Configuration")

st.sidebar.markdown(
    """
    <style>
    /* Control de selección cerrado y labels */
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        font-size: 12px !important;
        line-height: 1.15em !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        font-size: 12px !important;
    }
    /* POPUP (menú abierto) — selectores amplios y con !important */
    .stApp div[data-baseweb="popover"],
    .stApp div[data-baseweb="popover"] *,
    .stApp div[data-baseweb="popover"] div[data-baseweb="menu"],
    .stApp div[data-baseweb="popover"] div[role="listbox"],
    .stApp div[data-baseweb="popover"] ul[role="listbox"],
    .stApp div[data-baseweb="popover"] [role="option"],
    .stApp div[data-baseweb="popover"] [role="option"] * {
        font-size: 12px !important;
        line-height: 1.15em !important;
    }
    .stApp div[data-baseweb="popover"] [role="option"] {
        padding-top: 6px !important;
        padding-bottom: 6px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

indices = sorted(const_df["index"].unique()) if not const_df.empty else []
selected_index = st.sidebar.selectbox("Index", indices) if indices else None

df_ok = const_df[(const_df["index"]==selected_index)&(const_df["status"]=="ok")] if selected_index else pd.DataFrame()

if not df_ok.empty:
    opts_display = (df_ok[["name","yahoo"]]
                    .dropna()
                    .drop_duplicates()
                    .assign(display=lambda d: d["name"].str.strip() + " (" + d["yahoo"].str.strip() + ")"))
    display2ticker = dict(zip(opts_display["display"], opts_display["yahoo"]))
    options_for_user = sorted(opts_display["display"].tolist())
else:
    display2ticker = {}
    options_for_user = []

selected_display = st.sidebar.multiselect("Companies", options=options_for_user, default=[])
selected_tickers = [display2ticker[d] for d in selected_display]

years = st.sidebar.slider("Lookback (years)", 1, 10, 5, 1)

rf_pct = st.sidebar.number_input(
    "Risk-free (%)", value=0.00, step=1.00, format="%.2f"
)
rf_annual = float(rf_pct) / 100.0

use_log_returns = st.sidebar.checkbox("Use log returns", value=True)

try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

def _utc_now():
    return datetime.now(timezone.utc)

@st.cache_data(show_spinner=True, ttl=3600)
def fetch_prices_live(ticker: str, years: int, interval: str="1d") -> pd.DataFrame:
    if not YF_OK:
        raise RuntimeError("yfinance not available.")
    end_dt = _utc_now()
    start_dt = end_dt - timedelta(days=365*years + 7)
    tk = yf.Ticker(ticker)
    df_raw = tk.history(start=start_dt.date(), end=None, interval=interval, auto_adjust=False)
    if df_raw is None or df_raw.empty:
        df_raw = tk.history(period=f"{max(1, years)}y", interval=interval, auto_adjust=False)
    if df_raw is None or df_raw.empty:
        raise RuntimeError(f"No price data for {ticker}")

    df = df_raw.copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx

    out = pd.DataFrame(index=df.index)
    out["close"] = df.get("Close", df.get("close"))
    out["adj_close"] = df.get("Adj Close", df.get("adjclose", out["close"]))
    out["volume"] = df.get("Volume", df.get("volume"))
    out = out.sort_index().loc[~out.index.duplicated(keep="last")]
    return out.dropna(how="all")

@st.cache_data(show_spinner=True, ttl=3600)
def get_prices(tickers: Iterable[str], years: int, interval: str="1d"):
    cols, rows, errs = [], [], {}
    for t in tickers:
        try:
            df = fetch_prices_live(t, years, interval)
            cols.append(df["adj_close"].rename(t))
            tmp = df.copy()
            tmp = tmp.assign(date=pd.to_datetime(tmp.index).tz_localize(None), ticker=t).reset_index(drop=True)
            rows.append(tmp[["ticker","date","adj_close","close","volume"]])
        except Exception as e:
            errs[t] = str(e)
    wide = pd.concat(cols, axis=1) if cols else pd.DataFrame()
    if not wide.empty:
        wide = wide.sort_index().dropna(axis=1, how="all")
    long = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    return wide, long, errs

TRADING_DAYS = 252
MIN_OVERLAP_DAYS = 2*252

def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index()
    if "date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "date"})
    out["date"] = pd.to_datetime(out["date"])
    return out

def port_perf(w: np.ndarray, mu_ann: pd.Series, cov_ann: pd.DataFrame) -> Tuple[float,float]:
    r = float(np.dot(w, mu_ann.values))
    v = float(np.sqrt(np.dot(w, np.dot(cov_ann.values, w))))
    return r, v

def sharpe_ann(r: float, v: float, rf: float) -> float:
    return np.nan if v<=0 else (r - rf)/v

def sortino_ann(rp_daily: pd.Series, rf_daily: float, rf_annual: float) -> float:
    ex = rp_daily - rf_daily
    dd = np.minimum(0.0, ex).std(ddof=1)
    if dd<=0 or np.isnan(dd): return np.nan
    rp_ann = rp_daily.mean()*TRADING_DAYS
    return (rp_ann - rf_annual) / (dd*np.sqrt(TRADING_DAYS))

def treynor_ann(rp_daily: pd.Series, rm_daily: pd.Series, rf_annual: float) -> float:
    var_m = rm_daily.var(ddof=1)
    if var_m<=0: return np.nan
    beta = rp_daily.cov(rm_daily) / var_m
    if beta==0 or np.isnan(beta): return np.nan
    rp_ann = rp_daily.mean()*TRADING_DAYS
    return (rp_ann - rf_annual) / beta

def solve_min_variance(mu_ann: pd.Series, cov_ann: pd.DataFrame, n: int) -> np.ndarray:
    bounds = [(0,1)]*n
    cons = ({'type':'eq','fun': lambda w: np.sum(w)-1.0},)
    if SCIPY_OK:
        x0 = np.full(n, 1/n)
        res = minimize(lambda w: float(np.dot(w, np.dot(cov_ann.values, w))),
                       x0=x0, bounds=bounds, constraints=cons, method='SLSQP')
        if res.success: return res.x
        return x0
    rng = np.random.default_rng(42); best_w, best_v = None, np.inf
    for _ in range(20000):
        w = rng.random(n); w /= w.sum()
        _, v = port_perf(w, mu_ann, cov_ann)
        if v < best_v: best_v, best_w = v, w
    return best_w

def solve_max_sharpe(mu_ann: pd.Series, cov_ann: pd.DataFrame, rf_ann: float, n: int) -> np.ndarray:
    bounds = [(0,1)]*n
    cons = ({'type':'eq','fun': lambda w: np.sum(w)-1.0},)
    if SCIPY_OK:
        def neg_sharpe(w):
            r,v = port_perf(w, mu_ann, cov_ann)
            s = sharpe_ann(r, v, rf_ann)
            return -s if np.isfinite(s) else 1e6
        x0 = np.full(n, 1/n)
        res = minimize(neg_sharpe, x0=x0, bounds=bounds, constraints=cons, method='SLSQP')
        if res.success: return res.x
        return x0
    rng = np.random.default_rng(123); best_w, best_s = None, -np.inf
    for _ in range(30000):
        w = rng.random(n); w /= w.sum()
        r,v = port_perf(w, mu_ann, cov_ann); s = sharpe_ann(r, v, rf_ann)
        if s>best_s: best_s, best_w = s, w
    return best_w

def _clean_w(w: np.ndarray) -> np.ndarray:
    w = np.clip(w,0,1); s = w.sum()
    return w/s if s>0 else np.full_like(w, 1/len(w))

def _as_vega(ch: alt.Chart) -> dict:
    try:
        spec = ch.to_dict()
    except Exception:
        import json
        spec = json.loads(ch.to_json())
    return spec

def show_chart(ch: alt.Chart, use_container_width: bool = True):
    placeholder = st.empty()
    spec = _as_vega(ch)
    placeholder.vega_lite_chart(spec, use_container_width=use_container_width)

PORTFOLIO_ORDER = ["Max Sharpe", "Min Variance", "Baseline (User)"]
PORTFOLIO_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]  # Azul MS, Rojo MV, Verde Baseline
PORTFOLIO_SHAPES  = ["triangle-up", "triangle-down", "square"]

st.title("Investment Portfolio Analysis")

if UNIVERSE_PATH is None:
    st.error("Missing constituents file. Put it at data/ConstituentsWithYahoo.csv or data/constituents_with_yahoo.csv")

issues = {}
prices_wide = pd.DataFrame()
assets = []
data_ready = False

if selected_index is None:
    st.info("Select an index in the sidebar to begin.")
elif len(selected_tickers) < 2:
    st.info("Select at least two tickers to compute allocations and render charts.")
else:
    with st.spinner("Fetching Yahoo Finance prices..."):
        prices_wide, prices_long, issues = get_prices(selected_tickers, years, "1d")

    usable = [c for c in prices_wide.columns if prices_wide[c].dropna().size >= MIN_OVERLAP_DAYS]
    if len(usable) >= 2:
        prices_wide = prices_wide[usable].dropna(how="any")
        assets = list(prices_wide.columns)
        data_ready = True
    else:
        st.warning("Not enough overlapping history across selected tickers. Try different tickers or reduce the lookback.")
        if issues:
            st.caption(f"Non-fatal fetch issues: {issues}")

if data_ready:
    rets = (np.log(prices_wide / prices_wide.shift(1)) if use_log_returns else prices_wide.pct_change()).dropna()
    mu = rets.mean()*TRADING_DAYS
    cov = rets.cov()*TRADING_DAYS
    corr = rets.corr()

    RF_ANNUAL = rf_annual
    RF_DAILY = RF_ANNUAL / TRADING_DAYS

    W_MIN = _clean_w(solve_min_variance(mu, cov, len(assets)))
    W_MAX = _clean_w(solve_max_sharpe(mu, cov, RF_ANNUAL, len(assets)))

    st.caption(f"Index: {selected_index} | Selected tickers: {', '.join(assets)} | Lookback: {years}y")

    st.subheader("Asset Allocation")

    header_cols = st.columns([0.55, 0.25, 0.20])
    header_cols[0].markdown("**Asset**")
    header_cols[1].markdown("**Ticker**")
    header_cols[2].markdown("**Weight (%)**")

    weights_pct = []
    for t in assets:
        name = TICKER2NAME.get(t, t)
        c = st.columns([0.55, 0.25, 0.20])
        c[0].markdown(f"{name}")
        c[1].markdown(
            f"<span style='font-family:monospace; font-size:16px; font-weight:700;'>{t}</span>",
            unsafe_allow_html=True
        )
        val = c[2].number_input(
            label=f"weight_{t}",
            min_value=0.0, max_value=100.0,
            value=float(100/len(assets)),
            step=1.00,              
            format="%.2f",        
            key=f"w_{t}",
            label_visibility="collapsed",
            help="Usa las flechas (+/−) para 1 punto, o escribe decimales (p. ej., 12.50)."
        )
        weights_pct.append(val)

    w_user = np.array(weights_pct, dtype=float) / 100.0
    w_sum = float(w_user.sum())
    n_assets = len(assets)

    if n_assets >= 3:
        valid_sum = (0.999 <= w_sum <= 1.0000001)
    else:
        valid_sum = (abs(w_sum - 1.0) <= 1e-6)


    if not valid_sum:
        st.error("Adjust the weights: the sum must meet 100%. "
                 "Charts and metrics are enabled when the sum is valid.")
        st.stop()

    if w_sum > 0:
        w_user = w_user / w_sum

    def port_series(w: np.ndarray) -> pd.Series: return rets.dot(w)
    r_mkt = port_series(np.full(len(assets), 1/len(assets)))

    def perf_summary(w: np.ndarray) -> Dict[str,float]:
        r_ann, v_ann = port_perf(w, mu, cov)
        rp = port_series(w)
        return dict(
            expected_return=r_ann,
            volatility=v_ann,
            sharpe=sharpe_ann(r_ann, v_ann, RF_ANNUAL),
            sortino=sortino_ann(rp, RF_DAILY, RF_ANNUAL),
            treynor=treynor_ann(rp, r_mkt, RF_ANNUAL)
        )

    perf_user   = perf_summary(w_user)
    perf_minvar = perf_summary(W_MIN)
    perf_maxshp = perf_summary(W_MAX)

    c1, c2 = st.columns([1.1, 1.0])
    with c1:
        st.markdown("**Portfolio Allocation**")
        W = pd.DataFrame({
            "Min Variance": pd.Series(W_MIN, index=assets),
            "Max Sharpe":   pd.Series(W_MAX, index=assets),
            "Baseline (User)": pd.Series(w_user, index=assets),
        })
        W.index = [TICKER2NAME.get(t,t) for t in W.index]
        W_pct = W.applymap(lambda x: f"{x:.2%}")
        st.dataframe(W_pct, use_container_width=True)

    with c2:
        st.markdown("**Performance summary**")
        M = pd.DataFrame({
            "Max Sharpe":   perf_maxshp,
            "Min Variance": perf_minvar,
            "Baseline (User)": perf_user
        })
        M_fmt = M.copy()
        for metric in ["expected_return","volatility"]:
            M_fmt.loc[metric] = (M.loc[metric]*100).map(lambda v: f"{v:.2f}%")
        for metric in ["sharpe","sortino","treynor"]:
            M_fmt.loc[metric] = M.loc[metric].map(lambda v: f"{v:.4f}")
        M_fmt = M_fmt.rename(index={
            "expected_return":"Expected return",
            "volatility":"Volatility",
            "sharpe":"Sharpe",
            "sortino":"Sortino",
            "treynor":"Treynor"
        })
        st.dataframe(M_fmt, use_container_width=True)

        legend_html = """
        <div style="display:flex; gap:14px; align-items:center; margin-top:8px;">
          <div style="display:flex; align-items:center; gap:6px;">
            <span style="width:12px;height:12px;background:#1f77b4;display:inline-block;border-radius:2px;"></span>
            <span>Max Sharpe</span>
          </div>
          <div style="display:flex; align-items:center; gap:6px;">
            <span style="width:12px;height:12px;background:#d62728;display:inline-block;border-radius:2px;"></span>
            <span>Min Variance</span>
          </div>
          <div style="display:flex; align-items:center; gap:6px;">
            <span style="width:12px;height:12px;background:#2ca02c;display:inline-block;border-radius:2px;"></span>
            <span>Baseline (User)</span>
          </div>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)

    st.subheader("Portfolio Performance")
    def nav_from_returns(rp: pd.Series, initial: float=100.0) -> pd.Series:
        return initial * (np.exp(rp.cumsum()) if use_log_returns else (1.0+rp).cumprod())
    roi_df = pd.concat([
        (nav_from_returns(port_series(W_MIN))/100-1).rename("Min Variance"),
        (nav_from_returns(port_series(W_MAX))/100-1).rename("Max Sharpe"),
        (nav_from_returns(port_series(w_user))/100-1).rename("Baseline (User)"),
    ], axis=1)
    roi_long = ensure_date_column(roi_df).melt(id_vars="date", var_name="portfolio", value_name="roi_pct")

    ch_roi = (
        alt.Chart(roi_long).mark_line(size=2.3)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("roi_pct:Q", title="ROI (%)", axis=alt.Axis(format="%")),
            color=alt.Color("portfolio:N",
                            title="Portfolio",
                            sort=PORTFOLIO_ORDER,
                            scale=alt.Scale(domain=PORTFOLIO_ORDER, range=PORTFOLIO_COLORS)),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("portfolio:N"), alt.Tooltip("roi_pct:Q", format=".2%")]
        ).properties(height=360)
    )
    show_chart(ch_roi, use_container_width=True)

    st.subheader("Correlation Matrix")
    corr_df = corr.copy()
    corr_df.index = [TICKER2NAME.get(t,t) for t in corr_df.index]
    corr_df.columns = [TICKER2NAME.get(t,t) for t in corr_df.columns]
    corr_long = corr_df.reset_index().melt(id_vars="index", var_name="asset_y", value_name="rho").rename(columns={"index":"asset_x"})

    ch_heat = (
        alt.Chart(corr_long).mark_rect()
        .encode(
            x=alt.X("asset_x:N", sort=None, title=""),
            y=alt.Y("asset_y:N", sort=None, title=""),
            color=alt.Color("rho:Q", title="ρ", scale=alt.Scale(scheme="redblue", domain=[-1,1])),
            tooltip=["asset_x:N","asset_y:N",alt.Tooltip("rho:Q", format=".2f")]
        ).properties(height=420)
    )
    ch_text = (
        alt.Chart(corr_long).mark_text(fontSize=12)
        .encode(
            x="asset_x:N", y="asset_y:N",
            text=alt.Text("rho:Q", format=".2f"),
            color=alt.condition("abs(datum.rho) > 0.5", alt.value("white"), alt.value("black"))
        )
    )
    show_chart(ch_heat + ch_text, use_container_width=True)

    st.subheader("Efficient Frontier")
    def min_var_given_return(target_ret: float) -> Optional[np.ndarray]:
        if not SCIPY_OK:
            return None
        cons_sum = {'type':'eq','fun': lambda w: np.sum(w)-1.0}
        cons_ret = {'type':'eq','fun': lambda w, tr=target_ret: float(np.dot(mu.values, w)) - tr}
        x0 = np.full(len(assets), 1/len(assets))
        res = minimize(lambda w: float(np.dot(w, np.dot(cov.values, w))),
                       x0=x0, bounds=[(0,1)]*len(assets), constraints=(cons_sum, cons_ret), method='SLSQP')
        return res.x if res.success else None

    def random_cloud(n=10000):
        rng = np.random.default_rng()
        vols = np.empty(n); rets_sim = np.empty(n)
        for i in range(n):
            w = rng.random(len(assets)); w /= w.sum()
            r,v = port_perf(w, mu, cov)
            vols[i] = v; rets_sim[i] = r
        return vols, rets_sim

    def frontier_points(num=60):
        r_minvar, _ = port_perf(W_MIN, mu, cov)
        r_max_tgt = float(mu.max())
        targets = np.linspace(r_minvar, r_max_tgt, num=num)
        vols, rets_list = [], []
        for tr in targets:
            w = min_var_given_return(tr)
            if w is None:
                rng = np.random.default_rng(); best_w, best_v = None, np.inf
                for _ in range(6000):
                    cand = rng.random(len(assets)); cand /= cand.sum()
                    r,v = port_perf(cand, mu, cov)
                    if abs(r-tr)<1e-3 and v<best_v: best_v, best_w = v, cand
                w = best_w
            if w is not None:
                r,v = port_perf(w, mu, cov)
                rets_list.append(r); vols.append(v)
        return np.array(vols), np.array(rets_list)

    vol_c, ret_c = random_cloud(10000)
    vol_f, ret_f = frontier_points(60)

    r_min, v_min = port_perf(W_MIN, mu, cov)
    r_max, v_max = port_perf(W_MAX, mu, cov)
    r_usr, v_usr = port_perf(w_user, mu, cov)

    df_cloud = pd.DataFrame({"vol":vol_c, "ret":ret_c}).dropna()
    df_front = pd.DataFrame({"vol":vol_f, "ret":ret_f}).dropna().sort_values("vol")

    df_pts = pd.DataFrame({
        "portfolio": ["Max Sharpe", "Min Variance", "Baseline (User)"],
        "vol": [v_max, v_min, v_usr],
        "ret": [r_max, r_min, r_usr]
    })

    ch_cloud = alt.Chart(df_cloud).mark_circle(opacity=0.18, size=10).encode(
        x=alt.X("vol:Q", title="Volatility (annualized)", axis=alt.Axis(format="%")),
        y=alt.Y("ret:Q", title="Expected return (annualized)", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("vol:Q", format=".2%"), alt.Tooltip("ret:Q", format=".2%")]
    )
    ch_front = alt.Chart(df_front).mark_line(size=3).encode(
        x=alt.X("vol:Q", axis=alt.Axis(format="%")),
        y=alt.Y("ret:Q", axis=alt.Axis(format="%"))
    )
    ch_pts = (
        alt.Chart(df_pts)
        .mark_point(size=200, filled=True)
        .encode(
            x=alt.X("vol:Q", axis=alt.Axis(format="%")),
            y=alt.Y("ret:Q", axis=alt.Axis(format="%")),
            shape=alt.Shape("portfolio:N",
                            scale=alt.Scale(domain=PORTFOLIO_ORDER, range=PORTFOLIO_SHAPES),
                            legend=alt.Legend(title="Portfolios")),
            color=alt.Color("portfolio:N",
                            scale=alt.Scale(domain=PORTFOLIO_ORDER, range=PORTFOLIO_COLORS),
                            legend=alt.Legend(title="Portfolios")),
            tooltip=["portfolio:N", alt.Tooltip("vol:Q", format=".2%"), alt.Tooltip("ret:Q", format=".2%")]
        )
    )
    show_chart(ch_cloud + ch_front + ch_pts, use_container_width=True)

    st.subheader("Historical Performance — Portfolio & Assets")

    if use_log_returns:
        nav_assets = 100.0*np.exp(rets.cumsum())
    else:
        nav_assets = 100.0*(1.0+rets).cumprod()
    roi_assets = nav_assets/100.0 - 1.0

    def nav_from_returns_hist(rp: pd.Series) -> pd.Series:
        return 100.0*(np.exp(rp.cumsum()) if use_log_returns else (1.0+rp).cumprod())

    roi_min = nav_from_returns_hist(rets.dot(W_MIN))/100.0 - 1.0
    roi_max = nav_from_returns_hist(rets.dot(W_MAX))/100.0 - 1.0
    roi_usr = nav_from_returns_hist(rets.dot(w_user))/100.0 - 1.0

    final_roi_assets = roi_assets.iloc[-1].sort_values(ascending=False)
    ordered_assets = final_roi_assets.index.tolist()
    asset_labels = [f"{TICKER2NAME.get(t, t)} ({t})" for t in ordered_assets]

    pf_df = pd.concat([
        roi_max.rename("Max Sharpe"),
        roi_min.rename("Min Variance"),
        roi_usr.rename("Baseline (User)")
    ], axis=1)
    pf_long = ensure_date_column(pf_df).melt(id_vars="date", var_name="series", value_name="roi")
    pf_long["is_portfolio"] = True

    assets_long = roi_assets.reset_index().rename(columns={roi_assets.index.name or "index":"date"})
    if "date" not in assets_long.columns:
        assets_long = assets_long.rename(columns={assets_long.columns[0]:"date"})
    assets_long = assets_long.melt("date", var_name="ticker", value_name="roi")
    assets_long["series"] = assets_long["ticker"].map(lambda t: f"{TICKER2NAME.get(t, t)} ({t})")
    assets_long = assets_long.drop(columns=["ticker"])
    assets_long["is_portfolio"] = False

    series_long = pd.concat([pf_long, assets_long], axis=0, ignore_index=True)

    SERIES_ORDER = ["Max Sharpe", "Min Variance", "Baseline (User)"] + asset_labels

    color_range = ["#1f77b4", "#d62728", "#2ca02c"] + ["#9aa0a6"] * len(asset_labels)

    ch_hist = (
        alt.Chart(series_long)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("roi:Q", title="ROI (%)", axis=alt.Axis(format="%")),
            color=alt.Color("series:N",
                            sort=SERIES_ORDER,
                            scale=alt.Scale(domain=SERIES_ORDER, range=color_range),
                            legend=alt.Legend(title="Series (ordered)")),
            size=alt.Size("is_portfolio:N",
                          scale=alt.Scale(domain=[True, False], range=[2.6, 1.2]),
                          legend=None),
            opacity=alt.Opacity("is_portfolio:N",
                                scale=alt.Scale(domain=[True, False], range=[1.0, 0.18]),
                                legend=None),
            tooltip=[alt.Tooltip("date:T"),
                     alt.Tooltip("series:N"),
                     alt.Tooltip("roi:Q", format=".2%")]
        )
        .properties(height=420)
    )

    ch_zero = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(strokeDash=[6,4], color="black", opacity=0.8).encode(y="y:Q")
    show_chart(ch_hist + ch_zero, use_container_width=True)

st.caption("Data source: Yahoo Finance (on demand). Constituents validated and mapped to Yahoo Finance symbols.")
