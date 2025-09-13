# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math

# --- Safe import of norm (scipy optional) ---
try:
    from scipy.stats import norm
except Exception:
    class _NormFallback:
        @staticmethod
        def cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        @staticmethod
        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)
    norm = _NormFallback()

# -------------------------
# Page config & CSS
# -------------------------
st.set_page_config(page_title="Analyst Terminal — Valuation & Options", page_icon="💹", layout="wide")

st.markdown("""
<style>
body { background-color: #0e1117; color: #e6e6e6; }
h1,h2,h3 { color: #39FF14; font-weight:700; }
.block-container { padding: 1rem 2rem; }
.metric-card { background: #111316; padding: 10px; border-radius: 8px; border: 1px solid #222; }
.css-1d391kg { background-color: #0b0c0e !important; }
.stDataFrame { background-color: #121416; }
a { color: #7ef9a4; }
</style>
""", unsafe_allow_html=True)

st.title("💹 Analyst Terminal — Equity Valuation & Options")
st.caption("Real-time modelling, DCF, comps, Black–Scholes options, and news. Built for IB/PE/AM prep. — [Navjot Dhah](https://www.linkedin.com/in/navjot-dhah-57870b238)")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Search & Settings")
ticker = st.sidebar.text_input("Enter ticker (example: AAPL, WYNN, MSFT)", value="WYNN").upper().strip()
use_live = st.sidebar.checkbox("Use live yfinance data", value=True)

# DCF defaults
default_wacc = st.sidebar.number_input("Default WACC (%)", value=9.0, step=0.1)/100.0
default_tg = st.sidebar.number_input("Default Terminal growth (%)", value=2.5, step=0.1)/100.0
projection_years = st.sidebar.selectbox("Projection years", [5,7,10], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("Tip: If yfinance misses fields, use manual overrides shown in each section.")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(ttl=300)
def fetch_yf(t):
    tk = yf.Ticker(t)
    try:
        info = tk.info
    except:
        info = {}
    try:
        fin = tk.financials
    except:
        fin = pd.DataFrame()
    try:
        bs = tk.balance_sheet
    except:
        bs = pd.DataFrame()
    try:
        cf = tk.cashflow
    except:
        cf = pd.DataFrame()
    try:
        hist = tk.history(period="5y")
    except:
        hist = pd.DataFrame()
    return info, fin, bs, cf, hist

def safe_number(x):
    try:
        return float(x)
    except:
        return np.nan

def style_numeric(df):
    if df is None or df.empty:
        return df
    df_t = df.T
    numeric_cols = df_t.select_dtypes(include=[np.number]).columns
    fmt = {c: "{:,.0f}" for c in numeric_cols}
    return df_t.style.format(fmt)

def find_row_value(df, keywords):
    if df is None or df.empty:
        return None
    idx = df.index
    for k in keywords:
        for label in idx:
            if k.lower() in str(label).lower():
                try:
                    return df.loc[label].iloc[0]
                except:
                    continue
    return None

def dcf_from_fcf(last_fcf, growth, discount, tg, years):
    proj = [last_fcf * (1 + growth)**i for i in range(1, years+1)]
    pv = sum([proj[i] / ((1 + discount)**(i+1)) for i in range(len(proj))])
    terminal = proj[-1]*(1+tg)/(discount-tg)/((1+discount)**years) if discount>tg else np.nan
    enterprise = pv + (terminal if not np.isnan(terminal) else 0)
    return {"proj_nominal": proj, "proj_pv": [proj[i]/((1+discount)**(i+1)) for i in range(len(proj))],
            "terminal_pv": terminal, "enterprise_value": enterprise}

def black_scholes_price(S, K, T, r, sigma, option="call"):
    if T <= 0 or sigma <= 0:
        return max(0.0,S-K) if option=="call" else max(0.0,K-S)
    d1 = (np.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    if option=="call":
        price = S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
    # Greeks
    delta = norm.cdf(d1) if option=="call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2 if option=="call" else -d2)
    vega = S*norm.pdf(d1)*np.sqrt(T)
    rho = K*T*np.exp(-r*T)*norm.cdf(d2 if option=="call" else -d2)
    return price, delta, gamma, theta, vega, rho

def get_yahoo_news(ticker, limit=6):
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
        resp = requests.get(url, timeout=6).json()
        items = resp.get("news",[]) or resp.get("items",[]) or []
        out=[]
        for it in items[:limit]:
            title = it.get("title") or it.get("headline")
            link = it.get("link") or it.get("url")
            pub = it.get("publisher") or it.get("provider") or it.get("source")
            ts = it.get("providerPublishTime") or it.get("pubDate") or None
            time = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M") if ts else ""
            out.append({"title": title, "link": link, "source": pub, "time": time})
        return out
    except:
        return []

# -------------------------
# Fetch data
# -------------------------
info, fin, bs, cf, hist = fetch_yf(ticker) if use_live else ({}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
company_name = info.get("shortName") or info.get("longName") or ticker
st.header(f"{company_name} — {ticker}")

price = safe_number(info.get("currentPrice") or (hist["Close"].iloc[-1] if not hist.empty else np.nan))
market_cap = safe_number(info.get("marketCap"))
shares_out = safe_number(info.get("sharesOutstanding") or info.get("floatShares") or np.nan)
ev = (market_cap or 0) + (safe_number(info.get("totalDebt")) or 0) - (safe_number(info.get("totalCash")) or 0)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Price", f"${price:,.2f}" if not np.isnan(price) else "N/A")
col2.metric("Market Cap", f"${market_cap:,.0f}" if not np.isnan(market_cap) else "N/A")
col3.metric("Enterprise Value", f"${ev:,.0f}" if not np.isnan(ev) else "N/A")
col4.metric("Shares Outstanding", f"{int(shares_out):,}" if not np.isnan(shares_out) else "N/A")
col5.metric("Sector / Industry", f"{info.get('sector','N/A')} / {info.get('industry','N/A')}")
st.markdown("---")

# Price chart
st.subheader("Price Chart")
if not hist.empty:
    fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
    fig.update_layout(template="plotly_dark", height=400, margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Price history not available.")

# Financial statements
st.subheader("Financial Statements")
f1,f2,f3 = st.columns(3)
with f1: st.markdown("**Income Statement**"); st.dataframe(style_numeric(fin), use_container_width=True)
with f2: st.markdown("**Balance Sheet**"); st.dataframe(style_numeric(bs), use_container_width=True)
with f3: st.markdown("**Cash Flow**"); st.dataframe(style_numeric(cf), use_container_width=True)

st.markdown("---")

# DCF interactive
st.subheader("DCF Valuation (Interactive)")
# derive last FCF
ocf_val = find_row_value(cf, ["operat","cash from operating"])
capex_val = find_row_value(cf, ["capital expend","purchase of property"])
last_fcf = (safe_number(ocf_val) + safe_number(capex_val)) if ocf_val is not None else st.number_input("Manual FCF", value=500_000_000.0, step=1000000.0, format="%.0f")
last_fcf = st.number_input("Use this FCF", value=float(last_fcf), format="%.0f", step=1000000.0)

g = st.slider("FCF CAGR (%)", -10.0, 30.0, 5.0)/100.0
d = st.slider("Discount Rate / WACC (%)", 0.1, 30.0, float(default_wacc*100))/100.0
tg = st.slider("Terminal Growth (%)", -2.0,6.0,float(default_tg*100))/100.0
years = st.selectbox("Projection years", [3,5,7,10], index=1)

result = dcf_from_fcf(last_fcf, g, d, tg, years)
ev_calc = result["enterprise_value"]
terminal_pv = result["terminal_pv"]
proj_pv = result["proj_pv"]
equity_val = ev_calc - (safe_number(info.get("totalDebt") or 0)) + (safe_number(info.get("totalCash") or 0))
implied_price = equity_val / shares_out if shares_out and shares_out>0 else np.nan

st.metric("Enterprise Value (DCF)", f"${ev_calc:,.0f}")
st.metric("Equity Value", f"${equity_val:,.0f}")
st.metric("Implied Price", f"${implied_price:,.2f}" if not np.isnan(implied_price) else "N/A")

# DCF contribution plot
fig_dcf = go.Figure()
fig_dcf.add_trace(go.Bar(x=[f"Y{i}" for i in range(1, years+1)], y=result["proj_pv"], name="Discounted FCF", marker_color="#00CC96"))
fig_dcf.add_trace(go.Bar(x=["Terminal"], y=[terminal_pv if terminal_pv else 0], name="Terminal PV", marker_color="#f5c518"))
fig_dcf.update_layout(template="plotly_dark", barmode="stack", title="DCF PV Contributions")
st.plotly_chart(fig_dcf, use_container_width=True)

st.markdown("---")

# Options Pricing with Greeks
st.subheader("Options Pricing (Black-Scholes + Greeks)")
col1, col2, col3, col4, col5 = st.columns(5)
S_default = price if not np.isnan(price) else 100.0
S = col1.number_input("Underlying Price (S)", value=float(S_default))
K = col2.number_input("Strike (K)", value=float(S_default))
days = col3.number_input("Days to Expiry", min_value=1,max_value=3650,value=30)
r = col4.number_input("Risk-free rate (%)", value=0.5)/100.0
if not hist.empty:
    hist_ret = hist["Close"].pct_change().dropna()
    sigma_est = hist_ret.rolling(21).std().dropna().iloc[-1]*np.sqrt(252) if len(hist_ret)>21 else hist_ret.std()*np.sqrt(252)
    sigma_est = float(sigma_est) if not np.isnan(sigma_est) else 0.25
else: sigma_est = 0.25
sigma = col5.number_input("Volatility σ", value=float(sigma_est), format="%.4f")
T = days/365.0

call_val, call_delta, call_gamma, call_theta, call_vega, call_rho = black_scholes_price(S,K,T,r,sigma,"call")
put_val, put_delta, put_gamma, put_theta, put_vega, put_rho = black_scholes_price(S,K,T,r,sigma,"put")
st.write(f"Call Price ≈ ${call_val:,.2f} | Put Price ≈ ${put_val:,.2f}")
st.write(f"Call Greeks: Δ={call_delta:.4f}, Γ={call_gamma:.4f}, θ={call_theta:.2f}, ν={call_vega:.2f}, ρ={call_rho:.4f}")
st.write(f"Put Greeks: Δ={put_delta:.4f}, Γ={put_gamma:.4f}, θ={put_theta:.2f}, ν={put_vega:.2f}, ρ={put_rho:.4f}")

st.markdown("---")

# News feed
st.subheader("Company News (Yahoo)")
news_items = get_yahoo_news(ticker, limit=8)
if news_items:
    for n in news_items:
        tlink = n.get("link") or "#"
        title = n.get("title") or "No title"
        source = n.get("source") or ""
        time = n.get("time") or ""
        st.markdown(f"- [{title}]({tlink}) <small>({source}) {time})</small>", unsafe_allow_html=True)
else:
    st.info("No news found.")

