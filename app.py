# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
import math

# --- Safe import of norm ---
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
.stDataFrame { background-color: #121416; }
a { color: #7ef9a4; }
</style>
""", unsafe_allow_html=True)

st.title("💹 Analyst Terminal — Equity Valuation & Options")
st.caption("Real-time modelling, DCF, comps, Black–Scholes options, and news. Built for IB/PE/AM prep. — [Navjot Dhah](https://www.linkedin.com/in/navjot-dhah-57870b238)")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Search & settings")
ticker = st.sidebar.text_input("Enter ticker (e.g., AAPL, WYNN, MSFT)", "WYNN").upper().strip()
default_wacc = st.sidebar.number_input("Default WACC (%)", 9.0)/100
default_tg = st.sidebar.number_input("Default Terminal growth (%)", 2.5)/100
projection_years = st.sidebar.selectbox("Projection years", [5,7,10], index=0)

# -------------------------
# Helper functions
# -------------------------
def fetch_data(ticker):
    tk = yf.Ticker(ticker)
    try: info = tk.info
    except: info = {}
    try: fin = tk.financials
    except: fin = pd.DataFrame()
    try: bs = tk.balance_sheet
    except: bs = pd.DataFrame()
    try: cf = tk.cashflow
    except: cf = pd.DataFrame()
    try: hist = tk.history(period="5y")
    except: hist = pd.DataFrame()
    return info, fin, bs, cf, hist

def safe_float(x):
    try: return float(x)
    except: return np.nan

def find_value(df, keywords):
    if df.empty: return None
    for k in keywords:
        for label in df.index:
            if k.lower() in str(label).lower():
                try: return df.loc[label].iloc[0]
                except: continue
    return None

def dcf(fcf, growth, discount, tg, years):
    proj = [fcf*(1+growth)**i for i in range(1, years+1)]
    pv = sum([proj[i]/((1+discount)**(i+1)) for i in range(len(proj))])
    terminal = proj[-1]*(1+tg)/(discount-tg)/((1+discount)**years) if discount>tg else 0
    return {"proj_pv": [proj[i]/((1+discount)**(i+1)) for i in range(len(proj))], "terminal": terminal, "EV": pv+terminal}

def bs_price(S,K,T,r,sigma,option="call"):
    if T<=0 or sigma<=0:
        return max(0.0,S-K) if option=="call" else max(0.0,K-S)
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option=="call": return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    else: return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

def bs_greeks(S,K,T,r,sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta_c = norm.cdf(d1)
    delta_p = delta_c -1
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega = S*norm.pdf(d1)*np.sqrt(T)
    theta_c = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))
    theta_p = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))
    rho_c = K*T*np.exp(-r*T)*norm.cdf(d2)
    rho_p = -K*T*np.exp(-r*T)*norm.cdf(-d2)
    return {"delta_c":delta_c,"delta_p":delta_p,"gamma":gamma,"vega":vega,"theta_c":theta_c,"theta_p":theta_p,"rho_c":rho_c,"rho_p":rho_p}

def fetch_news(ticker, limit=6):
    try:
        url=f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
        resp=requests.get(url, timeout=5).json()
        items=resp.get("news",[])[:limit]
        news=[]
        for n in items:
            title=n.get("title") or n.get("headline")
            link=n.get("link") or n.get("url")
            source=n.get("publisher") or n.get("provider") or ""
            ts=n.get("providerPublishTime")
            time=datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M") if ts else ""
            news.append({"title":title,"link":link,"source":source,"time":time})
        return news
    except: return []

# -------------------------
# Fetch data
# -------------------------
info, fin, bs, cf, hist = fetch_data(ticker)

st.header(f"{info.get('shortName',ticker)} — {ticker}")

price = safe_float(info.get("currentPrice") or (hist['Close'].iloc[-1] if not hist.empty else np.nan))
market_cap = safe_float(info.get("marketCap"))
shares_out = safe_float(info.get("sharesOutstanding") or info.get("floatShares"))
ev = safe_float(info.get("enterpriseValue") or (market_cap or 0) + safe_float(info.get("totalDebt") or 0) - safe_float(info.get("totalCash") or 0))

col1,col2,col3,col4,col5 = st.columns(5)
col1.metric("Price", f"${price:,.2f}")
col2.metric("Market Cap", f"${market_cap:,.0f}")
col3.metric("Enterprise Value", f"${ev:,.0f}")
col4.metric("Shares Outstanding", f"{int(shares_out):,}" if shares_out else "N/A")
col5.metric("Sector/Industry", f"{info.get('sector','N/A')}/{info.get('industry','N/A')}")

# -------------------------
# Price chart
# -------------------------
st.subheader("Price chart (candles)")
if not hist.empty:
    fig=go.Figure(data=[go.Candlestick(x=hist.index,open=hist['Open'],high=hist['High'],low=hist['Low'],close=hist['Close'])])
    fig.update_layout(template="plotly_dark",height=400)
    st.plotly_chart(fig,use_container_width=True)
else: st.info("No price history available.")

# -------------------------
# Financials
# -------------------------
st.subheader("Financial Statements")
st.write("Income Statement")
st.dataframe(fin.T if not fin.empty else pd.DataFrame())
st.write("Balance Sheet")
st.dataframe(bs.T if not bs.empty else pd.DataFrame())
st.write("Cash Flow")
st.dataframe(cf.T if not cf.empty else pd.DataFrame())

# -------------------------
# DCF
# -------------------------
st.subheader("DCF Valuation")
ocf = safe_float(find_value(cf, ["operat", "cash from operating"]))
capex = safe_float(find_value(cf, ["capital expend","purchase of property"])) or 0
last_fcf = ocf + capex if ocf else st.number_input("Manual FCF",500_000_000,step=1_000_000)
g = st.slider("FCF growth %", -10,30,5)/100
d = st.slider("Discount rate / WACC %",0,30,float(default_wacc*100))/100
tg = st.slider("Terminal growth %",-2,6,float(default_tg*100))/100
years = st.selectbox("Projection years",[3,5,7,10],index=1)
dcf_res = dcf(last_fcf,g,d,tg,years)
EV_calc = dcf_res["EV"]
equity_val = EV_calc - safe_float(info.get("totalDebt") or 0) + safe_float(info.get("totalCash") or 0)
implied_price = equity_val / shares_out if shares_out else np.nan
st.metric("Enterprise Value (DCF)",f"${EV_calc:,.0f}")
st.metric("Equity Value",f"${equity_val:,.0f}")
st.metric("Implied Price",f"${implied_price:,.2f}" if not np.isnan(implied_price) else "N/A")

fig_dcf=go.Figure()
fig_dcf.add_trace(go.Bar(x=[f"Y{i}" for i in range(1,years+1)],y=dcf_res["proj_pv"],name="Discounted FCF",marker_color="#00CC96"))
fig_dcf.add_trace(go.Bar(x=["Terminal"],y=[dcf_res["terminal"]],name="Terminal PV",marker_color="#f5c518"))
fig_dcf.update_layout(template="plotly_dark",barmode="stack",title="DCF PV contributions")
st.plotly_chart(fig_dcf,use_container_width=True)

# -------------------------
# Options & Greeks
# -------------------------
st.subheader("Options Pricing (Black–Scholes)")
S_default = price or 100
col1,col2,col3,col4,col5 = st.columns(5)
S = col1.number_input("Price (S)",value=float(S_default))
K = col2.number_input("Strike (K)",value=float(S_default))
days = col3.number_input("Days to Expiry",1,3650,30)
r = col4.number_input("Risk-free rate %",0.5)/100
sigma = col5.number_input("Volatility σ",0.25)
T = days/365
st.write(f"Call: ${bs_price(S,K,T,r,sigma,'call'):.2f} — Put: ${bs_price(S,K,T,r,sigma,'put'):.2f}")
st.write("Greeks:")
st.json(bs_greeks(S,K,T,r,sigma))

# -------------------------
# News
# -------------------------
st.subheader("Company News")
news_items = fetch_news(ticker,8)
if news_items:
    for n in news_items:
        st.markdown(f"- [{n['title']}]({n['link']}) <small>({n['source']}) {n['time']}</small>",unsafe_allow_html=True)
else: st.info("No news found or blocked.")

