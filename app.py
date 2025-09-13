# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from scipy.stats import norm

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Equity Valuation Terminal",
    page_icon="💹",
    layout="wide"
)

# ------------------------------
# Terminal Dark Theme CSS
# ------------------------------
st.markdown("""
<style>
body, .stApp {
    background-color: #0a0a0a;
    color: #33ff33;
    font-family: 'Courier New', Courier, monospace;
}
section[data-testid="stSidebar"] {
    background-color: #111111 !important;
    color: #33ff33 !important;
}
section[data-testid="stSidebar"] * {
    color: #33ff33 !important;
    font-family: 'Courier New', Courier, monospace;
}
div[data-testid="stMetric"] {
    background-color: #111111;
    border: 1px solid #33ff33;
    padding: 10px;
    border-radius: 0px;
    margin-bottom: 5px;
}
.stDataFrame, .stTable {
    background-color: #111111;
    color: #33ff33 !important;
    border: 1px solid #33ff33;
}
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #33ff33 !important;
    font-family: 'Courier New', Courier, monospace;
}
</style>
""", unsafe_allow_html=True)

st.title("💹 Equity Valuation Terminal")
st.markdown("Built by **Navjot Dhah** | Professional financial modelling for IB / PE / AM roles")

# ------------------------------
# Sidebar: Navigation & Input
# ------------------------------
menu = st.sidebar.radio("MENU", ["Overview", "Valuation", "Financials", "Options", "News"])

st.sidebar.header("Company Search")
ticker = st.sidebar.text_input("Ticker:", "AAPL")
period = st.sidebar.selectbox("Historical Period:", ["1y", "3y", "5y", "10y"], index=1)

# ------------------------------
# Fetch Data
# ------------------------------
try:
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    fin = stock.financials
    bal = stock.balance_sheet
    cf = stock.cashflow
    info = stock.info
except Exception as e:
    st.error(f"Error fetching data for {ticker}: {e}")
    st.stop()

# ------------------------------
# Overview: Candlestick + Indicators
# ------------------------------
if menu == "Overview":
    st.subheader(f"📊 {ticker} Stock Overview")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"],
        high=hist["High"],
        low=hist["Low"],
        close=hist["Close"],
        name="Candlestick"
    ))
    fig.add_trace(go.Bar(
        x=hist.index, 
        y=hist["Volume"], 
        name="Volume", 
        yaxis="y2", 
        opacity=0.3
    ))
    fig.update_layout(
        template="plotly_dark",
        yaxis=dict(title="Price (USD)"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        xaxis_title="Date",
        height=500,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Technical indicators
    hist["SMA20"] = hist["Close"].rolling(20).mean()
    hist["SMA50"] = hist["Close"].rolling(50).mean()
    chg = hist["Close"].diff()
    gain = chg.where(chg > 0, 0).rolling(14).mean()
    loss = -chg.where(chg < 0, 0).rolling(14).mean()
    rs = gain / loss
    hist["RSI"] = 100 - (100 / (1 + rs))
    exp1 = hist["Close"].ewm(span=12, adjust=False).mean()
    exp2 = hist["Close"].ewm(span=26, adjust=False).mean()
    hist["MACD"] = exp1 - exp2
    hist["Signal"] = hist["MACD"].ewm(span=9, adjust=False).mean()

    st.subheader("📈 Technical Indicators (Last 10 rows)")
    st.dataframe(hist[['Close','SMA20','SMA50','RSI','MACD','Signal']].tail(10).style.format("{:.2f}"))

# ------------------------------
# Valuation
# ------------------------------
elif menu == "Valuation":
    st.subheader("⚖️ Valuation Metrics")
    
    years = st.slider("Projection Years:", 3, 10, 5)
    
    try:
        base_rev = fin.loc["Total Revenue"].iloc[0]
    except:
        base_rev = 1e9
    
    growth_rate = st.number_input("Revenue Growth Rate (%)", value=5.0)/100
    ebit_margin = st.number_input("EBIT Margin (%)", value=15.0)/100
    tax_rate = st.number_input("Tax Rate (%)", value=21.0)/100
    dep_pct = st.number_input("Depreciation % of Revenue", value=5.0)/100
    capex_pct = st.number_input("CapEx % of Revenue", value=6.0)/100
    nwc_pct = st.number_input("Change in NWC % of Revenue", value=2.0)/100
    
    proj = []
    rev = base_rev
    for yr in range(1, years+1):
        rev *= (1 + growth_rate)
        ebit = rev * ebit_margin
        tax = ebit * tax_rate
        nopat = ebit - tax
        dep = rev * dep_pct
        capex = rev * capex_pct
        nwc = rev * nwc_pct
        fcf = nopat + dep - capex - nwc
        proj.append([datetime.now().year+yr, rev, ebit, nopat, fcf])
    proj_df = pd.DataFrame(proj, columns=["Year","Revenue","EBIT","NOPAT","FCF"])
    
    # WACC
    rf = st.number_input("Risk-Free Rate (%)", value=4.0)/100
    beta = st.number_input("Beta", value=1.1)
    mkt_return = st.number_input("Expected Market Return (%)", value=9.0)/100
    cost_of_equity = rf + beta * (mkt_return - rf)
    
    pretax_cost_debt = st.number_input("Pre-Tax Cost of Debt (%)", value=5.0)/100
    tax_rate_wacc = st.number_input("Corporate Tax Rate (%)", value=21.0)/100
    cost_of_debt = pretax_cost_debt * (1 - tax_rate_wacc)
    
    equity_val = st.number_input("Equity Value ($B)", value=info.get("marketCap",1e10)/1e9)*1e9
    debt_val = st.number_input("Total Debt ($B)", value=bal.loc["Total Debt"].iloc[0]/1e9 if "Total Debt" in bal.index else 10.0)*1e9
    
    w_e = equity_val / (equity_val + debt_val)
    w_d = debt_val / (equity_val + debt_val)
    wacc = w_e * cost_of_equity + w_d * cost_of_debt
    
    st.metric("WACC", f"{wacc*100:.2f}%")
    
    # DCF
    discount_factors = [(1/(1+wacc)**i) for i in range(1, years+1)]
    dcf = (proj_df["FCF"] * discount_factors).sum()
    
    terminal_growth = st.number_input("Terminal Growth Rate (%)", value=2.0)/100
    terminal_value = proj_df["FCF"].iloc[-1]*(1+terminal_growth)/(wacc-terminal_growth)
    terminal_value_pv = terminal_value / ((1+wacc)**years)
    
    ev = dcf + terminal_value_pv
    equity_value = ev - debt_val
    intrinsic_price = equity_value / info.get("sharesOutstanding",1)
    
    st.metric("Enterprise Value", f"${ev/1e9:.2f}B")
    st.metric("Equity Value", f"${equity_value/1e9:.2f}B")
    st.metric("Intrinsic Price / Share", f"${intrinsic_price:.2f}")

# ------------------------------
# Financials
# ------------------------------
elif menu == "Financials":
    st.subheader("📄 Financial Statements")
    st.dataframe(fin.fillna(""))
    st.dataframe(bal.fillna(""))
    st.dataframe(cf.fillna(""))

# ------------------------------
# Options
# ------------------------------
elif menu == "Options":
    st.subheader("📊 Black-Scholes Option Pricing")
    
    S = st.number_input("Current Stock Price (S)", value=float(hist["Close"].iloc[-1]))
    K = st.number_input("Strike Price (K)", value=S*1.05)
    T = st.number_input("Time to Maturity (Years)", value=1.0)
    sigma = st.number_input("Volatility (σ)", value=0.3)
    r = rf
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    st.metric("Call Option Value", f"${call_price:.2f}")
    st.metric("Put Option Value", f"${put_price:.2f}")

# ------------------------------
# News
# ------------------------------
elif menu == "News":
    st.subheader("📰 Latest News / Summary")
    if "longBusinessSummary" in info:
        st.write(info["longBusinessSummary"])
    else:
        st.info("No news summary available.")
