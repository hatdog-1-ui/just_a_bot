"""Interactive dashboard for backtesting and live monitoring."""

import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtester import Backtester
from config import BotConfig, StrategyConfig, RiskConfig
from exchange import ExchangeClient
from strategy import compute_indicators, Signal

st.set_page_config(
    page_title="Crypto Trading Bot",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Crypto Trading Bot Dashboard")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"], index=0)
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
    limit = st.slider("Bars to fetch", 100, 1000, 500, step=50)
    initial_capital = st.number_input("Starting Capital (USDT)", min_value=100, value=10_000, step=100)

    st.divider()
    st.subheader("Strategy Parameters")
    rsi_oversold = st.slider("RSI Oversold", 20, 45, 35)
    rsi_overbought = st.slider("RSI Overbought", 55, 80, 65)
    bb_std = st.slider("Bollinger Band Std Dev", 1.0, 3.0, 2.0, step=0.1)

    st.divider()
    st.subheader("Risk Parameters")
    stop_loss_pct = st.slider("Stop Loss %", 0.5, 5.0, 2.0, step=0.5) / 100
    take_profit_pct = st.slider("Take Profit %", 1.0, 10.0, 4.0, step=0.5) / 100
    max_risk = st.slider("Max Risk per Trade %", 0.5, 3.0, 1.0, step=0.25) / 100

    run = st.button("▶  Run Backtest", use_container_width=True, type="primary")

# ── Fetch data and run backtest ───────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def load_data(symbol, timeframe, limit):
    cfg = BotConfig(symbol=symbol, timeframe=timeframe)
    client = ExchangeClient(cfg.exchange)
    df = client.fetch_ohlcv(symbol, timeframe, limit=limit)
    return df

def build_config(symbol, timeframe, rsi_oversold, rsi_overbought, bb_std, stop_loss_pct, take_profit_pct, max_risk):
    cfg = BotConfig(symbol=symbol, timeframe=timeframe)
    cfg.strategy.rsi_oversold = rsi_oversold
    cfg.strategy.rsi_overbought = rsi_overbought
    cfg.strategy.bb_std = bb_std
    cfg.risk.stop_loss_pct = stop_loss_pct
    cfg.risk.take_profit_pct = take_profit_pct
    cfg.risk.max_risk_per_trade = max_risk
    return cfg

if run:
    with st.spinner("Loading data and running backtest..."):
        try:
            df = load_data(symbol, timeframe, limit)
            cfg = build_config(symbol, timeframe, rsi_oversold, rsi_overbought, bb_std,
                               stop_loss_pct, take_profit_pct, max_risk)

            backtester = Backtester(cfg)
            result = backtester.run(df.copy(), initial_capital=float(initial_capital))

            df_ind = compute_indicators(df.copy(), cfg.strategy).dropna().reset_index()

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.subheader("Performance Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Return", f"{result.total_return_pct:+.2f}%",
              delta_color="normal" if result.total_return_pct >= 0 else "inverse")
    c2.metric("Win Rate", f"{result.win_rate_pct:.1f}%")
    c3.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    c4.metric("Max Drawdown", f"{result.max_drawdown_pct:.2f}%")
    c5.metric("Total Trades", result.num_trades,
              help=f"{result.num_wins} wins / {result.num_losses} losses")

    st.divider()

    # ── Main price chart with BB and signals ──────────────────────────────────
    st.subheader("Price Chart — Bollinger Bands & Trade Signals")

    buys = [(t.entry_bar, t.entry_price) for t in result.trades if t.side == "long"]
    sells_entry = [(t.entry_bar, t.entry_price) for t in result.trades if t.side == "short"]
    exits = [(t.exit_bar, t.exit_price, t.pnl) for t in result.trades if t.exit_price]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=("Price + Bollinger Bands", "RSI", "MACD"),
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df_ind["timestamp"],
        open=df_ind["open"], high=df_ind["high"],
        low=df_ind["low"], close=df_ind["close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df_ind["timestamp"], y=df_ind["bb_upper"],
                             line=dict(color="rgba(100,149,237,0.4)", width=1),
                             name="BB Upper", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ind["timestamp"], y=df_ind["bb_lower"],
                             fill="tonexty",
                             fillcolor="rgba(100,149,237,0.07)",
                             line=dict(color="rgba(100,149,237,0.4)", width=1),
                             name="BB Lower", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ind["timestamp"], y=df_ind["bb_mid"],
                             line=dict(color="rgba(100,149,237,0.6)", width=1, dash="dot"),
                             name="BB Mid", showlegend=False), row=1, col=1)

    # Buy signals
    if buys:
        bx = [df_ind["timestamp"].iloc[i] for i, _ in buys if i < len(df_ind)]
        by = [p for i, p in buys if i < len(df_ind)]
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="markers",
            marker=dict(symbol="triangle-up", size=14, color="#00e676"),
            name="Buy Signal",
        ), row=1, col=1)

    # Sell / short signals
    if sells_entry:
        sx = [df_ind["timestamp"].iloc[i] for i, _ in sells_entry if i < len(df_ind)]
        sy = [p for i, p in sells_entry if i < len(df_ind)]
        fig.add_trace(go.Scatter(
            x=sx, y=sy, mode="markers",
            marker=dict(symbol="triangle-down", size=14, color="#ff1744"),
            name="Sell Signal",
        ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df_ind["timestamp"], y=df_ind["rsi"],
                             line=dict(color="#ab47bc", width=1.5), name="RSI"), row=2, col=1)
    fig.add_hline(y=rsi_overbought, line=dict(color="red", dash="dash", width=1), row=2, col=1)
    fig.add_hline(y=rsi_oversold, line=dict(color="green", dash="dash", width=1), row=2, col=1)
    fig.add_hrect(y0=rsi_overbought, y1=100, fillcolor="red", opacity=0.05, row=2, col=1)
    fig.add_hrect(y0=0, y1=rsi_oversold, fillcolor="green", opacity=0.05, row=2, col=1)

    # MACD
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df_ind["macd_hist"]]
    fig.add_trace(go.Bar(x=df_ind["timestamp"], y=df_ind["macd_hist"],
                         marker_color=colors, name="MACD Hist", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_ind["timestamp"], y=df_ind["macd"],
                             line=dict(color="#42a5f5", width=1.5), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_ind["timestamp"], y=df_ind["macd_signal"],
                             line=dict(color="#ff7043", width=1.5), name="Signal"), row=3, col=1)

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Equity curve ──────────────────────────────────────────────────────────
    st.subheader("Portfolio Equity Curve")
    eq_df = pd.DataFrame({
        "Equity (USDT)": result.equity_curve.values,
        "Bar": range(len(result.equity_curve)),
    })
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        y=eq_df["Equity (USDT)"],
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(38,166,154,0.15)",
        line=dict(color="#26a69a", width=2),
        name="Equity",
    ))
    fig_eq.add_hline(y=initial_capital, line=dict(color="gray", dash="dash", width=1))
    fig_eq.update_layout(
        height=300, template="plotly_dark",
        yaxis_title="USDT", xaxis_title="Bar",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── Trade history table ───────────────────────────────────────────────────
    st.subheader("Trade History")
    if result.trades:
        rows = []
        for t in result.trades:
            rows.append({
                "Side": t.side.upper(),
                "Entry Bar": t.entry_bar,
                "Entry Price": f"${t.entry_price:,.2f}",
                "Exit Price": f"${t.exit_price:,.2f}" if t.exit_price else "—",
                "Stop Loss": f"${t.stop_loss:,.2f}",
                "Take Profit": f"${t.take_profit:,.2f}",
                "Qty": f"{t.quantity:.6f}",
                "PnL (USDT)": f"{t.pnl:+.2f}" if t.pnl else "—",
                "PnL %": f"{t.pnl_pct:+.2f}%" if t.pnl_pct else "—",
                "Result": "✅ Win" if t.pnl and t.pnl > 0 else "❌ Loss",
            })
        trades_df = pd.DataFrame(rows)

        def color_result(val):
            if "Win" in str(val):
                return "color: #26a69a"
            if "Loss" in str(val):
                return "color: #ef5350"
            if val and val[0] == "+":
                return "color: #26a69a"
            if val and val[0] == "-":
                return "color: #ef5350"
            return ""

        st.dataframe(
            trades_df.style.applymap(color_result, subset=["PnL (USDT)", "PnL %", "Result"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No trades were executed in this period. Try adjusting the strategy parameters.")

else:
    st.info("👈 Configure your settings in the sidebar and click **Run Backtest** to start.")
    st.markdown("""
    ### What this dashboard shows you:
    - **Price chart** with Bollinger Bands and buy/sell signal markers
    - **RSI** (Relative Strength Index) — shows overbought/oversold zones
    - **MACD** — shows momentum direction and crossovers
    - **Equity curve** — how your portfolio would have grown over time
    - **Trade history** — every trade with entry, exit, and profit/loss
    """)
