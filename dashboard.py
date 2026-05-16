"""Interactive dashboard: Auto-Optimize → Backtest → Paper Trading."""

from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from backtester import Backtester
from config import BotConfig
from exchange import ExchangeClient
from optimizer import optimize, OptimResult
from risk_manager import RiskManager
from strategy import compute_indicators, generate_signal, Signal

st.set_page_config(page_title="Crypto Trading Bot", page_icon="📈", layout="wide")
st.title("📈 Crypto Trading Bot Dashboard")

# ── Sidebar — only the essentials ────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("That's all you need to fill in — the bot figures out the rest.")

    symbol = st.selectbox(
        "Which coin do you want to trade?",
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
        help="Pick the crypto pair you want to trade.",
    )
    timeframe = st.selectbox(
        "How often should it check for signals?",
        ["15m", "1h", "4h", "1d"],
        index=1,
        format_func=lambda x: {
            "15m": "Every 15 minutes",
            "1h":  "Every 1 hour (recommended)",
            "4h":  "Every 4 hours",
            "1d":  "Once a day",
        }[x],
    )
    initial_capital = st.number_input(
        "How much are you starting with? (USDT)",
        min_value=100, value=10_000, step=100,
        help="Paper money — no real funds are used.",
    )

    st.divider()
    st.button("🔍 Auto-Optimize & Run", key="do_optimize", type="primary",
              use_container_width=True,
              help="Fetches historical data, tests hundreds of parameter combinations, "
                   "and picks the one with the best risk-adjusted return.")

    with st.expander("🔬 Advanced — manual overrides"):
        manual_on = st.toggle("Override auto settings", value=False)
        rsi_oversold   = st.slider("RSI Oversold",    20, 45, 35, disabled=not manual_on)
        rsi_overbought = st.slider("RSI Overbought",  55, 80, 65, disabled=not manual_on)
        bb_std         = st.slider("BB Std Dev",      1.0, 3.0, 2.0, step=0.1, disabled=not manual_on)
        stop_loss_pct  = st.slider("Stop Loss %",     0.5, 5.0, 2.0, step=0.5, disabled=not manual_on) / 100
        take_profit_pct= st.slider("Take Profit %",   1.0, 10.0, 4.0, step=0.5, disabled=not manual_on) / 100
        max_risk       = st.slider("Max Risk/Trade %",0.5, 3.0, 1.0, step=0.25, disabled=not manual_on) / 100


# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner="Fetching market data from Binance...")
def load_data(symbol, timeframe, limit=500):
    cfg = BotConfig(symbol=symbol, timeframe=timeframe)
    return ExchangeClient(cfg.exchange).fetch_ohlcv(symbol, timeframe, limit=limit)


@st.cache_data(ttl=300, show_spinner="🔍 Testing hundreds of parameter combinations...")
def run_optimize(symbol, timeframe, capital):
    df = load_data(symbol, timeframe, limit=500)
    return optimize(df, initial_capital=float(capital)), df


def make_config_from(opt: Optional[OptimResult]) -> BotConfig:
    cfg = BotConfig(symbol=symbol, timeframe=timeframe)
    if manual_on:
        cfg.strategy.rsi_oversold    = rsi_oversold
        cfg.strategy.rsi_overbought  = rsi_overbought
        cfg.strategy.bb_std          = bb_std
        cfg.risk.stop_loss_pct       = stop_loss_pct
        cfg.risk.take_profit_pct     = take_profit_pct
        cfg.risk.max_risk_per_trade  = max_risk
    elif opt:
        cfg.strategy.rsi_oversold    = opt.rsi_oversold
        cfg.strategy.rsi_overbought  = opt.rsi_overbought
        cfg.strategy.bb_std          = opt.bb_std
        cfg.risk.stop_loss_pct       = opt.stop_loss_pct
        cfg.risk.take_profit_pct     = opt.take_profit_pct
    return cfg


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_bt, tab_paper = st.tabs(["📊 Backtest", "🤖 Paper Trading"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — BACKTEST
# ════════════════════════════════════════════════════════════════════════════════
with tab_bt:
    # Run optimization only when button is clicked; persist results in session state
    if st.session_state.get("do_optimize"):
        with st.status("Working...", expanded=True) as status:
            st.write("📡 Fetching historical price data...")
            st.write("🔍 Running parameter optimization (this takes ~20 seconds)...")
            try:
                opt, df_raw = run_optimize(symbol, timeframe, initial_capital)
                cfg    = make_config_from(opt)
                result = Backtester(cfg).run(df_raw.copy(), initial_capital=float(initial_capital))
                df_ind = compute_indicators(df_raw.copy(), cfg.strategy).dropna().reset_index()
                # Save everything so auto-refresh doesn't wipe it
                st.session_state.bt_opt    = opt
                st.session_state.bt_result = result
                st.session_state.bt_df_ind = df_ind
                status.update(label="✅ Done!", state="complete")
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

    if "bt_result" in st.session_state:
        opt    = st.session_state.bt_opt
        result = st.session_state.bt_result
        df_ind = st.session_state.bt_df_ind

        # ── Best settings found ───────────────────────────────────────────────
        st.subheader("🏆 Best Settings Found Automatically")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("RSI Oversold",   f"{opt.rsi_oversold}",
                  help="Bot buys when RSI drops below this (oversold zone)")
        s2.metric("RSI Overbought", f"{opt.rsi_overbought}",
                  help="Bot sells when RSI rises above this (overbought zone)")
        s3.metric("BB Std Dev",     f"{opt.bb_std}",
                  help="Width of the Bollinger Bands")
        s4.metric("Stop Loss",      f"{opt.stop_loss_pct*100:.1f}%",
                  help="Auto-exit if trade goes this far against you")
        s5.metric("Take Profit",    f"{opt.take_profit_pct*100:.1f}%",
                  help="Auto-exit when this profit is reached")

        st.divider()

        # ── Performance metrics ───────────────────────────────────────────────
        st.subheader("📈 Performance with These Settings")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Return",  f"{result.total_return_pct:+.2f}%")
        c2.metric("Win Rate",      f"{result.win_rate_pct:.1f}%",
                  help="% of trades that closed in profit")
        c3.metric("Sharpe Ratio",  f"{result.sharpe_ratio:.2f}",
                  help="Risk-adjusted return. Above 1.0 is good, above 2.0 is great.")
        c4.metric("Max Drawdown",  f"{result.max_drawdown_pct:.2f}%",
                  help="Worst peak-to-trough drop. Smaller is safer.")
        c5.metric("Total Trades",  result.num_trades,
                  help=f"{result.num_wins} wins / {result.num_losses} losses")

        st.divider()

        # ── Price chart ───────────────────────────────────────────────────────
        st.subheader("Price Chart — Signals, Stop Loss & Take Profit")

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.25, 0.20], vertical_spacing=0.03,
            subplot_titles=("Price + Bollinger Bands", "RSI", "MACD"),
        )

        fig.add_trace(go.Candlestick(
            x=df_ind["timestamp"],
            open=df_ind["open"], high=df_ind["high"],
            low=df_ind["low"],   close=df_ind["close"],
            name="Price",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df_ind["timestamp"], y=df_ind["bb_upper"],
                                 line=dict(color="rgba(100,149,237,0.4)", width=1),
                                 name="BB Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ind["timestamp"], y=df_ind["bb_lower"],
                                 fill="tonexty", fillcolor="rgba(100,149,237,0.07)",
                                 line=dict(color="rgba(100,149,237,0.4)", width=1),
                                 name="BB Lower", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ind["timestamp"], y=df_ind["bb_mid"],
                                 line=dict(color="rgba(100,149,237,0.6)", width=1, dash="dot"),
                                 name="BB Mid", showlegend=False), row=1, col=1)

        for t in result.trades:
            if t.entry_bar >= len(df_ind) or t.exit_bar is None or t.exit_bar >= len(df_ind):
                continue
            x0 = df_ind["timestamp"].iloc[t.entry_bar]
            x1 = df_ind["timestamp"].iloc[t.exit_bar]

            fig.add_trace(go.Scatter(
                x=[x0], y=[t.entry_price], mode="markers",
                marker=dict(symbol="triangle-up" if t.side == "long" else "triangle-down",
                            size=14, color="#00e676" if t.side == "long" else "#ff1744"),
                showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[t.take_profit, t.take_profit], mode="lines",
                line=dict(color="#00e676", width=1.5, dash="dash"),
                showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[t.stop_loss, t.stop_loss], mode="lines",
                line=dict(color="#ef5350", width=1.5, dash="dash"),
                showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[x1], y=[t.exit_price or t.entry_price], mode="markers",
                marker=dict(symbol="x", size=10,
                            color="#00e676" if (t.pnl or 0) > 0 else "#ef5350"),
                showlegend=False,
            ), row=1, col=1)

        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(color="#00e676", dash="dash"),
                                 name="Take Profit 🟢"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(color="#ef5350", dash="dash"),
                                 name="Stop Loss 🔴"), row=1, col=1)

        fig.add_trace(go.Scatter(x=df_ind["timestamp"], y=df_ind["rsi"],
                                 line=dict(color="#ab47bc", width=1.5), name="RSI"), row=2, col=1)
        fig.add_hline(y=opt.rsi_overbought, line=dict(color="red",   dash="dash", width=1), row=2, col=1)
        fig.add_hline(y=opt.rsi_oversold,   line=dict(color="green", dash="dash", width=1), row=2, col=1)
        fig.add_hrect(y0=opt.rsi_overbought, y1=100,             fillcolor="red",   opacity=0.05, row=2, col=1)
        fig.add_hrect(y0=0,                  y1=opt.rsi_oversold, fillcolor="green", opacity=0.05, row=2, col=1)

        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df_ind["macd_hist"]]
        fig.add_trace(go.Bar(x=df_ind["timestamp"], y=df_ind["macd_hist"],
                             marker_color=colors, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_ind["timestamp"], y=df_ind["macd"],
                                 line=dict(color="#42a5f5", width=1.5), name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_ind["timestamp"], y=df_ind["macd_signal"],
                                 line=dict(color="#ff7043", width=1.5), name="Signal"), row=3, col=1)

        fig.update_layout(
            height=700, xaxis_rangeslider_visible=False, template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Equity curve ──────────────────────────────────────────────────────
        st.subheader("Portfolio Equity Curve")
        fig_eq = go.Figure(go.Scatter(
            y=result.equity_curve.values, mode="lines",
            fill="tozeroy", fillcolor="rgba(38,166,154,0.15)",
            line=dict(color="#26a69a", width=2), name="Equity",
        ))
        fig_eq.add_hline(y=initial_capital, line=dict(color="gray", dash="dash", width=1))
        fig_eq.update_layout(height=280, template="plotly_dark",
                             yaxis_title="USDT", xaxis_title="Bar",
                             margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_eq, use_container_width=True)

        # ── Trade table ───────────────────────────────────────────────────────
        st.subheader("Trade History")
        if result.trades:
            rows = []
            for t in result.trades:
                rows.append({
                    "Side":            t.side.upper(),
                    "Entry Price":     f"${t.entry_price:,.2f}",
                    "Stop Loss 🔴":    f"${t.stop_loss:,.2f}",
                    "Take Profit 🟢":  f"${t.take_profit:,.2f}",
                    "Exit Price":      f"${t.exit_price:,.2f}" if t.exit_price else "—",
                    "Qty":             f"{t.quantity:.6f}",
                    "PnL (USDT)":      f"{t.pnl:+.2f}" if t.pnl else "—",
                    "PnL %":           f"{t.pnl_pct:+.2f}%" if t.pnl_pct else "—",
                    "Result":          "✅ Win" if (t.pnl or 0) > 0 else "❌ Loss",
                })
            df_trades = pd.DataFrame(rows)

            def _color(val):
                s = str(val)
                if "Win"  in s or (s and s[0] == "+"): return "color:#26a69a"
                if "Loss" in s or (s and s[0] == "-"): return "color:#ef5350"
                return ""

            st.dataframe(df_trades.style.applymap(_color, subset=["PnL (USDT)", "PnL %", "Result"]),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No trades in this period. The optimizer will try different settings next time.")

    if "bt_result" not in st.session_state:
        st.info("👈 Choose your coin and capital in the sidebar, then click **🔍 Auto-Optimize & Run**.")
        st.markdown("""
        ### How it works:
        1. **You pick** the coin, timeframe, and starting capital — that's it
        2. **The bot tests** hundreds of parameter combinations on real historical data
        3. **It picks** the best settings automatically (highest risk-adjusted return)
        4. **You see** the full backtest: charts, signals, stop loss, take profit, and trade history
        5. **Switch to Paper Trading** to test it live with no real money
        """)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PAPER TRADING
# ════════════════════════════════════════════════════════════════════════════════
with tab_paper:
    st_autorefresh(interval=30_000, key="paper_refresh")

    if "paper_capital"  not in st.session_state:
        st.session_state.paper_capital  = float(initial_capital)
        st.session_state.paper_position = None
        st.session_state.paper_trades   = []
        st.session_state.paper_equity   = [float(initial_capital)]
        st.session_state.paper_log      = []
        st.session_state.paper_opt      = None

    col_reset, col_info = st.columns([1, 4])
    with col_reset:
        if st.button("🔄 Reset Portfolio"):
            for k in ["paper_capital","paper_position","paper_trades",
                      "paper_equity","paper_log","paper_opt"]:
                del st.session_state[k]
            st.rerun()
    with col_info:
        st.caption("Updates every 30 seconds automatically. No real money involved.")

    st.divider()

    with st.spinner("Fetching settings & live price..."):
        try:
            # Use cached optimized params if available, else run optimizer
            if st.session_state.paper_opt is None:
                opt, _ = run_optimize(symbol, timeframe, initial_capital)
                st.session_state.paper_opt = opt
            else:
                opt = st.session_state.paper_opt

            cfg    = make_config_from(opt)
            df_raw = load_data(symbol, timeframe, limit=200)
            df_ind = compute_indicators(df_raw.copy(), cfg.strategy).dropna()
            now_price = float(df_ind.iloc[-1]["close"])
            now_rsi   = float(df_ind.iloc[-1]["rsi"])
            now_ts    = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            risk_mgr  = RiskManager(cfg.risk)

            # Check exit
            pos = st.session_state.paper_position
            if pos:
                hit_sl = now_price <= pos["stop_loss"]
                hit_tp = now_price >= pos["take_profit"]
                if hit_sl or hit_tp:
                    reason  = "TAKE PROFIT ✅" if hit_tp else "STOP LOSS 🔴"
                    exit_px = pos["take_profit"] if hit_tp else pos["stop_loss"]
                    pnl     = (exit_px - pos["entry_price"]) * pos["quantity"]
                    if pos["side"] == "short":
                        pnl = -pnl
                    st.session_state.paper_capital += pos["quantity"] * pos["entry_price"] + pnl
                    st.session_state.paper_trades.append({
                        "time": now_ts, "side": pos["side"].upper(),
                        "entry": pos["entry_price"], "exit": exit_px,
                        "sl": pos["stop_loss"], "tp": pos["take_profit"],
                        "qty": pos["quantity"], "pnl": pnl, "result": reason,
                    })
                    st.session_state.paper_log.insert(0,
                        f"[{now_ts}] Closed {pos['side'].upper()} @ ${exit_px:,.2f} — {reason}  PnL: ${pnl:+.2f}")
                    st.session_state.paper_position = None

            # Check entry
            sig = generate_signal(df_ind, cfg.strategy)
            if st.session_state.paper_position is None:
                if sig.signal == Signal.BUY:
                    plan = risk_mgr.plan_long(now_price, st.session_state.paper_capital)
                    if plan.position_value <= st.session_state.paper_capital:
                        st.session_state.paper_capital -= plan.position_value
                        st.session_state.paper_position = {
                            "side": "long", "entry_price": plan.entry_price,
                            "stop_loss": plan.stop_loss, "take_profit": plan.take_profit,
                            "quantity": plan.quantity, "opened_at": now_ts,
                        }
                        st.session_state.paper_log.insert(0,
                            f"[{now_ts}] Opened LONG @ ${plan.entry_price:,.2f}  "
                            f"SL=${plan.stop_loss:,.2f}  TP=${plan.take_profit:,.2f}")
                elif sig.signal == Signal.SELL:
                    plan = risk_mgr.plan_short(now_price, st.session_state.paper_capital)
                    if plan.position_value <= st.session_state.paper_capital:
                        st.session_state.paper_capital -= plan.position_value
                        st.session_state.paper_position = {
                            "side": "short", "entry_price": plan.entry_price,
                            "stop_loss": plan.stop_loss, "take_profit": plan.take_profit,
                            "quantity": plan.quantity, "opened_at": now_ts,
                        }
                        st.session_state.paper_log.insert(0,
                            f"[{now_ts}] Opened SHORT @ ${plan.entry_price:,.2f}  "
                            f"SL=${plan.stop_loss:,.2f}  TP=${plan.take_profit:,.2f}")

            # Equity tracking
            unrealised = 0.0
            pos = st.session_state.paper_position
            if pos:
                unrealised = (now_price - pos["entry_price"]) * pos["quantity"]
                if pos["side"] == "short":
                    unrealised = -unrealised
            st.session_state.paper_equity.append(
                st.session_state.paper_capital + unrealised)

            fetch_ok = True
        except Exception as e:
            st.error(f"Could not fetch data: {e}")
            fetch_ok = False

    if fetch_ok:
        # ── Active settings banner ────────────────────────────────────────────
        with st.expander("🏆 Auto-selected settings being used", expanded=False):
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("RSI Oversold",   opt.rsi_oversold)
            s2.metric("RSI Overbought", opt.rsi_overbought)
            s3.metric("BB Std Dev",     opt.bb_std)
            s4.metric("Stop Loss",      f"{opt.stop_loss_pct*100:.1f}%")
            s5.metric("Take Profit",    f"{opt.take_profit_pct*100:.1f}%")

        # ── Status cards ──────────────────────────────────────────────────────
        st.subheader("Live Status")
        equity_now = st.session_state.paper_equity[-1]
        pnl_total  = equity_now - float(initial_capital)

        ca, cb, cc, cd = st.columns(4)
        ca.metric("Current Price",     f"${now_price:,.2f}")
        cb.metric("Portfolio Equity",  f"${equity_now:,.2f}", f"{pnl_total:+.2f} USDT")
        cc.metric("RSI",               f"{now_rsi:.1f}",
                  "Oversold 🟢" if now_rsi < opt.rsi_oversold
                  else ("Overbought 🔴" if now_rsi > opt.rsi_overbought else "Neutral ⚪"))
        cd.metric("Signal", {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "⚪ HOLD"}.get(
            sig.signal.value, sig.signal.value))

        st.divider()

        # ── Active position ───────────────────────────────────────────────────
        st.subheader("Active Position")
        pos = st.session_state.paper_position
        if pos:
            unrealised = (now_price - pos["entry_price"]) * pos["quantity"]
            if pos["side"] == "short":
                unrealised = -unrealised
            pct = unrealised / (pos["entry_price"] * pos["quantity"]) * 100

            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("Side",             "🟢 LONG" if pos["side"] == "long" else "🔴 SHORT")
            p2.metric("Entry Price",      f"${pos['entry_price']:,.2f}")
            p3.metric("Stop Loss 🔴",     f"${pos['stop_loss']:,.2f}",
                      f"−{opt.stop_loss_pct*100:.1f}% from entry")
            p4.metric("Take Profit 🟢",   f"${pos['take_profit']:,.2f}",
                      f"+{opt.take_profit_pct*100:.1f}% from entry")
            p5.metric("Unrealised PnL",   f"${unrealised:+.2f}", f"{pct:+.2f}%")

            sl, tp = pos["stop_loss"], pos["take_profit"]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=now_price,
                gauge={
                    "axis": {"range": [sl * 0.998, tp * 1.002]},
                    "bar":  {"color": "#42a5f5"},
                    "steps": [
                        {"range": [sl * 0.998, sl], "color": "#ef5350"},
                        {"range": [sl, tp],          "color": "#1e1e1e"},
                        {"range": [tp, tp * 1.002],  "color": "#26a69a"},
                    ],
                },
                title={"text": f"Price position between Stop Loss and Take Profit  (opened {pos['opened_at']})"},
                number={"prefix": "$", "valueformat": ",.2f"},
            ))
            fig_gauge.update_layout(height=280, template="plotly_dark",
                                    margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info("No open position right now. The bot is watching for a signal...")

        st.divider()

        # ── Equity curve ──────────────────────────────────────────────────────
        st.subheader("Paper Portfolio Equity")
        fig_eq = go.Figure(go.Scatter(
            y=st.session_state.paper_equity, mode="lines",
            fill="tozeroy", fillcolor="rgba(38,166,154,0.12)",
            line=dict(color="#26a69a", width=2),
        ))
        fig_eq.add_hline(y=float(initial_capital),
                         line=dict(color="gray", dash="dash", width=1))
        fig_eq.update_layout(height=240, template="plotly_dark",
                             yaxis_title="USDT", xaxis_title="Refresh #",
                             margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_eq, use_container_width=True)

        # ── Completed trades ──────────────────────────────────────────────────
        if st.session_state.paper_trades:
            st.subheader("Completed Trades")
            df_pt = pd.DataFrame(st.session_state.paper_trades)
            df_pt["entry"] = df_pt["entry"].map(lambda x: f"${x:,.2f}")
            df_pt["exit"]  = df_pt["exit"].map(lambda x: f"${x:,.2f}")
            df_pt["sl"]    = df_pt["sl"].map(lambda x: f"${x:,.2f}")
            df_pt["tp"]    = df_pt["tp"].map(lambda x: f"${x:,.2f}")
            df_pt["pnl"]   = df_pt["pnl"].map(lambda x: f"${x:+.2f}")
            df_pt.columns  = ["Time","Side","Entry","Exit",
                               "Stop Loss 🔴","Take Profit 🟢","Qty","PnL","Result"]
            def _c(v):
                s = str(v)
                if "PROFIT" in s or s.startswith("$+"): return "color:#26a69a"
                if "LOSS"   in s or s.startswith("$-"): return "color:#ef5350"
                return ""
            st.dataframe(df_pt.style.applymap(_c, subset=["PnL","Result"]),
                         use_container_width=True, hide_index=True)

        # ── Log ───────────────────────────────────────────────────────────────
        if st.session_state.paper_log:
            st.subheader("Activity Log")
            for entry in st.session_state.paper_log[:20]:
                st.text(entry)

        st.caption(f"Last updated: {now_ts}  •  Auto-refreshes every 30s")
