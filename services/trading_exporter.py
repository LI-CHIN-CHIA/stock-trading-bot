"""
Trading Prometheus Exporter
讀取各 bot 的 state JSON，轉為 Prometheus metrics，供 Grafana 監控。
Port: 9877

涵蓋：
  台股 live/paper  → trading_* {mode="live"/"paper"}
  美股 paper       → us_*      {mode="paper"}
  加密貨幣 paper   → crypto_*  {mode="paper"}
  當沖 paper       → day_*
"""
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

STATE_FILE      = Path(os.getenv("STATE_FILE",      "/data/trader_state.json"))
PAPER_FILE      = Path(os.getenv("PAPER_FILE",      "/data/paper_trader_state.json"))
DAY_FILE        = Path(os.getenv("DAY_FILE",        "/data/day_trader_state.json"))
OVERNIGHT_FILE  = Path(os.getenv("OVERNIGHT_FILE",  "/data/overnight_trader_state.json"))
US_FILE         = Path(os.getenv("US_FILE",         "/data/us/us_paper_state.json"))
CRYPTO_FILE     = Path(os.getenv("CRYPTO_FILE",     "/data/crypto_paper_state.json"))
PORT         = int(os.getenv("EXPORTER_PORT", "9877"))


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def build_metrics() -> str:
    lines: list[str] = []
    declared: set[str] = set()

    def gauge(name: str, value: float, labels: str = "", help_text: str = ""):
        if name not in declared:
            if help_text:
                lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            declared.add(name)
        label_str = f"{{{labels}}}" if labels else ""
        lines.append(f"{name}{label_str} {value}")

    # ── 台股（live / paper）────────────────────────────────────────────────────
    for mode, path in [("live", STATE_FILE), ("paper", PAPER_FILE)]:
        state = _load(path)
        if not state:
            continue

        cash        = float(state.get("cash", 0))
        holdings    = state.get("holdings", {})
        trade_log   = state.get("trade_log", [])
        mode_label  = "紙上交易" if mode == "paper" else "實盤"

        gauge("trading_cash_twd",        cash,            f'bot="tw",mode="{mode}",label="{mode_label}"',    "Taiwan bot cash TWD")
        gauge("trading_position_count",  float(len(holdings)), f'bot="tw",mode="{mode}",label="{mode_label}"', "Taiwan open positions")

        holding_value = 0.0
        for ticker, h in holdings.items():
            shares      = float(h.get("shares", 0))
            entry_price = float(h.get("entry_price", 0))
            cost        = float(h.get("cost", shares * entry_price))
            hold_days   = int(h.get("hold_days", 0))
            val         = shares * entry_price
            holding_value += val
            lbl = f'bot="tw",mode="{mode}",ticker="{ticker}"'
            gauge("trading_position_shares",      shares,      lbl, "Shares held")
            gauge("trading_position_entry_price", entry_price, lbl, "Entry price TWD")
            gauge("trading_position_cost",        cost,        lbl, "Total cost TWD")
            gauge("trading_position_hold_days",   float(hold_days), lbl, "Days held")
            gauge("trading_position_unrealized_pnl", val - cost,   lbl, "Unrealized PnL TWD")

        gauge("trading_holding_value_twd", holding_value,         f'bot="tw",mode="{mode}",label="{mode_label}"', "Taiwan holding value TWD")
        gauge("trading_total_value_twd",   cash + holding_value,  f'bot="tw",mode="{mode}",label="{mode_label}"', "Taiwan total value TWD")

        buys  = [t for t in trade_log if t.get("action") == "BUY"]
        sells = [t for t in trade_log if t.get("action") == "SELL"]
        gauge("trading_total_buy_trades",  float(len(buys)),  f'bot="tw",mode="{mode}"', "Total buy trades")
        gauge("trading_total_sell_trades", float(len(sells)), f'bot="tw",mode="{mode}"', "Total sell trades")

        realized = 0.0
        for t in sells:
            m = re.search(r'([+-]?\d+\.?\d*)%', t.get("reason", ""))
            if m:
                pct = float(m.group(1)) / 100
                realized += t.get("shares", 0) * t.get("price", 0) / (1 + pct) * pct if pct != -1 else 0
        gauge("trading_realized_pnl_twd", realized, f'bot="tw",mode="{mode}"', "Realized PnL TWD")

    # ── 美股（paper only 目前）────────────────────────────────────────────────
    us_state = _load(US_FILE)
    if us_state:
        us_cash     = float(us_state.get("cash", 0))
        us_holdings = us_state.get("holdings", {})
        us_log      = us_state.get("trade_log", [])
        us_mode     = "paper"

        gauge("us_cash_usd",           us_cash,                 f'mode="{us_mode}",label="紙上交易"', "US bot cash USD")
        gauge("us_position_count",     float(len(us_holdings)), f'mode="{us_mode}",label="紙上交易"', "US open positions")

        us_holding_val = 0.0
        for ticker, h in us_holdings.items():
            shares      = float(h.get("shares", 0))
            entry_price = float(h.get("entry_price", 0))
            cost        = float(h.get("cost", shares * entry_price))
            hold_days   = int(h.get("hold_days", 0))
            val         = shares * entry_price
            us_holding_val += val
            lbl = f'mode="{us_mode}",ticker="{ticker}"'
            gauge("us_position_shares",      shares,      lbl, "US shares held")
            gauge("us_position_entry_price", entry_price, lbl, "US entry price USD")
            gauge("us_position_cost",        cost,        lbl, "US total cost USD")
            gauge("us_position_hold_days",   float(hold_days), lbl, "US days held")
            gauge("us_position_unrealized_pnl", val - cost,   lbl, "US unrealized PnL USD")

        gauge("us_holding_value_usd",  us_holding_val,              f'mode="{us_mode}",label="紙上交易"', "US holding value USD")
        gauge("us_total_value_usd",    us_cash + us_holding_val,    f'mode="{us_mode}",label="紙上交易"', "US total value USD")

        us_buys  = [t for t in us_log if t.get("action") == "BUY"]
        us_sells = [t for t in us_log if t.get("action") == "SELL"]
        gauge("us_total_buy_trades",   float(len(us_buys)),  f'mode="{us_mode}"', "US total buy trades")
        gauge("us_total_sell_trades",  float(len(us_sells)), f'mode="{us_mode}"', "US total sell trades")

        us_realized = 0.0
        for t in us_sells:
            pnl = t.get("pnl", 0)
            if pnl:
                us_realized += float(pnl)
        gauge("us_realized_pnl_usd", us_realized, f'mode="{us_mode}"', "US realized PnL USD")

    # ── 加密貨幣（paper only 目前）───────────────────────────────────────────
    crypto_state = _load(CRYPTO_FILE)
    if crypto_state:
        c_cash     = float(crypto_state.get("cash_usdt", 0))
        c_holdings = crypto_state.get("holdings", {}) if isinstance(crypto_state.get("holdings"), dict) else {}
        c_log      = crypto_state.get("trade_log", [])
        c_mode_raw = crypto_state.get("mode", "stable")
        c_mode     = "paper"

        gauge("crypto_cash_usdt",      c_cash,                 f'mode="{c_mode}",strategy="{c_mode_raw}",label="紙上交易"', "Crypto cash USDT")
        gauge("crypto_position_count", float(len(c_holdings)), f'mode="{c_mode}",strategy="{c_mode_raw}",label="紙上交易"', "Crypto open positions")

        c_holding_val = 0.0
        for symbol, h in c_holdings.items():
            if isinstance(h, dict):
                amount     = float(h.get("amount", h.get("shares", 0)))
                avg_price  = float(h.get("avg_price", h.get("entry_price", 0)))
                cost       = float(h.get("cost", amount * avg_price))
                val        = amount * avg_price
                c_holding_val += val
                lbl = f'mode="{c_mode}",symbol="{symbol}"'
                gauge("crypto_position_amount",      amount,    lbl, "Crypto amount held")
                gauge("crypto_position_avg_price",   avg_price, lbl, "Crypto avg entry price USDT")
                gauge("crypto_position_cost",        cost,      lbl, "Crypto total cost USDT")
                gauge("crypto_position_unrealized_pnl", val - cost, lbl, "Crypto unrealized PnL USDT")

        gauge("crypto_holding_value_usdt", c_holding_val,           f'mode="{c_mode}",label="紙上交易"', "Crypto holding value USDT")
        gauge("crypto_total_value_usdt",   c_cash + c_holding_val,  f'mode="{c_mode}",label="紙上交易"', "Crypto total value USDT")

        c_buys  = [t for t in c_log if t.get("action") == "BUY"]
        c_sells = [t for t in c_log if t.get("action") == "SELL"]
        gauge("crypto_total_buy_trades",  float(len(c_buys)),  f'mode="{c_mode}"', "Crypto total buy trades")
        gauge("crypto_total_sell_trades", float(len(c_sells)), f'mode="{c_mode}"', "Crypto total sell trades")

    # ── 當沖（paper only）─────────────────────────────────────────────────────
    day_state = _load(DAY_FILE)
    if day_state:
        gauge("day_trade_cash_twd",     float(day_state.get("cash",      0)),   "", "Day trade paper cash TWD")
        gauge("day_trade_daily_pnl",    float(day_state.get("daily_pnl", 0)),   "", "Day trade daily PnL TWD")
        gauge("day_trade_positions",    float(len(day_state.get("holdings",  {}))), "", "Day trade open positions")
        gauge("day_trade_watch_count",  float(len(day_state.get("watch_log", {}))), "", "Stocks being watched")
        gauge("day_trade_trades_today", float(len(day_state.get("trade_log", []))), "", "Trades executed today")

    # ── 隔日沖（paper only）───────────────────────────────────────────────────
    ov_state = _load(OVERNIGHT_FILE)
    if ov_state:
        ov_trades = ov_state.get("trade_log", [])
        ov_sells  = [t for t in ov_trades if t.get("side") == "SELL"]
        gauge("overnight_cash_twd",     float(ov_state.get("cash",      0)),   'mode="paper",label="隔日沖"', "Overnight trade cash TWD")
        gauge("overnight_total_pnl",    float(ov_state.get("total_pnl", 0)),   'mode="paper",label="隔日沖"', "Overnight total PnL TWD")
        gauge("overnight_positions",    float(len(ov_state.get("holdings", {}))), 'mode="paper",label="隔日沖"', "Overnight open positions")
        gauge("overnight_trades_total", float(len(ov_trades)),                 'mode="paper",label="隔日沖"', "Overnight total trades")
        gauge("overnight_sells_total",  float(len(ov_sells)),                  'mode="paper",label="隔日沖"', "Overnight completed trades")

    return "\n".join(lines) + "\n"


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            body = build_metrics().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), MetricsHandler)
    print(f"Trading exporter listening on :{PORT}")
    server.serve_forever()
