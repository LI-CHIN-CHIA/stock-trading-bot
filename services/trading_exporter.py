"""
Trading Prometheus Exporter
讀取 /data/trader_state.json，轉為 Prometheus metrics，供 Grafana 監控持倉/損益。
Port: 9877
"""
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

STATE_FILE = Path(os.getenv("STATE_FILE", "/data/trader_state.json"))
PAPER_FILE = Path(os.getenv("PAPER_FILE", "/data/paper_trader_state.json"))
PORT = int(os.getenv("EXPORTER_PORT", "9877"))


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def build_metrics() -> str:
    lines: list[str] = []

    def gauge(name: str, value: float, labels: str = "", help_text: str = "", type_: str = "gauge"):
        if not lines or not any(f"# HELP {name}" in l for l in lines):
            if help_text:
                lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {type_}")
        label_str = f"{{{labels}}}" if labels else ""
        lines.append(f"{name}{label_str} {value}")

    for mode, path in [("live", STATE_FILE), ("paper", PAPER_FILE)]:
        state = _load(path)
        if not state:
            continue

        cash = float(state.get("cash", 0))
        holdings: dict = state.get("holdings", {})
        trade_log: list = state.get("trade_log", [])

        # ── 現金 ─────────────────────────────────────────────────────────────
        gauge("trading_cash_twd", cash,
              f'mode="{mode}"',
              "Available cash in TWD")

        # ── 持倉 ─────────────────────────────────────────────────────────────
        holding_value = 0.0
        for ticker, h in holdings.items():
            shares = float(h.get("shares", 0))
            entry_price = float(h.get("entry_price", 0))
            cost = float(h.get("cost", shares * entry_price))
            peak = float(h.get("peak_price", entry_price))
            hold_days = int(h.get("hold_days", 0))
            current_value = shares * entry_price  # 用 entry_price 估算（無即時價格）
            holding_value += current_value
            unrealized_pnl = current_value - cost

            lbl = f'mode="{mode}",ticker="{ticker}"'
            gauge("trading_position_shares",      shares,         lbl, "Shares held")
            gauge("trading_position_entry_price", entry_price,    lbl, "Entry price TWD")
            gauge("trading_position_cost",        cost,           lbl, "Total cost TWD")
            gauge("trading_position_peak_price",  peak,           lbl, "Peak price since entry")
            gauge("trading_position_hold_days",   hold_days,      lbl, "Days held")
            gauge("trading_position_unrealized_pnl", unrealized_pnl, lbl, "Unrealized PnL TWD")

        gauge("trading_holding_value_twd", holding_value,
              f'mode="{mode}"', "Total holding value TWD")
        gauge("trading_total_value_twd", cash + holding_value,
              f'mode="{mode}"', "Total portfolio value TWD")
        gauge("trading_position_count", len(holdings),
              f'mode="{mode}"', "Number of open positions")

        # ── 交易記錄統計 ──────────────────────────────────────────────────────
        buys  = [t for t in trade_log if t.get("action") == "BUY"]
        sells = [t for t in trade_log if t.get("action") == "SELL"]
        gauge("trading_total_buy_trades",  float(len(buys)),  f'mode="{mode}"', "Total buy trades")
        gauge("trading_total_sell_trades", float(len(sells)), f'mode="{mode}"', "Total sell trades")

        # 已實現損益（從 sell reason 估算：有 % 的才計）
        realized = 0.0
        for t in sells:
            reason = t.get("reason", "")
            # reason 格式: "停利 11.6%" 或 "停損 -5.2%"
            import re
            m = re.search(r'([+-]?\d+\.?\d*)%', reason)
            if m:
                pct = float(m.group(1)) / 100
                cost_approx = t.get("shares", 0) * t.get("price", 0) / (1 + pct) if pct != -1 else 0
                realized += cost_approx * pct
        gauge("trading_realized_pnl_twd", realized,
              f'mode="{mode}"', "Estimated realized PnL TWD")

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
        pass  # suppress access log


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), MetricsHandler)
    print(f"Trading exporter listening on :{PORT}")
    server.serve_forever()
