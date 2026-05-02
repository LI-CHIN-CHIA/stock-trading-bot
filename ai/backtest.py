"""
Backtesting engine for the AI trading strategy.
Simulates portfolio trading using model predictions on historical data.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from ai.features import build_feature_matrix, build_target
from ai.model import TradingModel
from trader.risk import ATR_MULTIPLIER
from utils.constants import STOCK_DB, OTC_SYMBOLS, TRADEABLE

logger = logging.getLogger(__name__)

CAPITAL = 20_000          # Starting capital (TWD)
MAX_POSITIONS = 5         # Max concurrent holdings
MAX_HOLD_DAYS = 10        # Force exit after 10 trading days (~2 weeks)
STOP_LOSS = -0.07         # -7% stop loss
TAKE_PROFIT = 0.10        # +10% take profit
MIN_BUY_PROBA = 0.52      # Minimum AI buy probability to enter
TRANSACTION_FEE = 0.001425  # 0.1425% buy/sell commission
TAX_RATE = 0.003          # 0.3% sell tax


def get_symbol(code: str) -> str:
    return f"{code}.TWO" if code in OTC_SYMBOLS else f"{code}.TW"


@dataclass
class Trade:
    ticker: str
    entry_date: str
    entry_price: float
    shares: int
    cost: float
    exit_date: str = ""
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    total_return_pct: float
    annualised_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    final_capital: float
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trades: list = field(default_factory=list)


def run_backtest(
    model: TradingModel,
    period: str = "2y",
    capital: float = CAPITAL,
    verbose: bool = True,
) -> BacktestResult:
    """
    Walk-forward backtest across all STOCK_DB tickers.
    Uses first 70% of data to train, remaining 30% to simulate trading.
    """
    codes = list(STOCK_DB.keys())

    # ── Fetch historical data ─────────────────────────────────────────────────
    if verbose:
        print(f"📥 下載 {len(codes)} 支股票歷史資料...")

    raw_data: dict[str, pd.DataFrame] = {}
    for code in codes:
        sym = get_symbol(code)
        try:
            df = yf.Ticker(sym).history(period=period)
            if len(df) >= 80:
                raw_data[code] = df
        except Exception:
            pass

    if verbose:
        print(f"✅ 成功取得 {len(raw_data)} 支")

    # ── Prepare features ──────────────────────────────────────────────────────
    all_feat, all_target = [], []
    ticker_dfs: dict[str, tuple[pd.DataFrame, pd.Series]] = {}

    for code, df in raw_data.items():
        try:
            feat = build_feature_matrix(df, code=code)  # 傳入 code，回測包含三大法人特徵（只爬最近 45 天）
            target = build_target(df)
            common = feat.index.intersection(target.index)
            feat = feat.loc[common]
            target = target.loc[common]
            if len(feat) >= 60:
                ticker_dfs[code] = (feat, target)
                all_feat.append(feat)
                all_target.append(target)
        except Exception as e:
            logger.debug(f"{code}: {e}")

    if not all_feat:
        raise RuntimeError("No valid feature data for backtest")

    X_all = pd.concat(all_feat).sort_index()
    y_all = pd.concat(all_target).sort_index()
    common = X_all.index.intersection(y_all.index)
    X_all = X_all.loc[common]
    y_all = y_all.loc[common]

    # ── Train on first 70% ────────────────────────────────────────────────────
    split = int(len(X_all) * 0.70)
    X_train = X_all.iloc[:split]
    y_train = y_all.iloc[:split]

    if verbose:
        print("🤖 訓練 AI 模型...")
    metrics = model.train(X_train, y_train)
    if verbose:
        print(f"   驗證 F1={metrics['val_macro_f1']:.3f}, Buy F1={metrics['val_buy_f1']:.3f}")

    # ── Simulate trading on remaining 30% ────────────────────────────────────
    test_start = X_all.index[split]
    if verbose:
        print(f"📊 模擬交易期間: {str(test_start)[:10]} 起")

    # ── Market regime filter: use 0050 as benchmark ───────────────────────────
    benchmark_df = raw_data.get("0050")
    def _market_is_bullish(date) -> bool:
        """Return False if 0050 is below its 60-day MA (bear market filter)."""
        if benchmark_df is None or benchmark_df.empty:
            return True
        try:
            sub = benchmark_df.loc[:date]["Close"]
            if len(sub) < 60:
                return True
            return float(sub.iloc[-1]) > float(sub.iloc[-60:].mean())
        except Exception:
            return True

    cash = capital  # ✅ persistent cash — never recalculated from scratch
    holdings: dict[str, dict] = {}   # code -> {shares, entry_price, entry_date, cost, hold_days}
    equity_curve: dict = {}
    completed_trades: list[Trade] = []

    # Get sorted unique test dates across all tickers
    test_dates = sorted(set(
        idx for code, (feat, _) in ticker_dfs.items()
        for idx in feat.index if idx >= test_start
    ))

    for date in test_dates:
        # ── Check stop-loss / take-profit / max hold for existing holdings ───
        for code in list(holdings.keys()):
            h = holdings[code]
            price = _get_price(raw_data.get(code), date)
            if price <= 0:
                continue
            pnl_pct = (price - h["entry_price"]) / h["entry_price"]
            h["hold_days"] = h.get("hold_days", 0) + 1

            # Update ATR trailing peak (never moves down)
            h["peak_price"] = max(h.get("peak_price", h["entry_price"]), price)

            reason = None
            feat, _ = ticker_dfs.get(code, (None, None))

            # Get current ATR from feature matrix if available
            atr = 0.0
            pred = None
            if feat is not None and date in feat.index:
                row = feat.loc[[date]]
                atr = float(row["ATR"].iloc[0]) if "ATR" in row.columns else 0.0
                pred = model.predict(row)

            # ── 1. ATR trailing stop (primary) ────────────────────────────────
            peak = h["peak_price"]
            if atr > 0 and price <= peak - ATR_MULTIPLIER * atr:
                reason = f"ATR追蹤停損 {pnl_pct:.1%}"
            # ── 2. Fixed stop-loss fallback (when ATR unavailable) ────────────
            elif atr <= 0 and pnl_pct <= STOP_LOSS:
                reason = f"停損 {pnl_pct:.1%}"
            # ── 3. Strategy B: AI confirms take-profit ────────────────────────
            elif pnl_pct >= TAKE_PROFIT:
                if pred and pred["signal"] == "BUY" and pred.get("buy_proba", 0) >= 0.70:
                    pass  # hold — AI still bullish
                else:
                    reason = f"停利 {pnl_pct:.1%}"
            # ── 4. Max hold period ────────────────────────────────────────────
            elif h["hold_days"] >= MAX_HOLD_DAYS:
                reason = f"超過持有期 ({pnl_pct:.1%})"
            # ── 5. AI sell signal ─────────────────────────────────────────────
            elif pred and pred["signal"] == "SELL" and pred["sell_proba"] >= 0.55:
                reason = f"AI賣出 {pred['sell_proba']:.0%}"

            if reason:
                sell_value = h["shares"] * price * (1 - TAX_RATE - TRANSACTION_FEE)
                trade_pnl = sell_value - h["cost"]
                completed_trades.append(Trade(
                    ticker=code,
                    entry_date=h["entry_date"],
                    entry_price=h["entry_price"],
                    shares=h["shares"],
                    cost=h["cost"],
                    exit_date=str(date)[:10],
                    exit_price=price,
                    pnl=round(trade_pnl, 2),
                    pnl_pct=round(pnl_pct, 4),
                    reason=reason,
                ))
                cash += sell_value  # ✅ cash grows when we sell
                del holdings[code]

        # ── Look for new BUY signals (only in bullish market, only TRADEABLE) ──
        if len(holdings) < MAX_POSITIONS and cash > 500 and _market_is_bullish(date):
            candidates = []
            for code, (feat, _) in ticker_dfs.items():
                if code not in TRADEABLE or code in holdings or date not in feat.index:
                    continue
                row = feat.loc[[date]]
                pred = model.predict(row)
                if pred["signal"] == "BUY" and pred["buy_proba"] >= MIN_BUY_PROBA:
                    entry_atr = float(row["ATR"].iloc[0]) if "ATR" in row.columns else 0.0
                    candidates.append((code, pred["buy_proba"], entry_atr))

            # Pick top candidates by buy probability
            candidates.sort(key=lambda x: x[1], reverse=True)
            slots = MAX_POSITIONS - len(holdings)
            budget_per_slot = cash / max(slots, 1)

            for code, buy_proba, entry_atr in candidates[:slots]:
                price = _get_price(raw_data.get(code), date)
                if price <= 0 or budget_per_slot < price:
                    continue
                shares = int(budget_per_slot / price)
                if shares < 1:
                    continue
                cost = shares * price * (1 + TRANSACTION_FEE)
                if cost > cash:
                    continue
                holdings[code] = {
                    "shares": shares,
                    "entry_price": price,
                    "entry_date": str(date)[:10],
                    "cost": cost,
                    "buy_proba": buy_proba,
                    "hold_days": 0,
                    "atr": entry_atr,
                    "peak_price": price,
                }
                cash -= cost  # ✅ cash decreases when we buy

        # ── Record portfolio value ────────────────────────────────────────────
        holding_value = sum(
            h["shares"] * _get_price(raw_data.get(c), date)
            for c, h in holdings.items()
        )
        equity_curve[date] = cash + holding_value  # ✅ reflects unrealized gains

    # ── Compute metrics ───────────────────────────────────────────────────────
    equity = pd.Series(equity_curve).sort_index()
    if equity.empty:
        equity = pd.Series([capital])

    final_cap = equity.iloc[-1]
    total_return = (final_cap - capital) / capital
    days = max((equity.index[-1] - equity.index[0]).days, 1) if len(equity) > 1 else 1
    ann_return = (1 + total_return) ** (365 / days) - 1

    daily_ret = equity.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = drawdown.min()

    wins = [t for t in completed_trades if t.pnl > 0]
    losses = [t for t in completed_trades if t.pnl <= 0]
    win_rate = len(wins) / max(len(completed_trades), 1)

    if verbose:
        print(f"\n{'='*40}")
        print(f"📈 回測結果")
        print(f"{'='*40}")
        print(f"總報酬:     {total_return:+.2%}")
        print(f"年化報酬:   {ann_return:+.2%}")
        print(f"夏普比率:   {sharpe:.2f}")
        print(f"最大回撤:   {max_dd:.2%}")
        print(f"勝率:       {win_rate:.1%}")
        print(f"總交易次數: {len(completed_trades)}")
        print(f"期末資金:   {final_cap:,.0f} 元")
        print(f"{'='*40}")

    return BacktestResult(
        total_return_pct=round(total_return * 100, 2),
        annualised_return_pct=round(ann_return * 100, 2),
        sharpe_ratio=round(float(sharpe), 3),
        max_drawdown_pct=round(float(max_dd) * 100, 2),
        win_rate_pct=round(win_rate * 100, 1),
        total_trades=len(completed_trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        final_capital=round(final_cap, 0),
        equity_curve=equity,
        trades=completed_trades,
    )


def _get_price(df: pd.DataFrame | None, date) -> float:
    """Get close price at or before a given date."""
    if df is None or df.empty:
        return 0.0
    try:
        row = df.loc[:date].iloc[-1]
        return float(row["Close"])
    except Exception:
        return 0.0
