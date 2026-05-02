"""
TradingAgents 台股整合模組
===========================
使用本地 Ollama (deepseek-r1:32b) 對台股進行多代理人分析，
作為 LightGBM 模型的第二意見，提高決策品質。

環境變數:
  ENABLE_TRADING_AGENTS=true   啟用（預設 false，避免影響現有流程）
  TA_DEEP_MODEL=deepseek-r1:32b  深度思考模型（研究員/風控）
  TA_QUICK_MODEL=qwen2.5:7b      快速模型（分析師/交易員）
  TA_RESULTS_DIR=./data/ta_results  分析結果儲存目錄
  TA_HOLDING_TTL_MIN=90        持倉監控快取時間（分鐘），預設90分鐘重新詢問
  TA_SCAN_TTL_MIN=480          掃股買入快取時間（分鐘），預設480分鐘（同一天不重複掃）
"""

import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ENABLE_TA        = os.getenv("ENABLE_TRADING_AGENTS", "false").lower() == "true"
DEEP_MODEL       = os.getenv("TA_DEEP_MODEL",  "deepseek-r1:32b")
QUICK_MODEL      = os.getenv("TA_QUICK_MODEL", "qwen2.5:7b")
RESULTS_DIR      = Path(os.getenv("TA_RESULTS_DIR",
                       Path(__file__).parent.parent / "data" / "ta_results"))
HOLDING_TTL_MIN  = int(os.getenv("TA_HOLDING_TTL_MIN", "90"))   # 持倉監控快取（分鐘）
SCAN_TTL_MIN     = int(os.getenv("TA_SCAN_TTL_MIN",    "480"))  # 掃股快取（分鐘）

# TTL 快取：{ cache_key -> {result: dict, expires_at: datetime} }
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()


def _build_graph():
    """建立 TradingAgentsGraph (只在首次使用時初始化)。"""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.config import TradingAgentsConfig

    config = TradingAgentsConfig(
        llm_provider="ollama",
        deep_think_llm=DEEP_MODEL,
        quick_think_llm=QUICK_MODEL,
        response_language="zh-TW",
        max_debate_rounds=1,
        max_risk_discuss_rounds=1,
        max_recur_limit=50,
        results_dir=RESULTS_DIR,
    )
    # 只用市場面與新聞面分析，減少 token 消耗與延遲
    graph = TradingAgentsGraph(
        selected_analysts=["market", "news", "fundamentals"],
        config=config,
    )
    return graph


_graph = None
_graph_lock = threading.Lock()


def _get_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                logger.info("初始化 TradingAgentsGraph (Ollama)…")
                _graph = _build_graph()
                logger.info("TradingAgentsGraph 初始化完成")
    return _graph


def get_ta_signal(
    code: str,
    trade_date: str | None = None,
    mode: Literal["holding", "scan"] = "scan",
) -> dict | None:
    """
    呼叫 TradingAgents 多代理人框架，取得台股分析訊號。

    Args:
        code: 台股代號，如 "2330"（不含後綴）
        trade_date: 分析日期 YYYY-MM-DD，預設為今天
        mode: "holding" = 持倉監控（快取 HOLDING_TTL_MIN 分鐘，頻繁重問）
              "scan"    = 掃股買入（快取 SCAN_TTL_MIN 分鐘，避免重複）

    Returns:
        dict with keys: signal ("BUY"/"SELL"/"HOLD"), confidence (0~1),
                        full_decision (str), source ("trading_agents")
        None if disabled or error.
    """
    if not ENABLE_TA:
        return None

    date_str  = trade_date or datetime.now().strftime("%Y-%m-%d")
    ttl_min   = HOLDING_TTL_MIN if mode == "holding" else SCAN_TTL_MIN
    cache_key = f"{code}:{date_str}:{mode}"
    now       = datetime.now()

    with _cache_lock:
        entry = _cache.get(cache_key)
        if entry and now < entry["expires_at"]:
            remaining = int((entry["expires_at"] - now).total_seconds() / 60)
            logger.debug(f"TradingAgents 快取命中: {code} ({mode}, 剩餘{remaining}分鐘)")
            return entry["result"]

    try:
        graph = _get_graph()
        logger.info(f"TradingAgents 分析 {code} ({date_str}, mode={mode})…")
        _state, decision = graph.propagate(code, date_str)

        signal = _parse_decision(decision)
        result = {
            "signal": signal,
            "confidence": _confidence_from_decision(decision),
            "full_decision": decision,
            "source": "trading_agents",
            "queried_at": now.strftime("%Y-%m-%d %H:%M"),
        }

        with _cache_lock:
            _cache[cache_key] = {
                "result": result,
                "expires_at": now + timedelta(minutes=ttl_min),
            }

        logger.info(f"TradingAgents {code}: {signal} 信心={result['confidence']:.0%} "
                    f"(決策: {decision[:60]}…)")
        return result

    except Exception as e:
        logger.warning(f"TradingAgents 分析 {code} 失敗: {e}")
        return None


def _parse_decision(decision: str) -> str:
    """從 TradingAgents 完整決策文字中萃取 BUY/SELL/HOLD。"""
    upper = decision.upper()
    # 優先抓最後出現的明確關鍵字
    for keyword in reversed(["BUY", "SELL", "HOLD"]):
        if keyword in upper:
            return keyword
    return "HOLD"


def _confidence_from_decision(decision: str) -> float:
    """根據決策文字的強度給一個粗略信心分數（0.5 ~ 0.95）。"""
    upper = decision.upper()
    strong_buy  = any(w in upper for w in ["STRONG BUY", "STRONGLY BUY", "CONFIDENTLY BUY"])
    strong_sell = any(w in upper for w in ["STRONG SELL", "STRONGLY SELL"])
    if strong_buy or strong_sell:
        return 0.90
    if "BUY" in upper or "SELL" in upper:
        return 0.70
    return 0.55  # HOLD or ambiguous


def combine_signals(lgbm_signal: dict | None, ta_signal: dict | None) -> dict | None:
    """
    合併 LightGBM 與 TradingAgents 訊號。

    策略:
    - 若只有一個來源，直接使用該來源。
    - 兩者都是 BUY → 提升 buy_proba (取最大值再加成 10%)
    - 兩者都是 SELL → 提升 sell_proba
    - 兩者相反 → 保守，改為 HOLD（除非 lgbm buy_proba > 0.75）
    - TA 為 HOLD → 維持 lgbm 訊號但信心微降

    Returns:
        合併後的 signal dict（欄位與 lgbm signal 相同，新增 ta_signal 欄位）
    """
    if lgbm_signal is None and ta_signal is None:
        return None
    if lgbm_signal is None:
        # 只有 TA，轉換成 lgbm 格式
        sig = ta_signal["signal"]
        conf = ta_signal["confidence"]
        return {
            "signal": sig,
            "buy_proba":  conf if sig == "BUY"  else 0.1,
            "sell_proba": conf if sig == "SELL" else 0.1,
            "hold_proba": conf if sig == "HOLD" else 0.1,
            "trained": False,
            "ta_signal": ta_signal,
        }
    if ta_signal is None:
        lgbm_signal["ta_signal"] = None
        return lgbm_signal

    result = dict(lgbm_signal)
    result["ta_signal"] = ta_signal
    lgbm_sig = lgbm_signal.get("signal", "HOLD")
    ta_sig   = ta_signal["signal"]

    if lgbm_sig == "BUY" and ta_sig == "BUY":
        # 兩者一致看多 → 提升信心
        result["buy_proba"] = min(lgbm_signal.get("buy_proba", 0.5) * 1.10, 0.98)
        result["signal"] = "BUY"
    elif lgbm_sig == "SELL" and ta_sig == "SELL":
        result["sell_proba"] = min(lgbm_signal.get("sell_proba", 0.5) * 1.10, 0.98)
        result["signal"] = "SELL"
    elif lgbm_sig == "BUY" and ta_sig == "SELL":
        # 訊號衝突 → 保守
        buy_p = lgbm_signal.get("buy_proba", 0.5)
        if buy_p >= 0.75:
            logger.info(f"訊號衝突但 LightGBM 強烈看多 ({buy_p:.0%})，維持 BUY")
            result["signal"] = "BUY"
            result["buy_proba"] = buy_p * 0.90  # 輕微降低
        else:
            result["signal"] = "HOLD"
    elif lgbm_sig == "SELL" and ta_sig == "BUY":
        result["signal"] = "HOLD"
    elif ta_sig == "HOLD":
        # TA 中性 → 維持 lgbm 但信心降 5%
        if lgbm_sig == "BUY":
            result["buy_proba"] = lgbm_signal.get("buy_proba", 0.5) * 0.95
        elif lgbm_sig == "SELL":
            result["sell_proba"] = lgbm_signal.get("sell_proba", 0.5) * 0.95

    return result
