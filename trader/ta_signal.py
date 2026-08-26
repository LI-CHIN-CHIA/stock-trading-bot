"""
TradingAgents 台股整合模組
===========================
支援 Ollama（本地）與 FreeToken（OpenAI 相容 API）兩種後端。

環境變數:
  ENABLE_TRADING_AGENTS=true   啟用（預設 false）
  TA_PROVIDER=ollama           LLM 後端：ollama | openai（FreeToken 用 openai）
  TA_BASE_URL=                 OpenAI 相容端點（FreeToken: http://host:1919/v1）
  TA_DEEP_MODEL=               深度思考模型
  TA_QUICK_MODEL=              快速模型
  TA_RESULTS_DIR=              分析結果目錄
  TA_HOLDING_TTL_MIN=90        持倉監控快取（分鐘）
  TA_SCAN_TTL_MIN=480          掃股快取（分鐘）
"""

import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ENABLE_TA        = os.getenv("ENABLE_TRADING_AGENTS", "false").lower() == "true"
ENABLE_PTT       = os.getenv("ENABLE_PTT_SENTIMENT", "true").lower() == "true"
TA_PROVIDER      = os.getenv("TA_PROVIDER", "ollama")   # "ollama" | "openai"（FreeToken）
TA_BASE_URL      = os.getenv("TA_BASE_URL", "")         # FreeToken: http://localhost:1919/v1
DEEP_MODEL       = os.getenv("TA_DEEP_MODEL",  "deepseek-r1:32b")
QUICK_MODEL      = os.getenv("TA_QUICK_MODEL", "qwen2.5:7b")
RESULTS_DIR      = Path(os.getenv("TA_RESULTS_DIR",
                       Path(__file__).parent.parent / "data" / "ta_results"))
HOLDING_TTL_MIN  = int(os.getenv("TA_HOLDING_TTL_MIN", "90"))
SCAN_TTL_MIN     = int(os.getenv("TA_SCAN_TTL_MIN",    "480"))

# TTL 快取：{ cache_key -> {result: dict, expires_at: datetime} }
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()
_log_lock   = threading.Lock()


def _build_graph():
    """建立 TradingAgentsGraph (只在首次使用時初始化)。"""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.config import TradingAgentsConfig

    provider = TA_PROVIDER  # "ollama" 或 "openai"（FreeToken）

    # FreeToken / 自訂 OpenAI 相容端點：設定環境變數讓 langchain-openai 自動接收
    if provider == "openai" and TA_BASE_URL:
        os.environ.setdefault("OPENAI_API_KEY", "freetoken")
        os.environ["OPENAI_BASE_URL"] = TA_BASE_URL
        logger.info(f"TradingAgents 使用 OpenAI 相容後端: {TA_BASE_URL}")
    elif provider == "ollama":
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        logger.info(f"TradingAgents 使用 Ollama 後端: {ollama_host}")

    config = TradingAgentsConfig(
        llm_provider=provider,
        deep_think_llm=DEEP_MODEL,
        quick_think_llm=QUICK_MODEL,
        response_language="繁體中文（Traditional Chinese）。請務必全程使用繁體中文，嚴禁使用簡體中文。",
        max_debate_rounds=1,
        max_risk_discuss_rounds=1,
        max_recur_limit=50,
        results_dir=RESULTS_DIR,
    )
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
    ttl_override: int | None = None,
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
    ttl_min   = ttl_override if ttl_override is not None else (
                HOLDING_TTL_MIN if mode == "holding" else SCAN_TTL_MIN)
    cache_key = f"{code}:{date_str}:{mode}:{ttl_min}"
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
        state, recommendation = graph.propagate(code, date_str)

        # v0.7.0+ returns TradeRecommendation object; v0.3.x returns plain str
        if isinstance(recommendation, str):
            decision = recommendation
            signal = _parse_decision(decision)
            confidence = _confidence_from_decision(decision)
        else:
            # TradeRecommendation Pydantic model
            sig_val = recommendation.signal
            decision = str(sig_val.value if hasattr(sig_val, "value") else sig_val)
            signal = _parse_decision(decision)
            confidence = float(recommendation.confidence)

        result = {
            "signal": signal,
            "confidence": confidence,
            "full_decision": decision,
            "source": "trading_agents",
            "queried_at": now.strftime("%Y-%m-%d %H:%M"),
        }

        with _cache_lock:
            _cache[cache_key] = {
                "result": result,
                "expires_at": now + timedelta(minutes=ttl_min),
            }
            # 清除已過期的 entry，防止 _cache 無限成長
            expired_keys = [k for k, v in _cache.items() if now >= v["expires_at"]]
            for k in expired_keys:
                del _cache[k]

        # ── PTT 情緒加權 ─────────────────────────────────────────────────
        ptt = None
        if ENABLE_PTT and mode == "scan":
            try:
                from utils.forum_sentiment import get_forum_sentiment
                from utils.constants import STOCK_DB
                company_name = STOCK_DB.get(code, {}).get("name", "") if isinstance(STOCK_DB.get(code), dict) else ""
                ptt = get_forum_sentiment(code, company_name, days=5)
                # 情緒與訊號方向一致 → 提升信心；方向相反 → 降低信心
                if ptt["post_count"] >= 3:
                    boost = ptt["score"] * 0.1   # 最多 ±10% 調整
                    if signal == "BUY":
                        result["confidence"] = min(0.95, result["confidence"] + boost)
                    elif signal == "SELL":
                        result["confidence"] = min(0.95, result["confidence"] - boost)
                    logger.info(
                        f"PTT 情緒 {code}: {ptt['label']} (分數={ptt['score']:+.2f}, "
                        f"文章={ptt['post_count']}篇) → 調整信心至 {result['confidence']:.0%}"
                    )
                result["ptt_sentiment"] = ptt
            except Exception as e:
                logger.debug(f"PTT 情緒整合失敗 ({code}): {e}")

        # ── 儲存完整溝通紀錄 ─────────────────────────────────────────────
        _save_ta_log(code, date_str, mode, signal, result["confidence"], decision, state, now, ptt)

        logger.info(f"TradingAgents {code}: {signal} 信心={result['confidence']:.0%} "
                    f"(決策: {str(decision)[:60]}…)")
        return result

    except Exception as e:
        logger.warning(f"TradingAgents 分析 {code} 失敗: {e}")
        return None


def _save_ta_log(
    code: str, date_str: str, mode: str,
    signal: str, confidence: float,
    decision: str, state: dict | None, queried_at: datetime,
    ptt: dict | None = None,
) -> None:
    """將 TradingAgents 完整溝通記錄寫入 JSONL 檔案。"""
    import json
    log_dir = Path(os.getenv("DATA_DIR", Path(__file__).parent.parent / "data"))
    log_path = log_dir / "ta_signal_log.jsonl"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 從 AgentState 取出各分析師報告
    messages: list[dict] = []
    if state is not None:
        report_fields = {
            "market":       "market_report",
            "news":         "news_report",
            "social_media": "sentiment_report",
            "fundamentals": "fundamentals_report",
        }
        for agent, field in report_fields.items():
            content = getattr(state, field, None)
            if content and isinstance(content, str) and content.strip():
                messages.append({"agent": agent, "content": content})
        # LangChain messages（對話歷史）
        lc_messages = getattr(state, "messages", None) or []
        for msg in lc_messages:
            content = getattr(msg, "content", None)
            role = getattr(msg, "type", getattr(msg, "role", "unknown"))
            if content and isinstance(content, str) and content.strip():
                messages.append({"agent": "conversation", "role": role, "content": content})

    record = {
        "queried_at": queried_at.strftime("%Y-%m-%d %H:%M:%S"),
        "code": code,
        "date": date_str,
        "mode": mode,
        "signal": signal,
        "confidence": round(confidence, 3),
        "final_decision": decision,
        "messages": messages,
        "ptt_sentiment": {
            "score":      ptt.get("score"),
            "label":      ptt.get("label"),
            "post_count": ptt.get("post_count"),
            "summary":    ptt.get("summary"),
        } if ptt else None,
    }

    try:
        with _log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"TradingAgents log 儲存失敗: {e}")


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
    strong_sell = any(w in upper for w in ["STRONG SELL", "STRONGLY SELL", "CONFIDENTLY SELL"])
    moderate_sell = any(w in upper for w in ["MODERATE SELL", "LEAN SELL", "SLIGHT SELL"])
    if strong_buy:
        return 0.90
    if strong_sell:
        return 0.85
    if moderate_sell:
        return 0.72
    if "BUY" in upper:
        return 0.70
    if "SELL" in upper:
        # 普通 SELL（無強度修飾詞）→ 0.68，低於一般股票門檻 0.75，需更明確才觸發
        return 0.68
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
