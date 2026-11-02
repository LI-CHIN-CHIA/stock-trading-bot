"""
多源社群情緒分析
=================
整合以下來源對特定台股的情緒：
  1. PTT 股版（散戶討論，推噓數加權）
  2. Meta Threads（台灣 KOL 與散戶短文）
  3. Google News RSS（含 CMoney、今周刊、UDN、商業周刊等）

注意：Dcard 被 Cloudflare 封鎖，目前無法直接爬取。

回傳:
    {
        "score":      float,    # -1.0（極看空）~ +1.0（極看多）
        "label":      str,      # "看多" / "看空" / "中性"
        "confidence": float,    # 0~1，文章數越多越可信
        "sources": {
            "ptt":   dict,      # PTT 各別結果
            "news":  dict,      # Google News 各別結果
        },
        "summary":    str,      # 綜合摘要
    }
"""

import logging
import os
import re
import time
import urllib.parse
from datetime import datetime, timedelta
from threading import Lock

logger = logging.getLogger(__name__)

_cache: dict = {}
_cache_lock = Lock()


# ── Google News RSS ──────────────────────────────────────────────────────────

def _fetch_google_news(stock_code: str, company_name: str = "", days: int = 7) -> list[dict]:
    """從 Google News RSS 抓取相關新聞與討論。"""
    try:
        from curl_cffi import requests as cf_requests

        keywords = stock_code
        if company_name:
            keywords = f"{company_name} OR {stock_code}"
        query = urllib.parse.quote(keywords)
        url = (
            f"https://news.google.com/rss/search"
            f"?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        )
        resp = cf_requests.get(url, impersonate="chrome124", timeout=10)
        if resp.status_code != 200:
            return []

        items = []
        cutoff = datetime.now() - timedelta(days=days)

        # 解析 RSS
        entry_blocks = re.findall(r"<item>(.*?)</item>", resp.text, re.DOTALL)
        for block in entry_blocks:
            title_m   = re.search(r"<title>(.*?)</title>", block, re.DOTALL)
            source_m  = re.search(r"<source[^>]*>(.*?)</source>", block, re.DOTALL)
            date_m    = re.search(r"<pubDate>(.*?)</pubDate>", block)
            link_m    = re.search(r"<link>(.*?)</link>", block, re.DOTALL)

            title  = re.sub(r"<[^>]+>", "", title_m.group(1)).strip() if title_m else ""
            source = re.sub(r"<[^>]+>", "", source_m.group(1)).strip() if source_m else ""
            date_s = date_m.group(1).strip() if date_m else ""
            link   = link_m.group(1).strip() if link_m else ""

            if not title:
                continue

            # 日期過濾
            try:
                from email.utils import parsedate_to_datetime
                pub_date = parsedate_to_datetime(date_s).replace(tzinfo=None)
                if pub_date < cutoff:
                    continue
            except Exception:
                pass

            items.append({
                "title":  title,
                "source": source,
                "date":   date_s[:16],
                "link":   link,
            })

        return items[:30]
    except Exception as e:
        logger.debug(f"Google News RSS 失敗: {e}")
        return []


def _analyze_news_sentiment(items: list[dict], stock_code: str, company_name: str) -> dict:
    """用 LLM 分析新聞情緒。"""
    if not items:
        return {"score": 0.0, "label": "中性", "summary": "無新聞資料"}
    try:
        import ollama
        model = os.getenv("TA_QUICK_MODEL", "qwen2.5:7b")

        text = ""
        for i, item in enumerate(items[:15], 1):
            text += f"[{i}] {item['source']}: {item['title']}\n"

        prompt = f"""分析以下有關 {stock_code}{' (' + company_name + ')' if company_name else ''} 的新聞標題，判斷市場情緒。

{text}

請回答（格式固定）：
情緒：【看多】/【看空】/【中性】
分數：（-1.0到+1.0）
摘要：（30字以內）"""

        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        raw = resp["message"]["content"]

        score_m    = re.search(r"分數[：:]\s*([+-]?\d+\.?\d*)", raw)
        sentiment_m = re.search(r"情緒[：:]\s*【(看多|看空|中性)】", raw)
        summary_m  = re.search(r"摘要[：:]\s*(.+)", raw)

        score = float(score_m.group(1)) if score_m else 0.0
        score = max(-1.0, min(1.0, score))
        return {
            "score":   score,
            "label":   sentiment_m.group(1) if sentiment_m else "中性",
            "summary": summary_m.group(1).strip() if summary_m else raw[:80],
            "count":   len(items),
        }
    except Exception as e:
        logger.warning(f"新聞情緒 LLM 失敗: {e}")
        return {"score": 0.0, "label": "中性", "summary": "分析失敗", "count": len(items)}


# ── 合併分析 ─────────────────────────────────────────────────────────────────

def get_forum_sentiment(
    stock_code: str,
    company_name: str = "",
    days: int = 5,
) -> dict:
    """
    多源社群情緒分析（PTT + Google News）。

    Args:
        stock_code:   股票代號，如 "2330"
        company_name: 公司名稱，如 "台積電"
        days:         近幾天的資料

    Returns:
        dict with keys: score, label, confidence, sources, summary
    """
    cache_key = f"forum:{stock_code}:{datetime.now().strftime('%Y%m%d_%H')}"
    with _cache_lock:
        if cache_key in _cache:
            logger.debug(f"論壇情緒快取命中: {stock_code}")
            return _cache[cache_key]

    results: dict[str, dict] = {}

    # ── PTT ──────────────────────────────────────────────────────────────
    try:
        from utils.ptt_sentiment import get_ptt_sentiment
        ptt = get_ptt_sentiment(stock_code, company_name, days=days, fetch_content=False)
        results["ptt"] = ptt
        logger.info(f"PTT 情緒 {stock_code}: {ptt['label']} ({ptt['score']:+.2f}, {ptt['post_count']}篇)")
    except Exception as e:
        logger.warning(f"PTT 情緒失敗: {e}")
        results["ptt"] = {"score": 0.0, "label": "中性", "post_count": 0, "summary": ""}

    # ── Meta Threads ──────────────────────────────────────────────────────
    try:
        from utils.threads_sentiment import get_threads_sentiment
        threads = get_threads_sentiment(stock_code, company_name)
        results["threads"] = threads
        logger.info(f"Threads 情緒 {stock_code}: {threads['label']} ({threads['score']:+.2f}, {threads['count']}篇)")
    except Exception as e:
        logger.warning(f"Threads 情緒失敗: {e}")
        results["threads"] = {"score": 0.0, "label": "中性", "count": 0, "summary": ""}

    # ── Google News ────────────────────────────────────────────────────────
    try:
        news_items = _fetch_google_news(stock_code, company_name, days=days)
        news = _analyze_news_sentiment(news_items, stock_code, company_name)
        news["items"] = [{"title": n["title"], "source": n["source"]} for n in news_items[:10]]
        results["news"] = news
        logger.info(f"Google News 情緒 {stock_code}: {news['label']} ({news['score']:+.2f}, {news['count']}篇)")
    except Exception as e:
        logger.warning(f"Google News 情緒失敗: {e}")
        results["news"] = {"score": 0.0, "label": "中性", "count": 0, "summary": ""}

    # ── 加權合併 ───────────────────────────────────────────────────────────
    # PTT 0.35（散戶情緒）、Threads 0.35（KOL+散戶）、Google News 0.30（媒體）
    ptt_count     = results["ptt"].get("post_count", 0)
    threads_count = results["threads"].get("count", 0)
    news_count    = results["news"].get("count", 0)
    total         = ptt_count + threads_count + news_count

    if total == 0:
        combined_score = 0.0
        confidence     = 0.0
    else:
        # 動態權重：依各來源文章數比例，再加基礎權重
        w_ptt     = (ptt_count     / total) * 0.35 + 0.117
        w_threads = (threads_count / total) * 0.35 + 0.117
        w_news    = (news_count    / total) * 0.30 + 0.100
        norm      = w_ptt + w_threads + w_news
        combined_score = (
            results["ptt"]["score"]     * w_ptt +
            results["threads"]["score"] * w_threads +
            results["news"]["score"]    * w_news
        ) / norm
        combined_score = max(-1.0, min(1.0, combined_score))
        confidence = min(0.9, total / 40)

    if combined_score >= 0.3:
        label = "看多"
    elif combined_score <= -0.3:
        label = "看空"
    else:
        label = "中性"

    # 彙整摘要
    parts = []
    if results["ptt"].get("summary"):
        parts.append(f"PTT: {results['ptt']['summary']}")
    if results.get("threads", {}).get("summary"):
        parts.append(f"Threads: {results['threads']['summary']}")
    if results["news"].get("summary"):
        parts.append(f"新聞: {results['news']['summary']}")
    summary = "；".join(parts) if parts else "無足夠資料"

    result = {
        "score":      round(combined_score, 3),
        "label":      label,
        "confidence": round(confidence, 3),
        "sources":    results,
        "summary":    summary,
    }

    with _cache_lock:
        _cache[cache_key] = result

    logger.info(
        f"綜合論壇情緒 {stock_code}: {label} "
        f"(分數={combined_score:+.2f}, 信心={confidence:.0%}, "
        f"PTT={ptt_count}篇, Threads={threads_count}篇, 新聞={news_count}篇)"
    )
    return result
