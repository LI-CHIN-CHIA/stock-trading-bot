"""
Stock Discovery Module
======================
每日 08:30 開盤前自動掃描社群與三大法人資料，
決定是否新增或移除科技股交易清單。

資料來源：
  1. PTT 股板最新文章 → 提取頻繁提及的股票代碼
  2. TWSE 三大法人買賣超 → 外資/投信/自營商淨買超股票
  3. yfinance 成交量 → 流動性驗證（日均成交值 > MIN_VOLUME_M 萬）
  4. PTT/Threads 情緒 → 輔助判斷移除現有股票

動態清單：data/dynamic_tradeable.json
  added:     {code: {name, added_date, reason, sector}}
  suspended: {code: {name, suspended_date, reason}}
"""

import json
import logging
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR          = Path(__file__).parent.parent / "data"
DYNAMIC_FILE      = DATA_DIR / "dynamic_tradeable.json"
MIN_VOLUME_M      = 3000       # 日均成交值下限（萬元）
MIN_PTT_MENTIONS  = 3          # PTT 前 100 篇至少提及幾次才列入候選
FOREIGN_NET_THRESHOLD = 500_000  # 外資淨買超股數門檻
MAX_DYNAMIC_ADDED = 10         # 動態新增上限（避免清單爆炸）

# TWSE 行業代碼屬於科技的範圍（24~35 為電子相關）
TECH_INDUSTRY_TWSE = {"24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "35"}


# ── 動態清單 IO ───────────────────────────────────────────────────────────────

def _load_dynamic() -> dict:
    if DYNAMIC_FILE.exists():
        try:
            return json.loads(DYNAMIC_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"added": {}, "suspended": {}}


def _save_dynamic(data: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DYNAMIC_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_effective_tradeable() -> set[str]:
    """
    回傳有效可交易集合：靜態核心 ∪ 動態新增 − 暫停移除。
    供 bot._scan_and_buy() 使用，取代直接引用 TRADEABLE。
    """
    from utils.constants import TRADEABLE
    dyn = _load_dynamic()
    added    = set(dyn.get("added",     {}).keys())
    suspended = set(dyn.get("suspended", {}).keys())
    effective = (TRADEABLE | added) - suspended
    return effective


# ── PTT 股板掃描 ──────────────────────────────────────────────────────────────

def _scan_ptt_board(pages: int = 3) -> Counter:
    """
    爬取 PTT 股板前 N 頁文章標題，統計 4~5 位數股票代碼出現次數。
    回傳 Counter: {code: count}
    """
    try:
        import requests
        headers = {"Cookie": "over18=1", "User-Agent": "Mozilla/5.0"}
        code_counter: Counter = Counter()
        url = "https://www.ptt.cc/bbs/Stock/index.html"

        for _ in range(pages):
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                break
            titles = re.findall(r'class="title"[^>]*>.*?</a>', resp.text, re.DOTALL)
            for t in titles:
                text = re.sub(r"<[^>]+>", "", t)
                # 找 4~5 位數字（台股代碼格式）
                codes = re.findall(r"\b(\d{4,5})\b", text)
                code_counter.update(codes)
            # 上一頁
            prev = re.search(r'href="(/bbs/Stock/index(\d+)\.html)"[^>]*>‹', resp.text)
            if prev:
                url = "https://www.ptt.cc" + prev.group(1)
            else:
                break

        return code_counter
    except Exception as e:
        logger.warning(f"PTT 股板掃描失敗: {e}")
        return Counter()


# ── TWSE 三大法人 ─────────────────────────────────────────────────────────────

def _fetch_twse_institutional(date_str: str = "") -> dict[str, dict]:
    """
    取得指定日期（YYYYMMDD）的 TWSE 三大法人買賣超資料。
    date_str 為空時使用今天。

    回傳：{code: {name, foreign_net, trust_net, dealer_net, total_net}}
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y%m%d")
    try:
        from curl_cffi import requests as cf
        url = (
            f"https://www.twse.com.tw/rwd/zh/fund/T86"
            f"?response=json&date={date_str}&selectType=ALLBUT0999"
        )
        resp = cf.get(url, impersonate="chrome124", timeout=15)
        if resp.status_code != 200:
            return {}
        payload = resp.json()
        if payload.get("stat") != "OK" or not payload.get("data"):
            return {}

        result: dict[str, dict] = {}
        for row in payload["data"]:
            if len(row) < 15:
                continue
            code = row[0].strip()
            name = row[1].strip()

            def _parse(s: str) -> int:
                try:
                    return int(s.replace(",", "").replace("+", ""))
                except Exception:
                    return 0

            result[code] = {
                "name":        name,
                "foreign_net": _parse(row[4]),
                "trust_net":   _parse(row[7]),
                "dealer_net":  _parse(row[10]),
                "total_net":   _parse(row[14]),
            }
        return result
    except Exception as e:
        logger.warning(f"TWSE 三大法人資料取得失敗: {e}")
        return {}


def _classify_buyers(inst: dict) -> str:
    """
    根據三大法人資料判斷主力買方。
    回傳 '外資' / '投信' / '自營商' / '法人合計' / '散戶' / '混合'
    """
    foreign = inst.get("foreign_net", 0)
    trust   = inst.get("trust_net",   0)
    dealer  = inst.get("dealer_net",  0)
    total   = inst.get("total_net",   0)

    buyers = []
    if foreign > FOREIGN_NET_THRESHOLD:
        buyers.append("外資")
    if trust > 0:
        buyers.append("投信")
    if dealer > 0:
        buyers.append("自營商")

    if not buyers:
        return "散戶" if total <= 0 else "混合"
    if len(buyers) == 3:
        return "法人合計"
    return "+".join(buyers)


# ── 股票資格驗證 ──────────────────────────────────────────────────────────────

def _is_tech_stock(code: str) -> bool:
    """
    判斷是否為科技類股：
    1. 先查 SECTOR_MAP tech sectors（已知股票最快）
    2. 再用 yfinance info.sector 判斷
    """
    from utils.constants import SECTOR_MAP, TECH_SECTORS
    # 已知科技股
    for sector in TECH_SECTORS:
        if code in SECTOR_MAP.get(sector, set()):
            return True
    # 新股票：用 yfinance 查行業
    try:
        from utils.constants import OTC_SYMBOLS
        suffix = ".TWO" if code in OTC_SYMBOLS else ".TW"
        info = yf.Ticker(f"{code}{suffix}").fast_info
        # yfinance 對台股 sector/industry 不一定有，嘗試 quoteType
        sector_str = getattr(info, "sector", "") or ""
        # Technology / Electronic / Semiconductor → 科技
        if any(k in sector_str.lower() for k in ["tech", "electron", "semi", "hardware"]):
            return True
    except Exception:
        pass
    # 粗略：代碼在 2000~3999 或 6000~6999 → 多數為電子股
    try:
        n = int(code)
        return (2000 <= n <= 3999) or (6000 <= n <= 6999)
    except ValueError:
        return False


def _check_volume(code: str) -> tuple[bool, float]:
    """
    取近 20 個交易日平均日成交值（萬元）。
    回傳 (符合門檻, 均日成交值)
    """
    try:
        from utils.constants import OTC_SYMBOLS
        suffix = ".TWO" if code in OTC_SYMBOLS else ".TW"
        df = yf.Ticker(f"{code}{suffix}").history(period="30d")
        if df.empty or len(df) < 5:
            return False, 0.0
        avg_value = (df["Close"] * df["Volume"]).mean() / 10_000  # 換算成萬元
        return avg_value >= MIN_VOLUME_M, round(avg_value, 0)
    except Exception:
        return False, 0.0


def _get_stock_name(code: str) -> str:
    from utils.constants import STOCK_DB
    if code in STOCK_DB:
        return STOCK_DB[code].split()[0]
    try:
        from utils.constants import OTC_SYMBOLS
        suffix = ".TWO" if code in OTC_SYMBOLS else ".TW"
        info = yf.Ticker(f"{code}{suffix}").fast_info
        return getattr(info, "long_name", code) or code
    except Exception:
        return code


def _get_sector_for(code: str) -> str:
    from utils.constants import SECTOR_MAP
    for sector, codes in SECTOR_MAP.items():
        if code in codes:
            return sector
    return "其他"


# ── 移除邏輯 ──────────────────────────────────────────────────────────────────

def _should_suspend(code: str, inst_data: dict) -> tuple[bool, str]:
    """
    判斷是否暫停交易該股票。
    條件：外資今日明顯賣超 AND 價格低於 20 日均線。
    回傳 (暫停, 原因)
    """
    inst = inst_data.get(code, {})
    foreign_net = inst.get("foreign_net", 0)
    total_net   = inst.get("total_net",   0)

    # 外資大幅賣超（負值超過門檻）且三大法人合計也賣超
    if foreign_net < -FOREIGN_NET_THRESHOLD and total_net < 0:
        # 再確認價格低於 MA20
        try:
            from utils.constants import OTC_SYMBOLS
            suffix = ".TWO" if code in OTC_SYMBOLS else ".TW"
            df = yf.Ticker(f"{code}{suffix}").history(period="30d")
            if len(df) >= 20:
                ma20  = df["Close"].rolling(20).mean().iloc[-1]
                price = df["Close"].iloc[-1]
                if price < ma20:
                    reason = (
                        f"外資淨賣超 {foreign_net:,} 股，三大法人合計 {total_net:,}，"
                        f"股價 {price:.1f} < MA20 {ma20:.1f}"
                    )
                    return True, reason
        except Exception:
            pass

    return False, ""


# ── 主流程 ───────────────────────────────────────────────────────────────────

def run_discovery() -> dict:
    """
    每日選股發現主流程：
    1. 掃描 PTT 股板 → 找頻繁提及的科技股
    2. 取 TWSE 三大法人買賣超 → 找外資/投信淨買超的科技股
    3. 驗證流動性
    4. 更新 dynamic_tradeable.json（新增 / 暫停）
    5. 回傳摘要 dict
    """
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info("=== 選股發現任務開始 ===")
    dyn = _load_dynamic()

    from utils.constants import TRADEABLE, SECTOR_MAP, TECH_SECTORS

    # ── Step 1: PTT 熱門股 ────────────────────────────────────────────────────
    ptt_counter = _scan_ptt_board(pages=3)
    ptt_candidates = {
        code for code, cnt in ptt_counter.items()
        if cnt >= MIN_PTT_MENTIONS and len(code) in (4, 5)
    }
    logger.info(f"PTT 候選: {len(ptt_candidates)} 支 (≥{MIN_PTT_MENTIONS}次提及)")

    # ── Step 2: 三大法人買賣超 ───────────────────────────────────────────────
    inst_data = _fetch_twse_institutional()
    foreign_buy_codes = {
        code for code, d in inst_data.items()
        if d["foreign_net"] >= FOREIGN_NET_THRESHOLD
    }
    logger.info(f"外資淨買超 (≥{FOREIGN_NET_THRESHOLD:,}股): {len(foreign_buy_codes)} 支")

    # ── Step 3: 候選合集（PTT 熱門 OR 外資大買）────────────────────────────
    effective = get_effective_tradeable()
    new_candidates = (ptt_candidates | foreign_buy_codes) - effective

    added_this_run:    list[dict] = []
    suspended_this_run: list[dict] = []

    # ── Step 4: 驗證並新增 ───────────────────────────────────────────────────
    current_dynamic_count = len(dyn["added"])
    for code in sorted(new_candidates):
        if current_dynamic_count >= MAX_DYNAMIC_ADDED:
            logger.info(f"動態清單已達上限 {MAX_DYNAMIC_ADDED}，停止新增")
            break
        if code in dyn["suspended"]:
            continue  # 已暫停的不重新加入

        if not _is_tech_stock(code):
            logger.debug(f"  {code} 非科技股，跳過")
            continue

        ok, avg_vol = _check_volume(code)
        if not ok:
            logger.debug(f"  {code} 日均成交值 {avg_vol:.0f} 萬 < 門檻，跳過")
            continue

        inst     = inst_data.get(code, {})
        buyer    = _classify_buyers(inst)
        name     = _get_stock_name(code)
        sector   = _get_sector_for(code)
        ptt_cnt  = ptt_counter.get(code, 0)
        f_net    = inst.get("foreign_net", 0)

        reason_parts = []
        if ptt_cnt >= MIN_PTT_MENTIONS:
            reason_parts.append(f"PTT {ptt_cnt} 次")
        if f_net >= FOREIGN_NET_THRESHOLD:
            reason_parts.append(f"外資淨買 {f_net:,} 股")
        reason = "、".join(reason_parts) + f"（主力:{buyer}，均日成交 {avg_vol:.0f} 萬）"

        dyn["added"][code] = {
            "name":       name,
            "added_date": today,
            "reason":     reason,
            "sector":     sector,
            "buyer":      buyer,
        }
        current_dynamic_count += 1
        added_this_run.append({"code": code, "name": name, "reason": reason})
        logger.info(f"  ✅ 新增 {code} {name} [{sector}]: {reason}")

    # ── Step 5: 檢查現有股票是否暫停 ────────────────────────────────────────
    all_tradeable = get_effective_tradeable()  # 含剛新增的
    for code in list(all_tradeable):
        if code in dyn["suspended"]:
            continue
        suspend, reason = _should_suspend(code, inst_data)
        if not suspend:
            continue

        name   = _get_stock_name(code)
        sector = _get_sector_for(code)
        dyn["suspended"][code] = {
            "name":           name,
            "suspended_date": today,
            "reason":         reason,
        }
        # 若在動態清單中也移掉
        dyn["added"].pop(code, None)
        suspended_this_run.append({"code": code, "name": name, "reason": reason})
        logger.warning(f"  ⏸️  暫停 {code} {name}: {reason}")

    _save_dynamic(dyn)

    # ── 摘要 ─────────────────────────────────────────────────────────────────
    effective_after = get_effective_tradeable()
    summary = {
        "date":        today,
        "ptt_scanned": len(ptt_candidates),
        "inst_data":   len(inst_data),
        "added":       added_this_run,
        "suspended":   suspended_this_run,
        "total_effective": len(effective_after),
    }
    logger.info(
        f"=== 選股發現完成 "
        f"新增 {len(added_this_run)} 支，暫停 {len(suspended_this_run)} 支，"
        f"有效清單共 {len(effective_after)} 支 ==="
    )
    return summary


def print_dynamic_status():
    """印出目前動態清單狀態（供手動查詢用）。"""
    from utils.constants import TRADEABLE, STOCK_DB
    dyn = _load_dynamic()
    effective = get_effective_tradeable()

    print(f"\n{'='*55}")
    print(f"有效交易清單：{len(effective)} 支")
    print(f"  靜態核心：{len(TRADEABLE)} 支")
    print(f"  動態新增：{len(dyn['added'])} 支")
    print(f"  暫停移除：{len(dyn['suspended'])} 支")

    if dyn["added"]:
        print(f"\n【動態新增】")
        for code, info in dyn["added"].items():
            print(f"  {code} {info['name']} [{info['sector']}]")
            print(f"       加入:{info['added_date']}  主力:{info.get('buyer','?')}")
            print(f"       原因:{info['reason']}")

    if dyn["suspended"]:
        print(f"\n【暫停中】")
        for code, info in dyn["suspended"].items():
            in_core = "（靜態核心）" if code in TRADEABLE else "（動態加入）"
            print(f"  {code} {info['name']} {in_core}")
            print(f"       暫停:{info['suspended_date']}")
            print(f"       原因:{info['reason']}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    summary = run_discovery()
    print_dynamic_status()
