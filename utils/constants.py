# Taiwan Stock Market Constants

MARKET_TZ = "Asia/Taipei"
MARKET_OPEN_HOUR = 9
MARKET_CLOSE_HOUR = 13
MARKET_CLOSE_MINUTE = 30

# Full stock database: code -> "中文名 English Name"
# 含非科技股，供 LightGBM 訓練使用；TRADEABLE 才是實際下單範圍
STOCK_DB = {
    # ── 半導體 Semiconductors ──
    "2330": "台積電 TSMC",
    "2303": "聯電 UMC",
    "2454": "聯發科 MediaTek",
    "2379": "瑞昱 Realtek",
    "3034": "聯詠 Novatek",
    "3711": "日月光投控 ASE Tech",
    "3443": "創意 Global Unichip",
    # ── 記憶體 Memory ──
    "2408": "南亞科 Nanya Tech",
    "2344": "華邦電 Winbond",
    "2337": "旺宏 Macronix",
    "6770": "力積電 PSMC",
    # ── PCB ──
    "3037": "欣興 Unimicron",
    "2367": "燿華 Unitech PCB",
    "6269": "台郡 Flexium",
    # ── 電子零組件 Electronic Components ──
    "2308": "台達電 Delta Electronics",
    "2395": "研華 Advantech",
    "3008": "大立光 Largan Precision",
    "2474": "可成 Catcher Tech",
    "2301": "光寶科 Lite-On Tech",
    "2360": "致茂 Chroma ATE",
    "3533": "嘉澤 Lotes",
    # ── 電子組裝 Electronic Assembly ──
    "2317": "鴻海 Hon Hai / Foxconn",
    "2382": "廣達 Quanta",
    "2356": "英業達 Inventec",
    "4938": "和碩 Pegatron",
    "2353": "宏碁 Acer",
    "3231": "緯創 Wistron",
    "2357": "華碩 ASUS",
    "2376": "技嘉 Gigabyte",
    "2385": "群光 Chicony Elec",
    # ── 面板 Display Panels ──
    "3481": "群創 Innolux",
    "2409": "友達 AUO",
    # ── 電信 Telecom（訓練用，不下單）──
    "2412": "中華電 Chunghwa Telecom",
    "4904": "遠傳 FarEasTone",
    "3045": "台灣大 Taiwan Mobile",
    # ── 金融 Finance（訓練用，不下單）──
    "2881": "富邦金 Fubon Financial",
    "2882": "國泰金 Cathay Financial",
    "2883": "開發金 CDIB Financial",
    "2884": "玉山金 E.Sun Financial",
    "2885": "元大金 Yuanta Financial",
    "2886": "兆豐金 Mega Financial",
    "2887": "台新金 Taishin Financial",
    "2891": "中信金 CTBC Financial",
    "2892": "第一金 First Financial",
    "5880": "合庫金 Taiwan Cooperative Fin",
    "2820": "華票 China Bills Finance",
    # ── 傳產 Traditional Industries（訓練用，不下單）──
    "2002": "中鋼 China Steel",
    "1301": "台塑 Formosa Plastics",
    "1303": "南亞 Nan Ya Plastics",
    "1326": "台化 Formosa Chemicals",
    "1216": "統一 Uni-President",
    "1101": "台泥 Taiwan Cement",
    "2207": "和泰車 Hotai Motor",
    "2912": "統一超 President Chain Store",
    # ── ETF（訓練用，不下單）──
    "0050": "元大台灣50 ETF",
    "0056": "元大高股息 ETF",
    "00878": "國泰永續高股息 ETF",
    "006208": "富邦台50 ETF",
    "00713": "元大台灣高息低波 ETF",
    # ── 生技醫療 Biotech（訓練用，不下單）──
    "4763": "材料-KY",
    "6547": "晟德 Standard Foods",
    "1707": "葡萄王 Grape King Bio",
}

# ETF 代碼集合（訓練用，不下單）
ETF_SYMBOLS = {"0050", "0056", "00878", "006208", "00713"}

# 生技醫療股（訓練用，不下單）
BIOTECH_SYMBOLS = {"4763", "6547", "1707"}

# ── 類股分類 ───────────────────────────────────────────────────────────────────
SECTOR_MAP: dict[str, set[str]] = {
    "半導體":   {"2330", "2303", "2454", "2379", "3034", "3711", "3443"},
    "記憶體":   {"2408", "2344", "2337", "6770"},
    "PCB":      {"3037", "2367", "6269"},
    "電子零組件": {"2308", "2395", "3008", "2474", "2301", "2360", "3533"},
    "電子組裝": {"2317", "2382", "2356", "4938", "2353", "3231", "2357", "2376", "2385"},
    "面板":     {"3481", "2409"},
    # 以下僅供類股顯示與分散計算，不加入 TRADEABLE
    "電信":     {"2412", "4904", "3045"},
    "金融":     {"2881", "2882", "2883", "2884", "2885", "2886", "2887",
                 "2891", "2892", "5880", "2820"},
    "傳產":     {"2002", "1301", "1303", "1326", "1216", "1101", "2207", "2912"},
}

# 實際交易的類股（科技股）
TECH_SECTORS = {"半導體", "記憶體", "PCB", "電子零組件", "電子組裝", "面板"}

# 實際可交易的股票：只取科技類股，排除 ETF 與生技
TRADEABLE: set[str] = {
    code
    for sector in TECH_SECTORS
    for code in SECTOR_MAP.get(sector, set())
    if code not in ETF_SYMBOLS and code not in BIOTECH_SYMBOLS
}

# Backward-compatible alias for code that still references TWSE_POPULAR
TWSE_POPULAR = STOCK_DB

# OTC/TPEX stocks use .TWO suffix
OTC_SYMBOLS = {
    "6547", "6488", "6670", "6183", "5904", "3529",
    "4961", "3653", "6415", "8299",
}

# ── 興櫃 (Emerging Stock Board) ────────────────────────────────────────────────
# 高風險，套用 EMERGING 風險設定檔：停損-5%、最多1支、必須TA確認、持倉5天
EMERGING_SYMBOLS: dict[str, str] = {
    # code: "中文名 English"
    # 範例（請換成你實際想交易的興櫃股）:
    # "6589": "台康生技 TaiGen Biotech",
    # "6803": "崑鼎 Kanding",
}

# 興櫃也加入 STOCK_DB（若有的話）
STOCK_DB.update(EMERGING_SYMBOLS)

# 興櫃可交易集合
EMERGING_TRADEABLE: set[str] = set(EMERGING_SYMBOLS.keys())

# Period options for UI: label -> yfinance period string
PERIOD_OPTIONS = {
    "當天": "1d",
    "1週": "5d",
    "1個月": "1mo",
    "3個月": "3mo",
    "6個月": "6mo",
    "1年": "1y",
    "2年": "2y",
}

# Matching interval for each period (K棒時間粒度)
PERIOD_INTERVALS = {
    "1d":  "5m",   # 當天: 5分鐘K棒
    "5d":  "1h",   # 1週: 1小時K棒
    "1mo": "1d",   # 1個月以上: 日K棒
    "3mo": "1d",
    "6mo": "1d",
    "1y":  "1d",
    "2y":  "1d",
}

# Intraday periods (no MA60, shorter indicators)
INTRADAY_PERIODS = {"1d", "5d"}

# Technical indicator parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_NEAR_OVERBOUGHT = 60
RSI_NEAR_OVERSOLD = 40

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BB_PERIOD = 20
BB_STD = 2.0

MA_PERIODS = [5, 10, 20, 60]

ATR_PERIOD = 14
VOLUME_MA_PERIOD = 20

# Signal confidence thresholds
STRONG_BUY_THRESHOLD = 4
MODERATE_BUY_THRESHOLD = 2
WEAK_BUY_THRESHOLD = 1
STRONG_SELL_THRESHOLD = -4
MODERATE_SELL_THRESHOLD = -2
WEAK_SELL_THRESHOLD = -1

# Volume confirmation
HIGH_VOLUME_RATIO = 2.0
LOW_VOLUME_RATIO = 0.5

# Minimum data points needed for all indicators
MIN_DATA_POINTS = 65  # MA60 needs 60, plus buffer
