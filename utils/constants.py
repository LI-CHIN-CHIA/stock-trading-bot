# Taiwan Stock Market Constants

MARKET_TZ = "Asia/Taipei"
MARKET_OPEN_HOUR = 9
MARKET_CLOSE_HOUR = 13
MARKET_CLOSE_MINUTE = 30

# Full stock database: code -> "中文名 English Name"
# Used for both display and name-based search
STOCK_DB = {
    # ── 半導體 Semiconductors ──
    "2330": "台積電 TSMC",
    "2303": "聯電 UMC",
    "2454": "聯發科 MediaTek",
    "2379": "瑞昱 Realtek",
    "3034": "聯詠 Novatek",
    "2408": "南亞科 Nanya Tech",
    "3711": "日月光投控 ASE Tech",
    "2344": "華邦電 Winbond",
    "2337": "旺宏 Macronix",
    "6770": "力積電 PSMC",
    "3533": "嘉澤 Lotes",
    "2385": "群光 Chicony Elec",
    "3231": "緯創 Wistron",
    "2376": "技嘉 Gigabyte",
    "2357": "華碩 ASUS",
    "2382": "廣達 Quanta",
    "2356": "英業達 Inventec",
    "2353": "宏碁 Acer",
    "3481": "群創 Innolux",
    "2409": "友達 AUO",
    # ── 電子零組件 Electronic Components ──
    "2308": "台達電 Delta Electronics",
    "2317": "鴻海 Hon Hai / Foxconn",
    "2395": "研華 Advantech",
    "3008": "大立光 Largan Precision",
    "2474": "可成 Catcher Tech",
    "4938": "和碩 Pegatron",
    "2301": "光寶科 Lite-On Tech",
    "2360": "致茂 Chroma ATE",
    "3037": "欣興 Unimicron",
    "2367": "燿華 Unitech PCB",
    "6269": "台郡 Flexium",
    "3443": "創意 Global Unichip",
    # ── 電信 Telecom ──
    "2412": "中華電 Chunghwa Telecom",
    "4904": "遠傳 FarEasTone",
    "3045": "台灣大 Taiwan Mobile",
    # ── 金融 Finance ──
    "2881": "富邦金 Fubon Financial",
    "2882": "國泰金 Cathay Financial",
    "2883": "開發金 CDIB Financial",
    "2884": "玉山金 E.Sun Financial",
    "2885": "元大金 Yuanta Financial",
    "2886": "兆豐金 Mega Financial",
    "2887": "台新金 Taishin Financial",
    "2889": "國票金 IBF Financial",
    "2890": "永豐金 SinoPac Holdings",
    "2891": "中信金 CTBC Financial",
    "2892": "第一金 First Financial",
    "5880": "合庫金 Taiwan Cooperative Fin",
    "2801": "彰銀 Chang Hwa Bank",
    "2820": "華票 China Bills Finance",
    # ── 傳產 Traditional Industries ──
    "2002": "中鋼 China Steel",
    "1301": "台塑 Formosa Plastics",
    "1303": "南亞 Nan Ya Plastics",
    "1326": "台化 Formosa Chemicals",
    "1216": "統一 Uni-President",
    "1101": "台泥 Taiwan Cement",
    "1102": "亞泥 Asia Cement",
    "2207": "和泰車 Hotai Motor",
    "2105": "正新 Cheng Shin Rubber",
    "2912": "統一超 President Chain Store",
    "2801": "彰銀 Chang Hwa Bank",
    "2347": "聯強 Synnex Tech",
    # ── ETF（僅用於模型訓練，不實際交易）──
    "0050": "元大台灣50 ETF",
    "0056": "元大高股息 ETF",
    "00878": "國泰永續高股息 ETF",
    "006208": "富邦台50 ETF",
    "00713": "元大台灣高息低波 ETF",
    # ── 生技醫療 Biotech ──
    "4763": "材料-KY",
    "6547": "晟德 Standard Foods",
    "1707": "葡萄王 Grape King Bio",
    # ── 遊戲娛樂 Gaming ──
    "3673": "TPK",
    "6488": "環球晶 GlobalWafers",
}

# ETF 代碼集合（訓練用，不下單）
ETF_SYMBOLS = {"0050", "0056", "00878", "006208", "00713"}

# 實際可交易的股票（排除 ETF）
TRADEABLE = {code for code in STOCK_DB if code not in ETF_SYMBOLS}

# Backward-compatible alias for code that still references TWSE_POPULAR
TWSE_POPULAR = STOCK_DB

# OTC/TPEX stocks use .TWO suffix
OTC_SYMBOLS = {
    "6547", "6488", "6670", "6183", "5904", "3529",
    "4961", "3653", "6415", "8299",
}

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
