# Changelog

所有版本的重要更新記錄。

---

## [0.10.0] — 2026-10-15

### Changed
- **盤中掃描間隔：15 分鐘 → 3 分鐘**（`trading_bot.py` CronTrigger）
  - 原本：`minute="10,25,40,55"` hour 9–12，另加 13:00/13:20 補掃
  - 現在：`minute="*/3"` hour 9–13，單一 job 涵蓋全盤
- **TWSE 官方分類校正**（`utils/constants.py`）
  - 1313 聯成、1326 台化：化學工業 → **塑膠工業**（TWSE 官方歸屬）
  - 2049 上銀：電子零組件 → **電機機械**
  - 3008 大立光、3481 群創、2409 友達：移至 **光電**（TWSE 光電業）
  - 2301 光寶科、2395 研華、6669 緯穎：移至 **電腦週邊**（TWSE 電腦及週邊設備業）
  - 3661 世芯-KY、3105 穩懋：AI伺服器/光電 → **半導體**（TWSE 半導體業）
  - 3105 穩懋 加入 `OTC_SYMBOLS`（上櫃市場）
  - 移除自訂類別：`AI伺服器`、`面板`

### Added
- **化學工業新成員**（依 TWSE 官方化學工業分類）：
  - 1710 東聯、1717 長興、1722 台肥
- **電機機械新成員**：1503 士電、1519 華城

### Stats
- TRADEABLE：37 支 → **52 支**
- 涵蓋類股：半導體、記憶體、PCB、光電、電子零組件、電子組裝、電腦週邊、塑膠工業、化學工業、電機機械（共 10 個類別）

---

## [0.9.0] — 2026-10-01

### Added
- 擴充特徵矩陣：K 線型態（Doji、Hammer、Engulfing 等）+ 買賣力道特徵
- 新增可交易股票至 37 支（新增 PCB、光電、電子組件、電腦週邊、塑膠、化學、電機機械類股）

### Fixed
- `TradingAgentsConfig.response_language` 改為合法值 `zh-TW`
- US bot `retrain/inference` 特徵矩陣傳入錯誤
- US bot `MAX_POSITIONS` import 錯誤

---

## [0.8.0] — 2026-09-23

### Added
- 新增美股紙上交易 bot（`trading_bot.py` US 模式）
- 修復 crypto-bot 啟動流程

---

## [0.7.0] — 2026-09-22

### Fixed
- SDK session 過期靜默失敗：加入 session 存活偵測與重新登入
- 提高買入門檻，減少雜訊進場

---

## [0.6.1] — 2026-08-31

### Fixed
- 修復 10 項交易核心 bug，含 race condition、停損邏輯、持倉計算錯誤

---

## [0.6.0] — 2026-08-25～26

### Added
- **FreeToken 整合**：本地 LLM（Qwen2.5-7B）作為 TradingAgents 備援後端
- FreeToken × TradingAgents 整合測試
- **自動 Git commit**：每日結算後將績效報告 commit 至 GitHub，供 cloud agent 分析

### Fixed
- SDK auto re-login：session 失效後自動重連
- 緊急重訓增加 2 小時冷卻，防止過度重訓

---

## [0.5.0] — 2026-08-24（Checkpoint 系列）

### CP4 — 模型升級
- 切換至 Ensemble 模型：LightGBM + XGBoost + CatBoost 軟投票

### CP3 — 分批停利
- 停利策略：+7% 賣出一半，剩餘跟 ATR 追蹤至 +15% 全出

### CP2 — ATR 停損活化
- 保護期（前 3 天）：固定停損 -5%
- 成長期：改用 ATR 動態追蹤停損

### CP1 — 進場品質濾網
- 新增 RSI 過熱濾網（RSI > 75 不進場）
- 要求成交量 > 20 日均量
- 外資連續買超確認

### Fixed
- 持倉同步從 `trade_log` 還原 `entry_date`，防止 pod 重啟後日期歸零

---

## [0.4.0] — 2026-07-20

### Added
- AI 特徵工程：技術面 + 籌碼面 + 市場情緒多維特徵
- 回測報告框架

### Fixed
- 移除最短持有天數限制，允許當天依 TradingAgents 建議賣出
- 修復 race condition、DB 連線洩漏、重複入帳等 8 項問題

---

## [0.3.0] — 2026-05-03

### Added
- **興櫃股票支援**（高風險獨立 RiskProfile：停損 -5%、最多持 1 支）
- 雙軌風險管理：`RiskProfile` 依股票類型套用不同參數
- 興櫃停利：+15%（R/R ≈ 3x）
- 全面單元測試 + 1 個月整合回測

---

## [0.2.0] — 2026-05-03

### Added
- **TradingAgents 多代理人整合**（OpenAI-compatible API，Qwen2.5 模型）
  - 持倉股票觸發 TA 分析，不受停損/停利門檻限制
  - LightGBM 訊號 + TA 第二意見雙重確認進出場
- **紙上交易模式**（`PAPER_TRADING=true`，獨立 state 檔）
- `start_paper_bot.sh` 啟動腳本

---

## [0.1.0] — 2026-05-03

### Added
- 初始版本：台股 LightGBM 交易機器人
- 每日 08:30 選股掃描（`job_scan`）→ 09:10 進場（`job_buy`）
- 每 15 分鐘盤中監控（`job_monitor`）
- 13:35 每日結算（`job_settle`）
- 週一 14:35 模型重訓（`job_retrain`）
- PostgreSQL 持倉管理 + 交易記錄
- Kubernetes 部署（k3s）
- Line Notify 交易通知
