import Foundation

// MARK: - Auth

struct TokenResponse: Codable {
    let accessToken: String
    let tokenType: String
    let userId: Int
    let displayName: String

    enum CodingKeys: String, CodingKey {
        case accessToken = "access_token"
        case tokenType   = "token_type"
        case userId      = "user_id"
        case displayName = "display_name"
    }
}

// MARK: - Config

struct TradingConfig: Codable, Equatable {
    var enableDay: Bool
    var enableShort: Bool
    var enableOvernight: Bool
    var autoTrade: Bool
    var dayBudget: Double
    var maxPositions: Int
    var stopLossPct: Double
    var takeProfitPct: Double
    var paperBudgets: [Int]?     // 模擬帳戶額度清單（舊版伺服器沒有此欄位）
    var updatedAt: String

    enum CodingKeys: String, CodingKey {
        case enableDay       = "enable_day"
        case enableShort     = "enable_short"
        case enableOvernight = "enable_overnight"
        case autoTrade       = "auto_trade"
        case dayBudget       = "day_budget"
        case maxPositions    = "max_positions"
        case stopLossPct     = "stop_loss_pct"
        case takeProfitPct   = "take_profit_pct"
        case paperBudgets    = "paper_budgets"
        case updatedAt       = "updated_at"
    }
}

struct ConfigUpdate: Codable {
    var enableDay: Bool?
    var enableShort: Bool?
    var enableOvernight: Bool?
    var autoTrade: Bool?
    var dayBudget: Double?
    var maxPositions: Int?
    var stopLossPct: Double?
    var takeProfitPct: Double?
    var paperBudgets: [Int]?

    enum CodingKeys: String, CodingKey {
        case enableDay       = "enable_day"
        case enableShort     = "enable_short"
        case enableOvernight = "enable_overnight"
        case autoTrade       = "auto_trade"
        case dayBudget       = "day_budget"
        case maxPositions    = "max_positions"
        case stopLossPct     = "stop_loss_pct"
        case takeProfitPct   = "take_profit_pct"
        case paperBudgets    = "paper_budgets"
    }
}

// MARK: - Account（真實 / 多個模擬額度）

struct TradingAccount: Codable, Identifiable, Equatable {
    let id: String          // "real" / "paper-500000"
    let kind: String        // "real" / "paper"
    let label: String
    let budget: Double
    let used: Double             // 目前持倉佔用金額（收盤平倉後為 0）
    var tradedToday: Double?     // 今日進場成交金額合計
    let todayPnl: Double
    let pnlPct: Double
    let tradeCount: Int
    let winCount: Int
    let winRate: Double
    let targetPnl: Double
    let hitTarget: Bool

    var isReal: Bool { kind == "real" }

    enum CodingKeys: String, CodingKey {
        case id, kind, label, budget, used
        case tradedToday = "traded_today"
        case todayPnl   = "today_pnl"
        case pnlPct     = "pnl_pct"
        case tradeCount = "trade_count"
        case winCount   = "win_count"
        case winRate    = "win_rate"
        case targetPnl  = "target_pnl"
        case hitTarget  = "hit_target"
    }
}

/// 帳戶 id → 顯示名稱（不需打 API 就能標示成交/持倉屬於哪個帳戶）
func accountLabel(_ id: String?, isReal: Bool) -> String {
    guard let id = id, !id.isEmpty else { return isReal ? "真實" : "模擬" }
    if id == "real" { return "真實" }
    // "<策略>-<額度>"：paper = 一般模擬，fade = 過熱放空策略
    let parts = id.split(separator: "-")
    if parts.count == 2, let cap = Double(parts[1]) {
        let name = ["paper": "模擬", "fade": "過熱放空"][String(parts[0])] ?? String(parts[0])
        let wan = cap / 10000
        return wan == wan.rounded() ? "\(name) \(Int(wan))萬" : "\(name) \(String(format: "%.1f", wan))萬"
    }
    return id
}

func formatWan(_ v: Double) -> String {
    let wan = v / 10000
    return wan == wan.rounded() ? "\(Int(wan)) 萬" : "\(String(format: "%.1f", wan)) 萬"
}

// MARK: - Trade

struct Trade: Codable, Identifiable, Equatable {
    let id: Int
    let tradeDate: String
    let code: String
    let action: String
    let shares: Int
    let price: Double
    let pnl: Double
    let reason: String
    let orderNo: String
    let isReal: Bool
    var account: String? = nil
    let createdAt: String

    enum CodingKeys: String, CodingKey {
        case id, code, action, shares, price, pnl, reason, account
        case tradeDate = "trade_date"
        case orderNo   = "order_no"
        case isReal    = "is_real"
        case createdAt = "created_at"
    }
}

// MARK: - Position

struct Position: Codable, Identifiable {
    let id: Int
    let code: String
    let direction: String
    let shares: Int
    let entryPrice: Double
    let isReal: Bool
    var account: String? = nil
    let openedAt: String
    let updatedAt: String

    enum CodingKeys: String, CodingKey {
        case id, code, direction, shares, account
        case entryPrice = "entry_price"
        case isReal     = "is_real"
        case openedAt   = "opened_at"
        case updatedAt  = "updated_at"
    }

    var unrealizedPnl: Double {
        // approximation: shown as 0 until we have market price
        return 0.0
    }

    var directionLabel: String {
        direction == "SHORT" ? "反沖" : "當沖"
    }
}

// MARK: - Signal

struct Signal: Codable, Identifiable, Equatable {
    let id: Int
    let code: String
    let action: String
    let score: Int
    let price: Double
    let reason: String
    let signalDate: String
    let createdAt: String

    enum CodingKeys: String, CodingKey {
        case id, code, action, score, price, reason
        case signalDate = "signal_date"
        case createdAt  = "created_at"
    }

    var actionColor: String {
        switch action {
        case "BUY":   return "green"
        case "SELL":  return "red"
        case "SHORT": return "orange"
        case "COVER": return "blue"
        default:      return "gray"
        }
    }
}

// MARK: - Summary

struct DailySummary: Codable {
    let date: String
    let totalPnl: Double
    let tradeCount: Int
    let winCount: Int
    let winRate: Double
    let targetPct: Double
    let targetPnl: Double
    let hitTarget: Bool

    enum CodingKeys: String, CodingKey {
        case date
        case totalPnl   = "total_pnl"
        case tradeCount = "trade_count"
        case winCount   = "win_count"
        case winRate    = "win_rate"
        case targetPct  = "target_pct"
        case targetPnl  = "target_pnl"
        case hitTarget  = "hit_target"
    }
}

// MARK: - WS Messages

struct WSMessage: Codable {
    let type: String
    let data: WSData?
    let userId: Int?

    enum CodingKeys: String, CodingKey {
        case type, data
        case userId = "user_id"
    }
}

struct WSData: Codable {
    // signal fields
    let id: Int?
    let code: String?
    let action: String?
    let score: Int?
    let price: Double?
    let reason: String?
    // trade fields
    let shares: Int?
    let pnl: Double?
    let isReal: Bool?
    let account: String?
    // position fields
    let direction: String?
    let entryPrice: Double?

    enum CodingKeys: String, CodingKey {
        case id, code, action, score, price, reason, shares, pnl, direction, account
        case isReal     = "is_real"
        case entryPrice = "entry_price"
    }
}
