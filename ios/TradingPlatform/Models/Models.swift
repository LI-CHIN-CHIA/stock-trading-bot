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

    enum CodingKeys: String, CodingKey {
        case enableDay       = "enable_day"
        case enableShort     = "enable_short"
        case enableOvernight = "enable_overnight"
        case autoTrade       = "auto_trade"
        case dayBudget       = "day_budget"
        case maxPositions    = "max_positions"
    }
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
    let createdAt: String

    enum CodingKeys: String, CodingKey {
        case id, code, action, shares, price, pnl, reason
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
    let openedAt: String
    let updatedAt: String

    enum CodingKeys: String, CodingKey {
        case id, code, direction, shares
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
    // position fields
    let direction: String?
    let entryPrice: Double?

    enum CodingKeys: String, CodingKey {
        case id, code, action, score, price, reason, shares, pnl, direction
        case isReal     = "is_real"
        case entryPrice = "entry_price"
    }
}
