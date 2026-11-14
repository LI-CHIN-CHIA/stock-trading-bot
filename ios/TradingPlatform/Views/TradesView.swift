import SwiftUI

// MARK: - Range options

private enum TradeRange: Int, CaseIterable, Identifiable {
    case today = 1, week = 7, month = 30, custom = 0

    var id: Int { rawValue }
    var label: String {
        switch self {
        case .today:  return "今天"
        case .week:   return "本週"
        case .month:  return "本月"
        case .custom: return "自訂"
        }
    }
}

// MARK: - Main View

struct TradesView: View {
    @EnvironmentObject var api: APIService
    @EnvironmentObject var ws: WSService

    @State private var trades: [Trade] = []
    @State private var summary: DailySummary? = nil
    @State private var accounts: [TradingAccount] = []
    @State private var isLoading = false
    @State private var errorMsg  = ""
    @State private var range: TradeRange = .today
    // 自訂區間（含頭尾）；起訖選同一天即為單日
    @State private var startDate = Date()
    @State private var endDate   = Date()

    private static let dayFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        f.timeZone = TimeZone(identifier: "Asia/Taipei")
        f.locale = Locale(identifier: "en_US_POSIX")
        return f
    }()

    // trades grouped by date, sorted newest first
    private var grouped: [(date: String, trades: [Trade])] {
        let dict = Dictionary(grouping: trades, by: \.tradeDate)
        return dict.keys.sorted(by: >).map { d in (date: d, trades: dict[d]!) }
    }

    var body: some View {
        NavigationView {
            // 篩選列、帳戶列與狀態訊息都放在同一個 List 裡：若橫向的 AccountPicker 是頁面中
            // 第一個捲動區，導覽列會把它當成主捲動區，上下滑它時大標題會跟著展開／收合
            List {
                Section {
                    // ── Range picker ──────────────────────────────
                    Picker("期間", selection: $range) {
                        ForEach(TradeRange.allCases) { r in
                            Text(r.label).tag(r)
                        }
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .headerRow()

                    if range == .custom {
                        HStack(spacing: 8) {
                            DatePicker("起", selection: $startDate, in: ...Date(), displayedComponents: .date)
                                .labelsHidden()
                            Text("～").foregroundColor(.secondary)
                            DatePicker("迄", selection: $endDate, in: ...Date(), displayedComponents: .date)
                                .labelsHidden()
                            Spacer()
                        }
                        .environment(\.locale, Locale(identifier: "zh_TW"))
                        .padding(.horizontal, 16)
                        .padding(.bottom, 8)
                        .headerRow()
                    }

                    if accounts.count > 1 {
                        AccountPicker(accounts: accounts)
                            .padding(.bottom, 8)
                            .headerRow()
                    }

                    // ── Summary banner ────────────────────────────
                    if let s = summary {
                        SummaryBanner(summary: s, range: range)
                            .padding(.horizontal, 16)
                            .padding(.bottom, 8)
                            .headerRow()
                    }
                }

                // ── Trade list ────────────────────────────────
                if !errorMsg.isEmpty {
                    ContentUnavailableView("交易紀錄載入失敗", systemImage: "exclamationmark.triangle",
                                           description: Text(errorMsg))
                        .listRowSeparator(.hidden)
                } else if isLoading && trades.isEmpty {
                    ProgressView("載入中...")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 40)
                        .listRowSeparator(.hidden)
                } else if trades.isEmpty {
                    ContentUnavailableView(
                        "此期間無交易記錄",
                        systemImage: "chart.line.flattrend.xyaxis"
                    )
                    .listRowSeparator(.hidden)
                } else {
                    ForEach(grouped, id: \.date) { group in
                        Section(header: DateHeader(dateStr: group.date)) {
                            ForEach(group.trades) { trade in
                                TradeRow(trade: trade)
                            }
                        }
                    }
                }
            }
            .listStyle(.plain)
            .navigationTitle("交易紀錄")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: load) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .refreshable { load() }
            .onAppear { load() }
            .onChange(of: range) { _ in load() }
            .onChange(of: startDate) { _ in if range == .custom { load() } }
            .onChange(of: endDate) { _ in if range == .custom { load() } }
            .onChange(of: api.selectedAccount) { _ in load() }
            .onChange(of: ws.latestTrade) { _ in if range == .today { load() } }
        }
    }

    private func load() {
        Task {
            isLoading = true
            accounts = await loadAccounts(api)
            let custom = range == .custom
            let lo = custom ? Self.dayFormatter.string(from: min(startDate, endDate)) : ""
            let hi = custom ? Self.dayFormatter.string(from: max(startDate, endDate)) : ""
            async let t = api.fetchTrades(days: range.rawValue, start: lo, end: hi)
            async let s = api.fetchSummary(days: range.rawValue, start: lo, end: hi)
            do {
                trades = try await t
                errorMsg = ""
            } catch {
                errorMsg = error.localizedDescription
            }
            summary = try? await s
            isLoading = false
        }
    }
}

private extension View {
    /// List 頂部篩選區的列：不要預設內距與分隔線，維持原本貼齊畫面的版面
    func headerRow() -> some View {
        listRowInsets(EdgeInsets()).listRowSeparator(.hidden)
    }
}

// MARK: - Summary Banner

private struct SummaryBanner: View {
    let summary: DailySummary
    let range: TradeRange

    var body: some View {
        HStack(spacing: 0) {
            StatCell(
                label: range == .today ? "今日損益" : (range == .custom ? "區間損益" : "\(range.label)損益"),
                value: "\(summary.totalPnl >= 0 ? "+" : "")\(Int(summary.totalPnl))",
                color: summary.totalPnl >= 0 ? .green : .red
            )
            Divider().frame(height: 36)
            StatCell(
                label: "交易次數",
                value: "\(summary.tradeCount)",
                color: .primary
            )
            Divider().frame(height: 36)
            StatCell(
                label: "勝率",
                value: summary.tradeCount > 0
                    ? String(format: "%.0f%%", summary.winRate * 100)
                    : "-",
                color: summary.winRate >= 0.5 ? .green : .orange
            )
        }
        .padding(.vertical, 10)
        .background(Color(.secondarySystemBackground))
        .cornerRadius(10)
    }
}

private struct StatCell: View {
    let label: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: 2) {
            Text(value).font(.headline).foregroundColor(color)
            Text(label).font(.caption2).foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Date Header

private struct DateHeader: View {
    let dateStr: String  // "2026-11-09"

    var label: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        guard let d = formatter.date(from: dateStr) else { return dateStr }
        let cal = Calendar.current
        if cal.isDateInToday(d)     { return "今天 · \(dateStr)" }
        if cal.isDateInYesterday(d) { return "昨天 · \(dateStr)" }
        formatter.dateFormat = "M/d (EEE)"
        formatter.locale = Locale(identifier: "zh_TW")
        return formatter.string(from: d)
    }

    var body: some View {
        Text(label)
            .font(.subheadline).bold()
            .foregroundColor(.secondary)
            .textCase(nil)
    }
}

// MARK: - Trade Row

/// 解析伺服器的 created_at。後端欄位是帶時區的 UTC 時間，輸出可能是
/// "...T02:51:00.123456Z"、"...+00:00"，或沒有時區／沒有小數秒（Python 在微秒為 0 時省略）。
private func parseServerDate(_ s: String) -> Date? {
    let fmt = DateFormatter()
    fmt.locale = Locale(identifier: "en_US_POSIX")
    fmt.timeZone = TimeZone(identifier: "UTC")  // 沒有時區標記時視為 UTC
    for f in ["yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXXXX", "yyyy-MM-dd'T'HH:mm:ssXXXXX",
              "yyyy-MM-dd'T'HH:mm:ss.SSSSSS", "yyyy-MM-dd'T'HH:mm:ss"] {
        fmt.dateFormat = f
        if let d = fmt.date(from: s) { return d }
    }
    return nil
}

struct TradeRow: View {
    let trade: Trade

    var isBuy: Bool { trade.action == "BUY" || trade.action == "SHORT" }

    var timeString: String {
        guard let date = parseServerDate(trade.createdAt) else { return "" }
        let out = DateFormatter()
        out.dateFormat = "HH:mm"
        out.timeZone = TimeZone(identifier: "Asia/Taipei")
        return out.string(from: date)
    }

    var body: some View {
        HStack(spacing: 12) {
            // Action badge
            Text(trade.action)
                .font(.caption).bold()
                .foregroundColor(.white)
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(Capsule().fill(isBuy ? Color.green : Color.red))

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(trade.code).font(.headline)
                    // 真實 / 紙上 badge
                    Text(accountLabel(trade.account, isReal: trade.isReal))
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(trade.isReal ? .white : .secondary)
                        .padding(.horizontal, 5).padding(.vertical, 2)
                        .background(
                            Capsule().fill(trade.isReal
                                ? Color.blue.opacity(0.85)
                                : Color(.systemGray5))
                        )
                }
                Text(trade.reason).font(.caption).foregroundColor(.secondary)
                    .lineLimit(1)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                if !timeString.isEmpty {
                    Text(timeString)
                        .font(.caption2).foregroundColor(.secondary)
                }
                Text("\(trade.shares) 股")
                    .font(.subheadline)
                Text("$ \(String(format: "%.1f", trade.price))")
                    .font(.caption).foregroundColor(.secondary)
                if trade.action == "SELL" || trade.action == "COVER" {
                    Text("\(trade.pnl >= 0 ? "+" : "")\(Int(trade.pnl))")
                        .font(.caption).bold()
                        .foregroundColor(trade.pnl >= 0 ? .green : .red)
                }
            }
        }
        .padding(.vertical, 4)
    }
}
