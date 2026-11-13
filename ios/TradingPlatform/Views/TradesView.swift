import SwiftUI

// MARK: - Range options

private enum TradeRange: Int, CaseIterable, Identifiable {
    case today = 1, week = 7, month = 30

    var id: Int { rawValue }
    var label: String {
        switch self {
        case .today: return "今天"
        case .week:  return "本週"
        case .month: return "本月"
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
    @State private var range: TradeRange = .today

    // trades grouped by date, sorted newest first
    private var grouped: [(date: String, trades: [Trade])] {
        let dict = Dictionary(grouping: trades, by: \.tradeDate)
        return dict.keys.sorted(by: >).map { d in (date: d, trades: dict[d]!) }
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // ── Range picker ──────────────────────────────
                Picker("期間", selection: $range) {
                    ForEach(TradeRange.allCases) { r in
                        Text(r.label).tag(r)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)

                if accounts.count > 1 {
                    AccountPicker(accounts: accounts).padding(.bottom, 8)
                }

                // ── Summary banner ────────────────────────────
                if let s = summary {
                    SummaryBanner(summary: s, range: range)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 8)
                }

                Divider()

                // ── Trade list ────────────────────────────────
                Group {
                    if isLoading && trades.isEmpty {
                        ProgressView("載入中...").frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else if trades.isEmpty {
                        ContentUnavailableView(
                            "此期間無交易記錄",
                            systemImage: "chart.line.flattrend.xyaxis"
                        )
                    } else {
                        List {
                            ForEach(grouped, id: \.date) { group in
                                Section(header: DateHeader(dateStr: group.date)) {
                                    ForEach(group.trades) { trade in
                                        TradeRow(trade: trade)
                                    }
                                }
                            }
                        }
                        .listStyle(.plain)
                    }
                }
            }
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
            .onChange(of: api.selectedAccount) { _ in load() }
            .onChange(of: ws.latestTrade) { _ in if range == .today { load() } }
        }
    }

    private func load() {
        Task {
            isLoading = true
            accounts = await loadAccounts(api)
            async let t = api.fetchTrades(days: range.rawValue)
            async let s = api.fetchSummary(days: range.rawValue)
            trades  = (try? await t) ?? []
            summary = try? await s
            isLoading = false
        }
    }
}

// MARK: - Summary Banner

private struct SummaryBanner: View {
    let summary: DailySummary
    let range: TradeRange

    var body: some View {
        HStack(spacing: 0) {
            StatCell(
                label: range == .today ? "今日損益" : "\(range.label)損益",
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

struct TradeRow: View {
    let trade: Trade

    var isBuy: Bool { trade.action == "BUY" || trade.action == "SHORT" }

    var timeString: String {
        // createdAt format: "2026-11-11T02:51:00.000000" (UTC) → "HH:mm"
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
        fmt.timeZone = TimeZone(identifier: "UTC")
        guard let date = fmt.date(from: trade.createdAt) else { return "" }
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
