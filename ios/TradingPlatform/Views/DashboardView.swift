import SwiftUI

struct DashboardView: View {
    @EnvironmentObject var api: APIService
    @EnvironmentObject var ws: WSService
    @State private var summary: DailySummary?
    @State private var config: TradingConfig?
    @State private var accounts: [TradingAccount] = []
    @State private var isLoading = false
    @State private var errorMsg  = ""

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 16) {
                    // Connection badge
                    HStack {
                        Circle()
                            .fill(ws.isConnected ? Color.green : Color.red)
                            .frame(width: 8, height: 8)
                        Text(ws.isConnected ? "Agent 已連線" : "Agent 離線")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text(summary?.date ?? todayString())
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.horizontal)

                    if !errorMsg.isEmpty {
                        Label(errorMsg, systemImage: "exclamationmark.triangle.fill")
                            .font(.footnote)
                            .foregroundColor(.red)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                            .background(RoundedRectangle(cornerRadius: 12).fill(Color.red.opacity(0.1)))
                            .padding(.horizontal)
                    }

                    // 帳戶切換 + 額度
                    if !accounts.isEmpty {
                        AccountPicker(accounts: accounts)
                        if let acc = accounts.first(where: { $0.id == api.selectedAccount }) {
                            BudgetCard(account: acc, maxPositions: config?.maxPositions ?? 3)
                        }
                    }

                    // P&L card
                    if let s = summary {
                        PnLCard(summary: s)
                    }

                    // 所有帳戶今日比較
                    if accounts.count > 1 {
                        AccountsCompareCard(accounts: accounts)
                    }

                    // Config quick toggles
                    if let cfg = config {
                        ConfigCard(config: cfg, onUpdate: updateConfig)
                    }

                    // Latest signal
                    if let sig = ws.latestSignal {
                        LatestSignalCard(signal: sig)
                    }

                    // Latest trade
                    if let t = ws.latestTrade {
                        LatestTradeCard(trade: t)
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("今日總覽")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: load) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .refreshable { load() }
            .onAppear { load() }
            .onChange(of: api.selectedAccount) { _ in load() }
            .onChange(of: ws.latestTrade) { _ in load() }
            .overlay {
                if isLoading && summary == nil {
                    ProgressView("載入中...")
                }
            }
        }
    }

    private func todayString() -> String {
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd"
        return f.string(from: Date())
    }

    private func load() {
        Task {
            isLoading = true
            do {
                accounts = await loadAccounts(api)
                async let s = api.fetchSummary()
                async let c = api.fetchConfig()
                summary = try await s
                config  = try await c
                errorMsg = ""
            } catch {
                errorMsg = error.localizedDescription
            }
            isLoading = false
        }
    }

    private func updateConfig(_ update: ConfigUpdate) {
        Task {
            do {
                config = try await api.updateConfig(update)
                errorMsg = ""
            } catch {
                errorMsg = "更新失敗：\(error.localizedDescription)"
            }
        }
    }
}

// MARK: - Sub-cards

struct PnLCard: View {
    let summary: DailySummary

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("今日損益").font(.subheadline).foregroundColor(.secondary)
                    Text("\(summary.totalPnl >= 0 ? "+" : "")\(Int(summary.totalPnl)) 元")
                        .font(.system(size: 32, weight: .bold, design: .rounded))
                        .foregroundColor(summary.totalPnl >= 0 ? .green : .red)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 4) {
                    Label(summary.hitTarget ? "達標" : "未達標",
                          systemImage: summary.hitTarget ? "checkmark.seal.fill" : "xmark.seal")
                        .foregroundColor(summary.hitTarget ? .green : .orange)
                        .font(.headline)
                    Text("目標 \(Int(summary.targetPnl)) 元")
                        .font(.caption).foregroundColor(.secondary)
                }
            }

            HStack(spacing: 20) {
                StatTile(title: "交易次數", value: "\(summary.tradeCount)")
                StatTile(title: "獲利次數", value: "\(summary.winCount)")
                StatTile(title: "勝率", value: String(format: "%.0f%%", summary.winRate * 100))
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(.secondarySystemBackground)))
        .padding(.horizontal)
    }
}

/// 目前選取帳戶的額度與使用狀況
struct BudgetCard: View {
    let account: TradingAccount
    let maxPositions: Int

    // 真實帳戶的 budget 是單筆額度，總額度 = 單筆 × 最大持倉
    private var total: Double { account.isReal ? account.budget * Double(maxPositions) : account.budget }
    private var ratio: Double { total > 0 ? min(account.used / total, 1) : 0 }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(account.isReal ? "真實交易額度" : "模擬交易額度")
                    .font(.subheadline).foregroundColor(.secondary)
                Spacer()
                Text(account.label)
                    .font(.caption2).bold()
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(Capsule().fill((account.isReal ? Color.orange : Color.blue).opacity(0.2)))
            }
            HStack(alignment: .firstTextBaseline) {
                Text(formatWan(total)).font(.title2).bold()
                if account.isReal {
                    Text("（單筆 \(formatWan(account.budget)) × \(maxPositions) 檔）")
                        .font(.caption).foregroundColor(.secondary)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text("持倉中 \(Int(account.used)) 元")
                    Text("今日動用 \(Int(account.tradedToday ?? 0)) 元")
                }
                .font(.caption).foregroundColor(.secondary)
            }
            ProgressView(value: ratio)
                .tint(account.isReal ? .orange : .blue)
            HStack {
                Text("今日報酬率").font(.caption).foregroundColor(.secondary)
                Spacer()
                Text(String(format: "%+.2f%%", account.pnlPct * 100))
                    .font(.caption).bold()
                    .foregroundColor(account.todayPnl >= 0 ? .green : .red)
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(.secondarySystemBackground)))
        .padding(.horizontal)
    }
}

/// 各帳戶今日損益並列
struct AccountsCompareCard: View {
    let accounts: [TradingAccount]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("各帳戶今日表現").font(.headline)
            ForEach(accounts) { acc in
                HStack {
                    Circle().fill(acc.isReal ? Color.orange : Color.blue).frame(width: 8, height: 8)
                    Text(acc.label).font(.subheadline)
                    Spacer()
                    Text("\(acc.tradeCount) 筆").font(.caption).foregroundColor(.secondary)
                    Text("\(acc.todayPnl >= 0 ? "+" : "")\(Int(acc.todayPnl))")
                        .font(.subheadline).bold()
                        .foregroundColor(acc.todayPnl >= 0 ? .green : .red)
                        .frame(minWidth: 70, alignment: .trailing)
                    Text(String(format: "%+.2f%%", acc.pnlPct * 100))
                        .font(.caption)
                        .foregroundColor(acc.todayPnl >= 0 ? .green : .red)
                        .frame(minWidth: 56, alignment: .trailing)
                }
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(.secondarySystemBackground)))
        .padding(.horizontal)
    }
}

struct StatTile: View {
    let title: String
    let value: String

    var body: some View {
        VStack(spacing: 2) {
            Text(value).font(.headline)
            Text(title).font(.caption2).foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

struct ConfigCard: View {
    let config: TradingConfig
    let onUpdate: (ConfigUpdate) -> Void
    @State private var confirmAutoTrade = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("交易設定").font(.headline).padding(.bottom, 2)

            HStack {
                // 自動下單會送出真實委託，開啟前先確認，避免捲動時誤觸；關閉則立即生效
                ToggleRow(label: "自動下單", value: config.autoTrade) { v in
                    if v { confirmAutoTrade = true } else { onUpdate(ConfigUpdate(autoTrade: false)) }
                }
                .confirmationDialog("開啟自動下單？", isPresented: $confirmAutoTrade, titleVisibility: .visible) {
                    Button("開啟自動下單", role: .destructive) { onUpdate(ConfigUpdate(autoTrade: true)) }
                } message: {
                    Text("開啟後機器人會依信號自動送出真實委託。")
                }
                Spacer()
                ToggleRow(label: "當沖", value: config.enableDay) { v in
                    onUpdate(ConfigUpdate(enableDay: v))
                }
                Spacer()
                ToggleRow(label: "反沖", value: config.enableShort) { v in
                    onUpdate(ConfigUpdate(enableShort: v))
                }
            }

            HStack {
                Text("真實單筆額度").font(.subheadline).foregroundColor(.secondary)
                Spacer()
                Text(formatWan(config.dayBudget)).font(.subheadline)
            }

            HStack {
                Text("模擬額度").font(.subheadline).foregroundColor(.secondary)
                Spacer()
                Text((config.paperBudgets ?? []).map { formatWan(Double($0)) }.joined(separator: "、"))
                    .font(.subheadline)
            }

            HStack {
                Text("最大持倉").font(.subheadline).foregroundColor(.secondary)
                Spacer()
                Text("\(config.maxPositions) 檔").font(.subheadline)
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(.secondarySystemBackground)))
        .padding(.horizontal)
    }
}

struct ToggleRow: View {
    let label: String
    let value: Bool
    let onChange: (Bool) -> Void

    var body: some View {
        VStack(spacing: 4) {
            Toggle("", isOn: Binding(get: { value }, set: onChange))
                .labelsHidden()
            Text(label).font(.caption2).foregroundColor(.secondary)
        }
    }
}

struct LatestSignalCard: View {
    let signal: Signal

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("最新信號").font(.caption).foregroundColor(.secondary)
                HStack(spacing: 6) {
                    Text(signal.action)
                        .font(.subheadline).bold()
                        .foregroundColor(signal.action.contains("BUY") || signal.action == "COVER" ? .green : .red)
                    Text(signal.code).font(.subheadline)
                }
                Text(signal.reason).font(.caption).foregroundColor(.secondary)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 4) {
                Text("$ \(String(format: "%.1f", signal.price))")
                    .font(.subheadline).bold()
                Text("Score: \(signal.score)")
                    .font(.caption).foregroundColor(.secondary)
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(.secondarySystemBackground)))
        .padding(.horizontal)
    }
}

struct LatestTradeCard: View {
    let trade: Trade

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("最新成交").font(.caption).foregroundColor(.secondary)
                HStack(spacing: 6) {
                    Text(trade.action)
                        .font(.subheadline).bold()
                        .foregroundColor(trade.action == "BUY" ? .green : .red)
                    Text(trade.code).font(.subheadline)
                }
                Text(trade.reason).font(.caption).foregroundColor(.secondary)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 4) {
                Text("\(trade.pnl >= 0 ? "+" : "")\(Int(trade.pnl)) 元")
                    .font(.subheadline).bold()
                    .foregroundColor(trade.pnl >= 0 ? .green : .red)
                Text(accountLabel(trade.account, isReal: trade.isReal))
                    .font(.caption2)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(Capsule().fill(trade.isReal ? Color.orange.opacity(0.2) : Color.blue.opacity(0.2)))
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(.secondarySystemBackground)))
        .padding(.horizontal)
    }
}

/// 帳戶切換列：真實 + 每個模擬額度各一顆，選取後其他頁面的資料跟著切換。
struct AccountPicker: View {
    @EnvironmentObject var api: APIService
    let accounts: [TradingAccount]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(accounts) { acc in
                    let selected = acc.id == api.selectedAccount
                    Button {
                        api.selectedAccount = acc.id
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: acc.isReal ? "banknote" : "doc.text")
                                .font(.caption2)
                            Text(acc.label).font(.subheadline)
                        }
                        .padding(.horizontal, 12).padding(.vertical, 6)
                        .foregroundColor(selected ? .white : .primary)
                        .background(
                            Capsule().fill(selected
                                ? (acc.isReal ? Color.orange : Color.blue)
                                : Color(.secondarySystemBackground))
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal)
        }
    }
}

/// 載入帳戶清單，並確保 selectedAccount 指向存在的帳戶。
@MainActor
func loadAccounts(_ api: APIService) async -> [TradingAccount] {
    let list = (try? await api.fetchAccounts()) ?? []
    if !list.isEmpty, !list.contains(where: { $0.id == api.selectedAccount }) {
        // 預設：今日有真實成交看真實，否則看第一個模擬帳戶
        let real = list.first { $0.isReal && $0.tradeCount > 0 }
        api.selectedAccount = (real ?? list.first { !$0.isReal } ?? list[0]).id
    }
    return list
}
