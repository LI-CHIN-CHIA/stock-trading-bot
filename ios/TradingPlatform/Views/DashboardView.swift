import SwiftUI

struct DashboardView: View {
    @EnvironmentObject var api: APIService
    @EnvironmentObject var ws: WSService
    @State private var summary: DailySummary?
    @State private var config: TradingConfig?
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

                    // P&L card
                    if let s = summary {
                        PnLCard(summary: s)
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
                async let s = api.fetchSummary()
                async let c = api.fetchConfig()
                summary = try await s
                config  = try await c
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
            } catch {
                errorMsg = error.localizedDescription
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

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("交易設定").font(.headline).padding(.bottom, 2)

            HStack {
                ToggleRow(label: "自動下單", value: config.autoTrade) { v in
                    onUpdate(ConfigUpdate(autoTrade: v))
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
                Text("單筆額度").font(.subheadline).foregroundColor(.secondary)
                Spacer()
                Text("\(Int(config.dayBudget / 10000)) 萬").font(.subheadline)
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
                Text(trade.isReal ? "真實" : "模擬")
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
