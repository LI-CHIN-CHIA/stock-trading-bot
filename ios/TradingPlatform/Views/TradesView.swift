import SwiftUI

struct TradesView: View {
    @EnvironmentObject var api: APIService
    @EnvironmentObject var ws: WSService
    @State private var trades: [Trade] = []
    @State private var isLoading = false

    var body: some View {
        NavigationView {
            Group {
                if trades.isEmpty && !isLoading {
                    ContentUnavailableView("今日無交易記錄", systemImage: "chart.line.flattrend.xyaxis")
                } else {
                    List(trades) { trade in
                        TradeRow(trade: trade)
                    }
                    .listStyle(.plain)
                }
            }
            .navigationTitle("今日交易")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: load) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .refreshable { load() }
            .onAppear { load() }
            .overlay { if isLoading && trades.isEmpty { ProgressView("載入中...") } }
            .onChange(of: ws.latestTrade) { _ in load() }
        }
    }

    private func load() {
        Task {
            isLoading = true
            trades = (try? await api.fetchTrades()) ?? []
            isLoading = false
        }
    }
}

struct TradeRow: View {
    let trade: Trade

    var isBuy: Bool { trade.action == "BUY" || trade.action == "SHORT" }

    var body: some View {
        HStack(spacing: 12) {
            // Action badge
            Text(trade.action)
                .font(.caption).bold()
                .foregroundColor(.white)
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(Capsule().fill(isBuy ? Color.green : Color.red))

            VStack(alignment: .leading, spacing: 2) {
                Text(trade.code).font(.headline)
                Text(trade.reason).font(.caption).foregroundColor(.secondary)
                    .lineLimit(1)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
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
