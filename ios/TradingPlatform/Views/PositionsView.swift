import SwiftUI

struct PositionsView: View {
    @EnvironmentObject var api: APIService
    @EnvironmentObject var ws: WSService
    @State private var positions: [Position] = []
    @State private var accounts: [TradingAccount] = []
    @State private var isLoading = false

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                if accounts.count > 1 {
                    AccountPicker(accounts: accounts).padding(.vertical, 8)
                }
                if positions.isEmpty && !isLoading {
                    ContentUnavailableView("目前無持倉", systemImage: "tray")
                } else {
                    List(positions) { pos in
                        PositionRow(position: pos)
                    }
                    .listStyle(.plain)
                }
            }
            .navigationTitle("持倉")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: load) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .refreshable { load() }
            .onAppear { load() }
            .overlay { if isLoading && positions.isEmpty { ProgressView("載入中...") } }
            .onChange(of: ws.positionUpdate) { _ in load() }
            .onChange(of: api.selectedAccount) { _ in load() }
        }
    }

    private func load() {
        Task {
            isLoading = true
            accounts = await loadAccounts(api)
            positions = (try? await api.fetchPositions()) ?? []
            isLoading = false
        }
    }
}

struct PositionRow: View {
    let position: Position

    var body: some View {
        HStack(spacing: 12) {
            // Direction badge
            Text(position.directionLabel)
                .font(.caption).bold()
                .foregroundColor(.white)
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(Capsule().fill(position.direction == "SHORT" ? Color.orange : Color.blue))

            VStack(alignment: .leading, spacing: 2) {
                Text(position.code).font(.headline)
                Text("成本 $ \(String(format: "%.1f", position.entryPrice))")
                    .font(.caption).foregroundColor(.secondary)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                Text("\(position.shares) 股").font(.subheadline)
                Text(accountLabel(position.account, isReal: position.isReal))
                    .font(.caption2)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(Capsule().fill(position.isReal ? Color.orange.opacity(0.2) : Color.blue.opacity(0.2)))
            }
        }
        .padding(.vertical, 4)
    }
}
