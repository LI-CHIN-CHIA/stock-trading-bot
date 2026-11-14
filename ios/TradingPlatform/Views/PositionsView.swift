import SwiftUI

struct PositionsView: View {
    @EnvironmentObject var api: APIService
    @EnvironmentObject var ws: WSService
    @State private var positions: [Position] = []
    @State private var accounts: [TradingAccount] = []
    @State private var isLoading = false
    @State private var errorMsg  = ""

    var body: some View {
        NavigationView {
            // 帳戶列與狀態訊息都放在同一個 List 裡：若橫向的 AccountPicker 是頁面最上方的
            // 捲動區，導覽列會把它當成主捲動區，上下滑它時大標題會跟著展開／收合
            List {
                if accounts.count > 1 {
                    AccountPicker(accounts: accounts)
                        .padding(.vertical, 8)
                        .listRowInsets(EdgeInsets())
                        .listRowSeparator(.hidden)
                }
                if !errorMsg.isEmpty {
                    // 載入失敗不能顯示成「無持倉」，否則會誤以為已經沒有部位
                    ContentUnavailableView("持倉載入失敗", systemImage: "exclamationmark.triangle",
                                           description: Text(errorMsg))
                        .listRowSeparator(.hidden)
                } else if positions.isEmpty && !isLoading {
                    ContentUnavailableView("目前無持倉", systemImage: "tray")
                        .listRowSeparator(.hidden)
                } else {
                    ForEach(positions) { pos in
                        PositionRow(position: pos)
                    }
                }
            }
            .listStyle(.plain)
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
            do {
                positions = try await api.fetchPositions()
                errorMsg = ""
            } catch {
                errorMsg = error.localizedDescription
            }
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
