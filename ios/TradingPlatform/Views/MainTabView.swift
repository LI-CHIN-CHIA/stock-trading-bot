import SwiftUI

struct MainTabView: View {
    @EnvironmentObject var ws: WSService
    @State private var newSignal = false
    @State private var newTrade  = false

    var body: some View {
        TabView {
            DashboardView()
                .tabItem {
                    Label("總覽", systemImage: "house.fill")
                }

            SignalsView()
                .tabItem {
                    Label("信號", systemImage: "antenna.radiowaves.left.and.right")
                }
                .badge(newSignal ? "新" : nil)

            TradesView()
                .tabItem {
                    Label("交易", systemImage: "arrow.left.arrow.right")
                }
                .badge(newTrade ? "新" : nil)

            PositionsView()
                .tabItem {
                    Label("持倉", systemImage: "chart.bar.fill")
                }

            SettingsView()
                .tabItem {
                    Label("設定", systemImage: "gearshape.fill")
                }
        }
        .onChange(of: ws.latestSignal) { _ in
            newSignal = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) { newSignal = false }
        }
        .onChange(of: ws.latestTrade) { _ in
            newTrade = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) { newTrade = false }
        }
    }
}
