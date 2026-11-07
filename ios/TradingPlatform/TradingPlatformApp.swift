import SwiftUI

@main
struct TradingPlatformApp: App {
    @StateObject private var api = APIService.shared
    @StateObject private var ws  = WSService.shared

    var body: some Scene {
        WindowGroup {
            if api.isLoggedIn {
                MainTabView()
                    .environmentObject(api)
                    .environmentObject(ws)
                    .onAppear {
                        ws.connect()
                    }
            } else {
                LoginView()
                    .environmentObject(api)
                    .environmentObject(ws)
            }
        }
    }
}
