import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var api: APIService
    @State private var config: TradingConfig?
    @State private var isLoading = false
    @State private var errorMsg  = ""
    @State private var showLogoutConfirm = false

    // Editable state
    @State private var budget: Double = 50000
    @State private var maxPos: Double = 3
    @State private var slPct: Double  = 1.5
    @State private var tpPct: Double  = 2.5

    var body: some View {
        NavigationView {
            Form {
                if let cfg = config {
                    Section("自動交易") {
                        Toggle("自動下單", isOn: bindBool(\.autoTrade))
                        Toggle("當沖（做多）", isOn: bindBool(\.enableDay))
                        Toggle("反沖（做空）", isOn: bindBool(\.enableShort))
                        Toggle("隔日沖", isOn: bindBool(\.enableOvernight))
                    }

                    Section("資金管理") {
                        HStack {
                            Text("單筆額度")
                            Spacer()
                            Text("\(Int(budget / 10000)) 萬元")
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $budget, in: 10000...500000, step: 10000)
                            .onChange(of: budget) { v in
                                update(ConfigUpdate(dayBudget: v))
                            }

                        HStack {
                            Text("最大持倉數")
                            Spacer()
                            Text("\(Int(maxPos)) 檔")
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $maxPos, in: 1...10, step: 1)
                            .onChange(of: maxPos) { v in
                                update(ConfigUpdate(maxPositions: Int(v)))
                            }
                    }

                    Section("風控") {
                        HStack {
                            Text("停損 %")
                            Spacer()
                            Text("-\(String(format: "%.1f", slPct))%")
                                .foregroundColor(.red)
                        }
                        Slider(value: $slPct, in: 0.5...5.0, step: 0.5)

                        HStack {
                            Text("停利 %")
                            Spacer()
                            Text("+\(String(format: "%.1f", tpPct))%")
                                .foregroundColor(.green)
                        }
                        Slider(value: $tpPct, in: 0.5...10.0, step: 0.5)

                        Button("儲存停損停利") {
                            update(ConfigUpdate())
                        }
                    }
                } else {
                    ProgressView("載入設定中...")
                }

                Section("帳號") {
                    HStack {
                        Text("Email")
                        Spacer()
                        Text(api.displayName).foregroundColor(.secondary)
                    }
                    Button("登出", role: .destructive) {
                        showLogoutConfirm = true
                    }
                }

                if !errorMsg.isEmpty {
                    Section {
                        Text(errorMsg).foregroundColor(.red).font(.footnote)
                    }
                }
            }
            .navigationTitle("設定")
            .onAppear { load() }
            .confirmationDialog("確認登出？", isPresented: $showLogoutConfirm) {
                Button("登出", role: .destructive) {
                    WSService.shared.disconnect()
                    api.logout()
                }
            }
        }
    }

    private func load() {
        Task {
            isLoading = true
            if let cfg = try? await api.fetchConfig() {
                config = cfg
                budget = cfg.dayBudget
                maxPos = Double(cfg.maxPositions)
                slPct  = abs(cfg.stopLossPct * 100)
                tpPct  = cfg.takeProfitPct * 100
            }
            isLoading = false
        }
    }

    private func bindBool(_ kp: WritableKeyPath<TradingConfig, Bool>) -> Binding<Bool> {
        Binding(
            get: { config?[keyPath: kp] ?? false },
            set: { v in
                config?[keyPath: kp] = v
                var u = ConfigUpdate()
                switch kp {
                case \TradingConfig.autoTrade:       u.autoTrade = v
                case \TradingConfig.enableDay:       u.enableDay = v
                case \TradingConfig.enableShort:     u.enableShort = v
                case \TradingConfig.enableOvernight: u.enableOvernight = v
                default: break
                }
                update(u)
            }
        )
    }

    private func update(_ u: ConfigUpdate) {
        Task {
            do {
                config = try await api.updateConfig(u)
            } catch {
                errorMsg = error.localizedDescription
            }
        }
    }
}
