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
    @State private var paperBudgets: [Int] = [500000]
    @State private var newPaperWan: Double = 100

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

                    Section(header: Text("模擬交易額度"),
                            footer: Text("每個額度是一個獨立的模擬帳戶，單筆約用額度的三分之一；第一個為主要模擬帳戶。變更於下一個交易日生效，最多 6 個。")) {
                        ForEach(paperBudgets, id: \.self) { b in
                            HStack {
                                Image(systemName: "doc.text").foregroundColor(.blue)
                                Text("模擬 \(formatWan(Double(b)))")
                                Spacer()
                                if b == paperBudgets.first {
                                    Text("主要").font(.caption).foregroundColor(.secondary)
                                }
                            }
                        }
                        .onDelete { idx in
                            guard paperBudgets.count > 1 else { return }
                            paperBudgets.remove(atOffsets: idx)
                            update(ConfigUpdate(paperBudgets: paperBudgets))
                        }

                        if paperBudgets.count < 6 {
                            HStack {
                                Stepper("新增 \(Int(newPaperWan)) 萬", value: $newPaperWan, in: 10...1000, step: 10)
                                Button("加入") {
                                    let v = Int(newPaperWan) * 10000
                                    guard !paperBudgets.contains(v) else { return }
                                    paperBudgets.append(v)
                                    update(ConfigUpdate(paperBudgets: paperBudgets))
                                }
                                .buttonStyle(.borderedProminent)
                            }
                        }
                    }

                    Section(header: Text("真實交易額度"),
                            footer: Text("真實下單每筆的金額上限；總額度 = 單筆額度 × 最大持倉數。")) {
                        HStack {
                            Text("單筆額度")
                            Spacer()
                            Text("\(Int(budget / 10000)) 萬元")
                                .foregroundColor(.secondary)
                        }
                        // 放開滑桿才送出：拖動中每一格都送會產生大量請求，且完成順序不保證，
                        // 伺服器可能存到中間值
                        Slider(value: $budget, in: 10000...500000, step: 10000) { editing in
                            if !editing { update(ConfigUpdate(dayBudget: budget)) }
                        }

                        HStack {
                            Text("最大持倉數")
                            Spacer()
                            Text("\(Int(maxPos)) 檔")
                                .foregroundColor(.secondary)
                        }
                        Slider(value: $maxPos, in: 1...10, step: 1) { editing in
                            if !editing { update(ConfigUpdate(maxPositions: Int(maxPos))) }
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
                            update(ConfigUpdate(stopLossPct: -slPct / 100, takeProfitPct: tpPct / 100))
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
            do {
                apply(try await api.fetchConfig())
            } catch {
                errorMsg = error.localizedDescription
            }
            isLoading = false
        }
    }

    /// 以伺服器的設定為準，同步所有編輯中的欄位
    private func apply(_ cfg: TradingConfig) {
        config = cfg
        budget = cfg.dayBudget
        maxPos = Double(cfg.maxPositions)
        slPct  = abs(cfg.stopLossPct * 100)
        tpPct  = cfg.takeProfitPct * 100
        paperBudgets = cfg.paperBudgets ?? [500000]
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
                // 成功時只更新 config，不覆蓋其他尚未儲存的欄位（例如還沒按儲存的停損停利）
                config = try await api.updateConfig(u)
                errorMsg = ""
            } catch {
                errorMsg = "更新失敗：\(error.localizedDescription)"
                // 畫面上的開關/額度已先改了，失敗時重新載入，避免顯示與伺服器不一致
                load()
            }
        }
    }
}
