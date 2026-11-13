import SwiftUI

struct LoginView: View {
    @EnvironmentObject var api: APIService
    @State private var email      = ""
    @State private var password   = ""
    @State private var serverURL  = ""
    @State private var isLoading  = false
    @State private var errorMsg   = ""
    @State private var mode: Mode = .login

    enum Mode { case login, register }

    var body: some View {
        NavigationView {
            Form {
                Section("伺服器") {
                    TextField("Server URL", text: $serverURL)
                        .autocapitalization(.none)
                        .keyboardType(.URL)
                }

                Section("帳號") {
                    TextField("Email", text: $email)
                        .autocapitalization(.none)
                        .keyboardType(.emailAddress)
                    SecureField("密碼", text: $password)
                }

                if !errorMsg.isEmpty {
                    Section {
                        Text(errorMsg).foregroundColor(.red).font(.footnote)
                    }
                }

                Section {
                    Button(action: submit) {
                        HStack {
                            Spacer()
                            if isLoading {
                                ProgressView()
                            } else {
                                Text(mode == .login ? "登入" : "註冊")
                                    .bold()
                            }
                            Spacer()
                        }
                    }
                    .disabled(isLoading || email.isEmpty || password.isEmpty)

                    Button(mode == .login ? "還沒有帳號？註冊" : "已有帳號？登入") {
                        mode = mode == .login ? .register : .login
                        errorMsg = ""
                    }
                    .foregroundColor(.secondary)
                    .font(.footnote)
                }
            }
            .navigationTitle(mode == .login ? "登入交易平台" : "建立帳號")
            .onAppear {
                serverURL = api.serverURL
            }
        }
    }

    private func submit() {
        guard !isLoading else { return }
        isLoading = true
        errorMsg = ""

        let url = APIService.normalizeServerURL(serverURL)
        if !url.isEmpty {
            serverURL = url
            api.serverURL = url
            UserDefaults.standard.set(url, forKey: "serverURL")
        }

        Task {
            do {
                if mode == .login {
                    try await api.login(email: email, password: password)
                } else {
                    try await api.register(email: email, password: password, displayName: "")
                }
                // WebSocket 由 TradingPlatformApp 在主畫面出現時連線，這裡不重複呼叫
            } catch {
                errorMsg = error.localizedDescription
            }
            isLoading = false
        }
    }
}
