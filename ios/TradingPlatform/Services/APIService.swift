import Foundation

@MainActor
final class APIService: ObservableObject {
    static let shared = APIService()

    @Published var serverURL: String = UserDefaults.standard.string(forKey: "serverURL") ?? "http://ljj-system-product-name:30880"
    @Published var token: String = Keychain.load("jwt_token") ?? ""
    @Published var userId: Int = UserDefaults.standard.integer(forKey: "userId")
    @Published var displayName: String = UserDefaults.standard.string(forKey: "displayName") ?? ""
    @Published var isLoggedIn: Bool = false

    private init() {
        isLoggedIn = !token.isEmpty && userId > 0
    }

    var isAuthenticated: Bool { !token.isEmpty && userId > 0 }

    // MARK: - Auth

    func login(email: String, password: String) async throws {
        let body = ["email": email, "password": password]
        let resp: TokenResponse = try await post("/auth/login", body: body, auth: false)
        storeSession(resp)
    }

    func register(email: String, password: String, displayName: String) async throws {
        let body = ["email": email, "password": password, "display_name": displayName]
        let resp: TokenResponse = try await post("/auth/register", body: body, auth: false)
        storeSession(resp)
    }

    func logout() {
        Keychain.delete("jwt_token")
        token = ""
        userId = 0
        displayName = ""
        isLoggedIn = false
        UserDefaults.standard.removeObject(forKey: "userId")
        UserDefaults.standard.removeObject(forKey: "displayName")
    }

    private func storeSession(_ resp: TokenResponse) {
        token = resp.accessToken
        userId = resp.userId
        displayName = resp.displayName
        isLoggedIn = true
        Keychain.save(resp.accessToken, for: "jwt_token")
        UserDefaults.standard.set(resp.userId, forKey: "userId")
        UserDefaults.standard.set(resp.displayName, forKey: "displayName")
    }

    // MARK: - Config

    func fetchConfig() async throws -> TradingConfig {
        try await get("/api/config")
    }

    func updateConfig(_ update: ConfigUpdate) async throws -> TradingConfig {
        try await put("/api/config", body: update)
    }

    // MARK: - Trades

    /// 目前在 App 選取的帳戶（"real" / "paper-<額度>"）；空字串 = 由伺服器自動決定
    @Published var selectedAccount: String = UserDefaults.standard.string(forKey: "selectedAccount") ?? "" {
        didSet { UserDefaults.standard.set(selectedAccount, forKey: "selectedAccount") }
    }

    private func query(_ items: [(String, String)]) -> String {
        let parts = items.filter { !$0.1.isEmpty }.map { "\($0.0)=\($0.1)" }
        return parts.isEmpty ? "" : "?" + parts.joined(separator: "&")
    }

    func fetchAccounts() async throws -> [TradingAccount] {
        try await get("/api/accounts")
    }

    /// start/end（YYYY-MM-DD，含頭尾）指定時優先於 days；兩者相同即為單日。
    private func rangeQuery(date: String, days: Int, start: String, end: String) -> [(String, String)] {
        let custom = !start.isEmpty || !end.isEmpty
        return [
            ("trade_date", date),
            ("start_date", date.isEmpty ? start : ""),
            ("end_date", date.isEmpty ? end : ""),
            ("days", date.isEmpty && !custom && days > 1 ? String(days) : ""),
            ("account", selectedAccount),
        ]
    }

    func fetchTrades(date: String = "", days: Int = 1,
                     start: String = "", end: String = "") async throws -> [Trade] {
        try await get("/api/trades" + query(rangeQuery(date: date, days: days, start: start, end: end)))
    }

    func fetchSummary(date: String = "", days: Int = 1,
                      start: String = "", end: String = "") async throws -> DailySummary {
        try await get("/api/summary" + query(rangeQuery(date: date, days: days, start: start, end: end)))
    }

    // MARK: - Positions

    func fetchPositions() async throws -> [Position] {
        try await get("/api/positions" + query([("account", selectedAccount)]))
    }

    // MARK: - Signals

    func fetchSignals(date: String = "") async throws -> [Signal] {
        let path = date.isEmpty ? "/api/signals" : "/api/signals?signal_date=\(date)"
        return try await get(path)
    }

    // MARK: - HTTP primitives

    private func headers() -> [String: String] {
        var h = ["Content-Type": "application/json"]
        if !token.isEmpty { h["Authorization"] = "Bearer \(token)" }
        return h
    }

    /// 去掉前後空白與結尾斜線，避免 "http://host/ " + "/api" 變成無效網址
    static func normalizeServerURL(_ s: String) -> String {
        var url = s.trimmingCharacters(in: .whitespacesAndNewlines)
        while url.hasSuffix("/") { url.removeLast() }
        return url
    }

    private func makeURL(_ path: String) throws -> URL {
        guard let url = URL(string: Self.normalizeServerURL(serverURL) + path),
              url.scheme != nil, url.host != nil else {
            throw APIError.server("Server URL 格式錯誤：\(serverURL)")
        }
        return url
    }

    private func get<T: Decodable>(_ path: String) async throws -> T {
        var req = URLRequest(url: try makeURL(path), timeoutInterval: 15)
        headers().forEach { req.setValue($1, forHTTPHeaderField: $0) }
        let (data, resp) = try await URLSession.shared.data(for: req)
        try checkStatus(resp, data, authed: true)
        return try decode(data)
    }

    private func post<B: Encodable, T: Decodable>(_ path: String, body: B, auth: Bool = true) async throws -> T {
        var req = URLRequest(url: try makeURL(path), timeoutInterval: 15)
        req.httpMethod = "POST"
        if auth { headers().forEach { req.setValue($1, forHTTPHeaderField: $0) } }
        else { req.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        req.httpBody = try JSONEncoder().encode(body)
        let (data, resp) = try await URLSession.shared.data(for: req)
        try checkStatus(resp, data, authed: auth)
        return try decode(data)
    }

    private func put<B: Encodable, T: Decodable>(_ path: String, body: B) async throws -> T {
        var req = URLRequest(url: try makeURL(path), timeoutInterval: 15)
        req.httpMethod = "PUT"
        headers().forEach { req.setValue($1, forHTTPHeaderField: $0) }
        req.httpBody = try JSONEncoder().encode(body)
        let (data, resp) = try await URLSession.shared.data(for: req)
        try checkStatus(resp, data, authed: true)
        return try decode(data)
    }

    private func checkStatus(_ resp: URLResponse, _ data: Data, authed: Bool) throws {
        guard let http = resp as? HTTPURLResponse else { return }
        // 帶 token 的請求被拒 → token 過期或失效，回到登入畫面（登入本身的 401 是帳密錯誤，不處理）
        if http.statusCode == 401 && authed && isLoggedIn {
            WSService.shared.disconnect()
            logout()
            throw APIError.server("登入已過期，請重新登入")
        }
        if http.statusCode >= 400 {
            let detail = (try? JSONDecoder().decode([String: String].self, from: data))?["detail"] ?? "HTTP \(http.statusCode)"
            throw APIError.server(detail)
        }
    }

    private func decode<T: Decodable>(_ data: Data) throws -> T {
        let dec = JSONDecoder()
        dec.keyDecodingStrategy = .useDefaultKeys
        return try dec.decode(T.self, from: data)
    }
}

enum APIError: LocalizedError {
    case server(String)
    var errorDescription: String? {
        switch self { case .server(let m): return m }
    }
}
