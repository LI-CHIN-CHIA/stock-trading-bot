import Foundation

@MainActor
final class APIService: ObservableObject {
    static let shared = APIService()

    @Published var serverURL: String = UserDefaults.standard.string(forKey: "serverURL") ?? "http://192.168.1.100:30880"
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

    func fetchTrades(date: String = "", days: Int = 1) async throws -> [Trade] {
        if !date.isEmpty {
            return try await get("/api/trades?trade_date=\(date)")
        } else if days > 1 {
            return try await get("/api/trades?days=\(days)")
        } else {
            return try await get("/api/trades")
        }
    }

    func fetchSummary(date: String = "", days: Int = 1) async throws -> DailySummary {
        if !date.isEmpty {
            return try await get("/api/summary?trade_date=\(date)")
        } else if days > 1 {
            return try await get("/api/summary?days=\(days)")
        } else {
            return try await get("/api/summary")
        }
    }

    // MARK: - Positions

    func fetchPositions() async throws -> [Position] {
        try await get("/api/positions")
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

    private func get<T: Decodable>(_ path: String) async throws -> T {
        let url = URL(string: serverURL + path)!
        var req = URLRequest(url: url, timeoutInterval: 15)
        headers().forEach { req.setValue($1, forHTTPHeaderField: $0) }
        let (data, resp) = try await URLSession.shared.data(for: req)
        try checkStatus(resp, data)
        return try decode(data)
    }

    private func post<B: Encodable, T: Decodable>(_ path: String, body: B, auth: Bool = true) async throws -> T {
        let url = URL(string: serverURL + path)!
        var req = URLRequest(url: url, timeoutInterval: 15)
        req.httpMethod = "POST"
        if auth { headers().forEach { req.setValue($1, forHTTPHeaderField: $0) } }
        else { req.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        req.httpBody = try JSONEncoder().encode(body)
        let (data, resp) = try await URLSession.shared.data(for: req)
        try checkStatus(resp, data)
        return try decode(data)
    }

    private func put<B: Encodable, T: Decodable>(_ path: String, body: B) async throws -> T {
        let url = URL(string: serverURL + path)!
        var req = URLRequest(url: url, timeoutInterval: 15)
        req.httpMethod = "PUT"
        headers().forEach { req.setValue($1, forHTTPHeaderField: $0) }
        req.httpBody = try JSONEncoder().encode(body)
        let (data, resp) = try await URLSession.shared.data(for: req)
        try checkStatus(resp, data)
        return try decode(data)
    }

    private func checkStatus(_ resp: URLResponse, _ data: Data) throws {
        guard let http = resp as? HTTPURLResponse else { return }
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
