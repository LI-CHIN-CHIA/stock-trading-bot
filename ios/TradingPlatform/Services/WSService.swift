import Foundation
import Combine

@MainActor
final class WSService: ObservableObject {
    static let shared = WSService()

    @Published var latestTrade: Trade?
    @Published var latestSignal: Signal?
    @Published var positionUpdate: PositionUpdate?
    @Published var isConnected: Bool = false

    private var task: URLSessionWebSocketTask?
    private var pingTimer: Timer?
    private var shouldReconnect = true
    private let api = APIService.shared

    private init() {}

    func connect() {
        guard !api.token.isEmpty, api.userId > 0 else { return }
        let base = APIService.normalizeServerURL(api.serverURL)
            .replacingOccurrences(of: "https://", with: "wss://")
            .replacingOccurrences(of: "http://", with: "ws://")
        let urlStr = "\(base)/ws/\(api.userId)?token=\(api.token)"
        guard let url = URL(string: urlStr) else { return }

        // 先關掉舊連線，避免同時存在多條 socket、每筆事件收到多次
        task?.cancel(with: .goingAway, reason: nil)

        shouldReconnect = true
        let newTask = URLSession.shared.webSocketTask(with: url)
        task = newTask
        newTask.resume()
        // 真的收到伺服器訊息才算連線成功（見 receive）
        isConnected = false
        receive(on: newTask)
        startPing()
    }

    func disconnect() {
        shouldReconnect = false
        pingTimer?.invalidate()
        pingTimer = nil
        task?.cancel(with: .normalClosure, reason: nil)
        task = nil
        isConnected = false
    }

    private func receive(on socket: URLSessionWebSocketTask) {
        socket.receive { [weak self] result in
            Task { @MainActor in
                // 已被新連線取代或已斷線的舊 socket：忽略它的結果，也不要再排重連
                guard let self, self.task === socket else { return }
                switch result {
                case .failure:
                    self.isConnected = false
                    self.scheduleReconnect()
                case .success(let msg):
                    self.isConnected = true
                    self.handle(msg)
                    self.receive(on: socket)
                }
            }
        }
    }

    private func handle(_ msg: URLSessionWebSocketTask.Message) {
        guard case .string(let text) = msg,
              let data = text.data(using: .utf8),
              let ws = try? JSONDecoder().decode(WSMessage.self, from: data)
        else { return }

        switch ws.type {
        case "connected":
            isConnected = true

        case "trade", "trade_update":
            if let d = ws.data, let id = d.id, let code = d.code,
               let action = d.action, let shares = d.shares,
               let price = d.price, let pnl = d.pnl {
                latestTrade = Trade(
                    id: id, tradeDate: "", code: code, action: action,
                    shares: shares, price: price, pnl: pnl,
                    reason: d.reason ?? "", orderNo: "",
                    isReal: d.isReal ?? false, account: d.account, createdAt: ""
                )
            }

        case "signal":
            if let d = ws.data, let id = d.id, let code = d.code,
               let action = d.action, let price = d.price {
                latestSignal = Signal(
                    id: id, code: code, action: action,
                    score: d.score ?? 0, price: price,
                    reason: d.reason ?? "", signalDate: "", createdAt: ""
                )
            }

        case "position_update":
            if let d = ws.data, let code = d.code, let shares = d.shares {
                positionUpdate = PositionUpdate(code: code, shares: shares)
            }

        case "ping":
            task?.send(.string("{\"type\":\"pong\"}")) { _ in }

        default:
            break
        }
    }

    private func startPing() {
        pingTimer?.invalidate()
        pingTimer = Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { [weak self] _ in
            self?.task?.send(.string("{\"type\":\"pong\"}")) { _ in }
        }
    }

    private func scheduleReconnect() {
        guard shouldReconnect else { return }
        let failed = task
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak self] in
            // 等待期間若已經重新連線（或登出），就不要再建一條
            guard let self, self.shouldReconnect, self.task === failed else { return }
            self.connect()
        }
    }
}

struct PositionUpdate: Equatable {
    let code: String
    let shares: Int
}
