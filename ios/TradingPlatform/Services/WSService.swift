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
        let base = api.serverURL
            .replacingOccurrences(of: "https://", with: "wss://")
            .replacingOccurrences(of: "http://", with: "ws://")
        let urlStr = "\(base)/ws/\(api.userId)?token=\(api.token)"
        guard let url = URL(string: urlStr) else { return }

        shouldReconnect = true
        task = URLSession.shared.webSocketTask(with: url)
        task?.resume()
        isConnected = true
        receive()
        startPing()
    }

    func disconnect() {
        shouldReconnect = false
        pingTimer?.invalidate()
        pingTimer = nil
        task?.cancel(with: .normalClosure, reason: nil)
        isConnected = false
    }

    private func receive() {
        task?.receive { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                switch result {
                case .failure:
                    self.isConnected = false
                    self.scheduleReconnect()
                case .success(let msg):
                    self.handle(msg)
                    self.receive()
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
                    isReal: d.isReal ?? false, createdAt: ""
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
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak self] in
            self?.connect()
        }
    }
}

struct PositionUpdate: Equatable {
    let code: String
    let shares: Int
}
