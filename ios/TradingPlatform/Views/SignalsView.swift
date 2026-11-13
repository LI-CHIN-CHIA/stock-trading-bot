import SwiftUI

struct SignalsView: View {
    @EnvironmentObject var api: APIService
    @EnvironmentObject var ws: WSService
    @State private var signals: [Signal] = []
    @State private var isLoading = false
    @State private var errorMsg  = ""

    var body: some View {
        NavigationView {
            Group {
                if !errorMsg.isEmpty {
                    ContentUnavailableView("信號載入失敗", systemImage: "exclamationmark.triangle",
                                           description: Text(errorMsg))
                } else if signals.isEmpty && !isLoading {
                    ContentUnavailableView("今日無信號", systemImage: "antenna.radiowaves.left.and.right")
                } else {
                    List(signals) { sig in
                        SignalRow(signal: sig)
                    }
                    .listStyle(.plain)
                }
            }
            .navigationTitle("AI 信號")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: load) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .refreshable { load() }
            .onAppear { load() }
            .overlay { if isLoading && signals.isEmpty { ProgressView("載入中...") } }
            .onChange(of: ws.latestSignal) { _ in load() }
        }
    }

    private func load() {
        Task {
            isLoading = true
            do {
                signals = try await api.fetchSignals()
                errorMsg = ""
            } catch {
                errorMsg = error.localizedDescription
            }
            isLoading = false
        }
    }
}

struct SignalRow: View {
    let signal: Signal

    private var accentColor: Color {
        switch signal.action {
        case "BUY":   return .green
        case "SELL":  return .red
        case "SHORT": return .orange
        case "COVER": return .blue
        default:      return .gray
        }
    }

    var body: some View {
        HStack(spacing: 12) {
            // Score circle
            ZStack {
                Circle().stroke(accentColor, lineWidth: 2).frame(width: 36, height: 36)
                Text("\(signal.score)").font(.caption).bold().foregroundColor(accentColor)
            }

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(signal.code).font(.headline)
                    Text(signal.action)
                        .font(.caption).bold()
                        .foregroundColor(.white)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Capsule().fill(accentColor))
                }
                Text(signal.reason).font(.caption).foregroundColor(.secondary).lineLimit(2)
            }

            Spacer()

            Text("$ \(String(format: "%.1f", signal.price))")
                .font(.subheadline).bold()
        }
        .padding(.vertical, 4)
    }
}
