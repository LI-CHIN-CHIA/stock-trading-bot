"""
FreeToken 整合測試
==================
驗證 FreeToken 伺服器 API 與 TradingAgents 的相容性。

使用方式：
  1. 先啟動 FreeToken 伺服器（在另一個 terminal）:
       ft serve --model Qwen/Qwen2.5-7B-Instruct --port 1919

  2. 執行測試:
       python tests/test_freetoken.py

  3. 若測試通過，執行完整 TradingAgents 整合測試:
       python tests/test_freetoken.py --full
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

FREETOKEN_BASE = os.getenv("FREETOKEN_BASE_URL", "http://localhost:1919")
TEST_MODEL = os.getenv("FREETOKEN_TEST_MODEL", "")  # 自動從 /v1/models 取得

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── 顏色輸出 ──────────────────────────────────────────────────────────────────

def ok(msg):  print(f"\033[32m✅ {msg}\033[0m")
def fail(msg): print(f"\033[31m❌ {msg}\033[0m"); sys.exit(1)
def info(msg): print(f"\033[34mℹ️  {msg}\033[0m")
def warn(msg): print(f"\033[33m⚠️  {msg}\033[0m")


# ── Test 1: 伺服器連線 ────────────────────────────────────────────────────────

def test_server_connection():
    info("Test 1: 測試 FreeToken 伺服器連線...")
    try:
        r = requests.get(f"{FREETOKEN_BASE}/v1/models", timeout=5)
        if r.status_code == 200:
            ok(f"伺服器連線成功 ({FREETOKEN_BASE})")
            return r.json()
        else:
            fail(f"伺服器回傳 {r.status_code}: {r.text[:200]}")
    except requests.ConnectionError:
        fail(f"無法連線到 {FREETOKEN_BASE} — 請先啟動 FreeToken:\n"
             f"  ft serve --model Qwen/Qwen2.5-7B-Instruct")


# ── Test 2: 模型列表 ──────────────────────────────────────────────────────────

def test_list_models(models_data):
    info("Test 2: 取得可用模型列表...")
    models = models_data.get("data", [])
    if not models:
        fail("沒有可用模型，請確認 FreeToken 已載入模型")
    for m in models:
        print(f"   - {m.get('id', '?')}")
    ok(f"找到 {len(models)} 個模型")
    return models[0]["id"]





# ── Test 3: OpenAI Chat Completions API ───────────────────────────────────────

def test_chat_completions(model_id):
    info(f"Test 3: 測試 /v1/chat/completions (model={model_id})...")
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "請用一句話描述台積電（2330）的主要業務。"}
        ],
        "max_tokens": 100,
        "temperature": 0.1,
    }
    try:
        r = requests.post(
            f"{FREETOKEN_BASE}/v1/chat/completions",
            json=payload,
            timeout=60,
        )
        if r.status_code != 200:
            fail(f"Chat completions 失敗 {r.status_code}: {r.text[:300]}")
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        print(f"   回應: {content[:150]}")
        ok("OpenAI Chat Completions API 相容")
        return content
    except Exception as e:
        fail(f"Chat completions 錯誤: {e}")


# ── Test 4: Streaming ─────────────────────────────────────────────────────────

def test_streaming(model_id):
    info(f"Test 4: 測試 Streaming (model={model_id})...")
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "說「串流測試成功」"}],
        "max_tokens": 20,
        "stream": True,
    }
    try:
        chunks = []
        with requests.post(
            f"{FREETOKEN_BASE}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=30,
        ) as r:
            for line in r.iter_lines():
                if line and line.startswith(b"data: "):
                    chunk = line[6:]
                    if chunk == b"[DONE]":
                        break
                    data = json.loads(chunk)
                    delta = data["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        chunks.append(delta)
        result = "".join(chunks)
        print(f"   串流回應: {result}")
        ok("Streaming 相容")
    except Exception as e:
        warn(f"Streaming 測試失敗（非必須）: {e}")


# ── Test 5: LangChain ChatOpenAI 整合 ────────────────────────────────────────

def test_langchain_integration(model_id):
    info(f"Test 5: 測試 LangChain ChatOpenAI 整合...")
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(
            model=model_id,
            base_url=f"{FREETOKEN_BASE}/v1",
            api_key="freetoken",
            temperature=0.1,
            max_tokens=100,
        )
        response = llm.invoke([HumanMessage(content="台股代號 2330 是什麼公司？")])
        print(f"   LangChain 回應: {response.content[:150]}")
        ok("LangChain ChatOpenAI 整合成功")
        return True
    except ImportError:
        warn("langchain_openai 未安裝，跳過 LangChain 測試")
        return False
    except Exception as e:
        fail(f"LangChain 整合失敗: {e}")


# ── Test 6: TradingAgents 整合（需要 --full 旗標）────────────────────────────

def test_trading_agents_integration(model_id):
    info("Test 6: 測試 TradingAgents 整合...")
    os.environ["OPENAI_API_KEY"] = "freetoken"
    os.environ["OPENAI_BASE_URL"] = f"{FREETOKEN_BASE}/v1"
    os.environ["ENABLE_TRADING_AGENTS"] = "true"
    os.environ["TA_PROVIDER"] = "openai"
    os.environ["TA_DEEP_MODEL"] = model_id
    os.environ["TA_QUICK_MODEL"] = model_id

    try:
        # 直接呼叫 ta_signal 的 _build_graph（需要先修改 ta_signal.py）
        from trader.ta_signal import _build_graph
        info("建立 TradingAgentsGraph...")
        graph = _build_graph()
        ok("TradingAgentsGraph 建立成功")

        info("分析 2330（台積電）...")
        start = time.time()
        result, _ = graph.propagate("2330", "2026-08-25")
        elapsed = time.time() - start
        print(f"   決策: {result}")
        print(f"   耗時: {elapsed:.1f}s")
        ok("TradingAgents 完整分析成功")
    except ImportError as e:
        warn(f"TradingAgents 模組未找到: {e}")
    except Exception as e:
        fail(f"TradingAgents 整合失敗: {e}")


# ── 效能基準測試 ──────────────────────────────────────────────────────────────

def benchmark(model_id, n=3):
    info(f"Benchmark: 測試推理速度 ({n} 次)...")
    times = []
    for i in range(n):
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": f"第{i+1}次：台股現在幾點開盤？"}],
            "max_tokens": 50,
        }
        t0 = time.time()
        r = requests.post(f"{FREETOKEN_BASE}/v1/chat/completions", json=payload, timeout=30)
        t1 = time.time()
        if r.status_code == 200:
            usage = r.json().get("usage", {})
            tps = usage.get("completion_tokens", 0) / max(t1 - t0, 0.01)
            times.append(t1 - t0)
            print(f"   [{i+1}] {t1-t0:.2f}s  ~{tps:.0f} tok/s")
    if times:
        avg = sum(times) / len(times)
        ok(f"平均推理時間: {avg:.2f}s")


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    global FREETOKEN_BASE
    parser = argparse.ArgumentParser(description="FreeToken 整合測試")
    parser.add_argument("--full", action="store_true", help="執行完整 TradingAgents 整合測試")
    parser.add_argument("--base-url", default=FREETOKEN_BASE, help="FreeToken 伺服器 URL")
    parser.add_argument("--model", default="", help="指定模型 ID（預設自動偵測）")
    parser.add_argument("--bench", action="store_true", help="執行效能基準測試")
    args = parser.parse_args()

    FREETOKEN_BASE = args.base_url

    print("\n" + "="*60)
    print("  FreeToken × TradingAgents 整合測試")
    print("="*60 + "\n")

    # 基本測試
    models_data = test_server_connection()
    model_id = args.model or test_list_models(models_data)
    test_chat_completions(model_id)
    test_streaming(model_id)
    test_langchain_integration(model_id)

    if args.bench:
        benchmark(model_id)

    if args.full:
        test_trading_agents_integration(model_id)

    print("\n" + "="*60)
    print("  所有基本測試通過！")
    print(f"  FreeToken 端點: {FREETOKEN_BASE}/v1")
    print(f"  模型: {model_id}")
    print("\n  下一步：修改 ta_signal.py 使用 FreeToken")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
