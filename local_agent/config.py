"""
Local Agent 設定。
讀取 ~/.trading-agent/config.json，或環境變數 AGENT_* 覆蓋。
"""
import json
import os
from pathlib import Path
from dataclasses import dataclass, field


CONFIG_PATH = Path.home() / ".trading-agent" / "config.json"


@dataclass
class AgentConfig:
    server_url: str = "http://localhost:30880"   # 5090 server
    token: str = ""                               # JWT token（login 後填入）
    cert_path: str = ""                           # 富邦憑證路徑（本機）
    cert_pass_env: str = "FUBON_CERT_PASS"        # 憑證密碼的環境變數名稱
    fubon_pass_env: str = "FUBON_PASS"            # 富邦登入密碼的環境變數名稱
    fubon_account: str = ""                       # 富邦帳號（身分證號，用於 login）
    paper_mode: bool = True                       # True = 不下真實單


def load() -> AgentConfig:
    cfg = AgentConfig()

    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        except Exception as e:
            print(f"[Config] 讀取失敗: {e}，使用預設值")

    # 環境變數優先
    if v := os.getenv("AGENT_SERVER_URL"):  cfg.server_url   = v
    if v := os.getenv("AGENT_TOKEN"):       cfg.token        = v
    if v := os.getenv("AGENT_CERT_PATH"):   cfg.cert_path    = v
    if v := os.getenv("AGENT_ACCOUNT"):     cfg.fubon_account= v
    if v := os.getenv("AGENT_PAPER"):       cfg.paper_mode   = v.lower() != "false"

    return cfg


def save(cfg: AgentConfig):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({
        "server_url":    cfg.server_url,
        "token":         cfg.token,
        "cert_path":     cfg.cert_path,
        "cert_pass_env": cfg.cert_pass_env,
        "fubon_account": cfg.fubon_account,
        "paper_mode":    cfg.paper_mode,
    }, indent=2, ensure_ascii=False))
