"""python -m local_agent [login|run]"""
import asyncio
import sys
import getpass
import aiohttp
from .config import load, save, AgentConfig


async def _login(cfg: AgentConfig):
    server = input(f"Server URL [{cfg.server_url}]: ").strip() or cfg.server_url
    email  = input("Email: ").strip()
    pw     = getpass.getpass("Password: ")
    cfg.server_url = server

    async with aiohttp.ClientSession() as s:
        async with s.post(f"{server}/auth/login",
                          json={"email": email, "password": pw}) as r:
            data = await r.json()

    if "access_token" not in data:
        print(f"登入失敗: {data.get('detail', data)}")
        sys.exit(1)

    cfg.token = data["access_token"]
    save(cfg)
    print(f"✅ 登入成功 user_id={data['user_id']} display_name={data['display_name']}")
    print(f"設定已存至 ~/.trading-agent/config.json")


def main():
    cfg = load()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"

    if cmd == "login":
        asyncio.run(_login(cfg))
    elif cmd == "run":
        if not cfg.token:
            print("尚未登入，請先執行: python -m local_agent login")
            sys.exit(1)
        from .agent import LocalAgent
        asyncio.run(LocalAgent(cfg).run())
    else:
        print(f"未知指令: {cmd}")
        print("用法: python -m local_agent [login|run]")
        sys.exit(1)


main()
