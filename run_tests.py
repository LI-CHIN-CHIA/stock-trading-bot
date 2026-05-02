"""
全面測試執行器
==============
執行所有單元測試 + 整合回測，輸出摘要報告。

用法:
  venv/bin/python run_tests.py              # 單元測試 + 1個月回測
  venv/bin/python run_tests.py --unit-only  # 只跑單元測試（快速）
  venv/bin/python run_tests.py --period 3mo # 回測改用3個月
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent


def run_unit_tests() -> bool:
    print("\n" + "=" * 60)
    print("  單元測試 (pytest)")
    print("=" * 60)
    start = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_risk.py",
         "tests/test_signals.py",
         "-v", "--tb=short", "--no-header",
         "--color=yes"],
        cwd=BASE,
    )
    elapsed = time.time() - start
    ok = result.returncode == 0
    print(f"\n{'✅ 單元測試全部通過' if ok else '❌ 單元測試有失敗'} ({elapsed:.1f}s)")
    return ok


def run_backtest(period: str) -> bool:
    print("\n" + "=" * 60)
    print(f"  整合回測 (period={period})")
    print("=" * 60)
    start = time.time()
    result = subprocess.run(
        [sys.executable, "tests/backtest_1m.py", "--period", period],
        cwd=BASE,
    )
    elapsed = time.time() - start
    ok = result.returncode == 0
    print(f"\n{'✅ 回測完成' if ok else '❌ 回測失敗'} ({elapsed:.1f}s)")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-only", action="store_true", help="只跑單元測試")
    parser.add_argument("--backtest-only", action="store_true", help="只跑回測")
    parser.add_argument("--period", default="1mo", help="回測期間（預設 1mo）")
    args = parser.parse_args()

    total_start = time.time()
    all_ok = True

    if not args.backtest_only:
        unit_ok = run_unit_tests()
        all_ok = all_ok and unit_ok
        if not unit_ok:
            print("\n⚠️  單元測試有失敗，但繼續執行回測…")

    if not args.unit_only:
        bt_ok = run_backtest(args.period)
        all_ok = all_ok and bt_ok

    total = time.time() - total_start
    print("\n" + "=" * 60)
    if all_ok:
        print(f"✅ 全部測試通過 ({total:.1f}s)")
    else:
        print(f"❌ 部分測試失敗 ({total:.1f}s)")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
