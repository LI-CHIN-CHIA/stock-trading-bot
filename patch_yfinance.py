"""Patch yfinance utils.py to remove inplace=True calls that trigger Pandas4Warning."""
import re
import importlib
import importlib.util
import pathlib

spec = importlib.util.find_spec("yfinance.utils")
path = pathlib.Path(spec.origin)
src = path.read_text()

src = re.sub(r'(\w+)\.sort_index\(inplace=True\)', r'\1 = \1.sort_index()', src)
src = re.sub(r'(\w+)\.set_index\("date", inplace=True\)', r'\1 = \1.set_index("date")', src)
src = re.sub(r'(\w+)\.sort_index\(inplace=False\)', r'\1 = \1.sort_index()', src)
src = re.sub(r'(\w+)\.set_index\("date", inplace=False\)', r'\1 = \1.set_index("date")', src)
# 多行 drop/rename：移除 inplace=True/False 參數，並在呼叫前加回傳值賦值
src = re.sub(r',\s*inplace=(True|False)\)', ')', src)
# 為未賦值的 df.drop( / df.rename( 加上 df = 前綴
src = re.sub(r'^(\s+)(df)\.(drop|rename)\(', r'\1\2 = \2.\3(', src, flags=re.MULTILINE)

path.write_text(src)
print(f"Patched {path}")

# 刪掉預編譯的 .pyc，強制 Python 重新從修改後的 .py 編譯
pyc = pathlib.Path(importlib.util.cache_from_source(str(path)))
if pyc.exists():
    pyc.unlink()
    print(f"Removed cached bytecode {pyc}")
