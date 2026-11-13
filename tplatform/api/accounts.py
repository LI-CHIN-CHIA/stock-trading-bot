"""帳戶標籤：'real'（真實）或 'paper-<額度>'（模擬，可多個）。"""
from .models import UserConfig

REAL = "real"
ALL = "all"
DEFAULT_PAPER = 500_000
MAX_PAPER_ACCOUNTS = 6


def parse_budgets(raw) -> list[int]:
    if isinstance(raw, str):
        raw = [x for x in raw.split(",") if x.strip()]
    out: list[int] = []
    for x in raw or []:
        try:
            v = int(float(x))
        except (TypeError, ValueError):
            continue
        if v > 0 and v not in out:
            out.append(v)
    return out[:MAX_PAPER_ACCOUNTS] or [DEFAULT_PAPER]


def paper_id(capital: float) -> str:
    return f"paper-{int(capital)}"


def paper_budgets(cfg: UserConfig | None) -> list[int]:
    return parse_budgets(cfg.paper_budgets if cfg else "")


def resolve_write(account: str, is_real: bool, cfg: UserConfig | None) -> str:
    """寫入時：未帶 account 的舊客戶端依 is_real 歸戶。"""
    if account:
        return account
    return REAL if is_real else paper_id(paper_budgets(cfg)[0])


def read_filter(account: str, cfg: UserConfig | None) -> list[str] | None:
    """讀取時要篩選的帳戶清單；None = 不篩選。

    未指定 account（舊版 app）→ 真實 + 主要模擬帳戶，與加入多帳戶前的結果一致，
    且不會混入其他模擬額度的重複成交。
    """
    if account == ALL:
        return None
    if account:
        return [account]
    return [REAL, paper_id(paper_budgets(cfg)[0])]


def budget_of(account: str, cfg: UserConfig | None) -> float:
    """模擬/策略帳戶的額度寫在 id 尾端（paper-500000、fade-1000000）；真實帳戶用單筆額度。"""
    if account != REAL and "-" in account:
        try:
            return float(account.rsplit("-", 1)[1])
        except ValueError:
            return float(DEFAULT_PAPER)
    return cfg.day_budget if cfg else 50000.0


_STRATEGY_LABELS = {"paper": "模擬", "fade": "過熱放空"}


def label_of(account: str) -> str:
    if account == REAL:
        return "真實"
    prefix = account.split("-", 1)[0]
    return f"{_STRATEGY_LABELS.get(prefix, prefix)} {budget_of(account, None) / 10000:g}萬"
