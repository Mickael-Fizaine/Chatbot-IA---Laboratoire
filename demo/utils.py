"""Shared formatting and I/O helpers for all demo scripts."""

import json
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

W = 74  # default box width
RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Box / header helpers
# ---------------------------------------------------------------------------

def header(title: str, subtitle: str = "") -> str:
    inner = W - 2
    lines = [f"\n╔{'═' * inner}╗", f"║{title.center(inner)}║"]
    if subtitle:
        lines.append(f"║{subtitle.center(inner)}║")
    lines.append(f"╚{'═' * inner}╝")
    return "\n".join(lines)


def section(title: str) -> str:
    bar = "─" * (W - len(title) - 5)
    return f"\n┌─ {title} {bar}"


def box(lines: list[str]) -> str:
    inner = W - 4
    out = [f"  ┌{'─' * (W - 4)}┐"]
    for line in lines:
        truncated = line[:inner]
        out.append(f"  │ {truncated:<{inner - 2}} │")
    out.append(f"  └{'─' * (W - 4)}┘")
    return "\n".join(out)


def sep(char: str = "─") -> str:
    return char * W


def step_banner(number: int, total: int, title: str) -> str:
    tag = f"  [{number}/{total}] {title}  "
    pad = W - len(tag)
    return f"\n{'░' * 3}{tag}{'░' * max(0, pad)}"


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def table(headers: list[str], rows: list[list], max_col: int = 36) -> str:
    def trunc(s: str) -> str:
        s = str(s)
        return s[:max_col - 3] + "..." if len(s) > max_col else s

    trunc_rows = [[trunc(c) for c in row] for row in rows]
    col_w = [
        max(len(str(headers[i])), max((len(r[i]) for r in trunc_rows), default=0))
        for i in range(len(headers))
    ]

    def line(left, mid, right, fill):
        return left + mid.join(fill * (w + 2) for w in col_w) + right

    def data_row(cells):
        return "│" + "│".join(f" {str(c).ljust(col_w[i])} " for i, c in enumerate(cells)) + "│"

    return "\n".join([
        line("┌", "┬", "┐", "─"),
        data_row(headers),
        line("├", "┼", "┤", "─"),
        *[data_row(r) for r in trunc_rows],
        line("└", "┴", "┘", "─"),
    ])


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

def bar(value: float, width: int = 28, full: str = "█", empty: str = "░") -> str:
    filled = round(max(0.0, min(1.0, value)) * width)
    return full * filled + empty * (width - filled)


def metric_line(label: str, value: float, pct: bool = True) -> str:
    label_col = 32
    val_str = f"{value:.0%}" if pct else f"{value:.2f}"
    return f"  {label:<{label_col}} {val_str:>6}  {bar(value)}"


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_txt(name: str, content: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{name}_{ts}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  Resultats sauvegardes -> demo/results/{path.name}")


def save_json(name: str, data: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{name}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Resultats sauvegardes -> demo/results/{path.name}")
