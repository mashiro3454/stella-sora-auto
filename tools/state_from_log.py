"""봇 로그에서 지금 판의 상태(가진 잠재 레벨 등)를 다시 만들어 logs/run_state.json에 쓴다.

봇을 판 중간에 껐다 켤 때, 예전 코드라 상태 파일이 없으면 이걸로 만든다.

    python tools/state_from_log.py logs/night1.log
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stella_auto.gamedata import default_gamedata  # noqa: E402
from stella_auto.preset import Preset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main(log: str, preset: str = "presets/바람.json") -> None:
    lines = Path(log).read_text(encoding="utf-8").splitlines()
    start = max(i for i, l in enumerate(lines) if "새 판 1층" in l)
    run = lines[start:]
    gd = default_gamedata()
    pre = Preset.load(ROOT / preset)
    goals = {p.id: p for ch in pre.characters for p in ch.potentials}
    by_name = {}
    for pid, info in gd.potentials.items():
        if info.name:
            by_name.setdefault(info.name, pid)
    owned: dict[int, int] = {}
    lv3 = lv2 = rerolls_early = 0
    floor = 1
    gamble = any("650원: 성공" in l for l in run)
    for l in run:
        m = re.match(r"\[\S+\] (\d+)층", l)
        if m:
            floor = int(m.group(1))
        if "카드:" in l and " reroll " in l and floor <= 6:
            rerolls_early += 1
        m = re.search(r"카드: (?:강화 )?pick (?:강화: )?([^:|]+?)(?::| \|)", l)
        if not m:
            continue
        name = m.group(1).strip()
        lv = re.search(re.escape(name) + r"\((새|\d+)>(\d+)\)", l)
        pid = by_name.get(name)
        if pid is None or lv is None:
            continue
        to = int(lv.group(2))
        new = lv.group(1) == "새"
        info = gd.potentials[pid]
        if new and info.kind != "core":
            g = goals.get(pid)
            if to >= 3:
                lv3 += 1
            elif to == 2 and g and g.mark in ("필수", "다다익선"):
                lv2 += 1
        owned[pid] = max(owned.get(pid, 0), to)
    data = {"saved_at": time.time(), "floor": floor, "gold": 0, "owned": {str(k): v for k, v in owned.items()},
            "lv3_new_taken": lv3, "lv2_new_taken": lv2, "rerolls_early": rerolls_early,
            "gamble_won": gamble, "run_tracked": True}
    (ROOT / "logs" / "run_state.json").write_text(json.dumps(data), encoding="utf-8")
    names = {gd.potentials[k].name: v for k, v in owned.items()}
    print(f"{floor}층, 650원 {'성공' if gamble else '아직'}, Lv3 새 {lv3}, Lv2 새 {lv2}, 6층까지 리롤 {rerolls_early}")
    print(names)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "logs" / "night1.log"))
