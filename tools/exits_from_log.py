"""봇 로그의 "rNNNN 출구 자리 기억 (x, y)" 줄로 방마다 최근에 나간 자리(exits)를 채운다.

exits는 2026-10-02에 생긴 칸이라 그 전 기억에는 없다. 봇을 멈춘 뒤에 돌린다 (돌고 있는 봇이 덮어쓴다).

    python tools/exits_from_log.py logs/night1.log
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stella_auto.roommemory import EXIT_HISTORY, RoomMemory  # noqa: E402

LINE = re.compile(r"(r\d{4}) 출구 자리 기억 \((-?\d+), (-?\d+)\)")


def main(log: str) -> None:
    mem = RoomMemory()
    seen: dict[str, list[list[int]]] = {}
    for line in Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE.search(line)
        if m and m.group(1) in mem.rooms:
            seen.setdefault(m.group(1), []).append([int(m.group(2)), int(m.group(3))])
    for rid, pts in seen.items():
        r = mem.rooms[rid]
        if not r.exits:
            r.exits = pts[-EXIT_HISTORY:]
        flag = " (들쭉날쭉)" if mem.exit_scattered(r) else ""
        print(f"{rid}: 나간 자리 {len(r.exits)}개{flag}")
    mem.save()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/night1.log")
