"""봇 로그 요약: 몇 판 출발/완주했는지, 재시작 이유, 650원 도박, 저장한 기록.

    python tools/night_report.py logs/night1.log
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(path: str) -> None:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    starts = sum("새 판 1층" in l for l in lines)
    done = sum("판째 끝" in l for l in lines)
    gamble = sum("(650원 도박)" in l for l in lines)
    won = sum("650원: 성공" in l for l in lines)
    lost = sum("650원: 실패" in l for l in lines)
    reasons = Counter(re.sub(r"^.*재시작: ", "", l) for l in lines if "층 재시작:" in l)
    errors = [l for l in lines if "오류" in l or "Traceback" in l]
    first = next((l[1:9] for l in lines if l.startswith("[")), "?")
    last = next((l[1:9] for l in reversed(lines) if l.startswith("[")), "?")
    print(f"기간 {first} ~ {last}")
    print(f"새 판 출발 {starts}번, 20층까지 완주 {done}번")
    print(f"650원 도박 선택지 {gamble}번 (성공 {won}, 실패 {lost})")
    print("재시작 이유:", ", ".join(f"{k} {v}번" for k, v in reasons.most_common()))
    print(f"오류 {len(errors)}줄")
    led = ROOT / "logs" / "records.jsonl"
    if led.exists():
        print("저장한 기록:")
        for l in led.read_text(encoding="utf-8", errors="replace").splitlines():
            r = json.loads(l)
            print(f"  {r['saved_at']} {r['name']}: 평점 {r['record_level']}, 점수 {r['score']}"
                  f"{' (버릴 조건: ' + ', '.join(r['discard_reasons']) + ')' if r['discard_reasons'] else ''}"
                  f"{'' if r.get('tracked', True) else ' (중간에 켠 판)'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "logs" / "night1.log"))
