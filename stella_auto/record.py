"""탑이 끝난 기록 화면: 읽기와 저장 (docs/tower-rules.md "기록 보관").

2026-10-02 실험으로 확인한 흐름:
1. 20층이 끝나면 "탐색 완료" → 빈 곳 터치 → 기록 화면 (왼쪽 위 이름 옆 연필, 오른쪽 아래 "기록 저장").
2. 연필 → "이름 변경" 창. 이름 칸 클릭 → 입력 → "확인".
   이름에는 숫자, 글자, 밑줄만 들어간다 (점, 띄어쓰기, 쌍점은 지워진다). 그래서 "1002_0158" (월일_시분).
3. "기록 저장" → "안내: 기록 저장 성공" → "확인" → 난이도 선택 화면.

사용자 규칙 (2026-10-02): 기록 이름은 저장한 시각. 끝까지 간 판은 일단 다 저장하고, 점수와 버릴지 여부는
logs/records.jsonl 장부에 적는다 (분해는 되돌릴 수 없어서 밤새 혼자 돌 때는 하지 않는다).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .ocr import KoreanOcr

PENCIL = (552, 108)  # 이름 옆 연필
NAME_FIELD = (960, 553)  # 이름 변경 창의 입력 칸
RENAME_OK = (1170, 760)
SAVE_BUTTON = (1705, 980)  # "기록 저장"
NOTICE_OK = (960, 760)  # "기록 저장 성공" 창의 확인
TAB_POTENTIALS = (948, 115)
TAB_SKILLS = (1557, 115)

NAME_BOX = (230, 80, 520, 140)
RECORD_LEVEL_BOX = (100, 85, 215, 160)
RENAME_TITLE_BOX = (480, 236, 900, 296)  # "이름 변경"
NOTICE_TEXT_BOX = (480, 420, 1440, 620)  # "기록 저장 성공"
ENSEMBLE_LEVEL_BOXES = [(765, y - 24, 900, y + 24) for y in (475, 594, 713)] + \
                       [(1360, y - 24, 1500, y + 24) for y in (475, 594, 713)]
NAME_RE = re.compile(r"^\d{4}_\d{4}$")


def record_name(t: float | None = None) -> str:
    """저장한 시각 이름: "1002_0158" (10월 2일 01:58)."""
    return time.strftime("%m%d_%H%M", time.localtime(t))


def is_bot_record_name(name: str) -> bool:
    return bool(NAME_RE.match(name.replace(" ", "")))


def read_record_level(img: np.ndarray, ocr: KoreanOcr) -> int | None:
    """왼쪽 위 육각형 안의 도자기 평점 레벨 (두 자리). 배경 색(금/보라)에 따라 잘 읽히는 크기가 달라서
    여러 칸/크기로 읽고 가장 많이 나온 값을 쓴다. 테두리를 "1"로 읽어 "241"이 되기도 해서 앞 두 자리만."""
    votes: dict[int, int] = {}
    for box in ((100, 85, 215, 160), (115, 92, 200, 150), (120, 95, 195, 148)):
        for scale in (1.0, 1.5, 3.0):
            digits = re.sub(r"\D", "", ocr.text(img, box, scale=scale))
            if len(digits) >= 2 and 10 <= int(digits[:2]) <= 60:
                v = int(digits[:2])
                votes[v] = votes.get(v, 0) + 1
    return max(votes, key=votes.get) if votes else None


def read_ensemble_levels(img: np.ndarray, ocr: KoreanOcr) -> list[int | None]:
    """레코드 스킬 탭의 협주스킬 6개 "레벨 N"."""
    out = []
    for box in ENSEMBLE_LEVEL_BOXES:
        t = ocr.text(img, box).replace("레멜", "레벨").replace(" ", "")
        m = re.search(r"(\d+)", t)
        out.append(int(m.group(1)) if m else (0 if "활성" in t or "성화" in t else None))  # "활성화 전" = 0레벨
    return out


@dataclass
class SavedRecord:
    name: str
    saved_at: str
    floor_reached: int
    record_level: int | None
    ensemble_levels: list
    potentials: dict  # 잠재력 이름 -> 레벨 (봇이 판 내내 기억한 것)
    score: float | None
    discard_reasons: list = field(default_factory=list)
    tracked: bool = True  # 봇이 1층부터 끝까지 본 판인지 (중간에 켰으면 잠재 레벨이 빠져 있다)
    note: str = ""


def append_ledger(path: Path, rec: SavedRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
