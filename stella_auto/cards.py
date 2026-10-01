"""잠재력 카드 선택 화면 읽기 (레벨업, 강화머신, 상점 음료, 선택의 방 공통).

OCR로 카드 이름, 레벨 줄("레벨 3", "레벨 2 > 3", "레벨 1+1"), 추천 표시("6레벨 추천", "0/6")를
읽고, 이름은 우리 팀이 받을 수 있는 잠재력 이름 중 가장 비슷한 것으로 맞춘다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz, process

from .gamedata import GameData, PotentialInfo, default_gamedata
from .ocr import KoreanOcr, OcrLine
from .preset import Preset

CARD_AREA = (150, 200, 1780, 1000)  # 카드 3장이 놓이는 곳 (1920x1080 기준)
NAME_MIN_HEIGHT = 25  # 카드 이름은 설명 글자(높이 ~21)보다 크다
MATCH_THRESHOLD = 70

_LEVEL_FIX = str.maketrans({"멜": "벨", "빌": "벨", "▶": ">", "▸": ">", "→": ">"})
_LEVEL_RE = re.compile(r"레\s*벨\s*(\d+)\s*(?:\+\s*(\d+))?\s*(?:>\s*(\d+)\s*(?:\+\s*(\d+))?)?")
_COUNTER_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


@dataclass
class Card:
    slot: int  # 왼쪽부터 0, 1, 2
    click: tuple[int, int]  # 카드를 고를 때 누를 곳 (카드 그림 쪽)
    raw_name: str
    potential: PotentialInfo | None
    match_score: float
    level_from: int | None  # 업그레이드 전 레벨. 새 잠재력이면 None
    level_to: int  # 고르면 되는 레벨 (보너스 뺀 기본값)
    bonus: int  # "레벨 1+2"의 2 같은 보너스
    raw_level: str
    recommend: str | None = None  # "6레벨 추천" (게임 프리셋 표시)
    counter: tuple[int, int] | None = None  # "0/6" (지금/목표)
    level_known: bool = True  # 레벨 줄을 못 읽었으면 False (코어 카드는 레벨 줄이 없어서 항상 True)

    @property
    def is_new(self) -> bool:
        return self.level_from is None

    @property
    def gain(self) -> int:
        return self.level_to - (self.level_from or 0)


def team_pool(preset: Preset, gd: GameData | None = None) -> dict[int, PotentialInfo]:
    """이 팀이 탑에서 받을 수 있는 잠재력 전부 (메인은 메인용, 지원은 지원용)."""
    gd = gd or default_gamedata()
    pool: dict[int, PotentialInfo] = {}
    for ch in preset.characters:
        info = gd.characters.get(ch.char_id)
        if info is None:
            continue
        for pid in info.pool("master" if ch.slot == "master" else "assist"):
            if pid in gd.potentials:
                pool[pid] = gd.potentials[pid]
    return pool


def parse_level(text: str) -> tuple[int | None, int, int] | None:
    """'레벨 3' -> (None, 3, 0), '레벨 2 > 3' -> (2, 3, 0), '레벨 1+2' -> (None, 1, 2), '레벨 4+3 > 6+3' -> (4, 6, 3)."""
    m = _LEVEL_RE.search(text.translate(_LEVEL_FIX))
    if not m:
        return None
    a, a_bonus, b, b_bonus = m.groups()
    if b is None:
        return None, int(a), int(a_bonus or 0)
    return int(a), int(b), int(b_bonus or a_bonus or 0)


def _center(line: OcrLine) -> tuple[float, float]:
    x, y, w, h = line.box
    return x + w / 2, y + h / 2


def read_cards(img: np.ndarray, ocr: KoreanOcr, pool: dict[int, PotentialInfo]) -> list[Card]:
    lines = ocr.read(img, CARD_AREA)
    names_by_id = {pid: p.name for pid, p in pool.items() if p.name}
    choices = list(names_by_id.values())
    ids = list(names_by_id.keys())

    # 카드 이름 후보: 큰 글자이고, 바로 아래 비슷한 x에 "레벨" 줄이 있거나 이름 목록과 잘 맞는 것
    name_lines = []
    for l in lines:
        if l.box[3] < NAME_MIN_HEIGHT or "레벨" in l.text.translate(_LEVEL_FIX) or "달성" in l.text:
            continue
        best = process.extractOne(l.text.replace(" ", ""), [c.replace(" ", "") for c in choices], scorer=fuzz.ratio)
        if best and best[1] >= MATCH_THRESHOLD:
            name_lines.append((l, ids[best[2]], best[1]))
    name_lines.sort(key=lambda t: _center(t[0])[0])

    cards = []
    for slot, (line, pid, score) in enumerate(name_lines):
        cx, cy = _center(line)
        level = None
        raw_level = ""
        recommend = None
        counter = None
        for other in lines:
            ox, oy = _center(other)
            if abs(ox - cx) > 240:  # "0/6"은 카드 오른쪽 위 모서리라 이름 가운데에서 꽤 멀다
                continue
            if 0 < oy - cy < 70 and abs(ox - cx) < 170 and level is None:
                parsed = parse_level(other.text)
                if parsed:
                    level, raw_level = parsed, other.text
            elif -400 < oy - cy < -150:
                t = other.text.replace(" ", "")
                if "추" in t and "레벨" in t.translate(_LEVEL_FIX):
                    recommend = other.text
                m = _COUNTER_RE.search(t)
                if m and len(t) <= 6:
                    counter = (int(m.group(1)), int(m.group(2)))
        info = pool[pid]
        known = True
        if level is None:
            if info.kind == "core":  # 코어(분홍) 카드는 레벨 줄이 없다
                level = (None, 1, 0)
            else:  # 레벨을 못 읽음. Lv1로 치면 멀쩡한 카드를 버리게 된다
                level, known = (None, 0, 0), False
        cards.append(Card(
            slot=slot,
            click=(int(cx), int(cy - 250)),
            raw_name=line.text,
            potential=info,
            match_score=score,
            level_from=level[0],
            level_to=level[1],
            bonus=level[2],
            raw_level=raw_level,
            recommend=recommend,
            counter=counter,
            level_known=known,
        ))
    return cards
