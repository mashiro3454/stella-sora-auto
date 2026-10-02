"""카드 화면 읽기 (실제 OCR). 샘플 화면은 tests/screens."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from stella_auto.cards import match_name, parse_level, read_cards, team_pool
from stella_auto.ocr import KoreanOcr
from stella_auto.preset import Preset

ROOT = Path(__file__).parent
POOL = team_pool(Preset.load(ROOT.parent / "presets" / "바람.json"))


@pytest.fixture(scope="module")
def ocr():
    return KoreanOcr()


def read(ocr, name):
    img = cv2.imdecode(np.fromfile(str(ROOT / "screens" / name), np.uint8), cv2.IMREAD_COLOR)
    return [(c.potential.name, c.level_from, c.level_to) for c in read_cards(img, ocr, POOL)]


@pytest.mark.parametrize("name, expected", [
    # 전체 화면 OCR이 이름을 통째로 빼먹던 화면 (금색 카드 3장)
    ("card_select__live_gold.jpg", [("숙청 명령", None, 1), ("정밀 영점 조절", None, 2), ("전투력 증폭", None, 3)]),
    ("card_select__v1_410.jpg", [("가속 돌파", None, 3), ("파멸의 질풍", 5, 6), ("혼란스러운 흐름", 2, 3)]),
    ("enhance_select__v1_490.jpg", [("관통 탄도", 3, 4), ("섬광 발도", 4, 5), ("바람 장벽", 3, 4)]),
    ("card_select__v1_25.jpg", [("연쇄 폭발", None, 1), ("기능 지속", None, 1), ("진원 확장", None, 1)]),
    # 사용자 영상 (2026-10-02): "섬멸의 잔향"을 "성별의 잔향"으로 읽어 세 번째 카드를 통째로 놓쳤다
    ("enhance_select__v2_bonus_misread.jpg", [("혼란스러운 흐름", 2, 3), ("결전의 순간", 3, 4), ("섬멸의 잔향", 3, 4)]),
])
def test_read_cards(ocr, name, expected):
    assert read(ocr, name) == expected


@pytest.mark.parametrize("text, expected", [
    ("성별의 잔향", "섬멸의 잔향"),  # 받침/초성 하나씩 틀림: 글자 비율 60, 자모로는 맞는다
    ("섬멸의 잔향", "섬멸의 잔향"),
    ("획 득", None),
    ("6레벨 추천", None),
])
def test_match_name(text, expected):
    names = [p.name for p in POOL.values() if p.name]
    m = match_name(text, names)
    assert (names[m[0]] if m else None) == expected


@pytest.mark.parametrize("text, expected", [
    ("레벨 3", (None, 3, 0)), ("레벨 2 > 3", (2, 3, 0)), ("레벨 1+2", (None, 1, 2)),
    ("레벨 4+3 > 6+3", (4, 6, 3)), ("레멜 3>4 公", (3, 4, 0)), ("설명 스크롤", None),
])
def test_parse_level(text, expected):
    assert parse_level(text) == expected
