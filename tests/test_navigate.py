from pathlib import Path

import cv2
import numpy as np
import pytest

from stella_auto import navigate as nv

SCREENS = Path(__file__).parent / "screens"


def load(name: str) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(SCREENS / name), np.uint8), cv2.IMREAD_COLOR)


@pytest.mark.parametrize(
    "angle, keys",
    [(0, ("d",)), (44, ("w", "d")), (90, ("w",)), (179, ("a",)), (-179, ("a",)), (-100, ("s",)), (-40, ("s", "d"))],
)
def test_keys_for_angle(angle, keys):
    assert nv.keys_for_angle(angle) == keys


def test_find_character():
    # 실제 캡처: 체력바 가운데 (984, 615) -> 발은 체력바보다 60px 위
    x, y = nv.find_character(load("field__live_marker.jpg"))
    assert abs(x - 984) < 10 and abs(y - 555) < 10


def test_find_character_none_on_menu():
    assert nv.find_character(load("record_detail__live.jpg")) is None


def test_find_exit_marker():
    img = load("field__live_marker.jpg")
    m = nv.find_exit_marker(img, nv.find_character(img))
    assert m is not None and abs(m.icon[0] - 1152) < 15 and abs(m.icon[1] - 160) < 15
    # 삼각형은 위쪽, 살짝 오른쪽
    assert m.direction is not None and 45 < nv.angle_of(*m.direction) < 90


def test_find_talk_prompt():
    assert nv.find_talk_prompt(load("field__v3_5.5.jpg")) is not None
    assert nv.find_talk_prompt(load("field__live_npc.jpg")) is not None
    assert nv.find_talk_prompt(load("field__v1_20.jpg")) is None


def test_template_tracker_follows_shift():
    img = load("field__live_marker.jpg")
    t = nv.TemplateTracker(img, (700, 300, 800, 380))
    shifted = np.roll(img, (40, -60), axis=(0, 1))  # 아래로 40, 왼쪽으로 60
    cx, cy = t.update(shifted)
    assert abs(cx - (750 - 60)) < 3 and abs(cy - (340 + 40)) < 3


def test_walk_toward_detours_when_blocked():
    # 캐릭터가 전혀 안 움직이는 가짜 게임: 막힘을 감지하고 옆으로 비켜 가야 한다
    img = load("field__live_marker.jpg")
    held = []
    res = nv.walk_toward(lambda k, s: held.append(k), lambda: img, lambda im, ch: (400, 200), max_steps=12)
    assert res.reason == "steps"
    assert ("s", "a") in held or ("w", "d") in held  # 목표(왼쪽 위)에서 90도 꺾은 방향
