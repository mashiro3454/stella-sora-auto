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


def test_markers_exit_and_npc():
    # 사용자 스크린샷 조각: 왼쪽 문 아이콘(출구), 오른쪽 물음표 상자(NPC)
    crop = cv2.imdecode(np.fromfile(str(SCREENS / "markers_crop.png"), np.uint8), cv2.IMREAD_COLOR)
    canvas = np.full((1080, 1920, 3), 120, np.uint8)
    canvas[300:300 + crop.shape[0], 800:800 + crop.shape[1]] = crop
    kinds = sorted((m.icon[0], m.kind) for m in nv.find_markers(canvas, None))
    assert [k for _, k in kinds] == ["exit", "npc"]
    assert all(m.direction is not None for m in nv.find_markers(canvas, None))


@pytest.mark.parametrize("name, kind", [
    ("field__v3_5.5.jpg", "talk"), ("field__live_npc.jpg", "talk"),
    ("field__v1_130.2_enhance.jpg", "enhance"), ("field__v1_20.jpg", None), ("field__live_marker.jpg", None),
])
def test_find_prompt(name, kind):
    assert nv.find_prompt(load(name)) == kind


def test_prompt_circle_is_not_exit_door():
    # "F 대화" 표시의 흰 동그라미를 출구 문양으로 착각하던 문제 (8층에서 5분 헤맴)
    for name in ("field__v3_5.5.jpg", "field__v1_130.2_enhance.jpg"):
        img = load(name)
        d = nv.find_exit_door(img, nv.find_character(img))
        assert d is None or (d.center[0] - 1261) ** 2 + (d.center[1] - 676) ** 2 > 90 ** 2


def test_door_with_colored_ring():
    # 15층: 다음 방(선택) 문양 고리가 흰색이 아니라 하늘색이고, 레벨 표시에 반쯤 가렸다
    img = load("field__live_door15.jpg")
    ch = nv.find_character(img)
    d = nv.find_exit_door(img, ch, nv.DOOR_HUE["선택"][0])
    assert d is not None and abs(d.center[0] - 1588) < 20 and abs(d.center[1] - 100) < 25
    assert nv.door_matches(d, "선택")
    assert nv.find_markers(img, ch) == []  # 문이 화면에 있으면 가장자리 표시는 없다


def test_score_badge_is_not_marker():
    # 왼쪽 위 기록 점수 메달(파란 동그라미)을 출구 표시로 잡던 문제 (13층)
    img = load("odo/walk13_036.jpg")
    img = cv2.resize(img, (1920, 1080))
    assert all(not (m.icon[0] < 330 and m.icon[1] < 200) for m in nv.find_markers(img, nv.find_character(img)))


def test_cyan_gift_boxes_are_not_markers():
    # 14층: 청록 선물 상자를 출구 표시로 잡아 진짜 표시(왼쪽)와 번갈아 따라가다 7분을 넘겼다
    img = load("field__live_giftboxes14.jpg")
    # (이 사진은 1280 저장본을 키운 것이라 진짜 표시는 조각나서 안 잡힌다. 상자만 안 잡히면 된다)
    ms = nv.find_markers(img, nv.find_character(img))
    assert all(m.icon[0] < 1100 for m in ms)


def test_find_character_with_low_hp():
    # 체력이 반쯤 깎여 초록 칸이 2.5칸: 빈 칸(회청색)도 세서 찾아야 한다 (8층에서 4분 멈춤)
    x, y = nv.find_character(load("field__live_lowhp.jpg"))
    assert abs(x - 985) < 30 and abs(y - 515) < 20
