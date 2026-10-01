from pathlib import Path

import cv2
import numpy as np
import pytest

from stella_auto.screen import ANCHORS, ScreenDetector, StableDetector

SCREENS = sorted((Path(__file__).parent / "screens").glob("*.jpg"))


@pytest.fixture(scope="module")
def detector():
    return ScreenDetector()


def test_all_templates_exist(detector):
    assert set(detector.templates) == {a.name for a in ANCHORS}


@pytest.mark.parametrize("path", SCREENS, ids=lambda p: p.stem)
def test_detect_sample_screens(detector, path):
    # 파일 이름 앞부분(__ 앞)이 정답 화면 이름
    expected = path.stem.split("__")[0]
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    d = detector.detect(img)
    top = sorted(d.scores.items(), key=lambda kv: -kv[1])[:3]
    assert d.state == expected, f"{d.state} (상위 점수 {top})"


def test_rejects_wrong_size(detector):
    with pytest.raises(ValueError, match="1920x1080"):
        detector.detect(np.full((720, 1280, 3), 128, np.uint8))


def test_stable_detector_waits_for_repeats(detector):
    field = cv2.imdecode(np.fromfile(str(Path(__file__).parent / "screens" / "field__v1_20.jpg"), np.uint8), cv2.IMREAD_COLOR)
    card = cv2.imdecode(np.fromfile(str(Path(__file__).parent / "screens" / "card_select__v1_25.jpg"), np.uint8), cv2.IMREAD_COLOR)
    s = StableDetector(detector, frames=3)
    assert [s.update(field).state for _ in range(3)] == ["transition", "transition", "field"]
    assert s.update(card).state == "transition"  # 화면이 바뀌면 다시 센다
    assert s.update(card).state == "transition"
    assert s.update(card).state == "card_select"
