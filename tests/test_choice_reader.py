from pathlib import Path

import cv2
import numpy as np

from stella_auto.choices import find_option_boxes

SCREENS = Path(__file__).parent / "screens"


def load(name):
    return cv2.imdecode(np.fromfile(str(SCREENS / name), np.uint8), cv2.IMREAD_COLOR)


def test_option_boxes_two_and_three():
    assert len(find_option_boxes(load("npc_choice__v1_40.jpg"))) == 2
    assert len(find_option_boxes(load("npc_choice__v3_2.jpg"))) == 3


def test_option_boxes_four():
    assert len(find_option_boxes(load("npc_choice__live_notes4.jpg"))) == 4


def test_no_boxes_elsewhere():
    for name in ("field__v1_20.jpg", "card_select__v1_25.jpg", "dialog__v1_80.jpg"):
        assert find_option_boxes(load(name)) == []
