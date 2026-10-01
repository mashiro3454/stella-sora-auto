"""화면 인식용 기준 조각(data/templates/*.png)을 샘플 화면에서 잘라 만든다.

    python tools/make_templates.py                       # SOURCES에 적힌 대로 전부 다시 만들기
    python tools/make_templates.py hint_card shot.png    # 기준 조각 하나를 다른 화면에서 다시 자르기

좌표는 stella_auto/screen.py 의 ANCHORS에 있다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stella_auto.screen import ANCHOR_BY_NAME, TEMPLATE_DIR  # noqa: E402

SAMPLES = ROOT / "tests" / "screens"

# 기준 조각 → 잘라낼 샘플 화면 (tests/screens)
SOURCES = {
    "toolbar_field": "field__v1_20.jpg",
    "toolbar_back_bag": "npc_choice__v1_40.jpg",
    "toolbar_bag": "card_select__v1_25.jpg",
    "hint_card": "card_select__v1_25.jpg",
    "hint_npc": "npc_choice__v1_40.jpg",
    "hint_dialog": "dialog__v1_80.jpg",
    "hint_shop": "shop__v1_330.jpg",
    "header_record_detail": "record_detail__live.jpg",
    "header_record_manage": "record_manage__v3_23.jpg",
    "header_tower": "difficulty_select__v1_605.jpg",
    "header_bag": "bag__v2_1.5.jpg",
    "popup_notice": "notice__v3_12.jpg",
    "popup_buy": "shop_buy__v1_325.jpg",
    "popup_filter": "filter__v3_33.jpg",
    "esc_giveup": "esc_map__v3_7.jpg",
    "notes_banner": "notes_gain__v1_75.jpg",
    "explore_done": "explore_done__v3_9.5.jpg",
    "btn_save_record": "record_result__v1_595.jpg",
    "enhance_banner": "enhance_select__v1_490.jpg",
    "enhance_done": "enhance_select__v1_135.jpg",
    "ensemble_banner": "ensemble_up__v1_388.jpg",
    "touch_continue": "notes_gain__v1_75.jpg",
}


def read(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def make(anchor: str, image: Path) -> Path:
    x0, y0, x1, y1 = ANCHOR_BY_NAME[anchor].box
    crop = cv2.cvtColor(read(image)[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    out = TEMPLATE_DIR / f"{anchor}.png"
    cv2.imencode(".png", crop)[1].tofile(str(out))
    return out


def main(argv: list[str]) -> int:
    if len(argv) == 2:
        print(make(argv[0], Path(argv[1])))
        return 0
    if argv:
        print(__doc__)
        return 1
    missing = set(ANCHOR_BY_NAME) - set(SOURCES)
    if missing:
        print("SOURCES에 없는 기준 조각:", ", ".join(sorted(missing)))
        return 1
    for anchor, src in SOURCES.items():
        print(make(anchor, SAMPLES / src).name, "<-", src)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
