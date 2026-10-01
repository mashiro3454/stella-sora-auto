"""NPC 선택지 화면 읽기.

보기는 오른쪽에 흰 상자로 2~4개 세로로 쌓이고, 각 상자 바로 아래 오른쪽 끝에 파란 띠로 효과
("140 소모, 10개의 폭발의 소리 획득")가 붙는다. 질문은 첫 상자 위에 있다.
글자 높이로 질문과 보기를 가르면 보기 수에 따라 위치가 달라 틀리므로, 흰 상자를 그림으로 먼저 찾는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .ocr import KoreanOcr

# 상자 줄인지는 글자가 없는 상자 오른쪽 부분(✦ 장식만 있음)으로 본다. 상자 전체 폭으로 보면 긴 보기 글자
# ("어차피 이렇게 된 거, 전부 다 걸어 보지!")가 줄을 어둡게 해서 상자가 두 동강 나 빠졌다 (4개 중 2개만 읽음)
BOX_X = (1560, 1760)
SEARCH_Y = (250, 1000)
BRIGHT = 200
ROW_FILL = 0.85  # 한 줄에서 밝은 픽셀이 이 비율 이상이면 상자 줄
BOX_HEIGHT = (50, 130)


@dataclass
class Option:
    text: str
    effect: str
    box: tuple[int, int, int, int]  # (x0, y0, x1, y1)

    @property
    def center(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.box
        return (x0 + x1) // 2, (y0 + y1) // 2


def find_option_boxes(img: np.ndarray) -> list[tuple[int, int]]:
    """흰 보기 상자들의 (위 y, 아래 y)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reg = gray[SEARCH_Y[0]:SEARCH_Y[1], BOX_X[0]:BOX_X[1]]
    rows = (reg > BRIGHT).mean(axis=1) > ROW_FILL
    out, start = [], None
    for i, on in enumerate(list(rows) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if BOX_HEIGHT[0] <= i - start <= BOX_HEIGHT[1]:
                out.append((start + SEARCH_Y[0], i + SEARCH_Y[0]))
            start = None
    return out


def read_choices(img: np.ndarray, ocr: KoreanOcr) -> tuple[str, list[Option]]:
    """(질문, 보기들). 보기를 못 찾으면 빈 목록."""
    bands = find_option_boxes(img)
    if not bands:
        return "", []
    options = []
    for top, bottom in bands:
        text = " ".join(l.text for l in ocr.read(img, (1080, top, 1720, bottom))).strip()
        effect = " ".join(l.text for l in ocr.read(img, (1300, bottom - 6, 1890, bottom + 48))).strip()
        options.append(Option(text, effect, (1045, top, 1860, bottom)))
    first_top = bands[0][0]
    question = " ".join(l.text for l in ocr.read(img, (1000, max(0, first_top - 130), 1900, first_top - 8))).strip()
    return question.lstrip("@").strip(), options
