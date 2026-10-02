"""소리(협주스킬 재료) 읽기: 가방의 협주스킬별 필요한 소리, 상점 소리 상품의 종류.

소리는 9종이고 종류마다 그림 색이 다르다 (가방 위쪽 줄 순서, OpenCV 색상값 0~180, 2026-10-02 12층에서 잼):

| 순서 | 이름 | 그림 | 색상 |
|---|---|---|---|
| 0 | 강공의 소리 | 빨간 칼 | 7 |
| 1 | 행운의 소리 | 주황 눈 | 15 |
| 2 | 폭발의 소리 | 보라 | 129 |
| 3 | 체력의 소리 | 초록 하트 | 71 |
| 4 | 집중의 소리 | 청록 주먹 | 93 |
| 5 | 기술의 소리 | 파란 칼 | 109 (상점 그림은 106) |
| 6 | 필살기의 소리 | 분홍 별 | 163 |
| 7 | (이름 모름) | 하늘 물방울 | 103 |
| 8 | 바람의 소리 | 연두 소용돌이 | 44 |

가방 → 레코드 스킬 탭의 협주스킬 칸마다 "그림 35/25" 꼴로 (가진 개수/다음 레벨에 필요한 개수)가 있다.
상점 소리 상품은 그림이 같은 색이라 색으로 종류를 안다 (이름 OCR이 "%15"처럼 깨져도).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .ocr import KoreanOcr

NOTE_NAMES = ["강공", "행운", "폭발", "체력", "집중", "기술", "필살기", "물방울", "바람"]
NOTE_HUES = [7, 15, 129, 71, 93, 109, 163, 103, 44]  # 바람은 가방 위 46, 협주 칸 39~50
HUE_TOL = 6
ENSEMBLE_AREA = (500, 520, 1360, 890)  # 가방 레코드 스킬 탭의 협주스킬 6칸
_FRACTION = re.compile(r"(\d+)\s*/\s*(\d+)")


def icon_hue(img: np.ndarray, box: tuple[int, int, int, int]) -> float | None:
    """box 안 진한 색 픽셀들의 평균 색상 (색상은 원처럼 돌아서 원형 평균)."""
    x0, y0, x1, y1 = box
    hsv = cv2.cvtColor(img[max(0, y0):y1, max(0, x0):x1], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    m = hsv[(hsv[:, 1] > 110) & (hsv[:, 2] > 110)]
    if len(m) < 15:
        return None
    ang = m[:, 0].astype(float) * np.pi / 90
    return float((np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360) / 2)


def classify(hue: float | None) -> int | None:
    if hue is None:
        return None
    diffs = [abs((hue - h + 90) % 180 - 90) for h in NOTE_HUES]
    i = int(np.argmin(diffs))
    return i if diffs[i] <= HUE_TOL else None


def type_from_name(name: str) -> int | None:
    n = name.replace(" ", "")
    for i, key in enumerate(NOTE_NAMES):
        if key in n:
            return i
    return None


@dataclass
class NoteNeeds:
    have: dict[int, int] = field(default_factory=dict)  # 종류 -> 가진 개수
    users: dict[int, int] = field(default_factory=dict)  # 종류 -> 이 소리가 필요한 협주스킬 수
    needs: dict[int, list[int]] = field(default_factory=dict)  # 종류 -> 협주스킬별 다음 레벨 필요 개수들

    def summary(self) -> str:
        return ", ".join(f"{NOTE_NAMES[t]} {self.have.get(t, '?')}개/협주 {n}개" for t, n in sorted(self.users.items()))

    def gain(self, t: int, k: int = 5) -> int:
        """이 소리 k개를 더 받으면 협주스킬 활성화/레벨업 진행이 실제로 몇 칸 차는가 (사용자 2026-10-03).
        스킬마다 min(k, 남은 필요량)을 더한다. 이미 필요량을 넘긴 스킬은 0 (40개 규칙이 자동으로 들어간다)."""
        h = self.have.get(t, 0)
        return sum(min(k, max(0, need - h)) for need in self.needs.get(t, []))


# 협주스킬 6칸 (왼쪽/오른쪽 열 x 범위, 소리 줄 y). 활성화 전 칸은 흐리게 보여서 채도 기준을 낮게 잡는다
ENSEMBLE_COLS = ((600, 930), (1030, 1350))
ENSEMBLE_ROWS = (586, 710, 833)


def read_needs(img: np.ndarray, ocr: KoreanOcr) -> NoteNeeds:
    """가방 레코드 스킬 탭: 협주스킬마다 필요한 소리 종류(그림 색)와 가진 개수(그림 오른쪽 "35/25")."""
    out = NoteNeeds()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for x0, x1 in ENSEMBLE_COLS:
        for yc in ENSEMBLE_ROWS:
            y0, y1 = yc - 4, yc + 32
            m = ((hsv[y0:y1, x0:x1, 1] > 70) & (hsv[y0:y1, x0:x1, 2] > 70)).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(m)
            seen: set[int] = set()
            for k in range(1, n):
                x, y, w, h, a = (int(v) for v in st[k])
                if not (18 <= w <= 34 and 18 <= h <= 34 and a >= 150):
                    continue  # 숫자 글자(가늘다)와 칸 테두리는 뺀다
                px = hsv[y0 + y:y0 + y + h, x0 + x:x0 + x + w][lab[y:y + h, x:x + w] == k]
                ang = px[:, 0].astype(float) * np.pi / 90
                t = classify(float((np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360) / 2))
                if t is None or t in seen:
                    continue
                seen.add(t)
                out.users[t] = out.users.get(t, 0) + 1
                frac = _FRACTION.search(ocr.text(img, (x0 + x + w, y0, x0 + x + w + 85, y1)))
                if frac:
                    out.have[t] = int(frac.group(1))
                    out.needs.setdefault(t, []).append(int(frac.group(2)))
    return out


def shop_note_type(img: np.ndarray, cx: int, icon_y: int, name: str) -> int | None:
    """상점 소리 상품의 종류: 이름 (기술/물방울처럼 색이 가까운 게 있어서), 못 읽으면 그림 색."""
    t = type_from_name(name)
    return t if t is not None else classify(icon_hue(img, (cx - 45, icon_y - 25, cx + 15, icon_y + 30)))
