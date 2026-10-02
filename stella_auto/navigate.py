"""필드에서 걸어다니기: 캐릭터 찾기, 출구 표시 찾기, 목표 쪽으로 걷기.

docs/tower-rules.md "층 이동"의 실험 결과를 바탕으로 한다.
- WASD는 화면 기준 (W 위, A 왼쪽, S 아래, D 오른쪽).
- 캐릭터는 발밑 초록 체력바로 찾는다.
- 출구가 화면 밖이면 가장자리에 청록색 문 아이콘 + 방향 삼각형이 뜬다.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

# 체력바 아래쪽에서 발까지 대략 이만큼 위
HP_BAR_TO_FEET = 60

# (화면 각도, 키) 화면 기준: 오른쪽 0도, 위쪽 90도.
# 카메라가 비스듬히 내려다봐서 위아래 움직임이 좌우의 약 0.73배로 보인다. 그래서 W+D는 화면에서
# 45도가 아니라 약 36도로 간다 (13층 실험: W+D 0.1초에 (+30, -22)px).
DIAG = 36.0
DIRECTIONS: tuple[tuple[float, tuple[str, ...]], ...] = (
    (0, ("d",)), (DIAG, ("w", "d")), (90, ("w",)), (180 - DIAG, ("w", "a")),
    (180, ("a",)), (DIAG - 180, ("s", "a")), (-90, ("s",)), (-DIAG, ("s", "d")),
)


def angle_of(dx: float, dy: float) -> float:
    """화면 벡터의 각도 (오른쪽 0도, 위쪽 90도)."""
    return math.degrees(math.atan2(-dy, dx))


def angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def keys_for_angle(angle: float) -> tuple[str, ...]:
    return min(DIRECTIONS, key=lambda d: angle_diff(angle, d[0]))[1]


def angle_of_keys(keys: tuple[str, ...]) -> float | None:
    return next((a for a, k in DIRECTIONS if k == tuple(keys)), None)


def find_character(img: np.ndarray) -> tuple[float, float] | None:
    """캐릭터 발 위치 (화면 좌표). 체력바(칸이 한 줄로 늘어선 것)를 못 찾으면 None.

    체력바 칸은 찬 칸이 초록, 빈 칸이 어두운 회청색이다. 체력이 반쯤 깎이면 초록 칸이 2개 남짓이라
    (8층: 2.5칸) 초록 칸만 세면 못 찾고 봇이 4분 동안 멈춰 있었다. 그래서 빈 칸도 함께 센다.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = ((hsv[..., 0] > 35) & (hsv[..., 0] < 85) & (hsv[..., 1] > 120) & (hsv[..., 2] > 150)).astype(np.uint8)
    empty = ((hsv[..., 0] > 100) & (hsv[..., 0] < 130) & (hsv[..., 1] > 15) & (hsv[..., 1] < 70)
             & (hsv[..., 2] > 70) & (hsv[..., 2] < 115)).astype(np.uint8)
    # 체력이 아주 적으면 빈 칸이 어두운 빨강이 된다 (17층: 초록 1칸 + 빨강 3칸 남짓, 1분 동안 캐릭터를 못 찾음)
    empty |= (((hsv[..., 0] >= 170) | (hsv[..., 0] <= 4)) & (hsv[..., 1] > 80) & (hsv[..., 1] < 160)
              & (hsv[..., 2] > 60) & (hsv[..., 2] < 120)).astype(np.uint8)

    def segs(mask: np.ndarray, max_w: int) -> list:
        n, _, st, _ = cv2.connectedComponentsWithStats(mask)
        return [st[i] for i in range(1, n) if 15 <= st[i][2] <= max_w and 7 <= st[i][3] <= 15]

    # 빈 칸은 칸 사이 줄이 흐려 여러 칸이 한 덩어리로 잡히기도 한다 (칸 하나 약 29px)
    greens, empties = segs(green, 40), segs(empty, 200)
    best: list | None = None
    for s0 in greens:
        row_g = [t for t in greens if abs(t[1] - s0[1]) <= 3 and abs(t[0] - s0[0]) <= 200]
        row_e = [t for t in empties if abs(t[1] - s0[1]) <= 3 and abs(t[0] - s0[0]) <= 200]
        n_empty = sum(max(1, round(int(t[2]) / 29)) for t in row_e)
        if len(row_g) >= 3 or (row_g and len(row_g) + n_empty >= 4):
            row = row_g + row_e
            if best is None or len(row) > len(best):
                best = row
    if not best:
        return None
    x0 = min(t[0] for t in best)
    x1 = max(t[0] + t[2] for t in best)
    y = float(np.mean([t[1] + t[3] / 2 for t in best]))
    return (x0 + x1) / 2, y - HP_BAR_TO_FEET


@dataclass
class ExitMarker:
    icon: tuple[float, float]  # 문 아이콘 가운데
    direction: tuple[float, float] | None  # 삼각형이 가리키는 방향 (없으면 None)


@dataclass
class Marker:
    kind: str  # "exit" (문 아이콘) 또는 "npc" (물음표 상자 아이콘)
    icon: tuple[float, float]
    direction: tuple[float, float] | None  # 삼각형이 가리키는 방향 (없으면 None)


_MARKER_TEMPLATES: dict[str, np.ndarray] = {}


def _marker_templates() -> dict[str, np.ndarray]:
    if not _MARKER_TEMPLATES:
        from .screen import TEMPLATE_DIR

        for kind in ("exit", "npc"):
            path = TEMPLATE_DIR / f"marker_{kind}.png"
            _MARKER_TEMPLATES[kind] = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    return _MARKER_TEMPLATES


MARKER_MIN_SCORE = 0.55  # 진짜 표시는 0.9 안팎, 14층 청록 선물 상자는 0.3~0.4였다


def _classify_marker(gray: np.ndarray, box: tuple[int, int, int, int]) -> str | None:
    """청록 아이콘이 출구(문)인지 NPC(물음표 상자)인지. 어느 템플릿과도 안 닮았으면 None (표시가 아님).
    둘 다 비슷하게 닮았으면 가로세로 비율로 (문이 더 길쭉)."""
    x, y, w, h = box
    cx, cy = x + w // 2, y + h // 2
    region = gray[max(0, cy - 40):cy + 40, max(0, cx - 40):cx + 40]
    scores = {}
    for kind, t in _marker_templates().items():
        if region.shape[0] >= t.shape[0] and region.shape[1] >= t.shape[1]:
            scores[kind] = float(cv2.matchTemplate(region, t, cv2.TM_CCOEFF_NORMED).max())
    if not scores or max(scores.values()) < MARKER_MIN_SCORE:
        return None  # 청록색 물건 (14층: 선물 상자를 출구 표시로 잡아 오른쪽/왼쪽을 7분 동안 오갔다)
    if abs(scores.get("exit", 0) - scores.get("npc", 0)) > 0.08:
        return max(scores, key=scores.get)
    return "exit" if w / max(h, 1) < 0.9 else "npc"


def find_markers(img: np.ndarray, char: tuple[float, float] | None) -> list[Marker]:
    """화면 가장자리의 청록 표시들: 출구(문 아이콘)와 NPC(물음표 상자) + 각자 방향 삼각형."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 0] > 80) & (hsv[..., 0] < 100) & (hsv[..., 1] > 150) & (hsv[..., 2] > 150)).astype(np.uint8)
    mask[:110, :420] = 0  # 왼쪽 위 HUD
    mask[:200, :330] = 0  # 기록 점수 메달 (파란 동그라미를 출구 표시로 잡은 일이 있었다)
    mask[960:, :] = 0  # 아래 HUD
    mask[:85, 1420:] = 0  # 오른쪽 위 레벨/돈
    mask[:230, 1780:] = 0  # 자동 전투 버튼
    n, _, st, cen = cv2.connectedComponentsWithStats(mask)
    blobs = [(int(st[i][4]), cen[i], st[i]) for i in range(1, n) if st[i][4] > 60]
    if char:  # 캐릭터 발밑의 방향 화살표도 청록색이라 뺀다
        blobs = [b for b in blobs if math.hypot(b[1][0] - char[0], b[1][1] - char[1] - HP_BAR_TO_FEET) > 130]
    icons = [b for b in blobs if 25 <= b[2][3] <= 60 and 20 <= b[2][2] <= 55]
    if not icons:
        return []

    def inside(c: np.ndarray, b) -> bool:  # 문 아이콘은 여러 조각으로 쪼개져 잡힌다
        x, y, w, h = b[2][:4]
        return x - 2 <= c[0] <= x + w + 2 and y - 2 <= c[1] <= y + h + 2

    icon_ids = {id(i) for i in icons}  # 덩어리 안에 numpy 배열이 있어서 `in`으로 비교하면 오류가 난다
    triangles = [b for b in blobs if id(b) not in icon_ids and not any(inside(b[1], i) for i in icons)]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = []
    for icon in icons:
        # 삼각형은 가장 가까운 아이콘 하나에만 붙인다 (아이콘 두 개가 나란히 있을 때)
        mine = [t for t in triangles if math.hypot(*(t[1] - icon[1])) < 60
                and min(icons, key=lambda i: math.hypot(*(t[1] - i[1]))) is icon]
        direction = None
        if mine:
            tri = min(mine, key=lambda t: math.hypot(*(t[1] - icon[1])))
            d = tri[1] - icon[1]
            direction = (float(d[0]), float(d[1]))
        kind = _classify_marker(gray, tuple(int(v) for v in icon[2][:4]))
        if kind is None:
            continue
        out.append(Marker(kind, (float(icon[1][0]), float(icon[1][1])), direction))
    return out


# 예전 이름 (출구 표시만)
ExitMarker = Marker


def find_exit_marker(img: np.ndarray, char: tuple[float, float] | None) -> Marker | None:
    exits = [m for m in find_markers(img, char) if m.kind == "exit"]
    return exits[0] if exits else None


def marker_target(marker: Marker, char: tuple[float, float]) -> tuple[float, float]:
    """표시 쪽으로 가려면 향할 화면 좌표 (삼각형이 있으면 그 방향으로 멀리)."""
    if marker.direction is not None:
        dx, dy = marker.direction
        norm = math.hypot(dx, dy) or 1
        return char[0] + dx / norm * 600, char[1] + dy / norm * 600
    return marker.icon


@dataclass
class ExitDoor:
    center: tuple[float, float]  # 문 안 문양의 가운데
    size: tuple[float, float]  # 문양 타원의 가로세로
    hue: float  # 문양 고리 색 (OpenCV 색상값 0~180). 다음 방 종류마다 다르다


def find_exit_door(img: np.ndarray, char: tuple[float, float] | None = None,
                   hue: float | None = None) -> ExitDoor | None:
    """출구 문 안의 문양: 밝은 타원 테두리 + 안쪽이 방 종류 색.

    문양은 반투명이라 안쪽 채도가 낮을 수 있고(선택의 방 파란 문양), 화면 가장자리에서 잘릴 수 있다.
    잘린 경우 가장자리에 붙은 윤곽 점은 빼고 타원을 맞춘다. "F 대화" 동그라미처럼 안이 회색인 것은
    채도로 거른다.
    """
    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 70) & (hsv[..., 2] > 215)).astype(np.uint8) * 255
    hud = np.zeros_like(white)
    hud[:110, :420] = 1
    hud[960:, :] = 1
    hud[:85, 1420:] = 1  # 레벨/돈 (15층에서 계단 위 출구 문이 (1585, 95)에 있었다)
    hud[:230, 1780:] = 1  # 자동 전투 버튼
    white[hud > 0] = 0
    contours, _ = cv2.findContours(white, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    best: tuple[float, ExitDoor] | None = None
    for c in contours:
        if len(c) < 60:
            continue
        pts = c.reshape(-1, 2)
        keep = (pts[:, 0] > 2) & (pts[:, 1] > 2) & (pts[:, 0] < w_img - 3) & (pts[:, 1] < h_img - 3)
        keep &= hud[np.clip(pts[:, 1] - 3, 0, h_img - 1), pts[:, 0]] == 0  # HUD 경계에 잘린 점도 뺀다
        pts = pts[keep]
        if len(pts) < 50:
            continue
        (cx, cy), (a, b), ang = cv2.fitEllipse(pts.reshape(-1, 1, 2).astype(np.int32))
        big, small = max(a, b), min(a, b)
        if not (75 <= big <= 320 and small / big > 0.5):  # 75 미만은 방 제목 글자(ㅇ) 같은 것
            continue
        if cy > 930 or (cx < 420 and cy < 110) or (cx > 1420 and cy < 85) or (cx > 1780 and cy < 230) or (cx > 1450 and cy > 760):
            continue  # HUD 자리 (아래/오른쪽 아래 스킬 버튼, 왼쪽 위 아이콘, 오른쪽 위 돈)
        if math.hypot(cx - 1261, cy - 676) < 90:
            continue  # "F 대화"/"F 강화" 표시의 흰 동그라미 (늘 이 자리에 뜬다)
        # 캐릭터 발밑의 흰 동그라미 (발 위치에서 체력바 쪽으로 조금 아래)
        if char and math.hypot(cx - char[0], cy - char[1] - HP_BAR_TO_FEET / 2) < 150:
            continue
        poly = cv2.ellipse2Poly((int(cx), int(cy)), (int(a / 2), int(b / 2)), int(ang), 0, 360, 5).reshape(-1, 1, 2)
        sample = pts[::3]
        fit = float(np.mean([abs(cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), True)) < 4 for p in sample]))
        if fit < 0.8:
            continue
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.ellipse(mask, ((cx, cy), (a * 0.85, b * 0.85), ang), 255, -1)
        inside = hsv[mask > 0]
        if len(inside) < 500:
            continue
        colorful = inside[inside[:, 1] > 80]
        if len(colorful) / len(inside) < 0.2:
            continue
        door = ExitDoor((cx, cy), (a, b), float(np.median(colorful[:, 0])))
        if best is None or fit * big > best[0]:
            best = (fit * big, door)
    if best:
        return best[1]
    return _find_color_ring(img, char, hue) if hue is not None else None


def _hud_or_prompt(cx: float, cy: float, char: tuple[float, float] | None) -> bool:
    if cy > 930 or (cx < 420 and cy < 110) or (cx > 1420 and cy < 85) or (cx > 1780 and cy < 230)             or (cx > 1450 and cy > 760):
        return True
    if math.hypot(cx - 1261, cy - 676) < 90:
        return True
    return bool(char and math.hypot(cx - char[0], cy - char[1] - HP_BAR_TO_FEET / 2) < 150)


def _find_color_ring(img: np.ndarray, char: tuple[float, float] | None, hue: float, tol: int = 10) -> ExitDoor | None:
    """문양 고리가 흰색이 아니라 방 종류 색일 때 (15층 선택의 방 문: 하늘색 고리, 레벨 표시에 반쯤 가림).
    다음 방 색(hue)의 밝은 픽셀로 고리 모양 타원을 찾는다. 둘레를 고르게 덮어야(cover) 고리로 본다."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dh = np.abs(((hsv[..., 0].astype(np.int16) - int(hue) + 90) % 180) - 90)
    m = ((dh <= tol) & (hsv[..., 1] > 60) & (hsv[..., 2] > 170)).astype(np.uint8) * 255
    m[:110, :420] = 0
    m[960:, :] = 0
    m[:85, 1420:] = 0
    m[:230, 1780:] = 0
    m[760:, 1450:] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best: tuple[float, ExitDoor] | None = None
    h_img, w_img = img.shape[:2]
    for c in contours:
        if len(c) < 80:
            continue
        pts = c.reshape(-1, 2)
        keep = (pts[:, 0] > 2) & (pts[:, 1] > 2) & (pts[:, 0] < w_img - 3) & (pts[:, 1] < h_img - 3)
        keep &= ~((pts[:, 1] < 88) & (pts[:, 0] > 1417))  # 레벨/돈 표시에 잘린 가장자리
        pts = pts[keep]
        if len(pts) < 60:
            continue
        (cx, cy), (a, b), ang = cv2.fitEllipse(pts.reshape(-1, 1, 2).astype(np.int32))
        big, small = max(a, b), min(a, b)
        if not (75 <= big <= 320 and small / big > 0.55) or _hud_or_prompt(cx, cy, char):
            continue
        poly = cv2.ellipse2Poly((int(cx), int(cy)), (int(a / 2), int(b / 2)), int(ang), 0, 360, 5).reshape(-1, 1, 2)
        fit = float(np.mean([abs(cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), True)) < 5 for p in pts[::3]]))
        cover = len(set((np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) // 30).astype(int).tolist())) / 12
        if fit < 0.85 or cover < 0.6:
            continue
        if best is None or fit * big > best[0]:
            best = (fit * big, ExitDoor((cx, cy), (a, b), float(hue)))
    return best[1] if best else None


# 출구 문양 안쪽 색 (OpenCV 색상값 0~180, 가운데와 허용 폭). 다음 방 종류마다 다르다.
# 전투/선택/리더는 실행 로그 스샷에서 잰 값 (21, 89~90, 7). 강적/거래는 아직 못 재서 넓게 둔다.
DOOR_HUE = {"전투": (21, 10), "선택": (90, 12), "리더": (6, 10), "강적": (12, 14), "거래": (70, 30)}


def door_matches(door: ExitDoor, next_room: str) -> bool:
    """문양 색이 다음 방 종류와 맞는지. 다음 방을 모르면 True."""
    if next_room not in DOOR_HUE:
        return True
    mid, tol = DOOR_HUE[next_room]
    return abs((door.hue - mid + 90) % 180 - 90) <= tol


def exit_target(img: np.ndarray, char: tuple[float, float]) -> tuple[float, float] | None:
    """출구로 가려면 향할 화면 좌표. 문양이 보이면 문양, 아니면 가장자리 출구 표시 쪽. 둘 다 없으면 None."""
    door = find_exit_door(img, char)
    if door:
        return door.center
    marker = find_exit_marker(img, char)
    return marker_target(marker, char) if marker else None


PROMPT_THRESHOLD = 0.75
PROMPT_AREA = (1050, 480, 1450, 860)  # 상호작용 표시는 캐릭터 오른쪽 아래 같은 자리(약 1260, 680)에 뜬다
_PROMPTS: dict[str, np.ndarray] = {}


def find_prompt(img: np.ndarray) -> str | None:
    """상호작용 표시 종류: "talk"(F 대화, NPC/상점) 또는 "enhance"(F 강화, 강화머신). 없으면 None."""
    if not _PROMPTS:
        from .screen import TEMPLATE_DIR

        for kind, name in (("talk", "prompt_talk"), ("enhance", "prompt_enhance")):
            _PROMPTS[kind] = cv2.imdecode(np.fromfile(str(TEMPLATE_DIR / f"{name}.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
    x0, y0, x1, y1 = PROMPT_AREA
    gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    best, best_score = None, PROMPT_THRESHOLD
    for kind, t in _PROMPTS.items():
        score = float(cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED).max())
        if score >= best_score:
            best, best_score = kind, score
    return best


def find_talk_prompt(img: np.ndarray) -> tuple[float, float] | None:
    """NPC 근처에 뜨는 "F 대화" 표시가 있으면 그 자리. 없으면 None."""
    return (1261.0, 682.0) if find_prompt(img) == "talk" else None


class TemplateTracker:
    """처음 화면에서 잘라낸 조각을 다음 화면들에서 계속 찾는다 (카메라가 움직여도).

    가까이 갈수록 모양이 커지고 각도가 바뀌어서, 찾을 때마다 지금 화면에서 기준 모양을
    새로 잘라 둔다 (refresh). 그래야 조금씩 변하는 모양을 계속 따라간다.
    """

    def __init__(self, img: np.ndarray, box: tuple[int, int, int, int], threshold: float = 0.6,
                 refresh: bool = True):
        x0, y0, x1, y1 = box
        self.template = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        self.center = ((x0 + x1) / 2, (y0 + y1) / 2)
        self.threshold = threshold
        self.refresh = refresh
        self.score = 1.0

    def update(self, img: np.ndarray, radius: int = 400) -> tuple[float, float] | None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, tw = self.template.shape
        cx, cy = self.center
        x0 = int(max(0, cx - tw / 2 - radius))
        y0 = int(max(0, cy - th / 2 - radius))
        x1 = int(min(gray.shape[1], cx + tw / 2 + radius))
        y1 = int(min(gray.shape[0], cy + th / 2 + radius))
        if x1 - x0 < tw or y1 - y0 < th:
            return None
        res = cv2.matchTemplate(gray[y0:y1, x0:x1], self.template, cv2.TM_CCOEFF_NORMED)
        _, self.score, _, loc = cv2.minMaxLoc(res)
        if self.score < self.threshold:
            return None
        nx, ny = x0 + loc[0], y0 + loc[1]
        self.center = (nx + tw / 2, ny + th / 2)
        if self.refresh:
            self.template = gray[ny:ny + th, nx:nx + tw].copy()
        return self.center


# 이 걸음 수 동안 목표에 이만큼도 못 가까워지면 막힌 것으로 본다.
# (캐릭터나 배경 움직임으로 판단하면 카메라 흔들림, 이펙트, 돌아다니는 NPC에 속는다)
MAX_MISSES = 4  # 목표를 이만큼 연속으로 못 찾으면 포기
STUCK_STEPS = 4
STUCK_PROGRESS_PX = 20


@dataclass
class WalkResult:
    reason: str  # "arrived", "stopped", "lost", "stuck", "steps"
    steps: int


def walk_toward(
    hold: Callable[[tuple[str, ...], float], None],
    grab: Callable[[], np.ndarray],
    target: Callable[[np.ndarray, tuple[float, float]], tuple[float, float] | None],
    *,
    stop: Callable[[np.ndarray], bool] = lambda img: False,
    arrive_dist: float = 90,
    step_sec: float = 0.4,
    max_steps: int = 40,
    log: Callable[[str], None] = lambda s: None,
    initial_angle: float | None = None,
) -> WalkResult:
    """target(img, char)이 돌려주는 화면 좌표 쪽으로 걷는다.

    target은 목표의 화면 좌표(또는 가야 할 방향의 한 점)를 돌려준다. stop(img)이 참이면 멈춘다
    (예: 로딩 화면, "F 대화"). 목표에 가까워지지 않으면 옆으로 비켜 간다. 좌우를 번갈아
    가며 점점 길게.
    """
    dists: list[float] = []
    detours = 0
    last_angle: float | None = initial_angle  # 시작하자마자 목표를 놓쳐도 이 방향으로 간다
    misses = 0
    for step in range(max_steps):
        img = grab()
        if stop(img):
            return WalkResult("stopped", step)
        char = find_character(img)
        if char is None:
            log(f"{step}: 캐릭터를 못 찾음")
            return WalkResult("lost", step)
        tgt = target(img, char)
        if tgt is None:
            # 문양이 잠깐 안 보이거나 화면에 반쯤 걸쳐 있을 수 있다. 가던 방향으로 몇 걸음 더 간다
            if last_angle is None or misses >= MAX_MISSES:
                log(f"{step}: 목표를 못 찾음")
                return WalkResult("lost", step)
            misses += 1
            log(f"{step}: 목표가 안 보임, 가던 방향({last_angle:+.0f}도)으로 계속")
            hold(keys_for_angle(last_angle), step_sec)
            continue
        misses = 0
        dx, dy = tgt[0] - char[0], tgt[1] - char[1]
        dist = math.hypot(dx, dy)
        if dist < arrive_dist:
            return WalkResult("arrived", step)
        ang = angle_of(dx, dy)
        last_angle = ang
        dists.append(dist)
        if len(dists) > STUCK_STEPS and dists[-STUCK_STEPS - 1] - min(dists[-STUCK_STEPS:]) < STUCK_PROGRESS_PX:
            detours += 1
            side = 90 if detours % 2 else -90
            secs = step_sec * (1 + (detours + 1) // 2)
            log(f"{step}: 막힘 -> {ang + side:+.0f}도로 {secs:.1f}초 비켜 감")
            hold(keys_for_angle(ang + side), secs)
            hold(keys_for_angle(ang), step_sec)
            dists.clear()
            continue
        keys = keys_for_angle(ang)
        log(f"{step}: 캐릭터 ({char[0]:.0f},{char[1]:.0f}) 목표 ({tgt[0]:.0f},{tgt[1]:.0f}) 거리 {dist:.0f} {ang:+.0f}도 -> {'+'.join(keys)}")
        hold(keys, step_sec)
        time.sleep(0.05)
    return WalkResult("steps", max_steps)
