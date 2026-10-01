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

# (화면 각도, 키) 화면 기준: 오른쪽 0도, 위쪽 90도
DIRECTIONS: tuple[tuple[float, tuple[str, ...]], ...] = (
    (0, ("d",)), (45, ("w", "d")), (90, ("w",)), (135, ("w", "a")),
    (180, ("a",)), (-135, ("s", "a")), (-90, ("s",)), (-45, ("s", "d")),
)


def angle_of(dx: float, dy: float) -> float:
    """화면 벡터의 각도 (오른쪽 0도, 위쪽 90도)."""
    return math.degrees(math.atan2(-dy, dx))


def keys_for_angle(angle: float) -> tuple[str, ...]:
    def diff(a: float) -> float:
        return abs((angle - a + 180) % 360 - 180)

    return min(DIRECTIONS, key=lambda d: diff(d[0]))[1]


def find_character(img: np.ndarray) -> tuple[float, float] | None:
    """캐릭터 발 위치 (화면 좌표). 체력바(초록 칸 3개 이상이 한 줄)를 못 찾으면 None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 0] > 35) & (hsv[..., 0] < 85) & (hsv[..., 1] > 120) & (hsv[..., 2] > 150)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    segs = [st[i] for i in range(1, n) if 15 <= st[i][2] <= 40 and 7 <= st[i][3] <= 15]
    best: list | None = None
    for s in segs:
        row = [t for t in segs if abs(t[1] - s[1]) <= 3 and abs(t[0] - s[0]) <= 200]
        if len(row) >= 3 and (best is None or len(row) > len(best)):
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


def find_exit_marker(img: np.ndarray, char: tuple[float, float] | None) -> ExitMarker | None:
    """화면 가장자리의 출구 표시 (청록색 문 아이콘 + 삼각형)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 0] > 80) & (hsv[..., 0] < 100) & (hsv[..., 1] > 150) & (hsv[..., 2] > 150)).astype(np.uint8)
    mask[:110, :420] = 0  # 왼쪽 위 HUD
    mask[960:, :] = 0  # 아래 HUD
    mask[:200, 1500:] = 0  # 오른쪽 위 HUD
    n, _, st, cen = cv2.connectedComponentsWithStats(mask)
    blobs = [(int(st[i][4]), cen[i], st[i]) for i in range(1, n) if st[i][4] > 60]
    if char:  # 캐릭터 발밑의 방향 화살표도 청록색이라 뺀다
        blobs = [b for b in blobs if math.hypot(b[1][0] - char[0], b[1][1] - char[1] - HP_BAR_TO_FEET) > 130]
    icons = [b for b in blobs if 25 <= b[2][3] <= 60 and 20 <= b[2][2] <= 50]
    if not icons:
        return None
    icon = max(icons, key=lambda b: b[0])
    ix, iy, iw, ih = icon[2][:4]

    def inside_icon(c: np.ndarray) -> bool:  # 문 아이콘은 여러 조각으로 쪼개져 잡힌다
        return ix - 2 <= c[0] <= ix + iw + 2 and iy - 2 <= c[1] <= iy + ih + 2

    near = [b for b in blobs if b is not icon and not inside_icon(b[1]) and math.hypot(*(b[1] - icon[1])) < 60]
    if near:
        tri = min(near, key=lambda b: math.hypot(*(b[1] - icon[1])))
        d = tri[1] - icon[1]
        return ExitMarker((float(icon[1][0]), float(icon[1][1])), (float(d[0]), float(d[1])))
    return ExitMarker((float(icon[1][0]), float(icon[1][1])), None)


@dataclass
class ExitDoor:
    center: tuple[float, float]  # 문 안 문양의 가운데
    size: tuple[float, float]  # 문양 타원의 가로세로
    hue: float  # 문양 고리 색 (OpenCV 색상값 0~180). 다음 방 종류마다 다르다


def find_exit_door(img: np.ndarray, char: tuple[float, float] | None = None) -> ExitDoor | None:
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
    hud[:200, 1500:] = 1
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
    return best[1] if best else None


def exit_target(img: np.ndarray, char: tuple[float, float]) -> tuple[float, float] | None:
    """출구로 가려면 향할 화면 좌표. 문양이 보이면 문양, 아니면 가장자리 출구 표시 쪽. 둘 다 없으면 None."""
    door = find_exit_door(img, char)
    if door:
        return door.center
    marker = find_exit_marker(img, char)
    if marker is None:
        return None
    if marker.direction is not None:  # 삼각형 방향으로 멀리
        dx, dy = marker.direction
        norm = math.hypot(dx, dy) or 1
        return char[0] + dx / norm * 600, char[1] + dy / norm * 600
    return marker.icon


_TALK_TEMPLATE: np.ndarray | None = None
TALK_THRESHOLD = 0.75


def find_talk_prompt(img: np.ndarray) -> tuple[float, float] | None:
    """NPC 근처에 뜨는 "F 대화" 표시의 가운데. 없으면 None."""
    global _TALK_TEMPLATE
    if _TALK_TEMPLATE is None:
        from .screen import TEMPLATE_DIR

        path = TEMPLATE_DIR / "prompt_talk.png"
        _TALK_TEMPLATE = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(img[:960], cv2.COLOR_BGR2GRAY)  # 아래 HUD 제외
    res = cv2.matchTemplate(gray, _TALK_TEMPLATE, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < TALK_THRESHOLD:
        return None
    th, tw = _TALK_TEMPLATE.shape
    return loc[0] + tw / 2, loc[1] + th / 2


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
