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
) -> WalkResult:
    """target(img, char)이 돌려주는 화면 좌표 쪽으로 걷는다.

    target은 목표의 화면 좌표(또는 가야 할 방향의 한 점)를 돌려준다. stop(img)이 참이면 멈춘다
    (예: 로딩 화면, "F 대화"). 목표에 가까워지지 않으면 옆으로 비켜 간다. 좌우를 번갈아
    가며 점점 길게.
    """
    dists: list[float] = []
    detours = 0
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
            log(f"{step}: 목표를 못 찾음")
            return WalkResult("lost", step)
        dx, dy = tgt[0] - char[0], tgt[1] - char[1]
        dist = math.hypot(dx, dy)
        if dist < arrive_dist:
            return WalkResult("arrived", step)
        ang = angle_of(dx, dy)
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
