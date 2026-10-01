"""캐릭터가 방 안 어디쯤 있는지 재기.

카메라가 캐릭터를 따라다니므로, 화면 배경이 밀린 만큼 캐릭터가 움직인 것이다. 앞뒤 화면을
위상 상관(phase correlation)으로 비교해 배경이 얼마나 밀렸는지 잰다. HUD와 캐릭터는 화면에
붙어 있어서 가린다.

좌표("월드 좌표")는 방에 들어왔을 때 화면의 픽셀 좌표를 그대로 쓴다. 화면 점 p의 월드 좌표는
cam + p. 카메라는 비스듬히 내려다봐서 위아래가 좌우보다 짧게 보이지만, 길 찾기에는 화면 픽셀
그대로 써도 된다.

실험 (13층 리더의 방): W+D 1.5초에 약 (+460, -370)px, 멈춰 있을 때 응답 1.0, 걸을 때 0.2~0.6.
방향을 반대로 틀면 0.3~0.4초쯤 제자리에서 돈다. 300px 넘게 떨어진 화면끼리는 높이가 다른 물건들이
따로 밀려서(원근) 잘 안 맞는다. 그래서 걸을 때는 자주(초당 5번 이상) 잰다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

SCALE = 4  # 1/4로 줄여서 비교 (1920x1080 -> 480x270)
SMALL = (1920 // SCALE, 1080 // SCALE)
MIN_RESPONSE = 0.08  # 이보다 낮으면 비교 실패로 본다 (전투 이펙트, 화면 전환). 걸을 때 0.2~0.6, 엉뚱한 봉우리 0.02~0.05
MAX_STEP = 420  # 한 번에 이보다 많이 밀렸다고 나오면 믿지 않는다 (월드 px)
KEYFRAME_EVERY = 260  # 이만큼 움직일 때마다 기준 화면을 하나씩 남긴다

# 화면에 붙어 있는 것들 (HUD, 방 제목은 따로). (x0, y0, x1, y1)
FIXED_BOXES = (
    (0, 0, 420, 200),  # 왼쪽 위 아이콘, 기록 점수
    (1350, 0, 1920, 230),  # 오른쪽 위 레벨, 돈, 자동 전투
    (1450, 800, 1920, 1080),  # 오른쪽 아래 스킬
    (780, 930, 1150, 1080),  # "자동 전투 중..."
    (830, 380, 1090, 660),  # 화면 가운데 캐릭터
)
TITLE_BOX = (600, 40, 1320, 230)  # 방 제목 (들어온 직후 몇 초)

_WINDOW = np.outer(np.hanning(SMALL[1]), np.hanning(SMALL[0])).astype(np.float32)


def small_gray(img: np.ndarray) -> np.ndarray:
    """비교용 작은 흑백 화면. 화면에 붙은 것들(HUD, 캐릭터, 방 제목 자리)은 평균 밝기로 칠한다."""
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), SMALL, interpolation=cv2.INTER_AREA).astype(np.float32)
    mask = np.zeros(g.shape, bool)
    for x0, y0, x1, y1 in FIXED_BOXES + (TITLE_BOX,):
        mask[y0 // SCALE:y1 // SCALE, x0 // SCALE:x1 // SCALE] = True
    g[mask] = float(g[~mask].mean())
    return g


def shift_between(a: np.ndarray, b: np.ndarray) -> tuple[tuple[float, float], float]:
    """a에서 b로 배경이 밀린 양 (월드 px)과 응답(믿을 만한 정도, 0~1).

    cv2.phaseCorrelate는 OpenCV 5.0.0에서 같은 입력에도 부를 때마다 결과가 달라서(처음 한 번은
    엉뚱한 값) numpy FFT로 직접 계산한다.
    """
    h, w = a.shape
    fa = np.fft.rfft2((a - a.mean()) * _WINDOW)
    fb = np.fft.rfft2((b - b.mean()) * _WINDOW)
    r = np.conj(fa) * fb
    r /= np.abs(r) + 1e-9
    corr = np.fft.irfft2(r, s=(h, w))
    # 화면에 붙은 것(캐릭터 둘레 큰 원, 이펙트)은 정확히 0에서 뾰족한 봉우리를 만들고, 진짜 움직임은
    # 원근 때문에 조금 퍼진 봉우리가 된다. 그래서 3x3 합이 가장 큰 곳을 고른다.
    box = sum(np.roll(corr, (i, j), axis=(0, 1)) for i in (-1, 0, 1) for j in (-1, 0, 1))
    py, px = np.unravel_index(int(np.argmax(box)), box.shape)
    ys = [(py + k) % h for k in (-1, 0, 1)]
    xs = [(px + k) % w for k in (-1, 0, 1)]
    patch = corr[np.ix_(ys, xs)]
    resp = float(patch.sum())
    wsum = float(np.clip(patch, 0, None).sum()) or 1.0
    oy = float((np.clip(patch, 0, None).sum(axis=1) * (-1, 0, 1)).sum() / wsum)
    ox = float((np.clip(patch, 0, None).sum(axis=0) * (-1, 0, 1)).sum() / wsum)
    dy = (py if py < h / 2 else py - h) + oy
    dx = (px if px < w / 2 else px - w) + ox
    return (dx * SCALE, dy * SCALE), resp


class Odometry:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.cam = np.zeros(2)  # 지금 화면 왼쪽 위의 월드 좌표
        self.prev: np.ndarray | None = None
        self.keyframes: list[tuple[np.ndarray, np.ndarray]] = []  # (작은 화면, cam)
        self.failures = 0  # 연속으로 비교에 실패한 횟수
        self.last_response = 1.0

    def update(self, img: np.ndarray) -> bool:
        """새 화면으로 카메라 위치를 갱신. 비교에 실패하면 False (위치는 그대로)."""
        cur = small_gray(img)
        if self.prev is None:
            self.prev = cur
            self._maybe_keyframe(cur)
            return True
        (dx, dy), resp = shift_between(self.prev, cur)
        self.last_response = resp
        if resp >= MIN_RESPONSE and math.hypot(dx, dy) <= MAX_STEP:
            self.cam -= (dx, dy)
            self.prev = cur
            self.failures = 0
            self._maybe_keyframe(cur)
            return True
        # 많이 움직였거나 이펙트가 덮었다: 남겨 둔 기준 화면들과 비교해서 위치를 되찾는다
        if self._relocalize(cur):
            self.prev = cur
            self.failures = 0
            return True
        self.failures += 1
        if self.failures >= 3:
            self.prev = cur  # 계속 실패하면 지금 화면부터 다시 이어 간다 (위치는 조금 틀어진다)
        return False

    def _maybe_keyframe(self, cur: np.ndarray) -> None:
        if all(np.hypot(*(self.cam - c)) > KEYFRAME_EVERY for _, c in self.keyframes):
            self.keyframes.append((cur, self.cam.copy()))

    def _relocalize(self, cur: np.ndarray) -> bool:
        near = sorted(self.keyframes, key=lambda k: np.hypot(*(k[1] - self.cam)))[:4]
        best = None
        for kf, kcam in near:
            (dx, dy), resp = shift_between(kf, cur)
            if resp >= MIN_RESPONSE * 1.5 and math.hypot(dx, dy) <= 900 and (best is None or resp > best[0]):
                best = (resp, kcam - (dx, dy))
        if best is None:
            return False
        self.cam = best[1]
        return True

    def shift_origin(self, dx: float, dy: float) -> None:
        """좌표 기준을 옮긴다 (모든 월드 좌표에 (dx, dy)를 더한 것과 같다)."""
        self.cam += (dx, dy)
        self.keyframes = [(k, c + (dx, dy)) for k, c in self.keyframes]

    def to_world(self, p: tuple[float, float]) -> tuple[float, float]:
        return float(self.cam[0] + p[0]), float(self.cam[1] + p[1])

    def to_screen(self, w: tuple[float, float]) -> tuple[float, float]:
        return float(w[0] - self.cam[0]), float(w[1] - self.cam[1])
