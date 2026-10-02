"""지금 게임이 어떤 화면인지 알아보기.

화면마다 잘 안 변하는 UI 조각(기준 조각)을 정해 두고, 캡처한 화면의 그 자리에서
템플릿 매칭으로 찾는다. 좌표는 전부 게임 화면 1920x1080 기준.

기준 조각 이미지는 data/templates/<이름>.png 에 있고, tools/make_templates.py 로 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "data" / "templates"
MARGIN = 16  # 기준 조각을 찾을 때 원래 자리에서 상하좌우로 이만큼 더 본다
POPUP_AREA = (200, 80, 1400, 760)  # 팝업은 크기마다 뜨는 자리가 달라서 넓게 찾는다


@dataclass(frozen=True)
class Anchor:
    name: str
    box: tuple[int, int, int, int]  # (x0, y0, x1, y1) 템플릿을 자른 자리
    group: str | None = None  # 같은 자리에 비슷하게 생긴 것끼리 묶음. 묶음 안에선 1등만 인정
    threshold: float = 0.8
    search: tuple[int, int, int, int] | None = None  # None이면 box 주변 MARGIN 만큼

    def search_box(self) -> tuple[int, int, int, int]:
        if self.search:
            return self.search
        x0, y0, x1, y1 = self.box
        return max(0, x0 - MARGIN), max(0, y0 - MARGIN), min(1920, x1 + MARGIN), min(1080, y1 + MARGIN)


# 오른쪽 아래 키 안내 줄은 반투명이라 뒤 배경에 따라 점수가 0.75까지 떨어진다 (다른 화면은 0.45 이하)
HINT_THRESHOLD = 0.62

ANCHORS: tuple[Anchor, ...] = (
    # 왼쪽 위 아이콘 묶음
    Anchor("toolbar_field", (48, 22, 384, 98), group="toolbar"),  # 필드: 아이콘 3개
    Anchor("toolbar_back_bag", (44, 26, 266, 92), group="toolbar"),  # NPC, 상점: 뒤로 + 가방
    Anchor("toolbar_bag", (48, 22, 162, 92), group="toolbar"),  # 카드 선택, 소리 획득: 가방만
    # 오른쪽 아래 키 안내
    Anchor("hint_card", (1262, 1024, 1904, 1060), group="hint", threshold=HINT_THRESHOLD),
    Anchor("hint_npc", (1494, 1024, 1882, 1060), group="hint", threshold=HINT_THRESHOLD),
    Anchor("hint_dialog", (1754, 1024, 1882, 1060), group="hint", threshold=HINT_THRESHOLD),
    # 상점은 리롤을 다 쓰면 왼쪽 "Q 새로고침"이 사라져서 오른쪽 "Esc 돌아가기 Space 보기"만 본다
    Anchor("hint_shop", (1600, 1024, 1882, 1060), group="hint", threshold=HINT_THRESHOLD),
    # 왼쪽 위 제목줄 (뒤로 버튼 + 위치 아이콘 + 제목)
    Anchor("header_record_detail", (180, 36, 500, 92), group="header"),
    Anchor("header_record_manage", (180, 36, 500, 92), group="header"),
    Anchor("header_tower", (180, 36, 540, 92), group="header"),
    Anchor("header_bag", (180, 36, 540, 92), group="header"),
    Anchor("header_team", (180, 36, 500, 92), group="header"),  # 탑 입장 전 "팀 편성"
    Anchor("header_record_combo", (180, 36, 500, 92), group="header"),  # 탑 입장 전 "레코드 조합"
    # 팝업 제목 (위치 아이콘 + 글자)
    Anchor("popup_notice", (488, 200, 610, 246), group="popup", threshold=0.85, search=POPUP_AREA),
    Anchor("popup_buy", (488, 266, 612, 306), group="popup", threshold=0.85, search=POPUP_AREA),
    Anchor("popup_filter", (312, 156, 456, 202), group="popup", threshold=0.85, search=POPUP_AREA),
    # 그 밖에 화면마다 하나뿐인 것
    Anchor("esc_giveup", (44, 934, 312, 1008)),  # ESC 지도의 빨간 "포기" 버튼
    Anchor("notes_banner", (760, 40, 1180, 100), group="banner"),  # "소리 획득!"
    Anchor("ensemble_banner", (760, 40, 1180, 100), group="banner"),  # "협주 스킬 활성화!"
    Anchor("explore_done", (740, 260, 1180, 380)),  # "탐색 완료"
    # "빈 곳을 터치하여 계속하세요". 소리 획득, 탐색 완료 등에서 위아래로 조금씩 다른 자리에 뜬다
    Anchor("touch_continue", (766, 960, 1154, 998), search=(700, 820, 1220, 1040)),
    Anchor("btn_save_record", (1540, 936, 1872, 1026)),  # 기록 화면 "기록 저장"
    Anchor("enhance_banner", (600, 40, 1320, 100)),  # "잠재력 카드 1장을 선택해 강화하세요!"
    # "60 소모, 잠재력 강화 성공"에서 숫자 뒷부분. 숫자 자릿수에 따라 좌우로 밀린다
    Anchor("enhance_done", (860, 198, 1146, 240), search=(760, 186, 1260, 252)),
)
ANCHOR_BY_NAME = {a.name: a for a in ANCHORS}

# (화면 이름, 필요한 기준 조각들). 위에서부터 먼저 검사한다. 팝업은 다른 화면 위에 뜨니까 맨 위.
STATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("notice", ("popup_notice",)),  # "안내" 팝업 (포기 확인, 분해 확인 등)
    ("shop_buy", ("popup_buy",)),  # 상점 구매 팝업
    ("filter", ("popup_filter",)),  # 기록 관리 필터 팝업
    ("esc_map", ("esc_giveup",)),
    # "빈 곳을 터치하여 계속하세요" 화면들. 뒤에 어둡게 깔린 키 안내보다 먼저 본다
    ("notes_gain", ("notes_banner",)),
    ("ensemble_up", ("ensemble_banner",)),
    ("explore_done", ("explore_done",)),
    ("tap_continue", ("touch_continue",)),  # 위 배너가 아직 안 그려졌을 때 등
    ("enhance_select", ("hint_card", "enhance_banner")),  # 강화머신 카드 고르기
    ("enhance_select", ("hint_card", "enhance_done")),  # 화면이 막 열려서 제목이 흐릴 때
    ("card_select", ("hint_card",)),  # 레벨업 잠재력 카드 고르기
    ("npc_choice", ("hint_npc",)),
    ("shop", ("hint_shop",)),
    ("dialog", ("hint_dialog",)),  # "Space 다음" 대화
    ("record_result", ("btn_save_record",)),  # 탑이 끝난 뒤 기록 화면
    ("record_detail", ("header_record_detail",)),
    ("record_manage", ("header_record_manage",)),
    ("difficulty_select", ("header_tower",)),
    ("bag", ("header_bag",)),
    ("team_setup", ("header_team",)),
    ("record_combo", ("header_record_combo",)),
    ("field", ("toolbar_field",)),  # 전투/이동 중
)


@dataclass
class Detection:
    state: str  # STATES의 이름, "loading", "unknown"
    present: set[str] = field(default_factory=set)  # 인정된 기준 조각
    scores: dict[str, float] = field(default_factory=dict)


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def is_loading(gray: np.ndarray) -> bool:
    """검은 로딩 화면, 또는 위쪽이 검은 띠인 탑 입장 연출."""
    top = gray[:60]
    return gray.mean() < 25 or (top.mean() < 50 and top.std() < 12)


class ScreenDetector:
    def __init__(self, template_dir: Path = TEMPLATE_DIR):
        self.templates: dict[str, np.ndarray] = {}
        self._half: dict[str, np.ndarray] = {}  # 넓은 영역용 반 크기 조각
        for a in ANCHORS:
            path = template_dir / f"{a.name}.png"
            if path.exists():
                self.templates[a.name] = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)

    def scores(self, img: np.ndarray) -> dict[str, float]:
        gray = _gray(img)
        if gray.shape[:2] != (1080, 1920):
            raise ValueError(f"게임 화면은 1920x1080이어야 함 (지금 {gray.shape[1]}x{gray.shape[0]})")
        out = {}
        for name, tmpl in self.templates.items():
            x0, y0, x1, y1 = ANCHOR_BY_NAME[name].search_box()
            out[name] = self._match(gray[y0:y1, x0:x1], name, tmpl)
        return out

    def _match(self, region: np.ndarray, name: str, tmpl: np.ndarray) -> float:
        """기준 조각 점수. 찾을 영역이 넓으면(팝업) 반으로 줄여 자리를 찾고 원본 크기로 다시 확인한다.
        팝업 3개가 화면 판정 시간의 2/3를 먹어서 (영역 1200x680 × 3번) 두 단계로 나눴다 (56ms -> 22ms)."""
        if region.size < 200_000:
            return float(cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED).max())
        half = self._half.get(name)
        if half is None:
            half = self._half[name] = cv2.resize(tmpl, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        small = cv2.resize(region, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        _, _, _, loc = cv2.minMaxLoc(cv2.matchTemplate(small, half, cv2.TM_CCOEFF_NORMED))
        # 반 크기에서 찾은 자리 주변만 원본 크기로 다시 (오차 ±16px 여유)
        x, y = loc[0] * 2, loc[1] * 2
        th, tw = tmpl.shape
        y0, y1 = max(0, y - 16), min(region.shape[0], y + th + 16)
        x0, x1 = max(0, x - 16), min(region.shape[1], x + tw + 16)
        if y1 - y0 < th or x1 - x0 < tw:
            return float(cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED).max())
        return float(cv2.matchTemplate(region[y0:y1, x0:x1], tmpl, cv2.TM_CCOEFF_NORMED).max())

    def detect(self, img: np.ndarray) -> Detection:
        gray = _gray(img)
        if is_loading(gray):
            return Detection("loading")
        scores = self.scores(gray)
        present = {n for n, s in scores.items() if s >= ANCHOR_BY_NAME[n].threshold}
        # 묶음 안에서는 점수 1등만 남긴다
        best: dict[str, str] = {}
        for n in present:
            g = ANCHOR_BY_NAME[n].group
            if g and (g not in best or scores[n] > scores[best[g]]):
                best[g] = n
        present = {n for n in present if not ANCHOR_BY_NAME[n].group or best[ANCHOR_BY_NAME[n].group] == n}
        for state, required in STATES:
            if all(r in present for r in required):
                return Detection(state, present, scores)
        return Detection("unknown", present, scores)


class StableDetector:
    """같은 화면이 연속으로 `frames`번 나와야 인정한다.

    팝업이 커지는 중이거나 화면이 바뀌는 중인 프레임은 엉뚱하게 판정될 수 있다
    (예: 구매 팝업이 열리는 중에 "대화"로 보고 Space를 누르면 구매가 눌린다).
    """

    def __init__(self, detector: ScreenDetector | None = None, frames: int = 3):
        self.detector = detector or ScreenDetector()
        self.frames = frames
        self._last: str | None = None
        self._count = 0
        self.last_raw = ""  # 바로 직전 프레임 하나의 판정 (로딩처럼 짧게 지나가는 화면을 놓치지 않으려고)

    def update(self, img: np.ndarray) -> Detection:
        d = self.detector.detect(img)
        self.last_raw = d.state
        self._count = self._count + 1 if d.state == self._last else 1
        self._last = d.state
        if self._count < self.frames:
            return Detection("transition", d.present, d.scores)
        return d


def main(argv: list[str] | None = None) -> int:
    """켜져 있는 게임을 보면서 화면이 바뀔 때마다 무슨 화면인지 출력한다."""
    import argparse
    import sys
    import time

    from .capture import capture, find_game_window, restore

    ap = argparse.ArgumentParser(prog="python -m stella_auto.screen", description=main.__doc__)
    ap.add_argument("--interval", type=float, default=0.3, help="몇 초마다 볼지")
    ap.add_argument("--seconds", type=float, default=0, help="이만큼 보고 끝내기 (0이면 Ctrl+C까지)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    stable = StableDetector()
    last = None
    end = time.monotonic() + args.seconds if args.seconds else None
    try:
        while end is None or time.monotonic() < end:
            win = find_game_window()
            if win is None:
                print("게임 창을 못 찾음")
                return 1
            if win.minimized:
                win = restore(win)
            d = stable.update(capture(win))
            if d.state not in ("transition", last):
                top = ", ".join(f"{n} {s:.2f}" for n, s in sorted(d.scores.items(), key=lambda kv: -kv[1])[:2])
                print(f"{time.strftime('%H:%M:%S')}  {d.state:18s} ({top})")
                last = d.state
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
