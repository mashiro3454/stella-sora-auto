"""별의 탑 봇 본체: 화면을 보고 한 단계씩 행동한다.

    python -m stella_auto.runner presets/바람.json --floors 3

F12를 누르면 바로 멈춘다. 행동과 NPC 선택은 logs/ 에 기록한다.
아직 안 되는 것: 상점(들어가면 그냥 나옴), 강화머신, 사망 처리.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz, process

from . import navigate as nv
from .capture import capture, ensure_client_size, find_game_window, restore
from .cards import CARD_AREA, read_cards, team_pool
from .choices import find_option_boxes, read_choices
from .shop import drink_face, plan_purchases, read_reroll, read_shop
from .notes import NOTE_NAMES, NoteNeeds, read_needs, shop_note_type, type_from_name
from .gamedata import default_gamedata
from . import killswitch
from .input import GameInput, NotFocusedError, release_all_keys
from .ocr import KoreanOcr
from . import record as rc
from .recorder import Recorder
from .score import RecordResult, score_record
from .pathing import Goal, Navigator
from .preset import Preset
from .screen import ScreenDetector, StableDetector
from . import strategy
from .strategy import CardChooser, RunState

ROOT = Path(__file__).resolve().parent.parent
# 한 층씩 모드(--step): 멈춘 봇은 이 파일이 생기면 다음 층까지 간다 (파일은 봇이 지운다)
STEP_GO = ROOT / "logs" / "step.go"
VK_F12 = 0x7B
TAP_STATES = {"notes_gain", "ensemble_up", "tap_continue", "explore_done"}
EMPTY_SPOT = (1300, 1045)  # "빈 곳을 터치" 화면에서 누를 곳. 가운데는 목록, 오른쪽 아래는 기록 화면의 "기록 저장" 자리라 피한다
GAMBLE_LAST_FLOOR = 3  # 650원 도박은 1~3층에서만 나온다
TITLE_BOX = (600, 40, 1320, 230)  # 방에 들어가면 잠깐 뜨는 "선택의 방 / 2/20층"
# "14/20층". OCR이 "/"를 "1"로 읽기도 한다 ("14120층")
TITLE_FLOOR_RE = re.compile(r"(\d{1,2})\s*[/1lI|!\:;?]\s*20\s*층|(\d{1,2})\s*/\s*20")  # OCR이 /를 !?로도 읽는다 (00:26 "5!20층")
ROOM_NAMES = ("전투", "선택", "강적", "거래", "리더")
HUD_TEXT = ("기록점수", "점수", "자동전투", "전투중", "레벨", "레멜", "레텔", "간단히", "대화", "코인", "소리")
NPC_SCAN_EVERY = 2.5  # 초. 이름표 찾기(OCR 전체 화면)는 무거워서 가끔만
TALK_COOLDOWN = 8.0  # 대화가 끝나도 "F 대화"가 한동안 남아 있어서, 말 건 뒤 이만큼은 다시 안 건다
# NPC는 늘 출구 근처에 생긴다 (사용자 설명): 층마다 출구 이만큼 앞에서 멈춰 이름표를 찾고 나서 들어간다
EXIT_FRONT_DIST = 420  # 출구 피하기 거리(280+60)보다 밖: 안이면 출구 옆 NPC로 걸어갈 때 "출구가 가까워서 멈춤"에 걸린다
POST_COMBAT_NPC_WAIT = 2.5  # 전투가 끝나고 NPC가 생기는 데 걸리는 시간
# 알려진 NPC 이름. 오른쪽 위 레벨/돈 자리에 걸친 이름표도 이 이름이면 NPC로 본다
# (18:45 1층: 출구 옆 포셔 이름표가 화면 오른쪽 위 끝에 걸려서 HUD로 알고 버렸다)
KNOWN_NPCS = ("베아트리스", "포셔", "베르주", "베르너")
GAMBLE_EXIT_WAIT = 8.0  # 3층: 출구가 보여도 전투 뒤 NPC가 나올 수 있어서 이만큼 더 둘러본다
CHOICE_SEARCH = 30.0  # 선택의 방에서 NPC를 이만큼 찾아도 없으면 그냥 나간다
COMBAT_TIMEOUT = 150.0  # 전투방에서 "소리 획득"을 이만큼 못 보면 전투 끝을 놓친 것으로 보고 진행
# 층 구성 (사용자 설명): 1~6 전투/선택/전투/강적/거래/리더, 7~13 전투/강적/선택/전투/강적/거래/리더,
# 14~20 같은 구성. 전투가 있는 방은 "소리 획득"(전투 끝 보상)을 본 뒤에 나간다
ROOM_BY_FLOOR = {
    1: "전투", 2: "선택", 3: "전투", 4: "강적", 5: "거래", 6: "리더",
    7: "전투", 8: "강적", 9: "선택", 10: "전투", 11: "강적", 12: "거래", 13: "리더",
    14: "전투", 15: "강적", 16: "선택", 17: "전투", 18: "강적", 19: "거래", 20: "리더",
}
COMBAT_ROOMS = {"전투", "강적", "리더"}
SHOP_INDEX_BY_FLOOR = {5: 1, 12: 2, 19: 3, 20: 4}  # 몇 번째 상점인지 (20층은 보스 뒤 상점)
TRADE_TIMEOUT = 150.0  # 거래의 방에서 강화/상점을 이만큼 못 끝내면 그냥 나간다
ENHANCE_MAX_PRICE = 180  # 강화머신은 180원까지 누른다 (첫 상점 0-60-120-180, 그 뒤 60-120-180)

# 메뉴 버튼 (게임 화면 1920x1080 기준, 실험으로 확인)
BTN_DEPART = (1706, 978)  # 난이도 선택 "출발" (바로 왼쪽 "빠른 전투"는 절대 누르지 않는다)
BTN_NEXT = (1740, 994)  # 팀 편성 "다음", 레코드 조합 "전투 시작"
BTN_TRASH = (672, 982)  # 기록 화면 휴지통(분해)
BTN_CONFIRM = (1170, 805)  # 분해 확인 팝업 "확인"
# 메인 화면 → 별의 탑 (2026-10-02 05:17 새벽 5시 업데이트 뒤 직접 찾아 감)
LOBBY_BUTTONS = {
    "tap": EMPTY_SPOT,  # "아무 곳이나 터치하여 획득하세요" / "빈 곳을 터치하여 계속하세요"
    "attendance_close": (1668, 168),  # 매일 출석 창 X
    "lobby_depart": (1732, 972),  # 메인 화면 오른쪽 아래 "출발"
    "tower_hub": (1167, 520),  # "별의 탑 탐색"
    "enter_tower": (1420, 900),  # 탑 고르기 화면 "별의 탑 들어가기"
}
TOWER_NAME = "불꽃과 먼지"  # 바람 프리셋 탑 (물/바람 속성이 유리한 탑)
BOSS_NAMES = ("광기의요리사",)  # 20층 보스 이름 (화면 위에 떠서 NPC 이름표로 잘못 읽었다)
TOWER_PANEL_NAME = (1060, 700, 1400, 770)  # 탑 고르는 화면 오른쪽 사진 아래 고른 탑 이름
TOWER_TITLE_BOX = (680, 215, 1060, 280)  # 출발(난이도) 화면 정보 칸 맨 위 탑 이름
BTN_BACK = (97, 60)  # 왼쪽 위 뒤로


class Stop(Exception):
    pass


class Bot:
    def __init__(self, preset: Preset, log_dir: Path = ROOT / "logs"):
        self.preset = preset
        self.log_dir = log_dir
        self.gi = GameInput()
        self.ocr = KoreanOcr()
        self.det = ScreenDetector()
        self.stable = StableDetector(self.det, frames=2)
        self.chooser = CardChooser(preset)
        self.pool = team_pool(preset)
        self.run = RunState()
        self.nav = Navigator(self.gi, self.grab, lambda kind, msg: self.log(kind, msg))
        self.idle_since: float | None = None  # 출구를 못 찾기 시작한 때
        self.spots_tried: set = set()  # 이번 층에서 가 본 기억 속 NPC 자리
        self.npc_marker_rest_until = 0.0
        self.char_missing_since: float | None = None
        self.room = ""  # 이 층 방 종류 (전투/선택/강적/거래/리더)
        self.talked: set[str] = set()  # 이 층에서 말 건 NPC 이름
        self.talk_spots: list = []  # 이 층에서 F로 말을 건 자리들
        self.title_seen = 0.0  # 방 제목을 마지막으로 본 때
        self.title_checked = 0.0
        self.npc_scanned = 0.0
        self._card_tries = 0
        self._last_cards: tuple | None = None  # 직전에 읽은 카드 (두 번 연속 같아야 고른다)
        self._card_prev: np.ndarray | None = None  # 카드 멈춤 판정용 직전 화면 조각
        self.card_pool_char: int | None = None  # 방금 산 음료의 얼굴 캐릭터 (이어지는 카드 화면의 풀)
        self._card_wait_since = 0.0
        self._last_choice_bands: tuple | None = None
        self._declined: str | None = None  # ESC로 안 고르고 나간 선택지 질문
        self._quiz_waits = 0
        self.loading_at: float | None = None  # 층 사이 로딩을 본 때 (방 제목을 놓치면 이걸로 층을 센다)
        self.floor_changed_at = 0.0
        self.exit_fails = 0
        self.last_talk_at = 0.0
        self.exit_seen_at: float | None = None  # 이 층에서 출구를 처음 본 때
        self.floor_known = False  # 봇을 탑 중간에서 켜면 처음엔 몇 층인지 모른다
        self.floor_uncertain = False  # 지도에서 층을 못 읽어 짐작으로 둔 상태 (다음 제목을 그대로 믿는다)
        self.combat_done = True  # 이 층 전투가 끝났는지 ("소리 획득"을 봤는지)
        self.combat_done_at: float | None = None  # "소리 획득"을 본 시각
        self.exit_npc_checked = False  # 이 층에서 출구 앞 NPC 확인을 했는지
        self.joined_midway = False  # 봇을 층 중간에 켰는지 (전투 끝을 이미 지나쳤을 수 있다)
        self.enhance_count = 0  # 이 거래의 방에서 강화머신을 누른 횟수
        self.shop_done = False
        self.shop_search_start: float | None = None
        self.shop_plan = None
        self.shop_queue: list = []
        self.shop_rerolled = False
        self.shop_bought: set[tuple[bool, int]] = set()  # (리롤 뒤인지, 칸) 이 상점에서 산 것
        self.shop_potions_left = False  # 마지막으로 본 상점에 안 산 잠재력 음료가 남아 있었는지
        self.shop_reopens = 0  # 강화 우선 캐릭터 잠재를 얻으려고 상점에 다시 간 횟수
        self.trade_fails = 0
        self.shop_click_fails: dict[tuple[bool, int], int] = {}  # 눌렀는데 구매 창이 안 뜬 횟수
        self.shop_peeked = False
        self.note_needs = None  # 가방에서 읽은 협주스킬별 필요한 소리
        self.note_users = self._load_note_users()  # 소리 종류 -> 쓰는 협주스킬 수 (지난번 가방에서 읽은 것)
        self.trade_tick = 0
        self.gamble_won = False
        self.require_gamble = True
        self.step_mode = False  # 650원 이긴 뒤 층마다 멈추고 STEP_GO를 기다린다 (사용자가 한 층씩 보며 고칠 곳을 짚는다)
        self.step_pause_sec = 0.0  # 0이 아니면: 650원 이긴 판에서 출구로 들어가기 전에 이만큼만 멈춘다
        self.exit_paused = False  # 이 층에서 출구 앞 잠깐 멈춤을 했는지
        self.pause_pending = ""
        self.stop_after_tower = False
        self.max_runs = 0  # 이만큼 기록을 저장하면 멈춤 (0이면 계속)
        self.runs_done = 0
        self.restarts = 0
        self.deadline: float | None = None  # 이 시각이 지나면 새 판을 시작하지 않는다
        self.run_tracked = False  # 이 판을 봇이 1층부터 봤는지 (잠재 레벨 기억이 온전한지)
        self.restart_pending = ""
        self._gamble_at = 0.0
        self._gamble_verify: int | None = None  # 결과를 못 읽은 650원: 도박 전 돈 (카드 화면에서 다시 확인)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"run_{stamp}.jsonl"
        self.shot_dir = log_dir / "shots" / stamp
        self.shot_dir.mkdir(parents=True, exist_ok=True)
        self._prune_shots(log_dir / "shots", keep_days=3)
        gd = default_gamedata()
        # NPC 이름표로 착각하면 안 되는 글자: 전투 중 뜨는 스킬 이름, 잠재력 이름
        self.not_npc = [n.replace(" ", "") for n in gd.skill_names] + [p.name.replace(" ", "") for p in gd.potentials.values() if p.name] \
            + list(BOSS_NAMES)
        self.choice_path = log_dir / "choices.jsonl"

    @staticmethod
    def _prune_shots(shots: Path, keep_days: int) -> None:
        """오래된 판단 화면 폴더를 지운다 (디버그용 스냅샷일 뿐인데 하루 수백 MB씩 쌓여 OneDrive가 전부 올린다)."""
        cutoff = time.strftime("%Y%m%d", time.localtime(time.time() - keep_days * 86400))
        for d in shots.glob("20*"):
            if d.is_dir() and d.name[:8] < cutoff:
                for f in d.iterdir():
                    f.unlink(missing_ok=True)
                d.rmdir()

    # -- 기본 ---------------------------------------------------------------
    def check_stop(self) -> None:
        pass  # F12는 killswitch 스레드가 따로 지켜보다가 바로 끈다

    def grab(self) -> np.ndarray:
        self.check_stop()
        win = find_game_window()
        if win is None:
            raise Stop("게임 창이 없음")
        if win.minimized:
            win = restore(win)
        if (win.width, win.height) != (1920, 1080):
            self.log("화면", f"게임 창 크기가 {win.width}x{win.height}라서 1920x1080으로 맞춤")
            win = ensure_client_size(win)
        return capture(win)

    def shot(self, img: np.ndarray, tag: str) -> str:
        """판단한 순간의 화면을 남긴다 (나중에 왜 그렇게 했는지 보려고)."""
        name = f"{time.strftime('%H%M%S')}_{self.run.floor:02d}_{tag}.jpg"
        cv2.imencode(".jpg", cv2.resize(img, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tofile(str(self.shot_dir / name))
        return name

    def log(self, kind: str, msg: str = "", img: np.ndarray | None = None, **data) -> None:
        rec = {"t": time.strftime("%H:%M:%S"), "floor": self.run.floor, "kind": kind, "msg": msg, **data}
        if img is not None:
            rec["shot"] = self.shot(img, kind)
        print(f"[{rec['t']}] {self.run.floor}층 {kind}: {msg}", flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def read_gold_steady(self, tries: int = 3) -> int | None:
        """돈을 몇 번 읽어 가장 큰 값 (OCR이 "330"을 "30"처럼 한 자리 빼먹는 일이 있어서)."""
        vals = []
        for _ in range(tries):
            g = self.read_gold(self.grab())
            if g is not None:
                vals.append(g)
            time.sleep(0.15)
        return max(vals) if vals else None

    def judge_gamble(self) -> tuple[bool | None, str]:
        """650원 도박 결과. 화면 가운데 알림("200개를 잃었습니다", "650개를 획득했습니다")을 먼저 보고,
        없으면 돈이 얼마나 바뀌었는지로. (이김/짐/None=아직 모름, 이유)"""
        before = self._gamble_gold_before
        golds = []
        for _ in range(3):
            im = self.grab()
            banner = self.ocr.text(im, (480, 170, 1440, 280)).replace(" ", "")
            if "잃었" in banner or "잃" in banner and "개" in banner:
                return False, f"알림 '{banner}'"
            if "획득" in banner and "65" in banner:
                return True, f"알림 '{banner}'"
            g = self.read_gold(im)
            if g is not None:
                golds.append(g)
            time.sleep(0.2)
        if golds and before is not None:
            # 이기면 +650, 지면 -200 근처여야 한다. 코인 그림을 숫자로 읽어 480을 "7480"으로 본 적이 있어서
            # (06:03 진 판을 이긴 판으로 셈) 말이 안 되는 변화는 앞자리를 떼어 본다.
            # 맨 앞 1을 4로 읽기도 한다 (07:49 1180 -> "4180"). 도박 순간엔 다른 돈 변화가 없어서 +650/-200에 딱 맞춘다
            def cands(v: int, one: bool = False) -> list[int]:
                s = str(v)
                out = [v] + ([int(s[1:])] if len(s) >= 3 else [])
                out += [int("1" + s[1:])] if one and len(s) >= 4 and s[0] != "1" else []
                out += [int("1" + s)] if one and len(s) == 3 else []  # 맨 앞 1을 빠뜨림 (13:34 1130 -> "130")
                return out + ([v * 10] if one and len(s) <= 3 else [])  # 끝 0을 빠뜨림 (09:17 280 -> "28")

            for b in cands(before):
                for g in sorted(golds, reverse=True):
                    for after in cands(g, one=True):
                        diff = after - b
                        note = f"돈 {b} -> {after}" + (f" (읽은 값 {before} -> {g})" if (b, after) != (before, g) else "")
                        if 600 <= diff <= 700:
                            return True, note
                        if -250 <= diff <= -150:
                            return False, note
        return None, f"아직 모름 (돈 {before} -> {golds})"

    def press_enhance(self, img: np.ndarray) -> None:
        """"F 강화"가 떠 있을 때 강화머신을 한 번 쓴다."""
        price = self.next_enhance_price()
        self.enhance_count += 1
        self.last_talk_at = time.monotonic()
        self.log("강화", f"강화머신 {self.enhance_count}번째 ({price}원)", img)
        if self.run_tracked:
            self.save_state()
        self.nav.remember_spot("enhance")
        self.gi.key("f")
        time.sleep(1.2)

    def verify_gamble(self, gold: int) -> None:
        """650원 결과를 못 읽고 이긴 것으로 이어 간 판: 다음 카드 화면의 돈으로 다시 확인한다.
        (그사이 소리 판매/리롤로 조금 바뀔 수 있어서 넉넉하게 본다. 코인 그림을 숫자로 읽은 앞자리도 떼어 본다)"""
        before = getattr(self, "_gamble_verify", None)
        if before is None:
            return
        self._gamble_verify = None
        diffs = [g - before for g in [gold] + ([int(str(gold)[1:])] if len(str(gold)) >= 3 else [])]
        if any(350 <= d <= 1500 for d in diffs):
            self.log("650원", f"카드 화면 돈으로 다시 확인: 성공 ({before} -> {gold})")
        elif any(-450 <= d <= -100 for d in diffs):
            self.gamble_won = False
            self.run.keep_after_gamble = False
            self.restart_pending = "650원 도박 실패 (카드 화면 돈으로 다시 확인)"
            self.log("650원", f"카드 화면 돈으로 다시 확인: 실패 ({before} -> {gold})")
            if self.run_tracked:
                self.save_state()

    def read_gold(self, img: np.ndarray) -> int | None:
        text = self.ocr.text(img, (1700, 15, 1910, 85)).replace(",", "")
        nums = re.findall(r"\d+", text)
        return int(nums[-1]) if nums else None

    # -- 화면별 행동 -----------------------------------------------------------
    def cards_settled(self, img: np.ndarray) -> bool:
        """카드가 날아 들어오는 중인지: 카드 자리를 직전 화면과 비교한다 (OCR보다 30배 싸다).
        멈춘 화면끼리는 차이 3~4 (카드 반짝임), 움직이는 중엔 10~80 (영상 측정)."""
        x0, y0, x1, y1 = CARD_AREA
        g = cv2.resize(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (204, 100),
                       interpolation=cv2.INTER_AREA).astype(np.int16)
        prev, self._card_prev = self._card_prev, g
        if prev is None:
            self._card_wait_since = time.monotonic()
            return False
        if float(np.abs(g - prev).mean()) < 6.0:
            return True
        # 반짝임이 심해서 3초 넘게 안 멈춘 걸로 보이면, 예전처럼 두 번 읽어 똑같은지로 판단한다
        return time.monotonic() - self._card_wait_since > 3.0

    def on_cards(self, img: np.ndarray, enhance: bool) -> None:
        if not self.cards_settled(img):
            time.sleep(0.15)
            return
        gold = self.read_gold(img)
        if gold is not None:
            self.run.gold = gold
            self.verify_gamble(gold)
        cards = read_cards(img, self.ocr, self.pool)
        # 카드가 다 안 읽혔거나 레벨을 못 읽은 카드가 있으면 다시 읽는다 (그대로 고르면 엉뚱하게 리롤한다)
        unsure = len(cards) < 3 or any(not c.level_known for c in cards)
        if (not cards or unsure) and self._card_tries < 3:
            self._card_tries += 1
            time.sleep(0.4)
            return
        self._card_tries = 0
        if not cards:
            self.log("카드", "카드를 못 읽음, 잠깐 기다림")
            time.sleep(0.5)
            return
        # 멈춘 걸 확인 못 하고 3초가 지나 온 경우만 (반짝임 등): 예전처럼 두 번 읽어 똑같을 때만 고른다
        if time.monotonic() - self._card_wait_since > 3.0:
            sig = tuple((c.slot, c.potential.id if c.potential else c.raw_name, c.level_from, c.level_to, c.bonus)
                        for c in cards)
            if sig != self._last_cards:
                self._last_cards = sig
                time.sleep(0.3)
                return
        self._last_cards = None
        self._card_prev = None  # 행동하고 나면 (카드가 바뀌니) 멈춤 판정을 처음부터
        d = (self.chooser.choose_enhance(cards, self.run) if enhance
             else self.chooser.choose(cards, self.run, pool_char=self.card_pool_char))
        if not self.run_tracked and (d.action == "restart" or (d.action == "reroll" and
                                                               (not self.floor_known or self.run.rerolls_this_pick >= 2))):
            # 중간에 켠 판: 가진 잠재(와 층)를 다 몰라서 재시작 판단을 믿을 수 없다.
            # (20층에서 1층 규칙으로 리롤 5번 하고 판을 포기한 일이 있었다) 리롤은 2번까지만, 그 뒤엔 가장 나은 카드.
            # 층 규칙(Lv1 안 집기 등)을 지킨 점수를 먼저 보고, 다 안 되면 층 제한 없이.
            vals = {c.slot: self.chooser.value(c, self.run)[0] for c in cards}
            if all(v is None for v in vals.values()):
                vals = {c.slot: self.chooser.value(c, self.run, ignore_floor=True)[0] for c in cards}
            best = max(cards, key=lambda c: vals[c.slot] if vals[c.slot] is not None else -1)
            d = type(d)("pick", best, "중간에 켠 판이라 재시작/리롤 대신 가장 나은 카드", vals)
        desc = [f"{c.potential.name if c.potential else c.raw_name}({'새' if c.is_new else c.level_from}>{c.level_to})"
                f"={d.values.get(c.slot)}" for c in cards]
        counts = (f"+2 {self.run.plus2_taken}/{strategy.PLUS2_PER_RUN}" if enhance
                  else f"새Lv3 {self.run.lv3_new_taken}/{strategy.LV3_LIMIT}")
        self.log("카드", f"{'강화 ' if enhance else ''}{d.action} {d.reason} | {', '.join(desc)}"
                       f" | 돈 {self.run.gold} | {counts}", img)
        if self.step_pause_sec:
            time.sleep(1.0)  # 지켜보기 모드: 사용자가 제시된 카드를 볼 틈을 준다
        if d.action == "pick":
            self.gi.click(*d.card.click)
            time.sleep(0.35)
            self.gi.key("space")
            if enhance and any(c.level_from is not None and c.gain >= 2 for c in cards):
                self.run.plus2_taken += 1  # +2 강화 화면 (판에 5번뿐)
            if not enhance:
                self.card_pool_char = None  # 음료 카드 화면이 끝났다
            g = self.chooser.goals.get(d.card.potential.id) if d.card.potential else None
            if (d.card.is_new and d.card.level_to <= 1 and g
                    and (g.mark == "후순위" or (g.mark == "다다익선" and g.target_level < 6))):
                # 후순위/다다(3) 새 Lv1을 예외(필수·다다(6) 완집 or 잠재락)로 집었다: 화면을 남긴다 (사용자 확인용)
                dd = self.log_path.parent / "choice_alerts"
                dd.mkdir(parents=True, exist_ok=True)
                cv2.imencode(".png", img)[1].tofile(str(dd / f"{time.strftime('%m%d_%H%M%S')}_{self.run.floor:02d}_lv1pick.png"))
                self.log("카드", f"(알림) 후순위/다다(3) 새 Lv1 {d.card.potential.name}을 예외로 집음 — 화면 남김")
            self.chooser.record_pick(d.card, self.run)
            if self.run_tracked:
                self.save_state()
        elif d.action == "reroll":
            self.gi.key("q")
            self.chooser.record_reroll(self.run)
        else:  # restart: 카드 화면에선 못 나가니 아무거나 가져가고 필드에서 재시작
            self.restart_pending = d.reason
            self.gi.click(*cards[0].click)
            time.sleep(0.35)
            self.gi.key("space")
        time.sleep(0.4)  # 나머지 기다림은 화면 판정(연속 3번 일치)과 카드 멈춤 판정이 맡는다

    def on_choice(self, img: np.ndarray) -> None:
        # 보기 상자는 미끄러져 들어온다. 상자 위치가 두 번 연속 같을 때 읽는다
        bands = tuple(find_option_boxes(img))
        if not bands or bands != self._last_choice_bands:
            self._last_choice_bands = bands
            time.sleep(0.3)
            return
        self._last_choice_bands = None
        question, options = read_choices(img, self.ocr)
        if not options:
            time.sleep(0.4)
            return
        pairs = [(o.text, o.effect) for o in options]
        if len(pairs) == 1 and getattr(self, "_single_waits", 0) < 4:
            # 보기를 하나만 읽었다: 보기가 아직 다 안 나왔거나 못 읽은 것 (사용자: "선택지가 하나밖에 없다고?")
            self._single_waits = getattr(self, "_single_waits", 0) + 1
            time.sleep(0.5)
            return
        self._single_waits = 0
        quizish = is_quiz(question) or any("정답" in e or "정담" in e for _, e in pairs)
        if quizish and quiz_answer(question, pairs) is None and self._quiz_waits < 8:
            # 퀴즈인데 정답 보기가 아직 안 보인다: 보기가 하나씩 나타나는 중이다
            # (보기 하나만 읽고 틀린 답을 고른 일이 있었다)
            self._quiz_waits += 1
            time.sleep(0.5)
            return
        self._quiz_waits = 0
        idx, rule = choose_option(pairs, self.run.floor, question, note_users=self.note_users,
                                  note_have=self.note_needs.have if self.note_needs else None,
                                  note_needs=self.note_needs)
        if idx < 0 and self._declined == question:
            # ESC로 안 닫히는 선택지: 가장 싼 쪽 (첫 번째 숫자가 가장 작은 것)
            costs = [int(m.group(1) or m.group(2)) if (m := re.search(r"(\d+)\s*소모", t + e)) else 10 ** 6 for t, e in pairs]
            idx, rule = costs.index(min(costs)), "모름: ESC로 안 닫혀서 가장 싼 쪽 (기록)"
        gold_before = self.read_gold(img)
        self.log("선택지", f"{question} -> [{idx}] '{options[idx].text if idx >= 0 else 'ESC'}' ({rule})", img,
                 options=pairs, chosen=idx, rule=rule)
        if "알림" in rule:
            # 사용자가 나중에 보고 기준을 정할 선택지 (작은 내기, 보상을 고르는 퀴즈): 원본 화면을 남긴다
            d = self.log_path.parent / "choice_alerts"
            d.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".png", img)[1].tofile(str(d / f"{time.strftime('%m%d_%H%M%S')}_{self.run.floor:02d}.png"))
        with self.choice_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "floor": self.run.floor,
                                "question": question, "options": pairs, "chosen": idx, "rule": rule},
                               ensure_ascii=False) + "\n")
        if idx < 0:
            self._declined = question
            self.gi.key("esc")
            time.sleep(1.0)
            return
        self._declined = None
        self.gi.click(*options[idx].center)
        if rule == "650원 도박":
            self._gamble_gold_before = self.read_gold_steady() or gold_before
            self._gamble_at = time.monotonic()
        time.sleep(1.0)

    def saw_loading(self) -> None:
        self.nav.loading_seen()  # 문에 들어갔으면 그 자리를 출구로 기억, 다음 필드 화면이 새 방 입구
        if self.floor_known and self.loading_at is None:
            self.loading_at = time.monotonic()

    def exit_front_goal(self, img: np.ndarray, char: tuple[float, float], pos: tuple[float, float]):
        """출구 쪽 목표. 출구 EXIT_FRONT_DIST 안에 들어오면 None (걷기를 멈추고 NPC를 찾는다)."""
        g = self.nav.exit_goal(img, char, pos)
        if g is not None and math.hypot(g.pos[0] - pos[0], g.pos[1] - pos[1]) < EXIT_FRONT_DIST:
            return None
        return g

    def walk_stop(self, im: np.ndarray, *, talk: bool = False) -> bool:
        """걷기를 멈출 화면인지. 문에 들어가 로딩이 뜨는 순간도 여기서 기억해 둔다."""
        st = self.det.detect(im).state
        if st == "loading":
            self.saw_loading()
        return st != "field" or (talk and nv.find_talk_prompt(im) is not None)

    def new_floor(self, floor: int, room: str, confirmed: bool = False) -> None:
        """confirmed: 층 번호를 제목/지도에서 직접 읽었는지. 로딩을 세어 정한 층이면 False
        (00:26: 지도를 못 읽고 1층부터 세다가 가짜 3층에서 650원 재시작을 눌러 실제 7층 판을 버렸다)."""
        room = ROOM_BY_FLOOR.get(floor, "") or room  # OCR로 읽은 방 이름보다 층 구성표를 믿는다
        self.combat_done = room not in COMBAT_ROOMS
        self.combat_done_at = None
        self.exit_npc_checked = False
        self.exit_paused = False
        self.joined_midway = False
        self.enhance_count = 0
        self.shop_done = not self.has_shop(floor, room)
        self.shop_search_start = None
        self.shop_plan = None
        self.shop_queue = []
        self.shop_rerolled = False
        self.shop_bought = set()
        self.shop_click_fails = {}
        self.shop_potions_left = False
        self.shop_reopens = 0
        self.trade_fails = 0  # 거래의 방 스크립트가 표시를 못 찾은 연속 횟수 (많으면 평소 흐름으로)
        self.shop_peeked = False
        self.trade_tick = 0
        self.floor_changed_at = time.monotonic()
        self.loading_at = None
        restore = self.floor_uncertain
        if self.floor_uncertain and confirmed and floor <= GAMBLE_LAST_FLOOR:
            self.run_tracked = True  # 층을 몰랐다가 제목/지도로 1~3층이 확인됨: 650원 도박 규칙을 다시 쓴다
        self.run.floor = floor
        self.floor_known = True
        if self.step_mode and self.gamble_won and not self.step_pause_sec:
            # --step과 --step-pause를 같이 주면: 650원 이길 때만 멈춰 기다리고, 층마다는 5초 멈춤만
            self.pause_pending = f"{floor}층 시작"
        self.floor_uncertain = False
        self.exit_seen_at = None
        self.exit_fails = 0
        self._last_cards = None
        self._card_prev = None
        self.card_pool_char = None
        self.room = room
        self.talked = set()
        self.talk_spots: list = []  # 이 층에서 F로 말을 건 자리들
        self.idle_since = None
        self.spots_tried: set = set()
        self.log("층", f"{floor}층 {room + '의 방' if room else ''}")
        if restore:
            self.restore_state(floor)  # 층을 몰랐다가 제목으로 알게 됨: 저장해 둔 같은 판이면 이어받는다
        if self.run_tracked:
            self.save_state()
        self.nav.new_room(floor, room, ROOM_BY_FLOOR.get(floor + 1, ""))

    def check_title(self, img: np.ndarray) -> bool:
        """방 제목을 읽어 층이 바뀌었는지 본다. 새 층이면 True."""
        now = time.monotonic()
        if now - self.title_checked < 0.7:
            return False
        self.title_checked = now
        floor, room = None, ""
        for line in self.ocr.read(img, TITLE_BOX):
            t = line.text.replace(" ", "")
            m = TITLE_FLOOR_RE.search(t)
            if m:
                floor = int(m.group(1) or m.group(2))
            room = room or next((r for r in ROOM_NAMES if r in t), "")
        if floor is None:
            # "N/20층" 줄이 먼저 사라져도 "선택의 방" 같은 방 이름은 조금 더 남는다
            # 방 제목은 대화가 끝난 뒤 다시 뜨기도 해서, 지금 방과 종류가 다를 때만 다음 층으로 본다
            expected = ROOM_BY_FLOOR.get(self.run.floor + 1, "")
            if room and room != self.room and (not expected or room == expected) and now - self.floor_changed_at > 12:
                self.title_seen = now
                self.new_floor(self.run.floor + 1, room)
                self.log("층", "층 번호는 못 읽었지만 새 방 이름을 봐서 다음 층으로 셈")
                return True
            return False
        self.title_seen = now
        if self.floor_known and floor != self.run.floor and not self.floor_uncertain                 and self.loading_at is None and now - self.floor_changed_at < 15:
            # 방금 층이 바뀌었고 로딩도 안 봤다: 같은 제목을 잘못 읽은 것 ("1/20층"을 "11층"으로 읽어 2층으로 셌다)
            return False
        if self.floor_known and floor != self.run.floor and not self.floor_uncertain:
            # 층은 한 번에 1씩만 오른다. OCR이 "3/20층"을 "13/20층"으로 읽는 일이 있었다
            if floor < self.run.floor:
                return False
            if floor != self.run.floor + 1:
                self.log("층", f"제목을 {floor}층으로 읽었지만 {self.run.floor + 1}층으로 셈")
                floor = self.run.floor + 1
        if floor != self.run.floor or (room and not self.room):
            changed = floor != self.run.floor
            self.new_floor(floor, room or self.room, confirmed=True)
            return changed
        return False

    def npc_labels(self, img: np.ndarray) -> list[tuple[str, tuple[int, int, int, int], bool]]:
        """화면의 NPC 이름표들: (이름, 상자, 상점 NPC인지)."""
        lines = self.ocr.read(img, (0, 0, 1920, 960))
        out = []
        for l in lines:
            if re.search(r"[\[\]×xX*]|\d{2,}", l.text):
                continue  # 왼쪽에 뜨는 아이템 획득 알림 "[스텔라 코인]×80" (×를 *로 읽기도 한다)
            t = re.sub(r"[^가-힣]", "", l.text)  # OCR이 "4베0}트리스"처럼 섞어 읽을 때가 있어서 한글만
            x, y, w, h = l.box
            if not (13 <= h <= 32 and 2 <= len(t) <= 6) or len(t) < len(l.text.replace(" ", "")) / 2:
                continue
            top_right = (x > 1400 and y < 80) or (x > 1700 and y < 260)
            if top_right and process.extractOne(t, KNOWN_NPCS, scorer=fuzz.ratio, score_cutoff=60):
                top_right = False
            if (x < 480 and y < 320) or top_right or (x < 330 and 330 < y < 660):
                # 왼쪽 위 아이콘/기록 점수, 오른쪽 위 레벨/돈(y<80: 20층 상점 포셔 이름표가 y 90쯤이라 130이면 걸러졌다),
                # 자동 전투, 왼쪽 아이템 획득 알림 줄 ("[체력의 소리]×6"을 "력의"라는 NPC로 읽고 걸어갔다)
                continue
            if any(word in t for word in HUD_TEXT) or process.extractOne(t, HUD_TEXT, scorer=fuzz.ratio, score_cutoff=60):
                continue
            if fuzz.ratio(t, "강화") >= 80 or fuzz.ratio(t, "강화머신") >= 70:
                continue  # 강화머신 위 글자 (NPC가 아니다)
            if time.monotonic() - self.title_seen < 3 and TITLE_BOX[0] < x < TITLE_BOX[2] and y < TITLE_BOX[3]:
                continue
            if is_green_text(img, l.box):
                continue  # 이름 위의 초록 글자는 NPC 종류("사건", "회복", "상점")지 이름이 아니다
            if process.extractOne(t, self.not_npc, scorer=fuzz.ratio, score_cutoff=75):
                continue  # 전투 중 뜨는 스킬 이름 ("무장 습격" 등)
            if white_fraction(img, l.box) < 0.08:
                continue  # NPC 이름은 흰 글자 (실제 이름표 0.14~0.40). 바닥 무늬를 "수숗그"로 읽은 일이 있었다
            # 이름 바로 위의 초록 글자가 NPC 종류. OCR이 "회복"을 "화복"으로 읽기도 해서 비슷하게 맞춘다
            kind = next((o.text.replace(" ", "") for o in lines
                         if o is not l and 0 < y - o.box[1] < 80 and abs(o.box[0] + o.box[2] / 2 - (x + w / 2)) < 60
                         and is_green_text(img, o.box)), "")
            out.append((t, l.box, fuzz.ratio(kind, "상점") >= 50))
        return out

    def green_tag_npcs(self, img: np.ndarray, char: tuple[float, float]) -> list[tuple[str, tuple[int, int, int, int], bool]]:
        """이름표 OCR이 통째로 실패할 때 (22:03 1층: 베아트리스를 지나침): NPC 머리 위 초록
        딱지("사건"/"회복"/"상점")를 색 덩어리로 찾고, 그 조각만 따로 OCR해서 확인한다
        (전체 화면은 못 읽어도 조각은 읽힌다). 돌려주는 모양은 npc_labels와 같다."""
        hsv = cv2.cvtColor(img[:960], cv2.COLOR_BGR2HSV)
        mask = ((hsv[..., 0] > 35) & (hsv[..., 0] < 75) & (hsv[..., 1] > 70) & (hsv[..., 2] > 110)).astype(np.uint8)
        for x0, y0, x1, y1 in ((0, 0, 420, 200), (1350, 0, 1920, 230), (1450, 800, 1920, 960), (780, 930, 1150, 960)):
            mask[y0:y1, x0:x1] = 0  # HUD
        cx, cy = int(char[0]), int(char[1])
        mask[max(0, cy - 20):cy + 90, max(0, cx - 150):cx + 150] = 0  # 캐릭터 체력바(같은 초록)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 11), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(mask)
        out = []
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if not (14 <= h <= 36 and 28 <= w <= 120 and area >= 140):
                continue
            crop = img[max(0, y - 8):min(960, y + h + 8), max(0, x - 10):min(1920, x + w + 10)]
            big = cv2.copyMakeBorder(cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
                                     40, 40, 40, 40, cv2.BORDER_REPLICATE)  # 테두리가 있어야 작은 조각이 읽힌다
            tag = re.sub(r"[^가-힣]", "", self.ocr.text(big, None, 1.0))
            m = process.extractOne(tag, ("사건", "회복", "상점"), scorer=fuzz.ratio)
            if not m or m[1] < 50:
                continue
            # 딱지 아래쪽에서 흰 이름을 읽어 본다 (못 읽으면 "?" — 가서 F를 눌러 보면 된다)
            name = "?"
            for l in self.ocr.read(img, (max(0, x + w // 2 - 180), y + h, min(1920, x + w // 2 + 180), min(960, y + h + 190))):
                t = re.sub(r"[^가-힣]", "", l.text)
                if 2 <= len(t) <= 6 and not is_green_text(img, l.box) and white_fraction(img, l.box) >= 0.08:
                    name = t
                    break
            out.append((name, (x, y, w, h), m[0] == "상점"))
        return out

    def near_talk_spot(self, box: tuple[int, int, int, int]) -> bool:
        """이름표가 이번 층에서 이미 말 건 자리 근처인지. 이름을 "?"로 읽고 말 걸면 목록에 안 남아서,
        나중에 이름을 제대로 읽으면 같은 NPC에게 또 간다 (23:52 2층 베르주)."""
        x, y, w, h = box
        wx, wy = self.nav.odo.to_world((x + w / 2, y + h / 2))
        return any(math.hypot(wx - sx, wy - sy) < 250 for sx, sy in self.talk_spots)

    def near_shop_spot(self, box: tuple[int, int, int, int]) -> bool:
        """이름표가 기억해 둔 상점 자리 근처인지. 상점 NPC 이름을 "?"로 읽으면 '말 건 목록'에 안 남아서,
        상점이 끝난 뒤 일반 NPC인 줄 알고 또 걸어갔다 온다 (22:48 12층 포셔)."""
        x, y, w, h = box
        wx, wy = self.nav.odo.to_world((x + w / 2, y + h / 2))
        return any(math.hypot(wx - sx, wy - sy) < 300 for sx, sy in self.nav.spots("shop"))

    def already_talked(self, name: str) -> bool:
        """OCR이 같은 이름을 "베르너/베트너/베드너"처럼 다르게 읽어서, 비슷하면 같은 NPC로 친다."""
        return any(fuzz.ratio(name, t) >= 60 for t in self.talked if t != "?")

    # -- 거래의 방 (강화머신, 상점) ------------------------------------------------
    def has_shop(self, floor: int | None = None, room: str | None = None) -> bool:
        """이 층에 상점이 있는지: 거래의 방, 그리고 20층 (보스 뒤 마지막 상점)."""
        floor = self.run.floor if floor is None else floor
        room = self.room if room is None else room
        return room == "거래" or floor == 20

    @property
    def shop_index(self) -> int:
        return SHOP_INDEX_BY_FLOOR.get(self.run.floor, 1)

    def next_enhance_price(self) -> int:
        schedule = [0, 60, 120, 180] if self.shop_index == 1 else [60, 120, 180]
        return schedule[self.enhance_count] if self.enhance_count < len(schedule) else 10 ** 6

    def want_enhance(self, img: np.ndarray) -> bool:
        # 상점 4번(5, 12, 19, 20층) 모두 강화머신이 있다. 20층은 보스를 잡은 뒤에 나온다
        # (전엔 거래의 방만 봐서 20층 "강화 60"을 그냥 지나쳤다)
        if not (self.room == "거래" or (self.run.floor == 20 and self.combat_done)):
            return False
        price = self.next_enhance_price()
        if price > ENHANCE_MAX_PRICE:
            return False
        if self.run_tracked and not self.worth_enhancing():
            # 가진 잠재가 전부 6레벨이면 강화머신을 눌러도 카드가 안 나온다 (07:18 19층에서 두 번 헛걸음).
            # 후순위/다다익선(목표 3)/명함만뿐이어도 강화할 필요가 없다: 상점 음료부터 사서 올릴 잠재를 찾는다 (사용자 규칙)
            return False
        if self.defer_enhance_for_priority():
            return False  # 강화 우선 캐릭터 잠재가 없다: 상점 음료부터
        if self.defer_enhance_for_plus2():
            return False  # 강화할 게 전부 5레벨인데 +2가 남았다: 음료로 낮은 렙 잠재부터 만든다
        gold = self.read_gold(img)
        if gold is not None and gold < price:
            # 모자라 보이면 몇 번 더 읽는다 (08:01 20층: 1821원인데 강화를 건너뛰고 상점에서 돈을 다 썼다).
            # 돈이 정말 모자랄 때 매번 다시 읽지 않게 3초 동안은 앞의 판단을 쓴다
            now = time.monotonic()
            cached = getattr(self, "_enh_gold_check", None)
            if cached is None or now - cached[0] > 3:
                self._enh_gold_check = (now, self.read_gold_steady(3))
            gold = self._enh_gold_check[1]
        return gold is None or gold >= price

    def upgradable(self) -> list[int]:
        """강화머신이 올려 줄 수 있는 잠재: 가진 것 중 코어가 아니고 6레벨이 안 된 것."""
        gd = default_gamedata()
        return [pid for pid, lv in self.run.owned.items()
                if lv < 6 and pid in gd.potentials and gd.potentials[pid].kind != "core"]

    def worth_enhancing(self) -> list[int]:
        """강화할 만한 잠재: 6레벨이 안 된 필수와 다다익선(목표 6)."""
        goals = self.chooser.goals
        return [pid for pid in self.upgradable() if pid in goals and goals[pid].kind != "core"
                and (goals[pid].mark == "필수" or (goals[pid].mark == "다다익선" and goals[pid].target_level >= 6))]

    # -- 강화 우선 캐릭터 (프리셋 enhance_first, 사용자 2026-10-02 22시) --------------------
    # 이 캐릭터의 강화할 잠재가 없으면: 강화머신보다 상점 음료(할인부터)를 먼저 사서 얻는다.
    # 강화하다 떨어지면 상점에 다시 간다. 음료를 사도 못 얻으면 (돈/재고가 다하면) 그냥 다른 캐릭터를 강화한다.
    def priority_upgradable(self) -> list[int]:
        pc = self.preset.enhance_first
        if not pc:
            return []
        return [pid for pid in self.worth_enhancing() if self.chooser.char_of.get(pid) == pc]

    def shop_can_feed(self) -> bool:
        """상점 음료로 잠재를 더 얻어볼 수 있는지 (아직 안 갔거나, 남은 음료 + 돈이 있는지)."""
        if not self.shop_done:
            return True
        return self.shop_potions_left and self.shop_reopens < 2 and self.run.gold >= 700

    def defer_enhance_for_plus2(self) -> bool:
        """강화할 필수/다다익선(6)이 전부 5레벨 이상이고 +2 강화가 남아 있으면 강화를 미룬다
        (사용자 2026-10-03: +2가 5레벨에 떨어지면 1레벨을 버린다. 음료로 낮은 렙 잠재를 만들고 온다)."""
        if not self.run_tracked or self.run.plus2_taken >= strategy.PLUS2_PER_RUN:
            return False
        worth = self.worth_enhancing()
        if not worth or any(self.run.owned.get(pid, 0) <= 4 for pid in worth):
            return False
        return self.shop_can_feed()

    def defer_enhance_for_priority(self) -> bool:
        """강화를 미루고 상점부터 가야 하는지."""
        pc = self.preset.enhance_first
        if not pc or not self.run_tracked or self.priority_upgradable():
            return False
        return self.shop_can_feed()

    def enhance_reserve(self) -> int:
        """상점을 먼저 들를 때 강화머신에 쓸 돈 (아직 안 누른 값들, 180원까지)."""
        if self.enhance_count >= 90:
            return 0  # 강화머신을 못 찾아 포기함
        schedule = [0, 60, 120, 180] if self.shop_index == 1 else [60, 120, 180]
        return sum(p for p in schedule[self.enhance_count:] if p <= ENHANCE_MAX_PRICE)

    def find_green_label(self, img: np.ndarray, word: str) -> tuple[int, int, int, int] | None:
        """초록 글자 이름표(예: 강화머신 위 "강화")의 상자."""
        for l in self.ocr.read(img, (0, 0, 1920, 960)):
            t = re.sub(r"[^가-힣]", "", l.text)
            x, y, w, h = l.box
            if t and fuzz.ratio(t, word) >= 50 and is_green_text(img, l.box) and not (x < 480 and y < 320):
                return l.box
        return None

    def save_full(self, img: np.ndarray, tag: str) -> None:
        """나중에 소리 그림/이름 짝을 만들 자료: 원본 크기 화면을 logs/shop_data/에.
        png는 한 장에 1.4MB라 (OneDrive가 전부 올린다) 거의 같은 화질의 jpg로 바꿨다 (~150KB)."""
        d = self.log_path.parent / "shop_data"
        d.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
            str(d / f"{time.strftime('%m%d_%H%M%S')}_{self.run.floor:02d}_{tag}.jpg"))

    def _note_users_file(self) -> Path:
        return self.log_dir / "note_users.json"

    def _load_note_users(self) -> dict[int, int]:
        """프리셋마다 협주스킬이 쓰는 소리는 같다. 상점 가방에서 읽기 전(1~4층 선택지)에도 쓰려고 저장해 둔다."""
        try:
            d = json.loads(self._note_users_file().read_text(encoding="utf-8"))
            return {int(k): v for k, v in d.get(self.preset.share_code or self.preset.title, {}).items()}
        except (OSError, ValueError, AttributeError):
            return {}

    def _save_note_users(self) -> None:
        f = self._note_users_file()
        try:
            d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        except ValueError:
            d = {}
        d[self.preset.share_code or self.preset.title] = {str(k): v for k, v in self.note_users.items()}
        f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

    def peek_bag(self) -> None:
        """상점에서 B로 가방(협주스킬별 필요한 소리, 가진 소리 개수)을 열어 찍어 두고 ESC로 돌아온다."""
        self.gi.key("b")
        time.sleep(1.2)
        im = self.grab()
        if self.det.detect(im).state == "bag":
            self.save_full(im, "bag")
            # 레코드 스킬 탭 (왼쪽 두 번째)
            self.gi.click(220, 321)
            time.sleep(0.8)
            skills = self.grab()
            self.save_full(skills, "bag_skills")
            self.note_needs = read_needs(skills, self.ocr)
            self.log("상점", f"가방: {self.note_needs.summary() or '협주스킬 필요량을 못 읽음'}")
            if sum(self.note_needs.users.values()) >= 4:
                self.note_users = dict(self.note_needs.users)
                self._save_note_users()
            self.gi.key("esc")
            time.sleep(1.0)
        if self.det.detect(self.grab()).state == "bag":
            self.gi.key("esc")
            time.sleep(1.0)

    def on_shop(self, img: np.ndarray) -> None:
        if self.shop_plan is None and not self.shop_peeked:
            self.shop_peeked = True
            self.peek_bag()
            return
        if self.shop_plan is None:
            self.save_full(img, "shop")
        if self.shop_plan is None:
            gold = self.read_gold(img) or 0
            items = read_shop(img, self.ocr)
            for it in items:
                if (self.shop_rerolled, it.slot) in self.shop_bought:
                    it.sold_out = True  # 이미 산 칸 (품절 표시를 못 읽어도 다시 사지 않게)
                if it.kind == "potential" and not it.sold_out:
                    it.face = drink_face(img, it)
                    if it.face == -1:
                        self.save_full(img, f"face_unknown_{it.slot}")  # 모르는 얼굴: 나중에 템플릿을 뜬다
                if it.kind == "notes":
                    it.note_type = shop_note_type(img, it.click[0], it.click[1], it.name)
                    if self.note_needs is not None and it.note_type is not None:
                        it.users = self.note_needs.users.get(it.note_type, 0)
                        it.have = self.note_needs.have.get(it.note_type)
            price, left = read_reroll(img, self.ocr)
            if self.shop_rerolled:
                left = 0
            # 강화머신을 아직 덜 눌렀는데 강화할 만한 잠재가 없어서 상점부터 왔다: 나중에 강화할 돈을 남긴다
            reserve = self.enhance_reserve() if self.room == "거래" or self.run.floor == 20 else 0
            self.shop_plan = plan_purchases(items, gold, shop_index=self.shop_index, last_shop=self.run.floor >= 20,
                                            reroll_left=left, reroll_price=price,
                                            note_have=self.note_needs.have if self.note_needs else None,
                                            enhance_reserve=reserve, prefer_char=self.preset.enhance_first,
                                            note_needs=self.note_needs)
            self.shop_queue = list(self.shop_plan.buy)
            bought = {i.slot for i in self.shop_plan.buy}
            self.shop_potions_left = bool(self.shop_plan.buy) and any(
                i.kind == "potential" and not i.sold_out and i.slot not in bought for i in items)
            names = {c.char_id: c.name for c in self.preset.characters}
            desc = ", ".join(f"{i.slot}:{i.name}({i.price}{'/' + str(i.old_price) if i.discounted else ''}"
                             f"{' ' + names.get(i.face, '모르는 얼굴') + ' 얼굴' if i.face is not None else ''}"
                             f"{' 품절' if i.sold_out else ''}"
                             f"{' ' + NOTE_NAMES[i.note_type] + ' 협주' + str(i.users) if i.note_type is not None and i.users is not None else ''})"
                             for i in items)
            self.log("상점", f"{self.shop_index}번째 상점, {self.shop_plan.reason} | {desc}", img)
        if self.shop_queue:
            item = self.shop_queue[0]
            key = (self.shop_rerolled, item.slot)
            self.log("상점", f"구매: {item.name} {item.price}원")
            self.gi.click(*item.click)
            time.sleep(0.8)
            if self.det.detect(self.grab()).state == "shop":
                # 구매 창이 안 떴다: 안 눌린 것. 산 걸로 치면 안 된다 (18:39 20층: 카드 고르기에서 돌아오자마자 누른
                # 160원 음료가 안 눌렸는데 산 칸으로 적어서, 그 칸을 건너뛰고 200원짜리를 샀다)
                self.shop_click_fails[key] = self.shop_click_fails.get(key, 0) + 1
                if self.shop_click_fails[key] < 3:
                    self.log("상점", f"구매 창이 안 뜸, 다시 누름 ({self.shop_click_fails[key]}번째)")
                    time.sleep(0.5)
                    return
                self.log("상점", f"{item.name}: 구매 창이 세 번 안 떠서 건너뜀")
            self.shop_queue.pop(0)
            self.shop_bought.add(key)
            if item.kind == "potential":
                # 음료를 사면 카드 고르기가 나오고 거기서 리롤로 돈이 바뀐다. 돌아오면 돈을 다시 읽고 다시 계획한다
                self.shop_plan = None
                self.shop_queue = []
                self.card_pool_char = item.face  # 얼굴 음료면 카드 화면에 그 캐릭터 잠재만 나온다
            return
        if self.shop_plan.reroll and not self.shop_rerolled:
            self.shop_rerolled = True
            self.shop_plan = None
            self.log("상점", "상점 새로고침 (Q)")
            self.gi.key("q")
            time.sleep(1.2)
            return
        self.log("상점", "상점 끝, 나감")
        self.shop_done = True
        if self.run_tracked:
            self.save_state()
        self.gi.key("esc")
        time.sleep(0.8)

    def nudge_until(self, prompt: str, first: str = "s") -> bool:
        """목표 앞에 왔는데 상호작용 표시가 안 뜨면 아래/위/왼쪽/오른쪽으로 조금씩 움직여 본다.
        first="w": NPC는 이름표가 발밑보다 아래라 위쪽부터 (20층 상점 포셔 앞에서 표시가 안 떠서 포기했다)."""
        back = "w" if first == "s" else "s"
        for keys in ([first], [back], ["a"], ["d"], [first], [first], [back], [back]):
            self.gi.hold(keys, 0.25)
            time.sleep(0.15)
            if nv.find_prompt(self.grab()) == prompt:
                self.log("이동", f"조금씩 움직여서 '{prompt}' 표시를 찾음")
                return True
        return False

    def nearest_label(self, img: np.ndarray, char: tuple[float, float]) -> str:
        labels = self.npc_labels(img)
        if not labels:
            return "?"
        name, _, _ = min(labels, key=lambda lb: abs(lb[1][0] + lb[1][2] / 2 - char[0]) + abs(lb[1][1] - char[1]))
        return name

    def on_field(self, img: np.ndarray) -> str | None:
        if getattr(self, "_gamble_gold_before", None) is not None:
            won, why = self.judge_gamble()
            if won is None and time.monotonic() - self._gamble_at <= 12:
                # 결과를 읽을 때까지 제자리에서 기다린다 (07:07: 결과를 못 읽은 채 출구로 가서 다음 층에서 짐작했다)
                time.sleep(0.3)
                return None
            if won is None:
                # 끝까지 모르면 이어 가되 (잘못 재시작하면 이긴 판을 버린다), 다음 카드 화면의 돈으로 다시 확인한다
                self._gamble_verify = self._gamble_gold_before
                won = True
            self.gamble_won = self.gamble_won or won
            if won and self.step_mode:
                self.pause_pending = "650원 성공"
            self.run.keep_after_gamble = self.gamble_won
            self.log("650원", f"{'성공' if won else '실패'} ({why})", img)
            if self.run_tracked:
                self.save_state()
            self._gamble_gold_before = None
            if not won:
                self.restart_pending = "650원 도박 실패"
            if self.pause_pending:
                return None  # 한 층씩 모드: 출구로 걷기 전에 바로 멈춘다
        if self.restart_pending:
            return "restart"
        if self.check_title(img):
            return "next_floor"
        if self.loading_at is not None and time.monotonic() - self.loading_at > 5:
            self.new_floor(self.run.floor + 1, "")
            self.log("층", "방 제목을 못 읽었지만 로딩을 지나서 다음 층으로 셈")
            return "next_floor"
        if not self.floor_known:
            self.read_floor_from_map()
            return None
        char = nv.find_character(img)
        pos = self.nav.observe(img, char)  # 방 안 위치 (처음 보는 화면이면 어떤 지도인지도 알아본다)
        if self.loading_at is not None:
            # 로딩을 지나 새 방에 왔는데 아직 방 제목을 못 읽었다. 지난 방 기준으로 움직이지 않고 제목을 기다린다
            time.sleep(0.15)
            return None
        if char is None:
            if self.char_missing_since is None:
                self.char_missing_since = time.monotonic()
            if time.monotonic() - self.char_missing_since < 3:
                time.sleep(0.2)
                return None
            # 체력바를 계속 못 찾는다: 카메라가 캐릭터를 따라가니 화면 가운데에 있다고 보고 계속한다
            char = (960.0, 555.0)
        else:
            self.char_missing_since = None
        now = time.monotonic()

        prompt = nv.find_prompt(img)

        # 0-0) 거래의 방(5·12·19층)은 구조가 늘 같다 (사용자): 자리를 아는 지도에선 이름표를 훑지 않고
        # 상점·강화머신 두 자리만 오간다. 순서(상점 먼저냐 강화 먼저냐)는 want_enhance가 정한다
        if self.trade_step(img, pos, prompt):
            return None

        # 0) 거래의 방 강화머신: 가까이 가면 "F 강화". 180원까지 누른다
        if prompt == "enhance" and now - self.last_talk_at > 2.5 and self.want_enhance(img):
            self.press_enhance(img)
            return None

        # 1) "F 대화"가 떠 있으면, 아직 말 안 건 NPC면 말을 건다 (거래의 방에선 상점이 열린다)
        if (prompt == "talk" and now - self.last_talk_at > TALK_COOLDOWN
                and not (self.has_shop() and self.shop_done
                         and any(math.hypot(pos[0] - sx, pos[1] - sy) < 300 for sx, sy in self.nav.spots("shop")))):
            # 마지막 조건: 끝난 상점 NPC 앞의 "F 대화"로 상점을 또 열지 않는다 (23:02 19층 포셔)
            name = self.nearest_label(img, char)
            # 이름을 못 읽었는데 이 층에서 바로 이 근처에서 말을 걸었다면, 대개 방금 그 NPC다
            # (전엔 "말 건 목록"이 비어 있지만 않으면 건너뛰어서, 보스 이름을 목록에 넣은 20층에서 상점을 놓쳤다)
            here = self.nav.last_pos
            near_last = here is not None and any(math.hypot(here[0] - p[0], here[1] - p[1]) < 250 for p in self.talk_spots)
            if not self.already_talked(name) and not near_last:
                self.talked.add(name)
                if here is not None:
                    self.talk_spots.append(here)
                self.last_talk_at = now
                self.log("NPC", f"{name}에게 F로 말 걸기", img)
                self.gi.key("f")
                time.sleep(1.0)
                after = self.grab()
                after_state = self.det.detect(after).state
                if after_state != "field":
                    self.nav.remember_spot("shop" if after_state in ("shop", "shop_buy") or
                                           (self.room == "거래" and self.want_enhance(after) is False) else "npc")
                if after_state == "field":
                    self.log("NPC", f"{name}: 대화가 안 열림 (이미 끝난 이벤트일 수 있음)")
                    # 이름을 다르게 읽어("베아틔원"/"베아트리스") 같은 NPC에게 또 가지 않게, 근처 이름표를 다 말 건 것으로
                    for other, (bx, by, bw, bh), _ in self.npc_labels(after):
                        if abs(bx + bw / 2 - char[0]) < 250 and abs(by - char[1]) < 300:
                            self.talked.add(other)
                return None

        # 1-1) 거래의 방: 강화머신("강화" 초록 글자) 쪽으로 먼저 간다 (거래의 방은 매번 찾는다)
        if self.want_enhance(img):
            box = self.find_green_label(img, "강화")
            if box:
                x, y, w, h = box
                self.log("강화", f"강화머신 쪽으로 걸어감 ({x + w // 2}, {y + h // 2})", img)
                res = self.nav.walk(self.label_goal(img, box, 90), self.stop_for("enhance"), max_sec=15,
                                    avoid_exit=True, why="강화머신")
                self.log("강화", f"걷기 결과 {res.reason} ({res.secs:.0f}초)")
                if nv.find_prompt(self.grab()) != "enhance":
                    self.nudge_until("enhance")
                return None

        # 2) 화면에 말 안 건 NPC 이름표가 있으면 그쪽으로 걸어간다 (상점 NPC는 거래의 방에서 강화 뒤에만)
        # 20층 보스 뒤: 강화머신 자리를 알면 강화부터 (규칙 순서. 상점에서 돈을 다 써서 강화를 1번만 했다)
        # (기억한 자리에 아직 안 가 봤을 때만: 가 봤는데 못 찾았으면 상점이라도 들른다)
        enhance_first = (self.run.floor == 20 and self.combat_done and self.want_enhance(img)
                         and any(("enhance", round(x), round(y)) not in self.spots_tried
                                 for x, y in self.nav.spots("enhance")))
        if now - self.npc_scanned > NPC_SCAN_EVERY and not enhance_first:
            self.npc_scanned = now
            shop_ok = self.has_shop() and not self.shop_done and not self.want_enhance(img)
            def fresh(lb) -> bool:
                return (not self.already_talked(lb[0]) and not self.near_talk_spot(lb[1])
                        and (not (lb[2] or (self.has_shop() and self.near_shop_spot(lb[1]))) or shop_ok))
            todo = [lb for lb in self.npc_labels(img) if fresh(lb)]
            if not todo:  # 이름표 OCR이 실패했을 수 있다: 초록 딱지를 색으로 찾는다
                todo = [lb for lb in self.green_tag_npcs(img, char) if fresh(lb)]
            if todo:
                name, box, _ = min(todo, key=lambda lb: abs(lb[1][0] - char[0]) + abs(lb[1][1] - char[1]))
                x, y, w, h = box
                self.log("NPC", f"{name} 쪽으로 걸어감 ({x + w // 2}, {y + h // 2})", img)
                res = self.nav.walk(self.label_goal(img, box, 60), self.stop_for("talk"), max_sec=15,
                                    avoid_exit=True, why=f"NPC {name}")
                self.log("NPC", f"걷기 결과 {res.reason} ({res.secs:.0f}초)")
                if res.reason == "arrived" and self.nudge_until("talk", first="w"):
                    return None  # 말 걸기 표시가 떴다: 다음 화면에서 F
                if res.reason not in ("stopped", "near_exit"):
                    self.talked.add(name)  # 못 가면 이 NPC는 포기 (무한 반복 방지)
                elif res.reason == "near_exit":
                    # 출구 옆이라 멈춤: 포기하진 않지만 같은 길로 바로 다시 가지 않게 잠깐 쉰다
                    # (5층: 상점 NPC를 포기 목록에 넣어 상점을 못 들렀다)
                    self.npc_scanned = now + 6
                return None

        # 2-1) 화면 밖 NPC는 물음표 상자 표시를 따라간다 (전투가 끝나야 NPC가 생기는 방도 있다)
        if self.combat_done and not self.talked and now > self.npc_marker_rest_until:
            npc = next((m for m in nv.find_markers(img, char) if m.kind == "npc"), None)
            if npc:
                self.log("NPC", "물음표 표시를 따라감", img)
                res = self.nav.walk(self.marker_goal("npc"), self.stop_for("talk"), max_sec=15,
                                    avoid_exit=True, why="NPC 표시")
                self.log("NPC", f"표시 따라가기 결과 {res.reason} ({res.secs:.0f}초)")
                self.npc_scanned = 0.0  # NPC가 화면에 들어왔을 테니 바로 이름표를 찾는다
                if res.reason in ("near_exit", "no_path", "no_progress"):
                    # 출구 옆이라 못 가면 잠깐 쉰다 (3층: 1초에 두 번씩 따라가기→멈춤만 반복했다)
                    self.npc_marker_rest_until = time.monotonic() + 15
                return None

        # 2-1-0) 강화 우선 캐릭터: 강화머신을 더 쓸 수 있는데 그 캐릭터 잠재가 없으면 상점에 다시 간다
        if (self.preset.enhance_first and self.run_tracked and self.has_shop() and self.combat_done
                and self.shop_done and self.next_enhance_price() <= ENHANCE_MAX_PRICE
                and not self.priority_upgradable() and self.shop_can_feed()):
            self.shop_reopens += 1
            self.shop_done = False
            self.shop_plan, self.shop_queue = None, []
            names = {c.char_id: c.name for c in self.preset.characters}
            self.log("상점", f"{names.get(self.preset.enhance_first, '우선 캐릭터')} 잠재가 없어서 상점을 다시 간다"
                           f" ({self.shop_reopens}번째, 돈 {self.run.gold})")
            return None

        # 2-1-1) 거래의 방은 강화머신과 상점이 끝나야 나간다
        # 20층은 보스를 잡은 뒤 마지막 상점이 있다 (돈을 다 쓴다)
        if self.has_shop() and (self.want_enhance(img) or not self.shop_done) and self.combat_done:
            if self.shop_search_start is None:
                self.shop_search_start = now  # 20층은 보스전이 길어서 층 시작이 아니라 여기서부터 잰다
            if now - self.shop_search_start < TRADE_TIMEOUT:
                # 다음 루프에서 이름표부터 다시 찾고, 둘러보기는 두 번에 한 번만 (근처에서 멀어지지 않게)
                self.npc_scanned = 0.0
                self.trade_tick += 1
                if self.go_spot("enhance" if self.want_enhance(img) else "shop"):
                    return None
                if not self.shop_done and not self.want_enhance(img) and self.trade_tick % 2 == 1:
                    # 상점 NPC 머리 위 초록 "상점" 글자 (이름표를 못 읽어도 보인다)
                    box = self.find_green_label(img, "상점")
                    if box:
                        x, y, w, h = box
                        self.log("상점", f"'상점' 글자 쪽으로 걸어감 ({x + w // 2}, {y + h // 2})", img)
                        res = self.nav.walk(self.label_goal(img, box, 130), self.stop_for("talk"), max_sec=15,
                                            avoid_exit=True, why="상점 글자")
                        self.log("상점", f"걷기 결과 {res.reason} ({res.secs:.0f}초)")
                        if res.reason == "arrived":
                            self.nudge_until("talk", first="w")
                        return None
                if self.trade_tick % 2 == 0:
                    self._explore("강화머신/상점을 찾으려고 둘러봄", avoid_exit=True)
                else:
                    time.sleep(0.3)
                return None
            self.shop_done = True
            self.enhance_count = 99
            self.log("상점", f"{TRADE_TIMEOUT:.0f}초 동안 강화/상점을 못 끝냄, 그냥 나감", img)

        # 2-2) 전투방은 전투가 끝나야("소리 획득") 나간다. 출구는 전투 중에도 보인다
        if not self.combat_done:
            limit = COMBAT_TIMEOUT if not self.joined_midway else (120.0 if self.run.floor == 20 else 40.0)
            if now - self.floor_changed_at < limit:
                time.sleep(0.15)  # 자동 전투로 캐릭터가 움직이니 위치를 자주 잰다
                return None
            self.combat_done = True
            self.log("전투", f"{limit:.0f}초 동안 소리 획득을 못 봄, 전투가 끝난 것으로 보고 진행", img)

        # 3) 선택의 방은 NPC와 이야기하기 전엔 나가지 않는다
        if self.room == "선택" and not self.talked and now - self.floor_changed_at < CHOICE_SEARCH:
            if self.go_spot("npc"):
                return None
            self._explore("선택의 방 NPC를 찾으려고 둘러봄", avoid_exit=True)
            return None

        # 방 제목이 떠 있는 동안은 제목 글자(ㅇ)를 출구 문양으로 착각할 수 있어서 기다린다
        if now - self.title_seen < 2.5:
            time.sleep(0.2)
            return None

        goal = self.nav.exit_goal(img, char, pos)
        if goal is None:
            if self.idle_since is None:
                self.idle_since = now
            if now - self.idle_since < 3:
                time.sleep(0.3)  # 아직 전투 중이거나 출구가 안 열렸을 수 있다
                return None
            self._explore("출구를 찾으려고 둘러봄")
            return None
        self.idle_since = None
        if self.exit_seen_at is None:
            self.exit_seen_at = now
        if not self.exit_npc_checked:
            # NPC는 늘 출구 근처에 생긴다 (사용자 설명, 1~20층 모두). 들어가기 전에 출구 앞에서 멈춰 한 번 찾는다
            # (18:45 1층: 전투가 끝나고 2초 만에 출구로 가서, 출구 옆에 생긴 포셔를 지나쳤다)
            self.exit_npc_checked = True
            self.log("이동", f"출구 앞까지 가서 NPC를 찾음: {goal.kind} ({goal.pos[0]:.0f}, {goal.pos[1]:.0f})", img)
            res = self.nav.walk(self.exit_front_goal, self.walk_stop, max_sec=30, why="출구 앞")
            self.log("이동", f"걷기 결과 {res.reason} ({res.secs:.0f}초)")
            if self.combat_done_at is not None:
                left = POST_COMBAT_NPC_WAIT - (time.monotonic() - self.combat_done_at)
                if left > 0:
                    time.sleep(left)  # 전투가 막 끝났으면 NPC가 생길 때까지
            self.npc_scanned = 0.0  # 다음 화면에서 바로 이름표를 찾는다
            self.npc_marker_rest_until = 0.0
            return None
        if self.require_gamble and self.run_tracked and not self.gamble_won and self.run.floor >= GAMBLE_LAST_FLOOR:
            # 3층에서도 650원 NPC가 나올 수 있다. 출구가 보인 뒤에도 잠깐 NPC를 더 찾고 나서 판단한다
            if self.run.floor == GAMBLE_LAST_FLOOR and now - self.exit_seen_at < GAMBLE_EXIT_WAIT:
                time.sleep(0.5)
                return None
            self.restart_pending = f"{GAMBLE_LAST_FLOOR}층까지 650원 선택지를 못 받음"
            return "restart"
        if self.step_pause_sec and self.gamble_won and not self.exit_paused:
            # 사용자가 지켜보며 고칠 곳을 짚는 가벼운 모드: 층을 떠나기 전에 잠깐만 멈춘다
            self.exit_paused = True
            self.gi.release_all()
            self.log("멈춤", f"출구 앞 잠깐 멈춤 ({self.step_pause_sec:.0f}초)", img)
            time.sleep(self.step_pause_sec)
            return None
        self.log("이동", f"출구로 감: {goal.kind} ({goal.pos[0]:.0f}, {goal.pos[1]:.0f}), 지금 ({pos[0]:.0f}, {pos[1]:.0f})", img)
        res = self.nav.walk(self.nav.exit_goal, self.walk_stop, max_sec=45, why="출구")
        self.log("이동", f"걷기 결과 {res.reason} ({res.secs:.0f}초)")
        if res.reason in ("no_progress", "no_path", "timeout"):
            self.exit_fails += 1
            if self.exit_fails % 3 == 0:
                # 잘못 적은 막힌 칸 때문에 길이 막혀 구석으로 가는 경우가 있다 (8층: NPC에 막힌 뒤 왼쪽 끝 술통까지 감)
                n = self.nav.map.forget_blocks()
                self.log("이동", f"출구로 {self.exit_fails}번 못 감: 막힌 칸 {n}개를 지우고 길을 다시 찾음")
                if self.nav.door_seen is not None:
                    self.nav.door_seen = None  # 고른 문으로 못 간다: 다시 고르게 한다
                    self.nav.door_candidate = None
            if self.nav.memory_exit is not None and (self.exit_fails >= 6 or
                                                     (self.exit_fails >= 2 and self.nav.ignored_doors >= 10) or
                                                     (self.exit_fails >= 1 and self.nav.ignored_doors >= 30)):
                # 15층: 화면에 진짜 문이 보이는데 믿는 출구 자리와 멀다고 무시하고 70초 헤맸다 (이번 판 위치가 어긋남)
                self.nav.memory_exit = None
                self.log("이동", f"기억한 출구로 {self.exit_fails}번 못 감 (무시한 문양 {self.nav.ignored_doors}번): "
                               "이번 층은 화면 표시와 문양만 따라감")
        elif res.reason == "stopped":
            self.exit_fails = 0
        return None

    # -- 걷기 도우미 -------------------------------------------------------------
    def trade_step(self, img: np.ndarray, pos: tuple[float, float], prompt: str | None) -> bool:
        """거래의 방 스크립트: 기억해 둔 상점/강화머신 자리만 오간다. 처리했으면 True.
        처음 보는 지도(자리 모름)나 20층 보스 뒤 상점은 평소 흐름대로 (False)."""
        if self.room != "거래" or not self.combat_done or self.trade_fails >= 4:
            return False  # 스크립트가 계속 빗나가면 평소 흐름(이름표/초록 글자 찾기)으로
        if not self.nav.spots("shop") or not self.nav.spots("enhance"):
            return False
        want_e = self.want_enhance(img)
        if not want_e and self.shop_done:
            return False  # 둘 다 끝 (또는 강화 우선 캐릭터 재방문 판단은 뒤 흐름에서)
        now = time.monotonic()
        if self.shop_search_start is None:
            self.shop_search_start = now
        if now - self.shop_search_start > TRADE_TIMEOUT:
            self.shop_done = True
            self.enhance_count = 99
            self.log("상점", f"{TRADE_TIMEOUT:.0f}초 동안 강화/상점을 못 끝냄, 그냥 나감 (스크립트)", img)
            return False
        kind = "enhance" if want_e else "shop"
        want_prompt = "enhance" if want_e else "talk"
        if prompt == want_prompt and now - self.last_talk_at > 2.5:
            if want_e:
                self.press_enhance(img)
            else:
                self.last_talk_at = now
                self.log("상점", "포셔에게 F (거래의 방 스크립트)", img)
                self.gi.key("f")
                time.sleep(1.0)
            return True
        spot = min(self.nav.spots(kind), key=lambda p: math.hypot(p[0] - pos[0], p[1] - pos[1]))
        if math.hypot(spot[0] - pos[0], spot[1] - pos[1]) > 80:
            self.log("이동", f"{kind} 자리로 감 (거래의 방 스크립트)")
            res = self.nav.walk(lambda im, ch, p: Goal(spot, kind, arrive=50), self.stop_for(want_prompt),
                                max_sec=20, avoid_exit=True, why=f"{kind} 자리")
            self.log("이동", f"걷기 결과 {res.reason} ({res.secs:.0f}초)")
            left = self.nav.last_pos
            gap = math.hypot(spot[0] - left[0], spot[1] - left[1]) if left else 0.0
            if res.reason in ("arrived", "no_progress", "no_path") and gap > 70:
                # 막힌 칸 때문에 길찾기가 자리까지 못 갔다 (00:51 5층: 100px 앞에서 걷기→꼼지락만 반복).
                # 자리 쪽으로 키를 직접 눌러 마저 간다
                ang = nv.angle_of(spot[0] - left[0], spot[1] - left[1])
                self.log("이동", f"{kind} 자리까지 {gap:.0f}px 남아 곧장 걸음 ({ang:.0f}도)")
                self.gi.hold(nv.keys_for_angle(ang), min(0.7, gap / 320))
                time.sleep(0.15)
            if nv.find_prompt(self.grab()) != want_prompt:
                if self.nudge_until(want_prompt, first="s" if want_e else "w"):
                    self.trade_fails = 0
                else:
                    self.trade_fails += 1
            else:
                self.trade_fails = 0
        elif now - self.last_talk_at > 2.5:
            # 자리엔 왔는데 표시가 안 뜬다: 조금씩 움직여 본다
            self.nudge_until(want_prompt, first="s" if want_e else "w")
        else:
            time.sleep(0.2)
        return True

    def go_spot(self, kind: str) -> bool:
        """이 지도에서 전에 말을 걸었던 자리(kind)로 간다. 이번 층에서 아직 안 가 본 자리가 없으면 False."""
        pos = self.nav.last_pos
        todo = [p for p in self.nav.spots(kind) if (kind, round(p[0]), round(p[1])) not in self.spots_tried]
        if not todo or pos is None:
            return False
        spot = min(todo, key=lambda p: math.hypot(p[0] - pos[0], p[1] - pos[1]))
        self.spots_tried.add((kind, round(spot[0]), round(spot[1])))
        prompt = "enhance" if kind == "enhance" else "talk"
        self.log("이동", f"기억해 둔 {kind} 자리로 감 ({spot[0]:.0f}, {spot[1]:.0f})")
        res = self.nav.walk(lambda im, ch, p: Goal(spot, kind, arrive=60), self.stop_for(prompt), max_sec=20,
                            avoid_exit=True, why=f"{kind} 자리")
        self.log("이동", f"걷기 결과 {res.reason} ({res.secs:.0f}초)")
        found = res.reason == "stopped" and nv.find_prompt(self.grab()) == prompt
        if res.reason == "arrived":
            found = self.nudge_until(prompt, first="w" if prompt == "talk" else "s")
        if found and prompt == "enhance" and self.want_enhance(self.grab()):
            # 표시를 찾은 그 자리에서 바로 누른다 (20층: 다음 화면에서 옆의 상점 NPC "대화" 표시가 잡혀 상점부터 갔다)
            self.press_enhance(self.grab())
        self.npc_scanned = 0.0
        return True

    def stop_for(self, prompt: str):
        """걷다가 멈출 때: 필드가 아니게 됐거나 원하는 상호작용 표시("talk"/"enhance")가 떴을 때."""
        return lambda im: self.walk_stop(im) or nv.find_prompt(im) == prompt

    def label_goal(self, img: np.ndarray, box: tuple[int, int, int, int], below: int):
        """화면의 이름표(NPC, 강화머신)를 따라가는 목표. 이름표 아래 below px쯤이 그 물건의 발밑."""
        x, y, w, h = box
        tracker = nv.TemplateTracker(img, (x - 4, y - 4, x + w + 4, y + h + 4), threshold=0.5)
        last = [self.nav.odo.to_world((x + w / 2, y + h / 2 + below))]

        def fn(im: np.ndarray, ch: tuple[float, float], p: tuple[float, float]) -> Goal | None:
            c = tracker.update(im)
            if c is not None:
                last[0] = self.nav.odo.to_world((c[0], c[1] + below))
            return Goal(last[0], "label", arrive=50)

        return fn

    def marker_goal(self, kind: str):
        """화면 가장자리 표시(kind: "npc"/"exit") 방향으로 멀리. 표시가 사라지면 목표 없음."""
        def fn(im: np.ndarray, ch: tuple[float, float], p: tuple[float, float]) -> Goal | None:
            m = next((m for m in nv.find_markers(im, ch) if m.kind == kind), None)
            if m is None:
                return None
            ang = math.radians(nv.angle_of(m.icon[0] - 960, m.icon[1] - 540))
            c = self.nav.odo.to_world((960, 540))
            return Goal((c[0] + math.cos(ang) * 1000, c[1] - math.sin(ang) * 1000), f"{kind}_marker")

        return fn

    def _explore(self, why: str, avoid_exit: bool = False) -> None:
        """아직 안 가 본 쪽 중 막히지 않은 방향으로 조금 걷는다 (무작정 사방을 돌지 않는다).
        avoid_exit이면 출구 쪽으로는 안 간다 (아직 나가면 안 되는 방에서 출구에 들어가 버린 일이 있었다)."""
        # 둘러보다 새로 상호작용 표시가 뜨면 멈춘다. 이미 떠 있던 표시(말 건 NPC 옆)로는 안 멈춘다
        # (15층에서 회복 NPC 옆에 서서 0.5초짜리 둘러보기만 반복했다)
        before = nv.find_prompt(self.grab())
        res = self.nav.explore(lambda im: self.walk_stop(im) or nv.find_prompt(im) not in (None, before),
                               max_sec=2.5, avoid_exit=avoid_exit, why=why)
        if res.reason == "no_goal":
            self.nav.map.free.clear()  # 다 가 봤으면 걸어 본 칸 기록을 지우고 다시 둘러본다 (막힌 칸은 남김)
            time.sleep(0.5)

    # -- 층, 판 ---------------------------------------------------------------
    def wait_for_go(self) -> float:
        """한 층씩 모드: STEP_GO 파일이 생길 때까지 멈춘다. 멈춘 시간(초)을 돌려준다 (층 시간 초과에 안 세게)."""
        why, self.pause_pending = self.pause_pending, ""
        self.gi.release_all()
        STEP_GO.unlink(missing_ok=True)
        self.log("멈춤", f"한 층씩 모드: {why}. 다음으로 가려면 {STEP_GO.name} 파일", self.grab())
        t0 = time.monotonic()
        while not STEP_GO.exists():
            time.sleep(0.2)
        STEP_GO.unlink(missing_ok=True)
        paused = time.monotonic() - t0
        # 멈춘 동안 지난 시간은 전투/상점 기다리는 시간에서도 뺀다
        self.floor_changed_at += paused
        if self.shop_search_start is not None:
            self.shop_search_start += paused
        self.log("멈춤", f"이어서 감 ({paused:.0f}초 멈춤)")
        return paused

    def run_floor(self, timeout: float = 420) -> str:
        """한 층을 끝까지. 'next_floor' 또는 'restart'를 돌려준다.
        650원을 이긴 판은 층 하나에서 늦어지는 게 판을 버리는 것보다 나아서 더 기다린다."""
        start = time.monotonic()
        warned = False
        last_state = None
        while True:
            if self.pause_pending:
                start += self.wait_for_go()
            el = time.monotonic() - start
            if el > timeout * (1.7 if self.gamble_won else 1.0):
                break
            if el > 180 and not warned:
                warned = True
                self.log("층", f"이 층에서 3분째 ({'650원 이긴 판이라 12분까지' if self.gamble_won else '7분까지'} 기다림)")
            img = self.grab()
            d = self.stable.update(img)
            s = d.state
            if s != last_state and s != "transition":
                last_state = s
            if self.stable.last_raw == "loading":
                self.saw_loading()
            if s in ("transition", "loading", "unknown"):
                time.sleep(0.2)
                continue
            if s == "field":
                r = self.on_field(img)
                if r:
                    return r
            elif s == "card_select":
                self.on_cards(img, enhance=False)
            elif s == "enhance_select":
                self.on_cards(img, enhance=True)
            elif s == "npc_choice":
                self.on_choice(img)
            elif s == "dialog":
                self.gi.key("space")
                time.sleep(0.5)
            elif s in TAP_STATES:
                if s == "notes_gain" and not self.combat_done:
                    self.combat_done = True
                    self.combat_done_at = time.monotonic()
                    self.log("전투", "소리 획득 -> 전투 끝")
                    if self.run_tracked:
                        self.save_state()
                if s == "explore_done" and self.run.floor == 20:
                    # 20층 출구는 로딩 대신 "탐색 완료"로 넘어가서 출구 자리가 한 번도 안 적혔다. 여기서 적는다
                    # (출구를 알면 바닥 무늬를 출구로 알고 강화머신/상점 쪽으로 못 가는 일이 줄어든다)
                    self.nav.left_by_exit()
                self.gi.click(*EMPTY_SPOT)
                time.sleep(0.7)
            elif s == "shop":
                self.on_shop(img)
            elif s == "bag":
                self.gi.key("esc")  # 가방이 열려 있으면 닫는다
                time.sleep(0.8)
            elif s == "shop_buy":
                self.gi.key("space")  # 구매
                time.sleep(1.0)
            elif s == "esc_map":
                self.gi.key("esc")
                time.sleep(0.5)
            elif s == "record_result":
                return "tower_done"  # 20층까지 끝나고 기록 화면
            else:
                self.log("화면", f"예상 못 한 화면 {s}, 기다림")
                time.sleep(1.0)
        self.log("층", "시간 초과")
        return "restart"

    def restart(self, reason: str) -> None:
        """포기 → 분해 → 다시 출발해서 1층 필드까지."""
        self.log("재시작", reason, self.grab())
        self.restart_pending = ""
        self.start_from_menu(give_up=True)

    def lobby_step(self, img: np.ndarray) -> str | None:
        """메인 화면에서 별의 탑 출발 화면까지 한 걸음 (새벽 5시 데이터 업데이트 뒤 메인 화면으로 튕긴다).
        메인 "출발" → "별의 탑 탐색" → 탑 고르기(프리셋 속성 탑) → "별의 탑 들어가기" → 난이도 선택(마지막 난이도 그대로).
        중간의 출석/택배 보상 창은 빈 곳을 누르거나 닫는다. 한 일을 돌려준다 (모르는 화면이면 None)."""
        lines = self.ocr.read(img, (0, 0, 1920, 1080), scale=1.0)
        t = "".join(l.text for l in lines).replace(" ", "")
        step = lobby_action(t)
        if step is None:
            return None
        if step == "tower":
            want = TOWER_NAME.replace(" ", "")
            box = next((l.box for l in lines if fuzz.ratio(l.text.replace(" ", ""), want) >= 70 and l.box[0] < 960), None)
            if box:
                self.gi.click(box[0] + box[2] // 2, box[1] + box[3] // 2)
                time.sleep(1.2)
            # 고른 탑이 맞는지 오른쪽 사진 아래 이름으로 확인하고서 들어간다
            # (11:27: 고르는 클릭이 안 먹어서 전에 골라져 있던 물과 그림자 탑으로 들어갔다)
            chosen = self.ocr.text(self.grab(), TOWER_PANEL_NAME).replace(" ", "")
            if fuzz.partial_ratio(want, chosen) < 70:
                self.log("메뉴", f"고른 탑이 {chosen!r}라서 아직 안 들어감 ({TOWER_NAME}을 다시 고름)")
                time.sleep(0.5)
                return step
            self.gi.click(*LOBBY_BUTTONS["enter_tower"])
        else:
            self.gi.click(*LOBBY_BUTTONS[step])
        self.log("메뉴", f"메인 화면 쪽: {step}")
        time.sleep(1.5)
        return step

    def start_from_menu(self, give_up: bool = False) -> None:
        deadline = time.monotonic() + 300
        departed = False
        gave_up = False  # 이번에 포기를 눌렀는지. 안 눌렀는데 기록 화면이면 끝까지 간 판이라 저장한다
        last = None
        repeats = 0
        while time.monotonic() < deadline:
            img = self.grab()
            s = self.det.detect(img).state
            if s != last:
                self.log("메뉴", f"화면 {s}")  # 재시작이 막힐 때 어디서 막혔는지 보려고
                last, repeats = s, 0
            else:
                repeats += 1
                if repeats in (15, 40):
                    self.log("메뉴", f"화면 {s}에서 {repeats}번째", img)
            if s == "field":
                if departed:
                    self.run = RunState(floor=1)
                    self.run_tracked = True
                    self.new_floor(1, "전투")
                    self.gamble_won = False
                    self._gamble_verify = None
                    self.note_needs = None  # 가진 소리 개수는 판마다 처음부터 (쓰는 협주스킬 수는 note_users에 남는다)
                    self.log("시작", "새 판 1층")
                    return
                if give_up:
                    self.gi.key("esc")
                else:
                    self.log("시작", "출발 전인 줄 알았는데 탑 안 필드라서 그대로 이어 감")
                    return
            elif s == "esc_map":
                if give_up:
                    self.gi.key("q")
                    gave_up = True
                else:
                    self.gi.key("esc")
            elif s == "notice":
                text = self.ocr.text(img, (300, 250, 1620, 750))
                if "분해" in text:
                    self.gi.click(*BTN_CONFIRM)
                elif "업데이트" in text or repeats >= 3:
                    # 새벽 5시 "데이터가 업데이트되었습니다. 메인 화면으로 돌아갑니다." 창은 Space로 안 닫힌다
                    # (2026-10-02 05:15에 3분 동안 Space만 눌렀다). 가운데 "확인"을 누른다
                    self.gi.click(*rc.NOTICE_OK)
                else:
                    self.gi.key("space")
            elif s == "unknown" and repeats >= 2:
                self.lobby_step(img)  # 메인 화면/출석 보상 등: 별의 탑 출발 화면까지 돌아간다
            elif s in TAP_STATES:
                self.gi.click(*EMPTY_SPOT)
            elif s == "record_result":
                if gave_up or give_up:  # 포기하려고 들어온 길이면 이 기록은 포기한 판
                    self.gi.click(*BTN_TRASH)  # 중간에 포기한 기록은 분해
                else:
                    self.save_record()  # 끝까지 간 판의 기록 (봇을 기록 화면에서 켰을 때)
            elif s == "difficulty_select":
                if (self.log_path.parent / "pause_at_menu").exists():
                    # 사람이(또는 Claude가) 메뉴에서 할 일이 있을 때: logs/pause_at_menu 파일을 만들어 두면 여기서 멈춘다
                    (self.log_path.parent / "pause_at_menu").unlink(missing_ok=True)
                    raise Stop("출발 화면에서 멈춤 요청 (logs/pause_at_menu)")
                tower = self.ocr.text(img, TOWER_TITLE_BOX).replace(" ", "")
                if fuzz.partial_ratio(TOWER_NAME.replace(" ", ""), tower) < 70:
                    # 다른 탑의 출발 화면이다 (11:27 물과 그림자 탑으로 출발했다): 뒤로 가서 탑을 다시 고른다
                    self.log("메뉴", f"출발 화면의 탑이 {tower!r}라서 뒤로 가서 {TOWER_NAME}을 고름", img)
                    self.gi.click(*BTN_BACK)
                    time.sleep(1.5)
                    continue
                self.gi.click(*BTN_DEPART)
            elif s in ("team_setup", "record_combo"):
                self.gi.click(*BTN_NEXT)
                departed = departed or s == "record_combo"
            elif s in ("npc_choice", "dialog", "shop", "shop_buy", "bag", "record_detail", "record_manage", "filter"):
                # 가방/기록 화면이 열린 채 봇이 켜졌을 때도 ESC로 닫는다 (23:10: 가방에서 40번 제자리걸음)
                self.gi.key("esc")
            elif s in ("card_select", "enhance_select"):
                self.gi.key("space")
            time.sleep(0.9)
        raise Stop("재시작이 3분 안에 안 끝남")

    # -- 기록 저장 -------------------------------------------------------------
    def save_record(self) -> None:
        """탑이 끝난 기록 화면: 점수를 장부에 적고, 이름을 저장한 시각으로 바꾸고, "기록 저장".
        (사용자 규칙 2026-10-02. 흐름은 stella_auto/record.py 맨 위)"""
        name = rc.record_name()
        img = self.grab()
        self.save_full(img, "record_potentials")  # 나중에 카드 레벨 숫자 읽기를 만들 자료
        # 1) 레코드 스킬 탭에서 평점 레벨, 협주스킬 레벨
        self.gi.click(*rc.TAB_SKILLS)
        time.sleep(1.0)
        skills = self.grab()
        self.save_full(skills, "record_skills")
        level = rc.read_record_level(skills, self.ocr)
        ensemble = rc.read_ensemble_levels(skills, self.ocr)
        self.gi.click(*rc.TAB_POTENTIALS)
        time.sleep(0.6)
        # 2) 점수 (잠재 레벨은 봇이 판 내내 기억한 것)
        gd = default_gamedata()
        result = score_record(self.preset, RecordResult(
            potential_levels=dict(self.run.owned), ensemble_levels=[v or 0 for v in ensemble], notes={},
            record_level=level or 0))
        pots = {gd.potentials[pid].name if pid in gd.potentials else str(pid): lv for pid, lv in self.run.owned.items()}
        if self.run_tracked and result.discard_reasons:
            # 사용자 규칙 (2026-10-02 11시): 필수가 0레벨인 조건 미달 도자기는 저장하지 않고 바로 깬다
            self.discard_record(img, level, ensemble, pots, result)
            return
        # 3) 이름 바꾸기
        renamed = self.rename_record(name)
        # 4) 기록 저장 → "기록 저장 성공" → 확인
        self.gi.click(*rc.SAVE_BUTTON)
        ok = False
        for _ in range(8):
            time.sleep(0.8)
            im = self.grab()
            if self.det.detect(im).state == "notice":
                text = self.ocr.text(im, rc.NOTICE_TEXT_BOX).replace(" ", "")
                ok = "저장성공" in text or "성공" in text
                if not ok:
                    self.log("기록", f"저장 안내가 예상과 다름: {text!r} (멈춤)", im)
                    raise Stop(f"기록 저장 중 모르는 안내: {text}")
                self.gi.click(*rc.NOTICE_OK)
                break
        rec = rc.SavedRecord(name=name if renamed else "이름 없는 기록", saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                             floor_reached=self.run.floor, record_level=level, ensemble_levels=ensemble,
                             potentials=pots, score=round(result.total), discard_reasons=result.discard_reasons,
                             tracked=self.run_tracked, note="" if ok else "저장 성공 안내를 못 봄")
        rc.append_ledger(self.log_path.parent / "records.jsonl", rec)
        self.log("기록", f"저장 {rec.name}: 평점 {level}, 협주 {ensemble}, 점수 {rec.score}"
                         f"{' (버릴 조건: ' + ', '.join(result.discard_reasons) + ')' if result.discard_reasons else ''}"
                         f"{'' if self.run_tracked else ' (중간에 켠 판이라 잠재 레벨이 빠져 있음)'}", img)
        self.run_tracked = False
        time.sleep(1.5)

    def discard_record(self, img: np.ndarray, level, ensemble, pots: dict, result) -> None:
        """기록 화면에서 (잠겨 있으면 왼쪽 아래 "잠김"을 눌러 풀고) 휴지통(분해) → "분해" 안내 → 확인.
        끝까지 간 판의 기록은 처음부터 잠겨 있어서 휴지통을 눌러도 안내가 안 뜬다 (12:26 분해 실패 → 저장됨).
        장부에는 분해했다고 적는다."""
        done = False
        for attempt in range(2):
            lock = self.ocr.text(self.grab(), rc.LOCK_TEXT_BOX).replace(" ", "")
            if "잠김" in lock:
                self.gi.click(*rc.LOCK_BUTTON)
                time.sleep(1.0)
                after = self.ocr.text(self.grab(), rc.LOCK_TEXT_BOX).replace(" ", "")
                self.log("기록", f"잠금 풀기: {lock!r} -> {after!r}")
            self.gi.click(*BTN_TRASH)
            for _ in range(8):
                time.sleep(0.8)
                im = self.grab()
                if self.det.detect(im).state == "notice":
                    text = self.ocr.text(im, (300, 250, 1620, 750)).replace(" ", "")
                    if "분해" in text:
                        self.gi.click(*BTN_CONFIRM)
                        done = True
                        break
                    self.log("기록", f"분해 안내가 예상과 다름: {text!r} (멈춤)", im)
                    raise Stop(f"기록 분해 중 모르는 안내: {text}")
            if done:
                break
        rec = rc.SavedRecord(name="(분해)", saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                             floor_reached=self.run.floor, record_level=level, ensemble_levels=ensemble,
                             potentials=pots, score=round(result.total), discard_reasons=result.discard_reasons,
                             tracked=self.run_tracked, note="조건 미달이라 분해" if done else "분해 안내를 못 봄")
        rc.append_ledger(self.log_path.parent / "records.jsonl", rec)
        self.log("기록", f"분해 (조건 미달: {', '.join(result.discard_reasons)}): 평점 {level}, 협주 {ensemble}, "
                         f"점수 {rec.score}{'' if done else ' - 분해 안내를 못 봄'}", img)
        self.run_tracked = False
        time.sleep(1.5)

    def rename_record(self, name: str) -> bool:
        """기록 화면 연필 → 이름 칸 → 이름 입력 → 확인. 바뀐 이름을 읽어 확인한다."""
        for attempt in range(3):
            self.gi.click(*((548, 108) if attempt % 2 == 0 else rc.PENCIL))
            opened = False
            for _ in range(6):  # 창이 뜨는 데 시간이 걸릴 수 있다
                time.sleep(0.5)
                im = self.grab()
                if "이름" in self.ocr.text(im, rc.RENAME_TITLE_BOX).replace(" ", "") or                         "이름입력" in self.ocr.text(im, (700, 440, 1220, 600)).replace(" ", ""):
                    opened = True
                    break
            if not opened:
                self.log("기록", f"이름 변경 창이 안 열림 ({attempt + 1}/3)", im)
                continue
            self.gi.click(*rc.NAME_FIELD)
            time.sleep(0.6)
            for _ in range(14):
                self.gi.key("backspace", hold=0.03)
                time.sleep(0.03)
            self.gi.type_text(name)
            time.sleep(0.4)
            self.gi.click(*rc.RENAME_OK)
            time.sleep(1.2)
            got = re.sub(r"[^0-9]", "", self.ocr.text(self.grab(), rc.NAME_BOX))
            if got == name.replace("_", ""):
                return True
            self.log("기록", f"이름이 {got!r}로 읽힘 (원하는 이름 {name})")
        return False

    # -- 판 상태 저장 (봇을 껐다 켜도 이어 가게) -----------------------------------
    def state_path(self) -> Path:
        return self.log_path.parent / "run_state.json"

    def save_state(self) -> None:
        r = self.run
        data = {"saved_at": time.time(), "floor": r.floor, "gold": r.gold, "owned": {str(k): v for k, v in r.owned.items()},
                "lv3_new_taken": r.lv3_new_taken, "lv2_new_taken": r.lv2_new_taken, "rerolls_early": r.rerolls_early,
                "gamble_won": self.gamble_won, "run_tracked": self.run_tracked, "combat_done": self.combat_done,
                # 같은 층에서 봇을 껐다 켜도 강화머신 횟수/상점 진행을 이어받는다 (23:04 19층: 횟수가 0으로
                # 리셋돼 180원까지 규칙인데 실제론 240원짜리를 눌렀다)
                "enhance_count": self.enhance_count, "shop_done": self.shop_done,
                "shop_rerolled": self.shop_rerolled, "shop_reopens": self.shop_reopens,
                "plus2_taken": r.plus2_taken}
        try:
            tmp = self.state_path().with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(self.state_path())
        except OSError:
            pass

    def restore_state(self, floor: int) -> bool:
        """봇을 다시 켰을 때: 몇 분 안에 저장한 같은 판 상태가 있으면 이어받는다 (층이 같거나 하나 위)."""
        try:
            d = json.loads(self.state_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if time.time() - d.get("saved_at", 0) > 15 * 60 or floor not in (d["floor"], d["floor"] + 1)                 or not d.get("run_tracked"):
            return False
        self.run.owned = {int(k): v for k, v in d["owned"].items()}
        self.run.lv3_new_taken = d["lv3_new_taken"]
        self.run.lv2_new_taken = d["lv2_new_taken"]
        self.run.rerolls_early = d["rerolls_early"]
        self.run.plus2_taken = d.get("plus2_taken", 0)
        self.gamble_won = d["gamble_won"]
        self.run.keep_after_gamble = self.gamble_won
        self.run_tracked = True
        if floor == d["floor"] and d.get("combat_done"):
            self.combat_done = True  # 같은 방에서 전투가 이미 끝났다 (40초 기다리지 않게)
        if floor == d["floor"]:
            self.enhance_count = d.get("enhance_count", self.enhance_count)
            self.shop_done = d.get("shop_done", self.shop_done)
            self.shop_rerolled = d.get("shop_rerolled", self.shop_rerolled)
            self.shop_reopens = d.get("shop_reopens", self.shop_reopens)
        self.log("시작", f"저장해 둔 판 상태를 이어받음 ({d['floor']}층, 잠재 {len(self.run.owned)}개, "
                       f"650원 {'성공' if self.gamble_won else '아직'})")
        return True

    def time_up(self) -> bool:
        return self.deadline is not None and time.monotonic() > self.deadline

    def read_floor_from_map(self) -> None:
        """탑 안에서 시작할 때: ESC 지도의 "3/20층 전투의 방"으로 지금 층을 안다.
        카드 고른 직후 같은 때 ESC를 누르면 지도가 안 열리기도 해서 세 번까지 해 본다."""
        for attempt in range(3):
            time.sleep(0.5)
            if self.det.detect(self.grab()).state != "field":
                return  # 다른 화면이 떴다 (카드 선택 등). 그 화면을 먼저 처리하고 다음에 다시
            self.gi.key("esc")
            time.sleep(1.5)  # 지도가 지금 층 쪽으로 움직이는 동안 기다린다
            img = self.grab()
            st = self.det.detect(img).state
            if st != "esc_map":
                self.log("층", f"ESC를 눌렀는데 지도가 아니라 {st} 화면 ({attempt + 1}/3)", img)
                if st == "field":
                    continue
                return  # 다른 화면 (메뉴 등): 그 화면부터 처리하고 다음에 다시
            # 지금 층 표시("2/20층 선택의 방")는 지도에서 지금 방 옆에 붙어서 아래쪽 끝에 있기도 하다
            text = self.ocr.text(img, (560, 40, 1880, 1075)).replace(" ", "")
            m = TITLE_FLOOR_RE.search(text)
            floor_read = int(m.group(1) or m.group(2)) if m else None
            if floor_read is None and "마왕의방" in text:
                # 20층 보스방은 지도에 "마왕의 방"으로 나온다: 층 숫자를 못 읽어도 20층 확정
                # (01:37: 층을 몰라서 마지막 상점을 안 들르고 돈을 남긴 채 나갔다)
                floor_read = 20
                self.log("층", "지도의 '마왕의 방'으로 20층 확정")
            if floor_read is None:
                self.log("층", f"지도 글자에서 층을 못 찾음: {text[:60]!r} ({attempt + 1}/3)", img)
            self.gi.key("esc")
            time.sleep(0.6)
            if floor_read is not None:
                self.new_floor(floor_read, next((r for r in ROOM_NAMES if r in text), ""), confirmed=True)
                self.joined_midway = True  # 이 층 전투가 이미 끝났을 수 있다
                # 1~2층이면 판 시작이나 다름없다: 650원 도박/리롤 규칙을 그대로 쓴다 (빠진 잠재는 한두 장)
                self.run_tracked = self.run.floor <= 2
                self.restore_state(self.run.floor)
                self.nav.new_room(self.run.floor, self.room, ROOM_BY_FLOOR.get(self.run.floor + 1, ""), at_entrance=False)
                return
        # 못 읽으면 계속 ESC를 누르지 않게 일단 넘어가고, 다음 방 제목의 층 번호를 그대로 믿는다
        self.floor_known = True
        self.floor_uncertain = True
        self.log("층", f"지도에서 층을 못 읽음, 다음 방 제목으로 층을 정함 (지금은 {self.run.floor}층으로 둠)")

    def play(self, max_floors: int) -> None:
        """게임 창이 잠깐 가려지거나 다른 창이 앞으로 와도 꺼지지 않고, 기다렸다가 화면을 다시 보고 이어 간다."""
        fails = 0
        crashes = 0
        while True:
            try:
                self._play(max_floors)
                return
            except NotFocusedError as e:
                self.gi.release_all()
                fails += 1
                if fails >= 20:
                    raise Stop(f"게임 창을 계속 앞으로 못 가져옴 ({e})")
                self.log("포커스", f"{e}. 3초 뒤 다시 ({fails}/20)")
                time.sleep(3)
            except Stop:
                raise
            except Exception as e:  # 밤새 돌다 생긴 예상 못 한 오류로 봇이 통째로 꺼지지 않게
                self.gi.release_all()
                crashes += 1
                import traceback
                self.log("오류", f"{type(e).__name__}: {e} ({crashes}/10)", self.grab(),
                         traceback=traceback.format_exc())
                if crashes >= 10:
                    raise
                time.sleep(3)

    def _play(self, max_floors: int) -> None:
        s = self.det.detect(self.grab()).state
        for _ in range(60):  # 로딩/화면 전환 중에 켜졌으면 화면이 자리 잡을 때까지 (메뉴로 착각하지 않게)
            if s not in ("loading", "transition", "unknown"):
                break
            time.sleep(0.5)
            s = self.det.detect(self.grab()).state
        self.log("시작", f"봇 시작, 지금 화면 {s}")
        if s in ("field", "card_select", "enhance_select", "npc_choice", "dialog", "shop", "shop_buy",
                 "esc_map", "notes_gain", "ensemble_up", "tap_continue"):
            pass  # 탑 안: 필드에 나오면 on_field가 ESC 지도로 층을 확인한다
        else:
            self.start_from_menu(give_up=False)
        while True:
            r = self.run_floor()
            if r == "tower_done":
                self.save_record()
                self.runs_done += 1
                self.log("끝", f"탑 {self.runs_done}판째 끝 (그사이 재시작 {self.restarts}번)")
                if self.stop_after_tower or (self.max_runs and self.runs_done >= self.max_runs) or self.time_up():
                    return
                self.start_from_menu(give_up=False)
            elif r == "next_floor":
                if self.run.floor - 1 >= max_floors:
                    self.log("끝", f"{max_floors}층까지 넘김")
                    return
            else:
                self.restarts += 1
                if self.time_up():
                    self.log("끝", f"정한 시간이 지나서 멈춤 (탑 {self.runs_done}판, 재시작 {self.restarts}번)")
                    return
                self.restart(self.restart_pending or "층 실패")


def white_fraction(img: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """글자 상자 안에서 흰색(채도 낮고 밝은) 픽셀 비율."""
    x, y, w, h = box
    hsv = cv2.cvtColor(img[max(0, y):y + h, max(0, x):x + w], cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 1] < 50) & (hsv[..., 2] > 210)).mean()) if hsv.size else 0.0


def is_green_text(img: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """OCR 글자 상자 안의 밝은 글자색이 초록인지 (NPC 종류 글자) 흰색인지 (이름)."""
    x, y, w, h = box
    hsv = cv2.cvtColor(img[max(0, y):y + h, max(0, x):x + w], cv2.COLOR_BGR2HSV)
    bright = hsv[hsv[..., 2] > 150]
    if len(bright) == 0:
        return False
    green = (bright[:, 0] > 30) & (bright[:, 0] < 90) & (bright[:, 1] > 90)
    return float(green.mean()) > 0.3


def find_npc_event(img: np.ndarray, threshold: float = 0.65) -> tuple[float, float] | None:
    """이벤트가 있는 NPC 머리 위 초록 "사건" 글자의 가운데."""
    from .screen import TEMPLATE_DIR

    if not hasattr(find_npc_event, "_t"):
        find_npc_event._t = cv2.imdecode(np.fromfile(str(TEMPLATE_DIR / "npc_event.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
    t = find_npc_event._t
    res = cv2.matchTemplate(cv2.cvtColor(img[:960], cv2.COLOR_BGR2GRAY), t, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < threshold:
        return None
    return loc[0] + t.shape[1] / 2, loc[1] + t.shape[0] / 2


# 별의 탑 퀴즈 정답지 (sstoy src/modules/app-summary.ts STAR_TOWER_QA_DATA. 사용자가 준 공략표 12개와 2026-10-02 대조:
# 전부 일치. NPC 이름과 질문은 일부러 매칭하지 않는다 — 포셔가 베르너 질문을 할 수도 있다, 사용자 설명)
QUIZ = {
    "음...... 별의 탑이 가장 좋아하는 숫자는 뭘까?": "3? 항상 그렇게 선택했으니까......",
    "몇시까지 버텨야 '밤샘' 이라고 생각해?": "12시?",
    "자, 시험이야. 2의 10제곱은 얼마일까?": "1024?",
    "자, 시험이야. 정육면체는 몇 개의 면이 있을까?": "6개?",
    "한번 맞혀봐...... 난 어떤 여행자와의 대화를 더 좋아할까?": "큰 꿈을 가진 사람.",
    "'큰 뜻을 품는다'는건 뭐라고 생각해?": "계획을 잘 세우고, 실행해야 해.",
    "욕망에 충실하다는건...... 어떤 걸 말하는 것 같아?": "현재를 즐기자!",
    "자, 시험이야. 한 옥타브엔 몇 개의 음이 있을까?": "12개?",
    "한번 맞혀봐. 난 어떤 여행가를 더 좋아할까?": "욕망에 충실한 사람.",
    "뭘 먹는 게 건강에 더 좋을까?": "야채를 많이 먹으라고?",
    "이 중에서 어떤 게 건강에 좋을까?": "균형적인 음식?",
    "이 중에서 어떤 걸 줄이는 게 건강에 좋을까?": "오래 앉아 있지 말라고?",
}
_QUIZ_KEYS = [q.replace(" ", "") for q in QUIZ]
_QUIZ_ANSWERS = list(QUIZ.values())


def is_quiz(question: str) -> bool:
    q = question.replace(" ", "")
    return bool(q) and process.extractOne(q, _QUIZ_KEYS, scorer=fuzz.ratio, score_cutoff=75) is not None


def quiz_answer(question: str, options: list[tuple[str, str]]) -> int | None:
    """퀴즈 질문이면 정답 보기 번호. 질문이 정답지와 비슷하지 않으면 None."""
    q = question.replace(" ", "")
    if not q:
        return None
    m = process.extractOne(q, _QUIZ_KEYS, scorer=fuzz.ratio, score_cutoff=75)
    if m:
        answer = _QUIZ_ANSWERS[m[2]].replace(" ", "")
        scores = [fuzz.ratio(t.replace(" ", ""), answer) for t, _ in options]
        best = max(range(len(options)), key=lambda i: scores[i])
        return best if scores[best] >= 60 else None
    # 같은 퀴즈가 말투만 다르게 나와 질문이 덜 비슷할 때 (22:42 10층: "한번 맞혀봐...... 난 어떤 여행자와의
    # 대화를 더 좋아할까?"가 "맞혀보시죠...... 저는 어떤 여행가와..."로 나와 71 < 75):
    # 보상 글 "정답을 선택하면 ..."이 퀴즈라는 표시니까, 보기 중에 정답지의 답이 있으면 그걸 고른다
    if not any("정답" in e or "정담" in e for _, e in options):
        return None
    flat = [a.replace(" ", "") for a in _QUIZ_ANSWERS]
    hits = []  # (보기 번호, 정답지 번호)
    for i, (t, _) in enumerate(options):
        a = process.extractOne(t.replace(" ", ""), flat, scorer=fuzz.ratio, score_cutoff=75)
        if a:
            hits.append((i, a[2]))
    if not hits:
        return None
    # 정답지의 답이 여럿 보이면 (여행자/여행가 퀴즈는 답이 다르다) 질문이 더 비슷한 쪽
    return max(hits, key=lambda h: fuzz.ratio(q, _QUIZ_KEYS[h[1]]))[0]


def choose_option(options: list[tuple[str, str]], floor: int, question: str = "",
                  note_users: dict[int, int] | None = None,
                  note_have: dict[int, int] | None = None,
                  note_needs: "NoteNeeds | None" = None) -> tuple[int, str]:
    """NPC 선택지 고르기 (docs/tower-rules.md "NPC 선택지 고르기"). (번호, 규칙 이름).
    note_users: 소리 종류 -> 그 소리를 쓰는 협주스킬 수, note_have: 가진 개수 (공짜 소리 고르기에 씀)."""
    q = quiz_answer(question, options)
    if q is not None:
        return q, "퀴즈 정답지"
    # OCR이 "획득"을 "획티", "획듣"처럼 읽는다 (운명과 흥정: 공짜 선택지를 놓치고 50원짜리를 골랐다)
    texts = [re.sub(r"획.", "획득", (t + " " + e).replace(" ", "")) for t, e in options]
    effects = [re.sub(r"획.", "획득", e.replace(" ", "")) for _, e in options]
    if is_quiz(question) and len(set(re.sub(r"[^가-힣]", "", e) for e in effects)) > 1 and \
            not all("정답" in e or "정담" in e for e in effects if e):
        # 퀴즈인데 보기마다 보상이 다르다 ("A랑 B 중에 뭐 가질래"): 사용자가 정한다 (2026-10-02)
        return 0, "퀴즈: 보상 고르기 (알림)"
    # 30원 받기 ("300 획득": 코인 그림을 0으로 읽는다). "30%"나 "1300"은 아니다
    thirty = [i for i, e in enumerate(effects) if re.search(r"(?<![\d%])300?획득", e)]
    for i, s in enumerate(texts):
        if "650" in s and floor <= GAMBLE_LAST_FLOOR:
            return i, "650원 도박"
    # 사용자 규칙 (2026-10-02 14시, docs/choices-list.md 번호)
    hp = [i for i, e in enumerate(effects)
          if "HP" in e and ("소모" in e or "차감" in e) and ("잠재력" in e or "획득" in e) and "회복" not in e]
    if hp:
        rare = [i for i in hp if "희귀" in effects[i]]
        return (rare[0], "HP 내고 희귀 잠재력 (15번)") if rare else (hp[0], "HP 내고 잠재력/돈")
    for i, s in enumerate(texts):
        if "33%" in s and "잠재력" in s and "HP" not in s:
            return i, "33% 잠재력"
    for i, s in enumerate(texts):
        # "코인으로 바꿔줘 / 랜덤 소리 5개 소모, 150 획득" 처럼 "판다"는 말이 없을 때도 있다.
        # 소리를 내고(소모) 돈을 받는(획득) 쪽만. "150 소모, 랜덤 소리 5개 획득"(돈 내고 소리 사기)과 헷갈렸다 (19:18 1층)
        sell = re.search(r"소리[^,]*소모", s) and not re.search(r"150\s*0?\s*소모", s)
        if "소리" in s and "150" in s and ("팔" in s or "판매" in s or sell):
            return i, "소리 팔고 150원"
    if thirty and any("소리" in e and "소모" in e and ("90" in e or "140" in e) for e in effects):
        return thirty[0], "소리 사지 말고 30원"
    change = [i for i, e in enumerate(effects) if "변화" in e]
    if change:
        i = change[0]
        if "잠재력" in effects[i]:
            return i, "돈·잠재력 랜덤 변화 (3번, 모든 층)"
        if floor <= 6 or not thirty:
            return i, "돈·HP 랜덤 변화 = 100원 쪽 (2번, 6층 이하)"
        return thirty[0], "30원 (2번, 7층 이상)"
    trial = [i for i, e in enumerate(effects) if "소실" in e and "소모" not in e]
    if trial:
        # 시약 (4번): 사용자 2026-10-02 21시 "항상 2번(30원) 고르지 말고 1번(시약) 골라"
        return trial[0], "시약 마시기 (4번)"
    for i, e in enumerate(effects):
        if "회복" in e and "확률" in e and ("잠재력" in e or "소리" in e) and "소모" not in e:
            return i, "호의: 50% 잠재력/소리 (10~12번)"
    for i, e in enumerate(effects):
        if "확률" in e and "획득" in e and "잠재력" in e and not any(w in e for w in ("소모", "차감", "회복", "HP")):
            return i, "작은 내기: 50% 쪽 (14번, 알림)"
    for i, e in enumerate(effects):
        if "소리" in e and "소모" in e and "잠재력" in e:
            return i, "소리 내고 희귀 잠재력 (17번)"
    buy = [i for i, e in enumerate(effects) if "소모" in e and "잠재력" in e and "HP" not in e]
    if buy:
        def cost(e: str) -> int:
            m = re.search(r"(\d+)을?소모", e)
            return int(m.group(1)) if m else 10 ** 6
        i = min(buy, key=lambda k: cost(effects[k]))
        return i, "돈 내고 잠재력, 가장 싼 것 (13·18번)"
    loss = ("소모", "차감", "소실", "감소", "잃", "변화")  # "랜덤 변화", "HP 30% 소실"도 잃을 수 있는 것
    pick = free_note_choice(texts, note_users, loss, note_have, note_needs)
    if pick is not None:
        return pick
    for i, s in enumerate(texts):
        if ("획득" in s or "회복" in s) and not any(w in s for w in loss):
            return i, "모름: 잃는 것 없는 쪽 (기록)"
    for i, (t, e) in enumerate(options):
        if not e.strip() and t.strip():
            # 효과가 없는 보기 ("지금은 안 돼", "됐어."): 아무것도 안 잃는다
            return i, "모름: 아무 일 없는 쪽 (기록)"
    if texts and all(any(w in s for w in ("소모", "차감")) for s in texts):
        # 전부 돈/소리를 내는 선택지 (예: 소리 10개를 140원/90원에): 사지 않고 ESC로 나간다
        return -1, "모름: 전부 돈이 들어서 안 고름 (기록)"
    return 0, "모름: 첫 번째 (기록)"


def lobby_action(t: str) -> str | None:
    """메인 화면 쪽 화면 글자(공백 뺀 것)로 할 일. 탑 안/출발 화면 글자와 겹치지 않는 말만 쓴다.
    OCR이 "탑"을 자주 빼먹고("별의탐색") 큰 "출발" 글자를 못 읽어서 주변 글자로 알아본다."""
    if "터치" in t:
        return "tap"
    if "출석" in t:
        return "attendance_close"
    if "들어가기" in t and ("원초적미로" in t or "폭풍과번개" in t or "물과그림자" in t):
        return "tower"
    if "현상수배" in t or "재앙의전선" in t or "연합토벌" in t:
        return "tower_hub"
    if ("레코드" in t and "모집" in t) or ("여행가" in t and "스토리" in t):
        return "lobby_depart"
    return None


def free_note_choice(texts: list[str], note_users: dict[int, int] | None,
                     loss: tuple[str, ...], note_have: dict[int, int] | None = None,
                     note_needs: "NoteNeeds | None" = None) -> tuple[int, str] | None:
    """보기가 전부 공짜 소리 ("강공의 소리 5개 획득 / 행운의 소리 5개 획득 / 랜덤 소리 5개 획득").

    가방에서 협주스킬별 필요량을 읽어 뒀으면(note_needs): 5개를 받았을 때 활성화/레벨업 진행이
    실제로 몇 칸 차는지(gain)가 큰 소리 (사용자 2026-10-03: 강공 +5는 꿈의 날개 한 칸뿐이지만
    행운 +5는 협주 둘을 5칸씩 채운다). 전부 0칸이면 랜덤 (필요량 넘긴 소리만 남은 것).
    아직 못 읽었으면 예전 규칙: 많이 쓰는 것 > 덜 가진 것, 1개만 쓰는데 40개 이상이면 랜덤."""
    have = note_have or {}
    if not note_users or len(texts) < 2:
        return None
    kinds: list[tuple[int, int | None, bool]] = []
    for i, s in enumerate(texts):
        if "소리" not in s or "획득" not in s or any(w in s for w in loss):
            return None
        kinds.append((i, type_from_name(s), "랜덤" in s))
    if note_needs is not None and note_needs.needs:
        ranked = [(note_needs.gain(t), note_users.get(t, 0), -have.get(t, 0), -i, i, t)
                  for i, t, rnd in kinds if t is not None and not rnd]
        if ranked and max(ranked)[0] > 0:
            g, n, _, _, i, t = max(ranked)
            return i, f"공짜 소리: {NOTE_NAMES[t]} +5개면 협주 진행 {g}칸 (협주 {n}개가 씀)"
        rnd = [i for i, _, r in kinds if r]
        if rnd:
            return rnd[0], "공짜 소리: 필요량 찬 소리뿐이라 랜덤"
    named = [(note_users.get(t, 0), -have.get(t, 0), -i, i) for i, t, rnd in kinds
             if t is not None and not rnd and not (note_users.get(t, 0) == 1 and have.get(t, 0) >= 40)]
    if named and max(named)[0] > 0:
        n, _, _, i = max(named)
        return i, f"공짜 소리: 협주 {n}개가 쓰는 {NOTE_NAMES[kinds[i][1]]}"
    rnd = [i for i, _, r in kinds if r]
    if rnd:
        return rnd[0], "공짜 소리: 쓰는 소리가 없어서 랜덤"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stella_auto.runner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", type=Path, help="프리셋 JSON (python -m stella_auto.preset ... -o 로 만든 것)")
    ap.add_argument("--floors", type=int, default=20, help="시험용: 이만큼 층을 넘기면 멈춤 (기본 20 = 끝까지)")
    ap.add_argument("--runs", type=int, default=0, help="탑을 이만큼 끝내면(기록 저장) 멈춤. 0이면 계속")
    ap.add_argument("--hours", type=float, default=0, help="이만큼 시간이 지나면 다음 판을 시작하지 않고 멈춤. 0이면 계속")
    ap.add_argument("--no-gamble", action="store_true", help="시험용: 650원 도박 조건 없이 계속 올라간다")
    ap.add_argument("--step", action="store_true",
                    help="한 층씩: 650원을 이기면 멈추고, 그 뒤 층마다 멈춘다. logs/step.go 파일을 만들면 다음 층까지")
    ap.add_argument("--step-pause", type=float, default=0.0,
                    help="650원 이긴 판에서 층을 떠나기 전에 이만큼(초)만 멈춘다 (지켜보기용 가벼운 모드)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # 봇은 한 번에 하나만. 두 개가 같이 돌면 서로 키를 눌러서 엉망이 된다
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    mutex = kernel32.CreateMutexW(None, True, r"Local\stella_auto_runner")
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        print("봇이 이미 돌고 있음. 먼저 그걸 멈춰줘 (F12)", file=sys.stderr)
        return 1
    pid_file = ROOT / "logs" / "bot.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    bot = Bot(Preset.load(args.preset))
    bot.require_gamble = not args.no_gamble
    bot.step_mode = args.step
    bot.step_pause_sec = args.step_pause
    bot.max_runs = args.runs
    bot.deadline = time.monotonic() + args.hours * 3600 if args.hours else None

    def emergency_stop() -> None:
        # 메인 스레드가 키를 누르는 중일 수 있으니, 누를 수 있는 키를 전부 뗀다
        release_all_keys()
        pid_file.unlink(missing_ok=True)
        bot.log("멈춤", "F12 (즉시)")

    killswitch.start(emergency_stop)
    # 봇이 도는 동안 윈도우가 절전으로 들어가거나 화면을 끄지 않게 (봇이 끝나면 저절로 풀린다. 설정은 안 바꾼다)
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002
    kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
    try:
        bot.play(args.floors)
    except Stop as e:
        bot.log("멈춤", str(e))
    finally:
        bot.gi.release_all()
        pid_file.unlink(missing_ok=True)
        kernel32.CloseHandle(ctypes.c_void_p(mutex))
    return 0


if __name__ == "__main__":
    sys.exit(main())
