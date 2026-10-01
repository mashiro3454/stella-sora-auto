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
from .capture import capture, find_game_window, restore
from .cards import read_cards, team_pool
from .choices import find_option_boxes, read_choices
from .shop import plan_purchases, read_reroll, read_shop
from .notes import NOTE_NAMES, read_needs, shop_note_type
from .gamedata import default_gamedata
from . import killswitch
from .input import GameInput, NotFocusedError, release_all_keys
from .ocr import KoreanOcr
from . import record as rc
from .score import RecordResult, score_record
from .pathing import Goal, Navigator
from .preset import Preset
from .screen import ScreenDetector, StableDetector
from .strategy import CardChooser, RunState

ROOT = Path(__file__).resolve().parent.parent
VK_F12 = 0x7B
TAP_STATES = {"notes_gain", "ensemble_up", "tap_continue", "explore_done"}
EMPTY_SPOT = (1300, 1045)  # "빈 곳을 터치" 화면에서 누를 곳. 가운데는 목록, 오른쪽 아래는 기록 화면의 "기록 저장" 자리라 피한다
GAMBLE_LAST_FLOOR = 3  # 650원 도박은 1~3층에서만 나온다
TITLE_BOX = (600, 40, 1320, 230)  # 방에 들어가면 잠깐 뜨는 "선택의 방 / 2/20층"
# "14/20층". OCR이 "/"를 "1"로 읽기도 한다 ("14120층")
TITLE_FLOOR_RE = re.compile(r"(\d{1,2})\s*[/1lI|]\s*20\s*층|(\d{1,2})\s*/\s*20")
ROOM_NAMES = ("전투", "선택", "강적", "거래", "리더")
HUD_TEXT = ("기록점수", "점수", "자동전투", "전투중", "레벨", "레멜", "레텔", "간단히", "대화", "코인", "소리")
NPC_SCAN_EVERY = 2.5  # 초. 이름표 찾기(OCR 전체 화면)는 무거워서 가끔만
TALK_COOLDOWN = 8.0  # 대화가 끝나도 "F 대화"가 한동안 남아 있어서, 말 건 뒤 이만큼은 다시 안 건다
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


class Stop(Exception):
    pass


class Bot:
    def __init__(self, preset: Preset, log_dir: Path = ROOT / "logs"):
        self.preset = preset
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
        self.title_seen = 0.0  # 방 제목을 마지막으로 본 때
        self.title_checked = 0.0
        self.npc_scanned = 0.0
        self._card_tries = 0
        self._last_cards: tuple | None = None  # 직전에 읽은 카드 (두 번 연속 같아야 고른다)
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
        self.joined_midway = False  # 봇을 층 중간에 켰는지 (전투 끝을 이미 지나쳤을 수 있다)
        self.enhance_count = 0  # 이 거래의 방에서 강화머신을 누른 횟수
        self.shop_done = False
        self.shop_search_start: float | None = None
        self.shop_plan = None
        self.shop_queue: list = []
        self.shop_rerolled = False
        self.shop_peeked = False
        self.note_needs = None  # 가방에서 읽은 협주스킬별 필요한 소리
        self.trade_tick = 0
        self.gamble_won = False
        self.require_gamble = True
        self.stop_after_tower = False
        self.max_runs = 0  # 이만큼 기록을 저장하면 멈춤 (0이면 계속)
        self.runs_done = 0
        self.restarts = 0
        self.deadline: float | None = None  # 이 시각이 지나면 새 판을 시작하지 않는다
        self.run_tracked = False  # 이 판을 봇이 1층부터 봤는지 (잠재 레벨 기억이 온전한지)
        self.restart_pending = ""
        self._gamble_at = 0.0
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"run_{stamp}.jsonl"
        self.shot_dir = log_dir / "shots" / stamp
        self.shot_dir.mkdir(parents=True, exist_ok=True)
        gd = default_gamedata()
        # NPC 이름표로 착각하면 안 되는 글자: 전투 중 뜨는 스킬 이름, 잠재력 이름
        self.not_npc = [n.replace(" ", "") for n in gd.skill_names] + [p.name.replace(" ", "") for p in gd.potentials.values() if p.name]
        self.choice_path = log_dir / "choices.jsonl"

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
            after = max(golds)
            if after - before >= 600:
                return True, f"돈 {before} -> {after}"
            if after - before <= -150:
                return False, f"돈 {before} -> {after}"
        return None, f"아직 모름 (돈 {before} -> {golds})"

    def read_gold(self, img: np.ndarray) -> int | None:
        text = self.ocr.text(img, (1700, 15, 1910, 85)).replace(",", "")
        nums = re.findall(r"\d+", text)
        return int(nums[-1]) if nums else None

    # -- 화면별 행동 -----------------------------------------------------------
    def on_cards(self, img: np.ndarray, enhance: bool) -> None:
        gold = self.read_gold(img)
        if gold is not None:
            self.run.gold = gold
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
        # 카드가 날아 들어오는 중에 읽으면 이름과 레벨 줄이 다른 카드끼리 섞인다.
        # 0.3초 간격으로 두 번 읽어서 똑같을 때만 고른다.
        sig = tuple((c.slot, c.potential.id if c.potential else c.raw_name, c.level_from, c.level_to, c.bonus)
                    for c in cards)
        if sig != self._last_cards:
            self._last_cards = sig
            time.sleep(0.3)
            return
        self._last_cards = None
        d = self.chooser.choose_enhance(cards, self.run) if enhance else self.chooser.choose(cards, self.run)
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
        self.log("카드", f"{'강화 ' if enhance else ''}{d.action} {d.reason} | {', '.join(desc)} | 돈 {self.run.gold}", img)
        if d.action == "pick":
            self.gi.click(*d.card.click)
            time.sleep(0.35)
            self.gi.key("space")
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
        time.sleep(0.8)

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
        if is_quiz(question) and quiz_answer(question, pairs) is None and self._quiz_waits < 8:
            # 정답지에 있는 퀴즈인데 정답 보기가 아직 안 보인다: 보기가 하나씩 나타나는 중이다
            # (보기 하나만 읽고 틀린 답을 고른 일이 있었다)
            self._quiz_waits += 1
            time.sleep(0.5)
            return
        self._quiz_waits = 0
        idx, rule = choose_option(pairs, self.run.floor, question)
        if idx < 0 and self._declined == question:
            # ESC로 안 닫히는 선택지: 가장 싼 쪽 (첫 번째 숫자가 가장 작은 것)
            costs = [int(m.group(1) or m.group(2)) if (m := re.search(r"(\d+)\s*소모", t + e)) else 10 ** 6 for t, e in pairs]
            idx, rule = costs.index(min(costs)), "모름: ESC로 안 닫혀서 가장 싼 쪽 (기록)"
        gold_before = self.read_gold(img)
        self.log("선택지", f"{question} -> [{idx}] '{options[idx].text if idx >= 0 else 'ESC'}' ({rule})", img,
                 options=pairs, chosen=idx, rule=rule)
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

    def walk_stop(self, im: np.ndarray, *, talk: bool = False) -> bool:
        """걷기를 멈출 화면인지. 문에 들어가 로딩이 뜨는 순간도 여기서 기억해 둔다."""
        st = self.det.detect(im).state
        if st == "loading":
            self.saw_loading()
        return st != "field" or (talk and nv.find_talk_prompt(im) is not None)

    def new_floor(self, floor: int, room: str) -> None:
        room = ROOM_BY_FLOOR.get(floor, "") or room  # OCR로 읽은 방 이름보다 층 구성표를 믿는다
        self.combat_done = room not in COMBAT_ROOMS
        self.joined_midway = False
        self.enhance_count = 0
        self.shop_done = not self.has_shop(floor, room)
        self.shop_search_start = None
        self.shop_plan = None
        self.shop_queue = []
        self.shop_rerolled = False
        self.shop_peeked = False
        self.trade_tick = 0
        self.floor_changed_at = time.monotonic()
        self.loading_at = None
        restore = self.floor_uncertain
        if self.floor_uncertain and floor <= GAMBLE_LAST_FLOOR:
            self.run_tracked = True  # 층을 몰랐다가 1~3층으로 밝혀짐: 650원 도박 규칙을 다시 쓴다
        self.run.floor = floor
        self.floor_known = True
        self.floor_uncertain = False
        self.exit_seen_at = None
        self.exit_fails = 0
        self._last_cards = None
        self.room = room
        self.talked = set()
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
            self.new_floor(floor, room or self.room)
            return changed
        return False

    def npc_labels(self, img: np.ndarray) -> list[tuple[str, tuple[int, int, int, int], bool]]:
        """화면의 NPC 이름표들: (이름, 상자, 상점 NPC인지)."""
        lines = self.ocr.read(img, (0, 0, 1920, 960))
        out = []
        for l in lines:
            if re.search(r"[\[\]×xX]|\d{2,}", l.text):
                continue  # 왼쪽에 뜨는 아이템 획득 알림 "[스텔라 코인]×80"
            t = re.sub(r"[^가-힣]", "", l.text)  # OCR이 "4베0}트리스"처럼 섞어 읽을 때가 있어서 한글만
            x, y, w, h = l.box
            if not (13 <= h <= 32 and 2 <= len(t) <= 6) or len(t) < len(l.text.replace(" ", "")) / 2:
                continue
            if (x < 480 and y < 320) or (x > 1400 and y < 130) or (x > 1700 and y < 260):
                continue  # 왼쪽 위 아이콘/기록 점수, 오른쪽 위 레벨/돈/자동 전투
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
        if self.room != "거래":
            return False
        price = self.next_enhance_price()
        if price > ENHANCE_MAX_PRICE:
            return False
        gold = self.read_gold(img)
        return gold is None or gold >= price

    def find_green_label(self, img: np.ndarray, word: str) -> tuple[int, int, int, int] | None:
        """초록 글자 이름표(예: 강화머신 위 "강화")의 상자."""
        for l in self.ocr.read(img, (0, 0, 1920, 960)):
            t = re.sub(r"[^가-힣]", "", l.text)
            x, y, w, h = l.box
            if t and fuzz.ratio(t, word) >= 50 and is_green_text(img, l.box) and not (x < 480 and y < 320):
                return l.box
        return None

    def save_full(self, img: np.ndarray, tag: str) -> None:
        """나중에 소리 그림/이름 짝을 만들 자료: 원본 크기 화면을 logs/shop_data/에."""
        d = self.log_path.parent / "shop_data"
        d.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", img)[1].tofile(str(d / f"{time.strftime('%m%d_%H%M%S')}_{self.run.floor:02d}_{tag}.png"))

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
                if it.kind == "notes":
                    it.note_type = shop_note_type(img, it.click[0], it.click[1], it.name)
                    if self.note_needs is not None and it.note_type is not None:
                        it.users = self.note_needs.users.get(it.note_type, 0)
                        it.have = self.note_needs.have.get(it.note_type)
            price, left = read_reroll(img, self.ocr)
            if self.shop_rerolled:
                left = 0
            self.shop_plan = plan_purchases(items, gold, shop_index=self.shop_index, last_shop=self.run.floor >= 20,
                                            reroll_left=left, reroll_price=price,
                                            note_have=self.note_needs.have if self.note_needs else None)
            self.shop_queue = list(self.shop_plan.buy)
            desc = ", ".join(f"{i.slot}:{i.name}({i.price}{'/' + str(i.old_price) if i.discounted else ''}"
                             f"{' 품절' if i.sold_out else ''}"
                             f"{' ' + NOTE_NAMES[i.note_type] + ' 협주' + str(i.users) if i.note_type is not None and i.users is not None else ''})"
                             for i in items)
            self.log("상점", f"{self.shop_index}번째 상점, {self.shop_plan.reason} | {desc}", img)
        if self.shop_queue:
            item = self.shop_queue.pop(0)
            self.log("상점", f"구매: {item.name} {item.price}원")
            self.gi.click(*item.click)
            time.sleep(0.8)
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
        self.gi.key("esc")
        time.sleep(0.8)

    def nudge_until(self, prompt: str) -> bool:
        """목표 앞에 왔는데 상호작용 표시가 안 뜨면 아래/위/왼쪽/오른쪽으로 조금씩 움직여 본다."""
        for keys in (["s"], ["w"], ["a"], ["d"], ["s"], ["s"], ["w"], ["w"]):
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
            if won is not None or time.monotonic() - self._gamble_at > 12:
                won = bool(won) if won is not None else True  # 끝까지 모르면 이어 간다 (잘못 재시작하면 이긴 판을 버린다)
                self.gamble_won = self.gamble_won or won
                self.run.keep_after_gamble = self.gamble_won
                self.log("650원", f"{'성공' if won else '실패'} ({why})", img)
                if self.run_tracked:
                    self.save_state()
                self._gamble_gold_before = None
                if not won:
                    self.restart_pending = "650원 도박 실패"
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

        # 0) 거래의 방 강화머신: 가까이 가면 "F 강화". 180원까지 누른다
        if prompt == "enhance" and now - self.last_talk_at > 2.5 and self.want_enhance(img):
            price = self.next_enhance_price()
            self.enhance_count += 1
            self.last_talk_at = now
            self.log("강화", f"강화머신 {self.enhance_count}번째 ({price}원)", img)
            self.nav.remember_spot("enhance")
            self.gi.key("f")
            time.sleep(1.2)
            return None

        # 1) "F 대화"가 떠 있으면, 아직 말 안 건 NPC면 말을 건다 (거래의 방에선 상점이 열린다)
        if prompt == "talk" and now - self.last_talk_at > TALK_COOLDOWN:
            name = self.nearest_label(img, char)
            # 이름을 못 읽었는데 이 층에서 이미 누군가와 이야기했다면, 대개 방금 그 NPC다
            if not self.already_talked(name) and not (name == "?" and self.talked):
                self.talked.add(name)
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
        if now - self.npc_scanned > NPC_SCAN_EVERY:
            self.npc_scanned = now
            shop_ok = self.has_shop() and not self.shop_done and not self.want_enhance(img)
            todo = [lb for lb in self.npc_labels(img) if not self.already_talked(lb[0]) and (not lb[2] or shop_ok)]
            if todo:
                name, box, _ = min(todo, key=lambda lb: abs(lb[1][0] - char[0]) + abs(lb[1][1] - char[1]))
                x, y, w, h = box
                self.log("NPC", f"{name} 쪽으로 걸어감 ({x + w // 2}, {y + h // 2})", img)
                res = self.nav.walk(self.label_goal(img, box, 60), self.stop_for("talk"), max_sec=15,
                                    avoid_exit=True, why=f"NPC {name}")
                self.log("NPC", f"걷기 결과 {res.reason} ({res.secs:.0f}초)")
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
        if self.require_gamble and self.run_tracked and not self.gamble_won and self.run.floor >= GAMBLE_LAST_FLOOR:
            # 3층에서도 650원 NPC가 나올 수 있다. 출구가 보인 뒤에도 잠깐 NPC를 더 찾고 나서 판단한다
            if self.run.floor == GAMBLE_LAST_FLOOR and now - self.exit_seen_at < GAMBLE_EXIT_WAIT:
                time.sleep(0.5)
                return None
            self.restart_pending = f"{GAMBLE_LAST_FLOOR}층까지 650원 선택지를 못 받음"
            return "restart"
        self.log("이동", f"출구로 감: {goal.kind} ({goal.pos[0]:.0f}, {goal.pos[1]:.0f}), 지금 ({pos[0]:.0f}, {pos[1]:.0f})", img)
        res = self.nav.walk(self.nav.exit_goal, self.walk_stop, max_sec=45, why="출구")
        self.log("이동", f"걷기 결과 {res.reason} ({res.secs:.0f}초)")
        if res.reason in ("no_progress", "no_path", "timeout"):
            self.exit_fails += 1
            if self.exit_fails % 3 == 0:
                # 잘못 적은 막힌 칸 때문에 길이 막혀 구석으로 가는 경우가 있다 (8층: NPC에 막힌 뒤 왼쪽 끝 술통까지 감)
                n = self.nav.map.forget_blocks()
                self.log("이동", f"출구로 {self.exit_fails}번 못 감: 막힌 칸 {n}개를 지우고 길을 다시 찾음")
            if self.exit_fails == 6 and self.nav.memory_exit is not None:
                self.nav.memory_exit = None
                self.log("이동", "기억한 출구로 6번 못 가서 이번 층은 화면 표시와 문양만 따라감")
        elif res.reason == "stopped":
            self.exit_fails = 0
        return None

    # -- 걷기 도우미 -------------------------------------------------------------
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
    def run_floor(self, timeout: float = 420) -> str:
        """한 층을 끝까지. 'next_floor' 또는 'restart'를 돌려준다.
        650원을 이긴 판은 층 하나에서 늦어지는 게 판을 버리는 것보다 나아서 더 기다린다."""
        start = time.monotonic()
        warned = False
        last_state = None
        while True:
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
                    self.log("전투", "소리 획득 -> 전투 끝")
                    if self.run_tracked:
                        self.save_state()
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

    def start_from_menu(self, give_up: bool = False) -> None:
        deadline = time.monotonic() + 180
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
                else:
                    self.gi.key("space")
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
                self.gi.click(*BTN_DEPART)
            elif s in ("team_setup", "record_combo"):
                self.gi.click(*BTN_NEXT)
                departed = departed or s == "record_combo"
            elif s in ("npc_choice", "dialog", "shop", "shop_buy"):
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
                "gamble_won": self.gamble_won, "run_tracked": self.run_tracked, "combat_done": self.combat_done}
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
        self.gamble_won = d["gamble_won"]
        self.run.keep_after_gamble = self.gamble_won
        self.run_tracked = True
        if floor == d["floor"] and d.get("combat_done"):
            self.combat_done = True  # 같은 방에서 전투가 이미 끝났다 (40초 기다리지 않게)
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
            if not m:
                self.log("층", f"지도 글자에서 층을 못 찾음: {text[:60]!r} ({attempt + 1}/3)", img)
            self.gi.key("esc")
            time.sleep(0.6)
            if m:
                self.new_floor(int(m.group(1) or m.group(2)), next((r for r in ROOM_NAMES if r in text), ""))
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


# 별의 탑 퀴즈 정답지 (sstoy src/modules/app-summary.ts STAR_TOWER_QA_DATA)
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
    if not m:
        return None
    answer = _QUIZ_ANSWERS[m[2]].replace(" ", "")
    scores = [fuzz.ratio(t.replace(" ", ""), answer) for t, _ in options]
    best = max(range(len(options)), key=lambda i: scores[i])
    return best if scores[best] >= 60 else None


def choose_option(options: list[tuple[str, str]], floor: int, question: str = "") -> tuple[int, str]:
    """NPC 선택지 고르기 (docs/tower-rules.md "NPC 선택지 고르기"). (번호, 규칙 이름)."""
    q = quiz_answer(question, options)
    if q is not None:
        return q, "퀴즈 정답지"
    # OCR이 "획득"을 "획티", "획듣"처럼 읽는다 (운명과 흥정: 공짜 선택지를 놓치고 50원짜리를 골랐다)
    texts = [re.sub(r"획.", "획득", (t + " " + e).replace(" ", "")) for t, e in options]
    for i, s in enumerate(texts):
        if "650" in s and floor <= GAMBLE_LAST_FLOOR:
            return i, "650원 도박"
    for i, s in enumerate(texts):
        if "HP" in s and ("소모" in s or "차감" in s) and ("잠재력" in s or "획득" in s) and "회복" not in s:
            return i, "HP 내고 잠재력/돈"
    for i, s in enumerate(texts):
        if "33%" in s and "잠재력" in s:
            return i, "33% 잠재력"
    for i, s in enumerate(texts):
        # "코인으로 바꿔줘 / 랜덤 소리 5개 소모, 150 획득" 처럼 "판다"는 말이 없을 때도 있다
        if "소리" in s and "150" in s and ("팔" in s or "판매" in s or ("소모" in s and "획득" in s)):
            return i, "소리 팔고 150원"
    if any("소리" in s and ("90" in s or "140" in s) for s in texts):
        for i, s in enumerate(texts):
            if "30" in s and "획득" in s and "소리" not in s:
                return i, "소리 사지 말고 30원"
    hundred = [i for i, s in enumerate(texts) if "100" in s and "차감" not in s]
    thirty = [i for i, s in enumerate(texts) if re.search(r"(?<!\d)30", s) and "획득" in s]
    if hundred and thirty and set(hundred) != set(thirty):
        return (hundred[0], "100원 (6층 이하)") if floor <= 6 else (thirty[0], "30원 (7층 이상)")
    loss = ("소모", "차감", "소실", "감소", "잃", "변화")  # "랜덤 변화", "HP 30% 소실"도 잃을 수 있는 것
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stella_auto.runner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", type=Path, help="프리셋 JSON (python -m stella_auto.preset ... -o 로 만든 것)")
    ap.add_argument("--floors", type=int, default=20, help="시험용: 이만큼 층을 넘기면 멈춤 (기본 20 = 끝까지)")
    ap.add_argument("--runs", type=int, default=0, help="탑을 이만큼 끝내면(기록 저장) 멈춤. 0이면 계속")
    ap.add_argument("--hours", type=float, default=0, help="이만큼 시간이 지나면 다음 판을 시작하지 않고 멈춤. 0이면 계속")
    ap.add_argument("--no-gamble", action="store_true", help="시험용: 650원 도박 조건 없이 계속 올라간다")
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
