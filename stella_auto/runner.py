"""별의 탑 봇 본체: 화면을 보고 한 단계씩 행동한다.

    python -m stella_auto.runner presets/바람.json --floors 3

F12를 누르면 바로 멈춘다. 행동과 NPC 선택은 logs/ 에 기록한다.
아직 안 되는 것: 상점(들어가면 그냥 나옴), 강화머신, 사망 처리.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz, process

from . import navigate as nv
from .capture import capture, find_game_window, restore
from .cards import read_cards, team_pool
from .choices import find_option_boxes, read_choices
from .shop import plan_purchases, read_reroll, read_shop
from .gamedata import default_gamedata
from . import killswitch
from .input import GameInput, NotFocusedError, release_all_keys
from .ocr import KoreanOcr
from .preset import Preset
from .screen import ScreenDetector, StableDetector
from .strategy import CardChooser, RunState

ROOT = Path(__file__).resolve().parent.parent
VK_F12 = 0x7B
TAP_STATES = {"notes_gain", "ensemble_up", "tap_continue", "explore_done"}
EMPTY_SPOT = (1300, 1045)  # "빈 곳을 터치" 화면에서 누를 곳. 가운데는 목록, 오른쪽 아래는 기록 화면의 "기록 저장" 자리라 피한다
GAMBLE_LAST_FLOOR = 3  # 650원 도박은 1~3층에서만 나온다
TITLE_BOX = (600, 40, 1320, 230)  # 방에 들어가면 잠깐 뜨는 "선택의 방 / 2/20층"
TITLE_FLOOR_RE = re.compile(r"(\d+)\s*/\s*20")
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
        self.exit_angle: float | None = None  # 이 층에서 마지막으로 본 출구 방향
        self.blind_walks = 0  # 출구가 안 보일 때 마지막 방향으로 간 횟수
        self.idle_since: float | None = None  # 출구를 못 찾기 시작한 때
        self.explore_i = 0
        self.room = ""  # 이 층 방 종류 (전투/선택/강적/거래/리더)
        self.talked: set[str] = set()  # 이 층에서 말 건 NPC 이름
        self.title_seen = 0.0  # 방 제목을 마지막으로 본 때
        self.title_checked = 0.0
        self.npc_scanned = 0.0
        self._card_tries = 0
        self._last_cards: tuple | None = None  # 직전에 읽은 카드 (두 번 연속 같아야 고른다)
        self._last_choice_bands: tuple | None = None
        self.loading_at: float | None = None  # 층 사이 로딩을 본 때 (방 제목을 놓치면 이걸로 층을 센다)
        self.floor_changed_at = 0.0
        self.last_talk_at = 0.0
        self.exit_seen_at: float | None = None  # 이 층에서 출구를 처음 본 때
        self.floor_known = False  # 봇을 탑 중간에서 켜면 처음엔 몇 층인지 모른다
        self.combat_done = True  # 이 층 전투가 끝났는지 ("소리 획득"을 봤는지)
        self.joined_midway = False  # 봇을 층 중간에 켰는지 (전투 끝을 이미 지나쳤을 수 있다)
        self.enhance_count = 0  # 이 거래의 방에서 강화머신을 누른 횟수
        self.shop_done = False
        self.shop_plan = None
        self.shop_queue: list = []
        self.shop_rerolled = False
        self.trade_tick = 0
        self.gamble_won = False
        self.require_gamble = True
        self.restart_pending = ""
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
        desc = [f"{c.potential.name if c.potential else c.raw_name}({'새' if c.is_new else c.level_from}>{c.level_to})"
                f"={d.values.get(c.slot)}" for c in cards]
        self.log("카드", f"{'강화 ' if enhance else ''}{d.action} {d.reason} | {', '.join(desc)} | 돈 {self.run.gold}", img)
        if d.action == "pick":
            self.gi.click(*d.card.click)
            time.sleep(0.35)
            self.gi.key("space")
            self.chooser.record_pick(d.card, self.run)
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
        idx, rule = choose_option(pairs, self.run.floor, question)
        gold_before = self.read_gold(img)
        self.log("선택지", f"{question} -> [{idx}] '{options[idx].text}' ({rule})", img,
                 options=pairs, chosen=idx, rule=rule)
        with self.choice_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "floor": self.run.floor,
                                "question": question, "options": pairs, "chosen": idx, "rule": rule},
                               ensure_ascii=False) + "\n")
        self.gi.click(*options[idx].center)
        if rule == "650원 도박":
            self._gamble_gold_before = gold_before
        time.sleep(1.0)

    def saw_loading(self) -> None:
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
        self.shop_done = room != "거래"
        self.shop_plan = None
        self.shop_queue = []
        self.shop_rerolled = False
        self.trade_tick = 0
        self.floor_changed_at = time.monotonic()
        self.loading_at = None
        self.run.floor = floor
        self.floor_known = True
        self.exit_seen_at = None
        self._last_cards = None
        self.room = room
        self.talked = set()
        self.exit_angle = None
        self.idle_since = None
        self.explore_i = 0
        self.blind_walks = 0
        self.log("층", f"{floor}층 {room + '의 방' if room else ''}")

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
                floor = int(m.group(1))
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
        if self.floor_known and floor != self.run.floor:
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
            if time.monotonic() - self.title_seen < 3 and TITLE_BOX[0] < x < TITLE_BOX[2] and y < TITLE_BOX[3]:
                continue
            if is_green_text(img, l.box):
                continue  # 이름 위의 초록 글자는 NPC 종류("사건", "회복", "상점")지 이름이 아니다
            if process.extractOne(t, self.not_npc, scorer=fuzz.ratio, score_cutoff=75):
                continue  # 전투 중 뜨는 스킬 이름 ("무장 습격" 등)
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

    def on_shop(self, img: np.ndarray) -> None:
        if self.shop_plan is None:
            gold = self.read_gold(img) or 0
            items = read_shop(img, self.ocr)
            price, left = read_reroll(img, self.ocr)
            if self.shop_rerolled:
                left = 0
            self.shop_plan = plan_purchases(items, gold, shop_index=self.shop_index, last_shop=self.run.floor >= 20,
                                            reroll_left=left, reroll_price=price)
            self.shop_queue = list(self.shop_plan.buy)
            desc = ", ".join(f"{i.slot}:{i.name}({i.price}{'/' + str(i.old_price) if i.discounted else ''}"
                             f"{' 품절' if i.sold_out else ''})" for i in items)
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
            after = self.read_gold(img)
            if after is not None:
                won = after - self._gamble_gold_before >= 600
                self.gamble_won = self.gamble_won or won
                self.log("650원", f"{'성공' if won else '실패'} ({self._gamble_gold_before} -> {after})")
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
        if char is None:
            time.sleep(0.3)
            return None
        now = time.monotonic()

        prompt = nv.find_prompt(img)

        # 0) 거래의 방 강화머신: 가까이 가면 "F 강화". 180원까지 누른다
        if prompt == "enhance" and now - self.last_talk_at > 2.5 and self.want_enhance(img):
            price = self.next_enhance_price()
            self.enhance_count += 1
            self.last_talk_at = now
            self.log("강화", f"강화머신 {self.enhance_count}번째 ({price}원)", img)
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
                if self.det.detect(self.grab()).state == "field":
                    self.log("NPC", f"{name}: 대화가 안 열림 (이미 끝난 이벤트일 수 있음)")
                return None

        # 1-1) 거래의 방: 강화머신("강화" 초록 글자) 쪽으로 먼저 간다 (거래의 방은 매번 찾는다)
        if self.want_enhance(img):
            box = self.find_green_label(img, "강화")
            if box:
                x, y, w, h = box
                self.log("강화", f"강화머신 쪽으로 걸어감 ({x + w // 2}, {y + h // 2})", img)
                tracker = nv.TemplateTracker(img, (x - 4, y - 4, x + w + 4, y + h + 4), threshold=0.5)
                res = nv.walk_toward(lambda k, sec: self.gi.hold(list(k), sec), self.grab,
                                     lambda im, ch: (lambda c: (c[0], c[1] + 90) if c else None)(tracker.update(im)),
                                     stop=lambda im: self.walk_stop(im) or nv.find_prompt(im) == "enhance",
                                     arrive_dist=50, max_steps=30)
                self.log("강화", f"걷기 결과 {res.reason} ({res.steps}걸음)")
                if nv.find_prompt(self.grab()) != "enhance":
                    self.nudge_until("enhance")
                return None

        # 2) 화면에 말 안 건 NPC 이름표가 있으면 그쪽으로 걸어간다 (상점 NPC는 거래의 방에서 강화 뒤에만)
        if now - self.npc_scanned > NPC_SCAN_EVERY:
            self.npc_scanned = now
            shop_ok = self.room == "거래" and not self.shop_done and not self.want_enhance(img)
            todo = [lb for lb in self.npc_labels(img) if not self.already_talked(lb[0]) and (not lb[2] or shop_ok)]
            if todo:
                name, (x, y, w, h), _ = min(todo, key=lambda lb: abs(lb[1][0] - char[0]) + abs(lb[1][1] - char[1]))
                self.log("NPC", f"{name} 쪽으로 걸어감 ({x + w // 2}, {y + h // 2})", img)
                tracker = nv.TemplateTracker(img, (x - 4, y - 4, x + w + 4, y + h + 4), threshold=0.5)
                res = nv.walk_toward(lambda k, sec: self.gi.hold(list(k), sec), self.grab,
                                     lambda im, ch: tracker.update(im),
                                     stop=lambda im: self.walk_stop(im, talk=True),
                                     arrive_dist=50, max_steps=30)
                self.log("NPC", f"걷기 결과 {res.reason} ({res.steps}걸음)")
                if res.reason != "stopped":
                    self.talked.add(name)  # 못 가면 이 NPC는 포기 (무한 반복 방지)
                return None

        # 2-1) 화면 밖 NPC는 물음표 상자 표시를 따라간다 (전투가 끝나야 NPC가 생기는 방도 있다)
        if self.combat_done and not self.talked:
            npc = next((m for m in nv.find_markers(img, char) if m.kind == "npc"), None)
            if npc:
                self.log("NPC", "물음표 표시를 따라감", img)
                res = nv.walk_toward(
                    lambda k, sec: self.gi.hold(list(k), sec), self.grab,
                    lambda im, ch: (lambda m: nv.marker_target(m, ch) if m else None)(
                        next((m for m in nv.find_markers(im, ch) if m.kind == "npc"), None)),
                    stop=lambda im: self.walk_stop(im, talk=True), arrive_dist=0, max_steps=25,
                    initial_angle=nv.angle_of(*(np.subtract(nv.marker_target(npc, char), char))))
                self.log("NPC", f"표시 따라가기 결과 {res.reason} ({res.steps}걸음)")
                self.npc_scanned = 0.0  # NPC가 화면에 들어왔을 테니 바로 이름표를 찾는다
                return None

        # 2-1-1) 거래의 방은 강화머신과 상점이 끝나야 나간다
        if self.room == "거래" and (self.want_enhance(img) or not self.shop_done):
            if now - self.floor_changed_at < TRADE_TIMEOUT:
                # 다음 루프에서 이름표부터 다시 찾고, 둘러보기는 두 번에 한 번만 (근처에서 멀어지지 않게)
                self.npc_scanned = 0.0
                self.trade_tick += 1
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
            limit = COMBAT_TIMEOUT if not self.joined_midway else 40.0
            if now - self.floor_changed_at < limit:
                time.sleep(0.4)
                return None
            self.combat_done = True
            self.log("전투", f"{limit:.0f}초 동안 소리 획득을 못 봄, 전투가 끝난 것으로 보고 진행", img)

        # 3) 선택의 방은 NPC와 이야기하기 전엔 나가지 않는다
        if self.room == "선택" and not self.talked and now - self.floor_changed_at < CHOICE_SEARCH:
            self._explore("선택의 방 NPC를 찾으려고 둘러봄", avoid_exit=True)
            return None

        # 방 제목이 떠 있는 동안은 제목 글자(ㅇ)를 출구 문양으로 착각할 수 있어서 기다린다
        if now - self.title_seen < 2.5:
            time.sleep(0.3)
            return None

        target = nv.exit_target(img, char)
        if target is None:
            if self.idle_since is None:
                self.idle_since = now
            if now - self.idle_since < 4:
                time.sleep(0.4)  # 아직 전투 중이거나 출구가 안 열렸을 수 있다
                return None
            if self.exit_angle is not None and self.blind_walks < 3:
                self.blind_walks += 1
                self.log("이동", f"출구가 안 보임, 마지막으로 본 방향({self.exit_angle:+.0f}도)으로 감 ({self.blind_walks}/3)")
                self.gi.hold(list(nv.keys_for_angle(self.exit_angle)), 1.2)
            else:  # 그 방향이 막혔거나 본 적이 없으면 사방을 둘러본다
                self.exit_angle = None
                self._explore("출구를 찾으려고 둘러봄")
            self.idle_since = now - 2
            return None
        self.exit_angle = nv.angle_of(target[0] - char[0], target[1] - char[1])
        self.idle_since = None
        self.blind_walks = 0
        if self.exit_seen_at is None:
            self.exit_seen_at = now
        if self.require_gamble and not self.gamble_won and self.run.floor >= GAMBLE_LAST_FLOOR:
            # 3층에서도 650원 NPC가 나올 수 있다. 출구가 보인 뒤에도 잠깐 NPC를 더 찾고 나서 판단한다
            if self.run.floor == GAMBLE_LAST_FLOOR and now - self.exit_seen_at < GAMBLE_EXIT_WAIT:
                time.sleep(0.5)
                return None
            self.restart_pending = f"{GAMBLE_LAST_FLOOR}층까지 650원 선택지를 못 받음"
            return "restart"
        self.log("이동", f"출구로 걸어감 {tuple(round(v) for v in target)}", img)

        def tgt(im: np.ndarray, ch: tuple[float, float]) -> tuple[float, float] | None:
            t = nv.exit_target(im, ch)
            if t is not None:
                self.exit_angle = nv.angle_of(t[0] - ch[0], t[1] - ch[1])
            return t

        res = nv.walk_toward(lambda k, sec: self.gi.hold(list(k), sec), self.grab, tgt,
                             stop=self.walk_stop, arrive_dist=0, max_steps=45, initial_angle=self.exit_angle)
        self.log("이동", f"걷기 결과 {res.reason} ({res.steps}걸음)")
        return None

    def _explore(self, why: str, avoid_exit: bool = False) -> None:
        """사방을 돌아가며 조금씩 걸어 본다. avoid_exit이면 출구 쪽(90도 이내)으로는 안 간다
        (아직 나가면 안 되는 방에서 둘러보다 출구 문에 들어가 버린 일이 있었다)."""
        exit_ang = None
        if avoid_exit:
            img = self.grab()
            char = nv.find_character(img)
            t = nv.exit_target(img, char) if char else None
            if t is not None:
                exit_ang = nv.angle_of(t[0] - char[0], t[1] - char[1])
        for _ in range(4):
            ang = (90, 0, 180, -90)[self.explore_i % 4]
            self.explore_i += 1
            if exit_ang is None or abs((ang - exit_ang + 180) % 360 - 180) > 90:
                break
        else:
            ang = exit_ang + 180
        self.log("이동", f"{why} ({ang:+.0f}도)")
        self.gi.hold(list(nv.keys_for_angle(ang)), 1.2)

    # -- 층, 판 ---------------------------------------------------------------
    def run_floor(self, timeout: float = 420) -> str:
        """한 층을 끝까지. 'next_floor' 또는 'restart'를 돌려준다."""
        end = time.monotonic() + timeout
        last_state = None
        while time.monotonic() < end:
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
                self.gi.click(*EMPTY_SPOT)
                time.sleep(0.7)
            elif s == "shop":
                self.on_shop(img)
            elif s == "shop_buy":
                self.gi.key("space")  # 구매
                time.sleep(1.0)
            elif s == "esc_map":
                self.gi.key("esc")
                time.sleep(0.5)
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
        while time.monotonic() < deadline:
            img = self.grab()
            s = self.det.detect(img).state
            if s == "field":
                if departed:
                    self.run = RunState(floor=1)
                    self.new_floor(1, "전투")
                    self.gamble_won = False
                    self.log("시작", "새 판 1층")
                    return
                if give_up:
                    self.gi.key("esc")
            elif s == "esc_map":
                self.gi.key("q")
            elif s == "notice":
                text = self.ocr.text(img, (300, 250, 1620, 750))
                if "분해" in text:
                    self.gi.click(*BTN_CONFIRM)
                else:
                    self.gi.key("space")
            elif s in TAP_STATES:
                self.gi.click(*EMPTY_SPOT)
            elif s == "record_result":
                self.gi.click(*BTN_TRASH)
            elif s == "difficulty_select":
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

    def read_floor_from_map(self) -> None:
        """탑 안에서 시작할 때: ESC 지도의 "3/20층 전투의 방"으로 지금 층을 안다."""
        self.gi.key("esc")
        time.sleep(1.0)
        img = self.grab()
        if self.det.detect(img).state == "esc_map":
            text = self.ocr.text(img, (600, 60, 1300, 1020)).replace(" ", "")
            m = TITLE_FLOOR_RE.search(text)
            if m:
                self.new_floor(int(m.group(1)), next((r for r in ROOM_NAMES if r in text), ""))
                self.joined_midway = True  # 이 층 전투가 이미 끝났을 수 있다
            self.gi.key("esc")
            time.sleep(0.6)
        if not self.floor_known:
            self.floor_known = True  # 못 읽어도 계속 ESC를 누르지 않게. 다음 방 제목에서 바로잡힌다
            self.log("층", f"지도에서 층을 못 읽음, {self.run.floor}층으로 둠")

    def play(self, max_floors: int) -> None:
        """게임 창이 잠깐 가려지거나 다른 창이 앞으로 와도 꺼지지 않고, 기다렸다가 화면을 다시 보고 이어 간다."""
        fails = 0
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

    def _play(self, max_floors: int) -> None:
        s = self.det.detect(self.grab()).state
        if s in ("field", "card_select", "enhance_select", "npc_choice", "dialog", "shop", "shop_buy",
                 "esc_map", "notes_gain", "ensemble_up", "tap_continue"):
            pass  # 탑 안: 필드에 나오면 on_field가 ESC 지도로 층을 확인한다
        else:
            self.start_from_menu(give_up=False)
        while True:
            r = self.run_floor()
            if r == "next_floor":
                if self.run.floor - 1 >= max_floors:
                    self.log("끝", f"{max_floors}층까지 넘김")
                    return
            else:
                self.restart(self.restart_pending or "층 실패")


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
    texts = [(t + " " + e).replace(" ", "") for t, e in options]
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
    for i, s in enumerate(texts):
        if "획득" in s and "소모" not in s and "차감" not in s:
            return i, "모름: 잃는 것 없는 쪽 (기록)"
    return 0, "모름: 첫 번째 (기록)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stella_auto.runner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", type=Path, help="프리셋 JSON (python -m stella_auto.preset ... -o 로 만든 것)")
    ap.add_argument("--floors", type=int, default=3, help="이만큼 층을 넘기면 멈춤")
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

    def emergency_stop() -> None:
        # 메인 스레드가 키를 누르는 중일 수 있으니, 누를 수 있는 키를 전부 뗀다
        release_all_keys()
        pid_file.unlink(missing_ok=True)
        bot.log("멈춤", "F12 (즉시)")

    killswitch.start(emergency_stop)
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
