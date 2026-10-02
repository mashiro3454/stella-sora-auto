"""상점 화면 읽기와 살 것 고르기 (docs/tower-rules.md "상점").

상점은 4칸 x 2줄. 칸마다 가격표에 가격과 상품 이름이 있고, 할인 상품은 옛 가격(줄 그음)과 새 가격이
같이 찍히고 빨간 "할인" 딱지가 붙는다. 팔린 칸에는 "품절"이 붙는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from .ocr import KoreanOcr

COLUMNS = (1050, 1275, 1500, 1725)  # 칸 가운데 x
ROWS = ((395, 280), (695, 590))  # (가격표 가운데 y, 상품 그림 y)
TICKET_HALF_W = 95
REROLL_PRICE_BOX = (1690, 850, 1880, 900)  # 새로고침 버튼 위 가격
REROLL_LEFT_BOX = (1560, 930, 1790, 980)  # "남은 횟수 1"
# 200원짜리 잠재력 음료를 살 때 상점마다 남길 돈 (사용자 규칙 2026-10-02): 첫 상점 300~500원,
# 두 번째는 탑이 끝날 때 돈이 안 남게 적당히, 세 번째(19층)는 20층 상점용으로 800원 이상, 마지막은 다 쓴다
FULL_PRICE_KEEP = {1: 400, 2: 500, 3: 800, 4: 0}


@dataclass
class ShopItem:
    slot: int  # 0~7 (윗줄 왼쪽부터)
    name: str
    price: int
    old_price: int | None  # 할인 전 가격 (할인 아니면 None)
    sold_out: bool
    click: tuple[int, int]  # 상품 그림
    note_type: int | None = None  # 소리 종류 (notes.NOTE_NAMES 순서)
    face: int | None = None  # 음료의 캐릭터 얼굴: char_id(그 캐릭터 잠재만 나옴), -1(모르는 얼굴), None(민무늬: 셋 다)
    users: int | None = None  # 이 소리가 필요한 협주스킬 수 (가방에서 읽음, 모르면 None)
    have: int | None = None  # 이 소리를 가진 개수

    @property
    def discounted(self) -> bool:
        return self.old_price is not None and self.old_price > self.price

    @property
    def kind(self) -> str:
        n = self.name.replace(" ", "")
        if "음료" in n or "잠재력" in n:
            return "potential"
        if "소리" in n:
            return "notes"
        return "other"

    @property
    def note_count(self) -> int:
        # OCR이 "×15"를 "%15", "*5"처럼 읽는다 (19층: "기술의 소리%15"를 5개짜리로 알고 200원에 샀다)
        m = re.search(r"[x×X%*]\s*(\d+)\s*$", self.name.strip())
        if m:
            return int(m.group(1))
        return 15 if re.search(r"1\s*5\s*$", self.name.strip()) else 5


# 상품별 가격은 정해져 있다: (정가, 20% 할인, 50% 할인)
PRICES = {
    "potential": (200, 160, 100),
    "notes5": (90, 72, 45),
    "notes15": (400, 320, 200),
}


def _price_key(kind: str, name: str) -> str | None:
    if kind == "potential":
        return "potential"
    if kind == "notes":
        return "notes15" if ShopItem(0, name, 0, None, False, (0, 0)).note_count == 15 else "notes5"
    return None


def _guess_price(key: str, tag: str, digits: str, big: str = "") -> tuple[int, int | None]:
    """(가격, 옛 가격). 큰 가격 숫자는 OCR이 "160"을 "9160", "100"을 "!00"처럼 깨뜨려서,
    딱지("할인")로 할인 여부를 정하고 숫자 조각으로 20%/50%를 고른다."""
    full, d20, d50 = PRICES[key]
    if "할" not in tag and "인" not in tag:
        return full, None
    # 오른쪽의 큰 새 가격만 따로 읽은 숫자(big)가 할인 가격 하나와 딱 맞으면 그것
    # (12층: "폭발의 소리x5" 45원을 72원으로 알고 안 샀다. 넓게 읽으면 옛 가격과 섞여 빈칸이 됐다)
    for cand in (d50, d20):
        if big == str(cand):
            return cand, full
    s20, s50 = str(d20), str(d50)
    if s20 in digits or s20[:2] in digits:
        return d20, full
    if s50 in digits or s50[:2] in digits:
        return d50, full
    return d20, full  # 모르면 덜 싼 쪽으로 (돈 계산이 모자라지 않게)


def tag_kind(img: np.ndarray, cx: int, price_y: int) -> str | None:
    """가격표 왼쪽 위 빨간 딱지: 동그라미면 "sale"(할인), 길쭉한 알약 모양이면 "sold"(품절)."""
    x0, y0 = cx - 140, price_y - 90
    reg = img[y0:price_y - 15, x0:cx + 70]
    hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 110) & (hsv[..., 2] > 110)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(red)
    kind = None
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in st[i])
        if area < 150:
            continue
        if w >= 45 and 0.8 <= w / h <= 1.25:
            return "sale"
        if w / h >= 1.5:
            kind = "sold"
    return kind


def read_shop(img: np.ndarray, ocr: KoreanOcr) -> list[ShopItem]:
    items = []
    for r, (price_y, icon_y) in enumerate(ROWS):
        for c, cx in enumerate(COLUMNS):
            name = " ".join(ocr.text(img, (cx - 95, price_y + 16, cx + 95, price_y + 54)).split())
            if not name:
                continue
            tk = tag_kind(img, cx, price_y)
            tag = {"sale": "할인", "sold": "품절"}.get(tk, "")
            digits = re.sub(r"\D", "", ocr.text(img, (cx - 50, price_y - 30, cx + 92, price_y + 20)))
            big = re.sub(r"\D", "", ocr.text(img, (cx - 5, price_y - 32, cx + 92, price_y + 22)))
            sold = tk == "sold"
            kind = ShopItem(0, name, 0, None, False, (0, 0)).kind
            key = _price_key(kind, name)
            if key is None:
                nums = re.findall(r"\d+", digits)
                price, old = (int(nums[-1]) if nums else 9999), None
            else:
                price, old = _guess_price(key, tag, digits + big, big)
            items.append(ShopItem(r * 4 + c, name, price, old, sold, (cx, icon_y)))
    return items


def read_reroll(img: np.ndarray, ocr: KoreanOcr) -> tuple[int | None, int | None]:
    """(리롤 가격, 남은 횟수). 리롤을 다 쓰면 버튼이 사라져 (None, 0)."""
    price = re.findall(r"\d+", ocr.text(img, REROLL_PRICE_BOX))
    left = re.findall(r"\d+", ocr.text(img, REROLL_LEFT_BOX))
    return (int(price[-1]) if price else None), (int(left[-1]) if left else 0)


_FACE_TEMPLATES: dict[int, "np.ndarray"] = {}


def _face_templates() -> dict[int, "np.ndarray"]:
    if not _FACE_TEMPLATES:
        from .screen import TEMPLATE_DIR

        for f in TEMPLATE_DIR.glob("drink_face_*.png"):
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_GRAYSCALE)
            _FACE_TEMPLATES[int(f.stem.split("_")[-1])] = img
    return _FACE_TEMPLATES


def drink_face(img: np.ndarray, item: ShopItem) -> int | None:
    """잠재력 음료에 그려진 캐릭터 얼굴 (사용자 2026-10-03: 얼굴 음료는 그 캐릭터 잠재만 나온다).

    컵 오른쪽 아래 동그란 초상화를 원 찾기로 찾고, 아는 얼굴 템플릿과 맞춘다.
    char_id / -1(모르는 얼굴: 템플릿에 없음) / None(민무늬). 상점 자료 700칸 검증: 얼굴 95, 오탐 0."""
    x, y = item.click
    cell = img[max(0, y - 80):y + 70, max(0, x - 95):x + 95]
    if cell.shape[0] < 150 or cell.shape[1] < 190:
        return None
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    circ = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1, 60, param1=120, param2=42, minRadius=26, maxRadius=38)
    if circ is None:
        return None
    for cx, cy, _r in circ[0]:
        if not (60 < cx < 185 and 30 < cy < 145) or not (26 <= cx < 164 and 26 <= cy < 124):
            continue
        disc = g[int(cy) - 26:int(cy) + 26, int(cx) - 26:int(cx) + 26]
        best_id, best = -1, 0.0
        for cid, t in _face_templates().items():
            score = float(cv2.matchTemplate(disc, t, cv2.TM_CCOEFF_NORMED).max())
            if score > best:
                best_id, best = cid, score
        return best_id if best >= 0.5 else -1
    return None


@dataclass
class ShopPlan:
    buy: list[ShopItem]
    reroll: bool
    reason: str


def plan_purchases(items: list[ShopItem], gold: int, *, shop_index: int, last_shop: bool,
                   reroll_left: int, reroll_price: int | None, reserve: int = 80,
                   note_have: dict[int, int] | None = None, enhance_reserve: int = 0,
                   prefer_char: int | None = None) -> ShopPlan:
    """살 것 고르기.

    - 할인하는 잠재력 음료는 전부 산다. prefer_char(강화 우선 캐릭터)가 있으면 그 캐릭터 잠재가
      나올 수 있는 음료부터: 그 캐릭터 얼굴 → 민무늬 → 모르는 얼굴 → 다른 캐릭터 얼굴 (돈이 모자라
      뒤쪽을 못 사도 앞쪽이 남게). 얼굴 음료는 그 캐릭터 잠재만 나온다 (사용자 2026-10-03).
    - 소리 (사용자 규칙, 협주스킬에 필요한 소리만): 5개 45원은 산다, 5개 72원은 가진 게 41개 미만이면,
      15개 200원은 협주스킬 2개 이상이 쓰면, 5개 90원은 4개 이상이 쓰면 (단 그 소리만 40개쯤 있고
      협주스킬에 필요한 다른 소리가 20개 미만이면 안 산다). 15개 320/400원은 안 산다.
      협주스킬 필요량(users)을 못 읽었으면 45원짜리만 산다. note_have: 협주스킬에 필요한 소리별 가진 개수.
    - 200원짜리 잠재력 음료는 상점마다 남길 돈(FULL_PRICE_KEEP)까지 산다 (사용자 규칙, 탑이 끝날 때 돈이 안 남게).
      상점 리롤을 할 차례면 리롤 뒤에 산다 (할인 → 리롤 → 할인 → 200원짜리).
    - 마지막 상점(20층)에선 돈이 안 남게 다 산다.
    - 카드 리롤용으로 reserve(기본 80원)는 남긴다 (마지막 상점 제외). 강화머신을 상점 뒤에 쓸 거면
      그 값(enhance_reserve)도 남긴다.
    - 상점 리롤은 2번째, 4번째 상점에서 할인 상품을 다 산 뒤에 한다 (남은 횟수가 있을 때).
    """
    buy: list[ShopItem] = []
    money = gold
    keep = (0 if last_shop else reserve) + enhance_reserve
    avail = [i for i in items if not i.sold_out]

    def take(item: ShopItem) -> bool:
        nonlocal money
        if money - item.price >= keep:
            buy.append(item)
            money -= item.price
            return True
        return False

    def face_rank(i: ShopItem) -> int:
        if prefer_char is None or i.face == prefer_char:
            return 0
        return {None: 1, -1: 2}.get(i.face, 3)

    for it in sorted([i for i in avail if i.kind == "potential" and i.discounted], key=lambda i: (face_rank(i), i.price)):
        take(it)
    notes = [i for i in avail if i.kind == "notes"]
    for it in [i for i in notes if i.note_count == 5 and i.price == PRICES["notes5"][2] and (i.users is None or i.users >= 1)]:
        take(it)  # 5개 45원
    for it in [i for i in notes if i.note_count == 5 and i.price == PRICES["notes5"][1]
               and i.users and (i.have is None or i.have < 41)]:
        take(it)  # 5개 72원
    for it in [i for i in notes if i.note_count == 15 and i.price == PRICES["notes15"][2] and (i.users or 0) >= 2]:
        take(it)  # 15개 200원
    for it in [i for i in notes if i.note_count == 5 and i.price == PRICES["notes5"][0] and (i.users or 0) >= 4]:
        others = [v for t, v in (note_have or {}).items() if t != it.note_type]
        if it.have is not None and it.have >= 40 and any(v < 20 for v in others):
            continue  # 이 소리만 많고 다른 필요한 소리가 모자라면 정가로는 안 산다
        take(it)  # 5개 90원
    reroll = (shop_index in (2, 4) and reroll_left > 0 and reroll_price is not None
              and money - reroll_price >= keep)
    if not reroll:
        full_keep = keep if last_shop else max(keep, FULL_PRICE_KEEP.get(shop_index, 1000) + enhance_reserve)
        full_potions = sorted([i for i in avail if i.kind == "potential" and not i.discounted],
                              key=lambda i: (face_rank(i), i.price))
        for it in full_potions:
            if money - it.price >= full_keep:
                take(it)
        if last_shop:
            # 마지막 상점: 탑이 끝나면 돈은 쓸모없다. 남은 돈으로 소리까지 다 산다 (쓰는 협주스킬이 많은 것부터)
            rest = [i for i in avail if i not in buy and i.kind == "notes"]
            for it in sorted(rest, key=lambda i: (-(i.users or 0), i.price)):
                take(it)
    reason = f"돈 {gold} -> {money}, 살 것 {len(buy)}개" + (", 리롤" if reroll else "")
    return ShopPlan(buy, reroll, reason)
