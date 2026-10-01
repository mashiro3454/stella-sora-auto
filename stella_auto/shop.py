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


@dataclass
class ShopItem:
    slot: int  # 0~7 (윗줄 왼쪽부터)
    name: str
    price: int
    old_price: int | None  # 할인 전 가격 (할인 아니면 None)
    sold_out: bool
    click: tuple[int, int]  # 상품 그림
    note_type: int | None = None  # 소리 종류 (notes.NOTE_NAMES 순서)
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


@dataclass
class ShopPlan:
    buy: list[ShopItem]
    reroll: bool
    reason: str


def plan_purchases(items: list[ShopItem], gold: int, *, shop_index: int, last_shop: bool,
                   reroll_left: int, reroll_price: int | None, reserve: int = 80) -> ShopPlan:
    """살 것 고르기.

    - 할인하는 잠재력 음료는 전부 산다.
    - 소리 (사용자 규칙): 5개 45원은 무조건, 5개 72원은 가진 게 41개 미만이고 쓰는 협주스킬이 있으면,
      15개 200원은 협주스킬 2개 이상이 쓰면, 5개 90원은 4개 이상이 쓰면. 15개 320/400원은 안 산다.
      협주스킬 필요량(users)을 못 읽었으면 45원짜리만 산다.
    - 할인 상품을 다 사고도 1000원 이상 남으면 200원짜리 잠재력 음료도 산다.
    - 마지막 상점(20층)에선 돈이 안 남게 200원짜리까지 산다.
    - 카드 리롤용으로 reserve(기본 80원)는 남긴다 (마지막 상점 제외).
    - 상점 리롤은 2번째, 4번째 상점에서 할인 상품을 다 산 뒤에 한다 (남은 횟수가 있을 때).
    """
    buy: list[ShopItem] = []
    money = gold
    keep = 0 if last_shop else reserve
    avail = [i for i in items if not i.sold_out]

    def take(item: ShopItem) -> bool:
        nonlocal money
        if money - item.price >= keep:
            buy.append(item)
            money -= item.price
            return True
        return False

    for it in sorted([i for i in avail if i.kind == "potential" and i.discounted], key=lambda i: i.price):
        take(it)
    notes = [i for i in avail if i.kind == "notes"]
    for it in [i for i in notes if i.note_count == 5 and i.price == PRICES["notes5"][2]]:
        take(it)  # 5개 45원
    for it in [i for i in notes if i.note_count == 5 and i.price == PRICES["notes5"][1]
               and i.users and (i.have is None or i.have < 41)]:
        take(it)  # 5개 72원
    for it in [i for i in notes if i.note_count == 15 and i.price == PRICES["notes15"][2] and (i.users or 0) >= 2]:
        take(it)  # 15개 200원
    for it in [i for i in notes if i.note_count == 5 and i.price == PRICES["notes5"][0] and (i.users or 0) >= 4]:
        take(it)  # 5개 90원
    full_potions = sorted([i for i in avail if i.kind == "potential" and not i.discounted], key=lambda i: i.price)
    for it in full_potions:
        if last_shop or money - it.price >= 1000:
            take(it)
    reroll = (shop_index in (2, 4) and reroll_left > 0 and reroll_price is not None
              and money - reroll_price >= keep)
    reason = f"돈 {gold} -> {money}, 살 것 {len(buy)}개" + (", 리롤" if reroll else "")
    return ShopPlan(buy, reroll, reason)
