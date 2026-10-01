from pathlib import Path

import cv2
import numpy as np
import pytest

from stella_auto.ocr import KoreanOcr
from stella_auto.shop import ShopItem, plan_purchases, read_shop

SCREENS = Path(__file__).parent / "screens"


@pytest.fixture(scope="module")
def ocr():
    return KoreanOcr()


def summary(ocr, name):
    img = cv2.imdecode(np.fromfile(str(SCREENS / name), np.uint8), cv2.IMREAD_COLOR)
    return [(i.kind, i.price, i.old_price, i.sold_out) for i in read_shop(img, ocr)]


def test_read_shop_discounts(ocr):
    s = summary(ocr, "shop__v1_330.jpg")
    assert s[1] == ("potential", 160, 200, False)
    assert s[2] == ("potential", 100, 200, False)
    assert s[5] == ("notes", 45, 90, False)
    assert s[7] == ("notes", 400, None, False)


def test_read_shop_sold_out(ocr):
    s = summary(ocr, "shop__v1_574.jpg")
    assert s[0][3] and s[1][3]  # 품절 딱지
    assert s[5] == ("notes", 320, 400, False)
    assert s[6] == ("notes", 72, 90, False)


def item(slot, name, price, old=None, sold=False):
    return ShopItem(slot, name, price, old, sold, (0, 0))


def test_plan_buys_discounted_potions_and_keeps_reroll_money():
    items = [item(0, "잠재력 특제 음료", 160, 200), item(1, "잠재력 특제 음료", 100, 200),
             item(2, "잠재력 특제 음료", 200), item(3, "집중의 소리x5", 45, 90)]
    p = plan_purchases(items, 400, shop_index=1, last_shop=False, reroll_left=2, reroll_price=100)
    assert [b.slot for b in p.buy] == [1, 0, 3]  # 싼 할인 음료부터, 50% 소리, 80원은 남김
    assert not p.reroll  # 1번째 상점은 리롤 안 함


def test_plan_full_price_when_rich_or_last_shop():
    items = [item(0, "잠재력 특제 음료", 200)]
    assert plan_purchases(items, 1300, shop_index=3, last_shop=False, reroll_left=0, reroll_price=None).buy
    assert not plan_purchases(items, 1100, shop_index=3, last_shop=False, reroll_left=0, reroll_price=None).buy
    assert plan_purchases(items, 200, shop_index=4, last_shop=True, reroll_left=0, reroll_price=None).buy


def test_plan_reroll_on_2nd_and_4th():
    p = plan_purchases([], 500, shop_index=2, last_shop=False, reroll_left=1, reroll_price=100)
    assert p.reroll


def test_note_count_reads_ocr_variants():
    # 19층: "기술의 소리×15"를 OCR이 "%15"로 읽어 5개짜리 반값으로 알고 샀다
    mk = lambda name: ShopItem(0, name, 200, 400, False, (0, 0))
    assert mk("기술의 소리%15").note_count == 15
    assert mk("폭발의 소리*5").note_count == 5
    assert mk("강공의 소리x5").note_count == 5
    assert mk("필살기의 소리 15").note_count == 15
    plan = plan_purchases([mk("기술의 소리%15")], 3000, shop_index=3, last_shop=False, reroll_left=0, reroll_price=None)
    assert plan.buy == []


def test_read_shop_half_price_notes(ocr):
    # 12층: "폭발의 소리x5" 45원(반값)을 넓게 읽으면 옛 가격과 섞여 72원으로 알았다
    img = cv2.imdecode(np.fromfile(str(SCREENS / "shop__live_12.jpg"), np.uint8), cv2.IMREAD_COLOR)
    items = {i.slot: i for i in read_shop(img, ocr)}
    assert items[4].price == 45 and items[4].old_price == 90
    assert items[3].price == 72 and items[0].price == 160 and items[1].price == 100


def test_plan_notes_by_ensemble_needs():
    def note(slot, price, old, count, users, have=10):
        return ShopItem(slot, f"소리x{count}", price, old, False, (0, 0), note_type=0, users=users, have=have)
    items = [note(0, 45, 90, 5, 1), note(1, 72, 90, 5, 1), note(2, 72, 90, 5, 1, have=45), note(3, 200, 400, 15, 2),
             note(4, 200, 400, 15, 1), note(5, 90, None, 5, 4), note(6, 90, None, 5, 3), note(7, 320, 400, 15, 6)]
    plan = plan_purchases(items, 3000, shop_index=3, last_shop=False, reroll_left=0, reroll_price=None)
    assert sorted(i.slot for i in plan.buy) == [0, 1, 3, 5]
    # 협주스킬에 안 쓰는 소리는 45원이어도 안 산다
    items[0].users = 0
    plan = plan_purchases(items, 3000, shop_index=3, last_shop=False, reroll_left=0, reroll_price=None)
    assert 0 not in [i.slot for i in plan.buy]
    items[0].users = 1
    # 필요량을 못 읽었으면 45원짜리만
    for i in items:
        i.users = None
    plan = plan_purchases(items, 3000, shop_index=3, last_shop=False, reroll_left=0, reroll_price=None)
    assert [i.slot for i in plan.buy] == [0]


def test_read_ensemble_needs(ocr):
    from stella_auto import notes
    img = cv2.imdecode(np.fromfile(str(SCREENS / "bag__live_skills.jpg"), np.uint8), cv2.IMREAD_COLOR)
    n = notes.read_needs(img, ocr)
    assert n.users == {0: 3, 1: 2, 4: 1, 6: 3, 8: 6}
    assert n.have[8] == 16 and n.have[6] == 35 and n.have[0] == 24
    shop = cv2.imdecode(np.fromfile(str(SCREENS / "shop__live_12.jpg"), np.uint8), cv2.IMREAD_COLOR)
    types = [notes.shop_note_type(shop, i.click[0], i.click[1], i.name) for i in read_shop(shop, ocr) if i.kind == "notes"]
    assert types == [0, 2, 4, 2, 0]


def test_last_shop_spends_leftover_on_notes():
    items = [ShopItem(0, "잠재력 특제 음료", 200, None, False, (0, 0)),
             ShopItem(1, "바람의 소리x15", 320, 400, False, (0, 0), note_type=8, users=6),
             ShopItem(2, "집중의 소리x5", 90, None, False, (0, 0), note_type=4, users=1)]
    plan = plan_purchases(items, 700, shop_index=4, last_shop=True, reroll_left=0, reroll_price=None)
    assert sorted(i.slot for i in plan.buy) == [0, 1, 2]


def test_full_price_notes_skipped_when_others_low():
    item = ShopItem(0, "바람의 소리x5", 90, None, False, (0, 0), note_type=8, users=6, have=42)
    plan = plan_purchases([item], 3000, shop_index=3, last_shop=False, reroll_left=0, reroll_price=None,
                          note_have={8: 42, 6: 15, 0: 30})
    assert plan.buy == []
    plan = plan_purchases([item], 3000, shop_index=3, last_shop=False, reroll_left=0, reroll_price=None,
                          note_have={8: 42, 6: 25, 0: 30})
    assert plan.buy == [item]
