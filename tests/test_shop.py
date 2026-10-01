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
