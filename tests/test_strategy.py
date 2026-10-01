"""사용자가 예로 든 카드 선택 상황들 (docs/tower-rules.md "고르는 순서")."""

from pathlib import Path

import pytest

from stella_auto.cards import Card, team_pool
from stella_auto.preset import Preset
from stella_auto.strategy import CardChooser, RunState

PRESET = Preset.load(Path(__file__).parent.parent / "presets" / "바람.json")
POOL = team_pool(PRESET)
BY_NAME = {p.name: p for p in POOL.values()}


def card(name: str, to: int, frm: int | None = None, slot: int = 0) -> Card:
    return Card(slot, (0, 0), name, BY_NAME[name], 100, frm, to, 0, "")


def cards(*specs) -> list[Card]:
    return [card(*s, slot=i) if len(s) == 3 else card(s[0], s[1], None, slot=i) for i, s in enumerate(specs)]


@pytest.fixture
def chooser():
    return CardChooser(PRESET)


def owned(**levels) -> dict[int, int]:
    return {BY_NAME[n.replace("_", " ")].id: v for n, v in levels.items()}


def test_lv2_budget(chooser):
    assert chooser.lv2_budget == 4  # 필수 8 + 다다익선 6 - 10


def test_new_essential_beats_upgrades(chooser):
    # 필수 2렙(새) / 필수 4->5 / 다다익선 3->4 -> 필수 2렙
    st = RunState(floor=5, owned=owned(파멸의_질풍=4, 결전의_순간=3))
    d = chooser.choose(cards(("혼란스러운 흐름", 2), ("파멸의 질풍", 5, 4), ("결전의 순간", 4, 3)), st)
    assert d.action == "pick" and d.card.potential.name == "혼란스러운 흐름"


def test_plenty6_lv3_beats_essential_lv2(chooser):
    d = chooser.choose(cards(("관통 탄도", 2), ("결전의 순간", 3)), RunState(floor=5))
    assert d.card.potential.name == "결전의 순간"  # 210 > 200


def test_essential_lv2_beats_plenty3_lv3(chooser):
    d = chooser.choose(cards(("관통 탄도", 2), ("칼날의 춤사위", 3)), RunState(floor=14))
    assert d.card.potential.name == "관통 탄도"  # 200 > 90


def test_plenty3_waits_until_floor_13(chooser):
    d = chooser.choose(cards(("칼날의 춤사위", 3), ("보스의 포효", 3)), RunState(floor=12, gold=100))
    assert d.action == "reroll"


def test_late_upgrade_beats_low_priority_new(chooser):
    st = RunState(floor=19, owned=owned(드높은_기개=3))
    d = chooser.choose(cards(("정밀 영점 조절", 2), ("드높은 기개", 4, 3)), st)
    assert d.card.potential.name == "드높은 기개"


def test_purple_before_gold_when_equal(chooser):
    d = chooser.choose(cards(("제압 분석", 3), ("관통 탄도", 3)), RunState(floor=5))
    assert d.card.potential.name == "관통 탄도"  # 보라
    d = chooser.choose(cards(("관통 탄도", 2), ("제압 분석", 3)), RunState(floor=5))
    assert d.card.potential.name == "제압 분석"  # 레벨이 색보다 먼저


def test_never_lv1_for_main(chooser):
    d = chooser.choose(cards(("관통 탄도", 1), ("보스의 포효", 3)), RunState(floor=5, gold=100))
    assert d.action == "reroll"


def test_low_priority_only_from_19(chooser):
    assert chooser.choose(cards(("가속 돌파", 1)), RunState(floor=18, gold=100)).action == "reroll"
    d = chooser.choose(cards(("가속 돌파", 1)), RunState(floor=19))
    assert d.action == "pick"


def test_lv2_budget_spent(chooser):
    st = RunState(floor=5, gold=100, lv2_new_taken=4)
    assert chooser.choose(cards(("관통 탄도", 2)), st).action == "reroll"


def test_essential_safeguard_from_13(chooser):
    # 13층부터 아직 없는 필수는 Lv2라도 먼저 (Lv2 예산을 다 썼어도)
    st = RunState(floor=13, lv2_new_taken=4)
    d = chooser.choose(cards(("결전의 순간", 3), ("관통 탄도", 2)), st)
    assert d.card.potential.name == "관통 탄도"
    # 그래도 Lv1은 안 집는다
    assert chooser.choose(cards(("관통 탄도", 1)), RunState(floor=15, gold=100)).action == "reroll"


def test_keep_lv3_for_main(chooser):
    # 19층, Lv3 남은 횟수 2번인데 아직 없는 필수/다다익선이 많으면 후순위 Lv3는 안 집는다
    st = RunState(floor=19, gold=100, lv3_new_taken=8)
    assert chooser.choose(cards(("정밀 영점 조절", 3)), st).action == "reroll"


def test_reroll_limits(chooser):
    nothing = cards(("보스의 포효", 3), ("산맥의 흡입", 3))
    assert chooser.choose(nothing, RunState(floor=3, gold=100, rerolls_this_pick=5)).action == "restart"
    assert chooser.choose(nothing, RunState(floor=6, gold=100, rerolls_early=10)).action == "restart"
    assert chooser.choose(nothing, RunState(floor=7, gold=100, rerolls_early=10)).action == "reroll"
    assert chooser.choose(nothing, RunState(floor=3, gold=30)).action == "pick"  # 리롤할 돈이 없음


def test_preset_core(chooser):
    d = chooser.choose(cards(("연쇄 폭발", 1), ("기능 지속", 1), ("진원 확장", 1)), RunState(floor=1))
    assert d.card.potential.name == "기능 지속"


def test_enhance_uses_character_priority(chooser):
    st = RunState(floor=5, owned=owned(관통_탄도=3, 파멸의_질풍=3, 결전의_순간=3))
    d = chooser.choose_enhance(cards(("파멸의 질풍", 4, 3), ("관통 탄도", 4, 3), ("결전의 순간", 4, 3)), st)
    assert d.card.potential.name == "관통 탄도"  # 필수끼리 같으면 엘레노어 > 안즈


def test_record_pick_counts(chooser):
    st = RunState(floor=5)
    chooser.record_pick(card("관통 탄도", 3), st)
    chooser.record_pick(card("제압 분석", 2), st)
    chooser.record_pick(card("관통 탄도", 4, 3), st)
    assert (st.lv3_new_taken, st.lv2_new_taken) == (1, 1)
    assert st.owned[BY_NAME["관통 탄도"].id] == 4


def test_unknown_level_is_not_lv1(chooser):
    c = card("관통 탄도", 0)
    c.level_known = False
    v, why = chooser.value(c, RunState(floor=5))
    assert v is None and "못 읽음" in why



def test_upgrade_value_uses_card_level_when_owned_unknown(chooser):
    # 봇을 중간에 켜서 owned가 비어 있어도 "3>4"는 +1로 친다
    v, _ = chooser.value(card("숲속 공주의 은총", 4, 3), RunState(floor=5), ignore_floor=True)
    assert v == 100


def test_no_reroll_restart_after_gamble_win(chooser):
    # 6층에 프리셋에 없거나 19층부터인 카드만: 원래는 재시작, 650원 이긴 판은 리롤을 더 하고 그다음 덜 나쁜 것
    bad = cards(("무영 사냥꾼", 1), ("가속 돌파", 3), ("칼날의 춤사위", 3))
    st = RunState(floor=6, gold=2000, rerolls_early=12, rerolls_this_pick=5, keep_after_gamble=True)
    assert chooser.choose(bad, st).action == "reroll"
    st.rerolls_this_pick = 8
    assert chooser.choose(bad, st).action == "pick"
    st.keep_after_gamble = False
    assert chooser.choose(bad, st).action == "restart"


def test_fallback_never_takes_main_lv1(chooser):
    bad = cards(("관통 탄도", 1), ("무영 사냥꾼", 1), ("과열 사격", 3) if "과열 사격" in BY_NAME else ("가속 돌파", 1))
    st = RunState(floor=6, gold=0, rerolls_this_pick=8, keep_after_gamble=True)
    d = chooser.choose(bad, st)
    assert d.action == "pick" and d.card.potential.name != "관통 탄도"
