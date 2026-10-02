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


def test_fish_for_missing_essential(chooser):
    # 시험: 13층부터 0레벨 필수가 있으면 그 카드가 Lv2 이상으로 뜰 때까지 한 화면에서 리롤 3번까지
    have_all_but_one = {g.id: 3 for g in chooser.goals.values() if g.mark == "필수" and g.kind != "core"}
    missing = BY_NAME["섬멸의 잔향"].id
    have_all_but_one.pop(missing)
    offer = cards(("칼날의 춤사위", 4, 3), ("무영 사냥꾼", 3), ("바람의 섬광", 3))
    st = RunState(floor=14, gold=1000, owned=dict(have_all_but_one), keep_after_gamble=True)
    for _ in range(3):
        d = chooser.choose(offer, st)
        assert d.action == "reroll" and "필수 찾기" in d.reason
        chooser.record_reroll(st)
    assert chooser.choose(offer, st).action == "pick"  # 3번 하면 평소대로
    # 그 필수가 Lv2 이상으로 뜨면 바로 집는다
    st2 = RunState(floor=14, gold=1000, owned=dict(have_all_but_one), keep_after_gamble=True)
    d = chooser.choose(cards(("칼날의 춤사위", 4, 3), ("섬멸의 잔향", 2), ("바람의 섬광", 3)), st2)
    assert d.action == "pick" and d.card.potential.name == "섬멸의 잔향"
    # 12층, 돈 부족, 이기지 않은 판에선 안 함
    for st3 in (RunState(floor=12, gold=1000, owned=dict(have_all_but_one), keep_after_gamble=True),
                RunState(floor=14, gold=200, owned=dict(have_all_but_one), keep_after_gamble=True)):
        assert "필수 찾기" not in chooser.choose(offer, st3).reason


def test_fish_keeps_preset_core(chooser):
    # 13층: 프리셋 코어가 떠 있으면 필수 찾기 리롤을 하지 않고 코어를 집는다
    have = {g.id: 3 for g in chooser.goals.values() if g.mark == "필수" and g.kind != "core"}
    have.pop(BY_NAME["관통 탄도"].id)
    core = next(g for g in chooser.goals.values() if g.kind == "core")
    offer = [card(core.name, 1, slot=0), card("칼날의 춤사위", 3, slot=1), card("바람의 섬광", 3, slot=2)]
    st = RunState(floor=13, gold=1100, owned=have, keep_after_gamble=True)
    d = chooser.choose(offer, st)
    assert d.action == "pick" and d.card.potential.name == core.name


# -- 사용자 규칙 (2026-10-02 영상 검토 뒤) ------------------------------------------------

def test_plenty3_late_weight(chooser):
    # 지원 캐릭터의 다다익선(목표 3)은 13층부터 레벨당 35점: 새 Lv3(105) > 필수 +1(100) (19층 바람 장벽 > 숲속 3→4)
    have = all_but()
    have[BY_NAME["숲속 공주의 은총"].id] = 3
    st = RunState(floor=13, owned=have)
    d = chooser.choose(cards(("바람 장벽", 3), ("숲속 공주의 은총", 4, 3)), st)
    assert d.card.potential.name == "바람 장벽"
    # 메인(레이스)의 다다익선(목표 3)은 18층부터
    offer = cards(("칼날의 춤사위", 3), ("숲속 공주의 은총", 4, 3))
    assert chooser.choose(offer, RunState(floor=17, owned=have)).card.potential.name == "숲속 공주의 은총"
    assert chooser.choose(offer, RunState(floor=18, owned=have)).card.potential.name == "칼날의 춤사위"


def test_tiebreak_below_target_then_character_then_level(chooser):
    # 목표(3) 못 채운 쪽 먼저: 섬광 발도 2→3 > 칼날의 춤사위 3→4
    d = chooser.choose(cards(("칼날의 춤사위", 4, 3), ("섬광 발도", 3, 2)), RunState(floor=20, gold=0))
    assert d.card.potential.name == "섬광 발도"
    # 같은 캐릭터면 6레벨에 가까운 것: 혼란스러운 흐름 4→5 > 숲속 3→4 (영상 18층)
    d = chooser.choose(cards(("숲속 공주의 은총", 4, 3), ("혼란스러운 흐름", 5, 4)), RunState(floor=18))
    assert d.card.potential.name == "혼란스러운 흐름"
    # 캐릭터 순서가 레벨보다 먼저: 전투력 증폭(엘레노어) 3→4 > 혼란스러운 흐름(안즈) 5→6, 숲속 3→4 (영상 19층 강화)
    d = chooser.choose_enhance(cards(("숲속 공주의 은총", 4, 3), ("전투력 증폭", 4, 3), ("혼란스러운 흐름", 6, 5)),
                               RunState(floor=19))
    assert d.card.potential.name == "전투력 증폭"


def test_lv2_budget_ignored_on_last_floor(chooser):
    offer = cards(("강풍의 자태", 2), ("섬광 발도", 5, 4))
    assert chooser.choose(offer, RunState(floor=19, gold=0, lv2_new_taken=4)).card.potential.name == "섬광 발도"
    assert chooser.choose(offer, RunState(floor=20, gold=0, lv2_new_taken=4)).card.potential.name == "강풍의 자태"


def test_reroll_once_when_only_plenty3_over_target(chooser):
    offer = cards(("섬광 발도", 4, 3), ("칼날의 춤사위", 4, 3), ("과열 사격", 1))
    st = RunState(floor=20, gold=1000, owned=all_but())
    d = chooser.choose(offer, st)
    assert d.action == "reroll" and "한 번" in d.reason
    chooser.record_reroll(st)
    assert chooser.choose(offer, st).action == "pick"
    # 목표 못 채운 다다익선(3)이 같이 뜨면 바로 집는다
    assert chooser.choose(cards(("섬광 발도", 3, 2), ("칼날의 춤사위", 4, 3)), RunState(floor=20, gold=1000, owned=all_but())).action == "pick"


def all_but(*missing: str) -> dict[int, int]:
    have = {g.id: 6 for g in CardChooser(PRESET).goals.values() if g.mark == "필수" and g.kind != "core"}
    for m in missing:
        have.pop(BY_NAME[m].id)
    return have


def test_fish_without_limit_on_last_floor(chooser):
    # 20층: 0레벨 필수가 뜰 때까지 리롤 (650원 안 이긴 판도, 한 화면 3번 제한 없이, 남길 돈 없이)
    offer = cards(("칼날의 춤사위", 4, 3), ("무영 사냥꾼", 3), ("바람의 섬광", 3))
    st = RunState(floor=20, gold=1000, owned=all_but("섬멸의 잔향"), fish_this_pick=10, rerolls_this_pick=10)
    assert "필수 찾기" in chooser.choose(offer, st).reason
    st.gold = 40
    assert chooser.choose(offer, st).action == "reroll"
    st.gold = 30
    assert chooser.choose(offer, st).action == "pick"  # 돈이 없으면 평소대로 (재시작 안 함)
    # Lv1로 뜨면 원칙대로 안 집고 계속 찾는다
    st.gold = 1000
    d = chooser.choose(cards(("칼날의 춤사위", 4, 3), ("섬멸의 잔향", 1)), st)
    assert d.action == "reroll"


def test_locked_character_upgrades_instead_of_fishing(chooser):
    # 엘레노어: 보라+금 6개, 그중 2개가 6레벨 미만 → 새 엘레노어 잠재(섬멸의 잔향)가 안 뜬다.
    # 리롤로 찾지 않고, 엘레노어 잠재를 6레벨로 올리는 카드를 먼저 집는다
    have = all_but("섬멸의 잔향")
    have.update({BY_NAME["관통 탄도"].id: 4, BY_NAME["전투력 증폭"].id: 5, BY_NAME["정밀 영점 조절"].id: 6,
                 BY_NAME["바람 정령의 공명"].id: 6, BY_NAME["동력 증압"].id: 6})
    st = RunState(floor=20, gold=1000, owned=have)
    assert chooser.char_locked(BY_NAME["관통 탄도"].char_id, st)
    d = chooser.choose(cards(("결전의 순간", 5, 4), ("관통 탄도", 5, 4)), st)
    assert d.action == "pick" and d.card.potential.name == "관통 탄도"
    # 6레벨 미만이 하나뿐이면 안 막혀 있다: 다시 리롤로 찾는다
    have[BY_NAME["관통 탄도"].id] = 6
    st = RunState(floor=20, gold=1000, owned=have)
    assert not chooser.char_locked(BY_NAME["관통 탄도"].char_id, st)
    assert "필수 찾기" in chooser.choose(cards(("칼날의 춤사위", 4, 3), ("바람의 섬광", 3)), st).reason


def test_fish_takes_upgrade_of_same_character(chooser):
    # 17:47 20층: 전투력 증폭(엘레노어)을 찾으며 섬멸의 잔향 5→6을 5번 넘기고 리롤 50번에 돈을 다 썼다.
    # 같은 캐릭터 잠재를 6레벨 쪽으로 올리는 카드가 뜨면 리롤 대신 집는다 (새 엘레노어 잠재가 잘 뜨게)
    have = all_but("전투력 증폭")
    have.update({BY_NAME["섬멸의 잔향"].id: 5, BY_NAME["바람 정령의 공명"].id: 2})
    st = RunState(floor=20, gold=1500, owned=have)
    d = chooser.choose(cards(("동력 증압", 1), ("바람 정령의 공명", 3, 2), ("섬멸의 잔향", 6, 5)), st)
    assert d.action == "pick" and d.card.potential.name == "섬멸의 잔향"
    # 필수 레벨업(숲속 4→5)이 떠 있으면 리롤 대신 집는다 (사용자 23시 규칙)
    d = chooser.choose(cards(("바람 장벽", 2), ("가속 돌파", 1), ("숲속 공주의 은총", 5, 4)), st)
    assert d.action == "pick" and d.card.potential.name == "숲속 공주의 은총"
    # 목표 넘은 다다익선(3) 레벨업뿐이면 계속 찾는다
    have2 = dict(st.owned); have2[BY_NAME["칼날의 춤사위"].id] = 3
    st2 = RunState(floor=20, gold=1500, owned=have2)
    d = chooser.choose(cards(("바람 장벽", 2), ("가속 돌파", 1), ("칼날의 춤사위", 4, 3)), st2)
    assert d.action == "reroll" and "필수 찾기" in d.reason


def test_preset_enhance_first_roundtrip(tmp_path):
    # 강화 우선 캐릭터 (사용자 2026-10-02 22시): 지원 캐릭터 잠재를 메인보다 우선하는 프리셋용 옵션
    p = Preset.load(Path(__file__).parent.parent / "presets" / "바람.json")
    assert p.enhance_first == 137  # 엘레노어
    p.set_enhance_first("레이스")
    assert p.enhance_first == 143
    p.set_enhance_first(None)
    assert p.enhance_first is None
    import pytest as _pytest
    with _pytest.raises(ValueError):
        p.set_enhance_first("없는애")
    p.set_enhance_first("엘레노어")
    f = tmp_path / "p.json"
    p.save(f)
    assert Preset.load(f).enhance_first == 137


def test_enhance_first_defers_to_shop():
    from stella_auto.runner import Bot
    b = Bot(Preset.load(Path(__file__).parent.parent / "presets" / "바람.json"))
    b.run_tracked = True
    b.run.owned = {BY_NAME["혼란스러운 흐름"].id: 3, BY_NAME["드높은 기개"].id: 2}
    b.shop_done = False
    assert b.defer_enhance_for_priority()  # 엘레노어 잠재가 없다: 상점 음료 먼저
    b.run.owned[BY_NAME["관통 탄도"].id] = 2
    assert not b.defer_enhance_for_priority()  # 생기면 강화 먼저
    b.run.owned[BY_NAME["관통 탄도"].id] = 6  # 다시 다 떨어짐
    b.shop_done, b.shop_potions_left, b.shop_reopens, b.run.gold = True, True, 0, 1000
    assert b.defer_enhance_for_priority()  # 음료가 남았고 돈이 있다: 상점 다시
    b.run.gold = 300
    assert not b.defer_enhance_for_priority()  # 돈이 없으면 그냥 다른 캐릭터 강화
    b.run.gold, b.shop_reopens = 1000, 2
    assert not b.defer_enhance_for_priority()  # 두 번 다시 갔으면 그만
    b.preset.enhance_first = None
    b.shop_reopens = 0
    assert not b.defer_enhance_for_priority()  # 옵션이 꺼진 프리셋은 예전 그대로


def test_fish_stops_for_essential_or_plenty6_upgrade(chooser):
    # 사용자 2026-10-02 23시: 필수/다다익선(6)의 n→n+1 레벨업이 떠 있으면 리롤하지 말고 집는다
    # (22:53 15층: 결전의 순간 5→6을 필수 찾기 리롤로 넘겼다)
    have = all_but("섬멸의 잔향")
    have[BY_NAME["결전의 순간"].id] = 5
    st = RunState(floor=15, gold=1500, owned=have, keep_after_gamble=True)
    d = chooser.choose(cards(("폭풍의 흡수", 3), ("결전의 순간", 6, 5), ("무영 사냥꾼", 4, 3)), st)
    assert d.action == "pick" and d.card.potential.name == "결전의 순간"
    # 다다익선(3)이 목표 아래(2→3)인 레벨업도 집는다
    have2 = all_but("섬멸의 잔향")
    have2[BY_NAME["섬광 발도"].id] = 2
    st2 = RunState(floor=15, gold=1500, owned=have2, keep_after_gamble=True)
    d = chooser.choose(cards(("폭풍의 흡수", 3), ("섬광 발도", 3, 2)), st2)
    assert d.action == "pick" and d.card.potential.name == "섬광 발도"


def test_weak_upgrade_ok_when_mains_high(chooser):
    # 사용자 23시: 필수가 다 5렙 이상 + 다다익선(6)이 다 4렙 이상이면 다다익선(3)을 목표 넘게 올려도 된다
    high = {g.id: 6 for g in chooser.goals.values() if g.kind != "core" and g.mark in ("필수", "다다익선")}
    high[BY_NAME["칼날의 춤사위"].id] = 3
    offer = cards(("바람의 섬광", 1), ("칼날의 춤사위", 4, 3))
    d = chooser.choose(offer, RunState(floor=20, gold=1000, owned=dict(high)))
    assert d.action == "pick" and d.card.potential.name == "칼날의 춤사위"
    # 필수가 아직 4렙이면 전처럼 한 번 리롤해 본다
    low = dict(high); low[BY_NAME["드높은 기개"].id] = 4
    d = chooser.choose(offer, RunState(floor=20, gold=1000, owned=low))
    assert d.action == "reroll" and "한 번" in d.reason


def test_enhance_avoids_stranding_at_5(chooser):
    # 사용자 2026-10-03: +2 강화(판에 5번)가 5레벨에 떨어지면 1레벨을 버린다.
    # +1 강화 동점이면 5레벨을 만드는 쪽(4→5) 대신 4레벨에 멈추는 쪽(3→4)을 고른다 (4,4를 만든다)
    st = RunState(floor=5, owned=owned(섬멸의_잔향=4, 관통_탄도=3))
    d = chooser.choose_enhance(cards(("섬멸의 잔향", 5, 4), ("관통 탄도", 4, 3)), st)
    assert d.card.potential.name == "관통 탄도"
    # +2를 5번 다 썼으면 예전 동점 규칙대로 (레벨 높은 쪽 먼저)
    st2 = RunState(floor=5, owned=owned(섬멸의_잔향=4, 관통_탄도=3), plus2_taken=5)
    d = chooser.choose_enhance(cards(("섬멸의 잔향", 5, 4), ("관통 탄도", 4, 3)), st2)
    assert d.card.potential.name == "섬멸의 잔향"
    # 5→6(마무리)은 5레벨을 만드는 게 아니라서 그대로
    d = chooser.choose_enhance(cards(("섬멸의 잔향", 6, 5), ("바람 정령의 공명", 3, 2)), RunState(floor=5))
    assert d.card.potential.name == "섬멸의 잔향"


def test_defer_enhance_when_all_at_5(chooser):
    from stella_auto.runner import Bot
    b = Bot(Preset.load(Path(__file__).parent.parent / "presets" / "바람.json"))
    b.run_tracked = True
    b.run.owned = {BY_NAME["섬멸의 잔향"].id: 5, BY_NAME["관통 탄도"].id: 5}
    b.shop_done = False
    assert b.defer_enhance_for_plus2()  # 전부 5레벨 + +2 남음: 음료부터
    b.run.plus2_taken = 5
    assert not b.defer_enhance_for_plus2()  # +2를 다 썼으면 그냥 강화 (5→6 마무리)
    b.run.plus2_taken = 0
    b.run.owned[BY_NAME["관통 탄도"].id] = 4
    assert not b.defer_enhance_for_plus2()  # 4레벨짜리가 있으면 +2를 받을 수 있다


def test_plus2_not_into_plenty3_before_13(chooser):
    # 사용자 2026-10-03: 13층 전에는 다다익선(3)에 +2가 꽂히는 것도 낭비
    offer = cards(("칼날의 춤사위", 4, 2), ("섬멸의 잔향", 6, 5))
    st = RunState(floor=5, owned=owned(칼날의_춤사위=2, 섬멸의_잔향=5))
    d = chooser.choose_enhance(offer, st)
    assert d.card.potential.name == "섬멸의 잔향"  # 60점이어도 칼날(다다3) 대신 필수 5→6
    # 13층부터는 다다익선(3)도 +2를 받아도 된다 (후순위보다 먼저)
    offer2 = cards(("칼날의 춤사위", 4, 2), ("무영 사냥꾼", 5, 3))
    d = chooser.choose_enhance(offer2, RunState(floor=13, owned=owned(칼날의_춤사위=2, 무영_사냥꾼=3)))
    assert d.card.potential.name == "칼날의 춤사위"
    # 13층 전엔 낭비끼리면 그나마 점수 높은 것 (칼날 > 후순위)
    d = chooser.choose_enhance(offer2, RunState(floor=5, owned=owned(칼날의_춤사위=2, 무영_사냥꾼=3)))
    assert d.card.potential.name == "칼날의 춤사위"


def test_fish_takes_plenty3_lv2_when_rest_useless(chooser):
    # 사용자 2026-10-03 01시: 13층부터 다다익선(3) 새 Lv2만 쓸 만하고 나머지가 쓸모없으면 리롤 대신 집는다
    have = all_but("관통 탄도", "섬멸의 잔향")
    st = RunState(floor=15, gold=1700, owned=dict(have), keep_after_gamble=True)
    d = chooser.choose(cards(("칼날의 춤사위", 2), ("격노의 바람", 3), ("새벽을 여는 칼날", 1)), st)
    assert d.action == "pick" and d.card.potential.name == "칼날의 춤사위"
    # Lv2 예산을 다 썼으면 (점수가 안 붙어) 그대로 리롤
    st2 = RunState(floor=15, gold=1700, owned=dict(have), keep_after_gamble=True, lv2_new_taken=4)
    d = chooser.choose(cards(("칼날의 춤사위", 2), ("격노의 바람", 3), ("새벽을 여는 칼날", 1)), st2)
    assert d.action == "reroll" and "필수 찾기" in d.reason


def test_no_fishing_on_faced_drink_screen(chooser):
    # 01:22 19층: 레이스 얼굴 음료의 카드 화면에서 엘레노어 필수를 찾아 리롤했다 (돈 낭비).
    # 얼굴 음료 화면(pool_char)에선 그 캐릭터의 필수를 찾는 게 아니면 필수 찾기를 끈다
    have = all_but("관통 탄도", "섬멸의 잔향")  # 엘레노어 필수 2개가 0레벨
    offer = cards(("칼날의 춤사위", 5, 4), ("무영 사냥꾼", 1), ("과열 사격", 3))
    st = RunState(floor=19, gold=1500, owned=dict(have), keep_after_gamble=True)
    d = chooser.choose(offer, st, pool_char=143)  # 레이스 음료
    assert d.action == "pick" and d.card.potential.name == "칼날의 춤사위"
    # 민무늬 음료(또는 일반 레벨업)에선 그대로 찾는다
    d = chooser.choose(offer, st)
    assert d.action == "reroll" and "필수 찾기" in d.reason
    # 엘레노어 얼굴 음료면 엘레노어 필수를 그대로 찾는다
    d = chooser.choose(offer, RunState(floor=19, gold=1500, owned=dict(have), keep_after_gamble=True), pool_char=137)
    assert d.action == "reroll" and "필수 찾기" in d.reason
