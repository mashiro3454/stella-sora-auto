"""카드 고르기 규칙 (docs/tower-rules.md "잠재력 카드", "강화머신").

카드마다 "고르면 오르는 도자기 점수"를 계산하고, 고르면 안 되는 카드(아직 이른 층, Lv1, 예산 초과)를
뺀 뒤 점수가 가장 높은 것을 고른다. 고를 게 없으면 리롤, 리롤을 너무 많이 하면 재시작.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cards import Card
from .preset import PotentialGoal, Preset
from .score import ScoreWeights, potential_weight

LV3_LIMIT = 10  # 한 판에 새 잠재력 Lv3 카드는 10번까지
REROLL_COST = 40
MAX_REROLLS_PER_PICK = 5  # 한 번의 카드 선택에서 연속 5번 이상 리롤하면 재시작
MAX_REROLLS_EARLY = 10  # 6층까지 리롤이 10번을 넘으면 재시작
EARLY_FLOOR = 6
ESSENTIAL_SAFEGUARD_FLOOR = 13  # 이 층부터 아직 없는 필수는 Lv2 이상이면 먼저 집는다
SAFEGUARD_BONUS = 1000
CARD_ONLY_VALUE = 50  # 명함만: 1레벨이라도 있으면 되는 것


@dataclass
class RunState:
    floor: int = 1
    gold: int = 0
    owned: dict[int, int] = field(default_factory=dict)  # 잠재력 ID -> 지금 레벨
    lv3_new_taken: int = 0
    lv2_new_taken: int = 0
    rerolls_early: int = 0  # 6층까지 한 리롤 수
    rerolls_this_pick: int = 0


@dataclass
class Decision:
    action: str  # "pick", "reroll", "restart"
    card: Card | None = None
    reason: str = ""
    values: dict[int, float | None] = field(default_factory=dict)  # 카드 slot -> 점수 (못 고르면 None)


def start_floor(goal: PotentialGoal) -> int:
    """이 표시의 잠재력을 고르기 시작하는 층."""
    if goal.mark == "필수":
        return 1
    if goal.mark == "다다익선":
        return 1 if goal.target_level >= 6 else 13
    return 19  # 후순위, 명함만


class CardChooser:
    def __init__(self, preset: Preset, weights: ScoreWeights | None = None):
        self.preset = preset
        self.w = weights or ScoreWeights()
        self.goals: dict[int, PotentialGoal] = {p.id: p for ch in preset.characters for p in ch.potentials}
        self.char_of: dict[int, int] = {p.id: ch.char_id for ch in preset.characters for p in ch.potentials}
        main_goals = [g for g in self.goals.values() if g.kind != "core" and g.mark in ("필수", "다다익선")]
        self.lv2_budget = max(0, len(main_goals) - LV3_LIMIT)

    # -- 점수 -------------------------------------------------------------
    def _unacquired_main(self, state: RunState) -> int:
        return sum(1 for g in self.goals.values()
                   if g.kind != "core" and g.mark in ("필수", "다다익선") and state.owned.get(g.id, 0) == 0)

    def value(self, card: Card, state: RunState, *, ignore_floor: bool = False) -> tuple[float | None, str]:
        """(점수, 이유). 점수가 None이면 고르면 안 되는 카드."""
        if card.potential is None:
            return None, "모르는 카드"
        if not card.level_known:
            return None, "레벨을 못 읽음"
        goal = self.goals.get(card.potential.id)
        if card.potential.kind == "core":
            return (1000.0, "프리셋 코어") if goal else (None, "프리셋에 없는 코어")
        if goal is None or goal.mark is None:
            return None, "프리셋에 없음"
        if not ignore_floor and state.floor < start_floor(goal):
            return None, f"{goal.mark}{'(목표 ' + str(goal.target_level) + ')' if goal.mark == '다다익선' else ''}은 {start_floor(goal)}층부터"

        # 업그레이드 카드는 카드에 적힌 "전 레벨"이 지금 레벨이다 (봇을 중간에 켜서 owned를 몰라도 맞게)
        cur = card.level_from if card.level_from is not None else state.owned.get(goal.id, 0)
        main = goal.mark in ("필수", "다다익선")
        safeguard = goal.mark == "필수" and cur == 0 and state.floor >= ESSENTIAL_SAFEGUARD_FLOOR
        if card.is_new and not ignore_floor:
            if card.level_to <= 1 and main:
                return None, "Lv1은 안 집음"
            if card.level_to == 2 and main and not safeguard and state.lv2_new_taken >= self.lv2_budget:
                return None, f"Lv2 예산({self.lv2_budget}) 다 씀"
            if card.level_to >= 3 and not main and LV3_LIMIT - state.lv3_new_taken <= self._unacquired_main(state):
                return None, "Lv3 횟수는 필수/다다익선용으로 남겨 둠"

        if goal.mark == "명함만":
            v = CARD_ONLY_VALUE if cur == 0 else 0.0
        else:
            cap = self.w.max_scored_level
            v = (min(card.level_to, cap) - min(cur, cap)) * potential_weight(goal.mark, goal.target_level, self.w)
        if safeguard and card.level_to >= 2:
            v += SAFEGUARD_BONUS
        return float(v), goal.mark

    def _tiebreak(self, card: Card) -> tuple:
        # 점수가 같으면: 보라(rarity 1)가 금보다 덜 나오니 먼저, 그다음 프리셋의 캐릭터 순서
        rarity = card.potential.rarity if card.potential and card.potential.rarity else 9
        order = self.preset.priority
        cid = self.char_of.get(card.potential.id) if card.potential else None
        prio = order.index(cid) if cid in order else len(order)
        return (-rarity, -prio)

    # -- 결정 -------------------------------------------------------------
    def choose(self, cards: list[Card], state: RunState, *, can_reroll: bool = True) -> Decision:
        vals = {c.slot: self.value(c, state) for c in cards}
        ok = [c for c in cards if vals[c.slot][0] is not None and vals[c.slot][0] > 0]
        values = {s: v for s, (v, _) in vals.items()}
        if ok:
            best = max(ok, key=lambda c: (vals[c.slot][0], self._tiebreak(c)))
            return Decision("pick", best, f"{best.potential.name}: {vals[best.slot][1]} +{vals[best.slot][0]:.0f}", values)
        if state.rerolls_this_pick >= MAX_REROLLS_PER_PICK:
            return Decision("restart", None, f"한 선택에서 리롤 {state.rerolls_this_pick}번", values)
        if state.floor <= EARLY_FLOOR and state.rerolls_early >= MAX_REROLLS_EARLY:
            return Decision("restart", None, f"{EARLY_FLOOR}층까지 리롤 {state.rerolls_early}번", values)
        if can_reroll and state.gold >= REROLL_COST:
            return Decision("reroll", None, "고를 카드가 없음", values)
        # 리롤도 못 하면: 프리셋에 있는 카드 중 점수 높은 것, 그것도 없으면 첫 카드
        fallback = [c for c in cards if c.potential and c.potential.id in self.goals]
        pick = max(fallback, key=lambda c: self.value(c, state, ignore_floor=True)[0] or 0) if fallback else cards[0]
        return Decision("pick", pick, "리롤 못 함, 덜 나쁜 것", values)

    def choose_enhance(self, cards: list[Card], state: RunState) -> Decision:
        """강화머신: 오르는 점수가 가장 큰 카드. 같으면 프리셋 캐릭터 순서."""
        scored = [(self.value(c, state, ignore_floor=True)[0] or 0.0, c) for c in cards]
        values = {c.slot: v for v, c in scored}
        best = max(scored, key=lambda t: (t[0], self._tiebreak(t[1])))[1]
        return Decision("pick", best, f"강화: {best.potential.name if best.potential else '?'}", values)

    def record_pick(self, card: Card, state: RunState) -> None:
        """카드를 가져간 뒤 상태 갱신."""
        if card.potential is None:
            return
        if card.is_new and card.potential.kind != "core":
            if card.level_to >= 3:
                state.lv3_new_taken += 1
            elif card.level_to == 2 and self.goals.get(card.potential.id) and self.goals[card.potential.id].mark in ("필수", "다다익선"):
                state.lv2_new_taken += 1
        state.owned[card.potential.id] = max(state.owned.get(card.potential.id, 0), card.level_to)
        state.rerolls_this_pick = 0

    def record_reroll(self, state: RunState) -> None:
        state.rerolls_this_pick += 1
        state.gold -= REROLL_COST
        if state.floor <= EARLY_FLOOR:
            state.rerolls_early += 1
