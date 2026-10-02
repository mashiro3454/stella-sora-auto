"""카드 고르기 규칙 (docs/tower-rules.md "잠재력 카드", "강화머신").

카드마다 "고르면 오르는 도자기 점수"를 계산하고, 고르면 안 되는 카드(아직 이른 층, Lv1, 예산 초과)를
뺀 뒤 점수가 가장 높은 것을 고른다. 고를 게 없으면 리롤, 리롤을 너무 많이 하면 재시작.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cards import Card
from .gamedata import default_gamedata
from .preset import PotentialGoal, Preset
from .score import ScoreWeights, potential_weight

LV3_LIMIT = 10  # 한 판에 새 잠재력 Lv3 카드는 10번까지
PLUS2_PER_RUN = 5  # 강화머신 +2 강화는 한 판에 5번 (사용자 2026-10-03). 6레벨에서 잘리면 1레벨을 버린다
REROLL_COST = 40
MAX_REROLLS_PER_PICK = 5  # 한 번의 카드 선택에서 연속 5번 이상 리롤하면 재시작
MAX_REROLLS_EARLY = 10  # 6층까지 리롤이 10번을 넘으면 재시작
MAX_REROLLS_KEEP = 8  # 650원 이긴 판(재시작 안 함): 한 선택에서 리롤은 이만큼까지, 그 뒤엔 덜 나쁜 것
EARLY_FLOOR = 6
ESSENTIAL_SAFEGUARD_FLOOR = 13  # 이 층부터 아직 없는 필수는 Lv2 이상이면 먼저 집는다
SAFEGUARD_BONUS = 1000
CARD_ONLY_VALUE = 50  # 명함만: 1레벨이라도 있으면 되는 것
# 시험 (2026-10-02 사용자: "테스트해보는건 어때"): 13층부터 아직 0레벨인 필수가 있으면 그 카드가 뜰 때까지 리롤을 더 쓴다.
# 필수가 0레벨이면 기록은 어차피 버려지니 돈을 아낄 까닭이 적다. 카드 한 장에 그 필수가 뜰 확률은 1~3%.
FISH_FLOOR = 13
FISH_PER_PICK = 3  # 한 선택 화면에서 필수 찾기 리롤은 이만큼까지 (그다음엔 평소대로 고른다)
FISH_RESERVE = 200  # 이만큼은 남긴다
LAST_FLOOR = 20
# 사용자 규칙 (2026-10-02 영상 검토 뒤):
# - 20층에서 0레벨 필수가 있으면 그게 (Lv2 이상으로) 뜰 때까지 리롤한다. 한 화면 횟수 제한 없음, 돈이 있는 만큼.
# - 한 캐릭터의 보라+금 잠재가 6개 이상이고 그중 2개 이상이 6레벨 미만이면 그 캐릭터의 새 잠재는 안 뜬다.
#   그땐 리롤로 찾지 말고 그 캐릭터의 잠재를 6레벨로 올려서 막힌 걸 푼다.
CHAR_LOCK_COUNT = 6
CHAR_LOCK_UNFINISHED = 2
UNLOCK_BONUS = 500
# 막히진 않았어도 13층부터 0레벨 필수가 있는 캐릭터의 잠재 레벨업은 조금 더 쳐 주고, 필수 찾기 리롤 대신 집는다
# (6레벨에 가까워질수록 그 캐릭터의 새 잠재가 잘 뜬다. 17:47 20층: 섬멸의 잔향 5→6을 5번 넘기며 리롤 50번, 돈을 다 씀)
HELP_BONUS = 60
# - 다다익선(목표 3)은 지원 캐릭터는 13층부터, 메인(레이스)은 18층부터 레벨당 35점 (새 Lv3 105점 > 필수 +1 100점)
PLENTY_LOW_LATE = 35
PLENTY_LOW_LATE_FLOOR = {"master": 18, "assist": 13}
# - 목표를 넘은 다다익선(목표 3) 레벨업만 뜨면 리롤을 한 번 해 보고, 그래도 없으면 그냥 집는다
WEAK_REROLL_RESERVE = 200


@dataclass
class RunState:
    floor: int = 1
    gold: int = 0
    owned: dict[int, int] = field(default_factory=dict)  # 잠재력 ID -> 지금 레벨
    lv3_new_taken: int = 0
    lv2_new_taken: int = 0
    rerolls_early: int = 0  # 6층까지 한 리롤 수
    rerolls_this_pick: int = 0
    # 시험 (2026-10-02 밤): 650원 도박에 이긴 판은 리롤 횟수로 재시작하지 않는다. 이긴 판 둘이 6층에서
    # "리롤 10번", "한 선택에서 리롤 5번"으로 끝났다 (리롤 대상은 프리셋에 없거나 13/19층부터인 카드였다)
    keep_after_gamble: bool = False
    fish_this_pick: int = 0  # 이번 선택 화면에서 필수 찾기로 한 리롤
    fish_total: int = 0  # 이번 판 필수 찾기 리롤 (시험 기록용)
    plus2_taken: int = 0  # 강화머신에서 +2 강화 화면을 쓴 횟수 (한 판에 PLUS2_PER_RUN번뿐)


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
        self.role_of: dict[int, str] = {ch.char_id: "master" if ch.slot == "master" else "assist" for ch in preset.characters}
        main_goals = [g for g in self.goals.values() if g.kind != "core" and g.mark in ("필수", "다다익선")]
        self.lv2_budget = max(0, len(main_goals) - LV3_LIMIT)

    # -- 점수 -------------------------------------------------------------
    def _unacquired_main(self, state: RunState) -> int:
        """아직 0레벨인 필수/다다익선(목표 6): 새 Lv3 기회(10번)의 원래 주인들 (사용자 2026-10-02 23시).
        이들이 Lv2로라도 잡히면 기회가 남고, 남는 기회는 다다익선(목표 3) 새 Lv3에 써도 된다."""
        return sum(1 for g in self.goals.values()
                   if g.kind != "core" and state.owned.get(g.id, 0) == 0
                   and (g.mark == "필수" or (g.mark == "다다익선" and g.target_level >= self.w.plenty_high_min_target)))

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
            # 20층(마지막 층)엔 아낄 까닭이 없다 (영상: 20층 강풍의 자태 새 Lv2 → 상점 음료로 Lv6)
            if (card.level_to == 2 and main and not safeguard and state.lv2_new_taken >= self.lv2_budget
                    and state.floor < LAST_FLOOR):
                return None, f"Lv2 예산({self.lv2_budget}) 다 씀"
            plenty_low = goal.mark == "다다익선" and goal.target_level < self.w.plenty_high_min_target
            if (card.level_to >= 3 and (not main or plenty_low)
                    and LV3_LIMIT - state.lv3_new_taken <= self._unacquired_main(state)):
                return None, "Lv3 횟수는 필수/다다익선(6)용으로 남겨 둠"

        if goal.mark == "명함만":
            v = CARD_ONLY_VALUE if cur == 0 else 0.0
        else:
            cap = self.w.max_scored_level
            v = (min(card.level_to, cap) - min(cur, cap)) * self.weight(goal, state)
        if safeguard and card.level_to >= 2:
            v += SAFEGUARD_BONUS
        if v > 0:
            v += self.help_bonus(card, state)
        return float(v), goal.mark

    def help_bonus(self, card: Card, state: RunState) -> float:
        """0레벨 필수가 있는 캐릭터의 6레벨 안 된 잠재 레벨업: 막혔으면 500, 13층부터는 60."""
        if card.potential is None or card.is_new or (card.level_from or 0) >= 6:
            return 0.0
        cid = card.potential.char_id
        if not any(self.char_of.get(g.id) == cid for g in self.missing_essentials(state)):
            return 0.0
        if self.char_locked(cid, state):
            return float(UNLOCK_BONUS)
        return float(HELP_BONUS) if state.floor >= FISH_FLOOR else 0.0

    def weight(self, goal: PotentialGoal, state: RunState) -> float:
        w = potential_weight(goal.mark, goal.target_level, self.w)
        if goal.mark == "다다익선" and goal.target_level < self.w.plenty_high_min_target:
            role = self.role_of.get(self.char_of.get(goal.id, -1), "assist")
            if state.floor >= PLENTY_LOW_LATE_FLOOR[role]:
                w = PLENTY_LOW_LATE
        return w

    # -- 캐릭터별 새 잠재 막힘 (사용자 설명) ------------------------------------------
    @staticmethod
    def char_levels(char_id: int, state: RunState) -> list[int]:
        """이 캐릭터의 가진 보라+금 잠재 레벨들."""
        gd = default_gamedata()
        return [lv for pid, lv in state.owned.items() if lv > 0 and pid in gd.potentials
                and gd.potentials[pid].char_id == char_id and gd.potentials[pid].kind != "core"]

    def char_locked(self, char_id: int, state: RunState) -> bool:
        lv = self.char_levels(char_id, state)
        return len(lv) >= CHAR_LOCK_COUNT and sum(1 for x in lv if x < 6) >= CHAR_LOCK_UNFINISHED


    def _tiebreak(self, card: Card) -> tuple:
        # 점수가 같으면 (사용자 규칙, 2026-10-02 영상 검토):
        # 1) 목표 레벨을 못 채운 쪽 (섬광 발도 2→3 > 칼날의 춤사위 3→4)
        # 2) 프리셋 캐릭터 순서 (엘레노어 > 안즈 > 레이스. 6레벨을 먼저 채워야 그 캐릭터의 새 잠재가 잘 뜬다)
        # 3) 지금 레벨이 높은 쪽 (6레벨에 가까운 것부터)  4) 보라 (영상에선 한 번도 기준이 안 됐다)
        p = card.potential
        goal = self.goals.get(p.id) if p else None
        cur = card.level_from or 0
        below = 1 if goal and goal.mark in ("필수", "다다익선") and cur < goal.target_level else 0
        order = self.preset.priority
        cid = p.char_id if p else None
        prio = order.index(cid) if cid in order else len(order)
        rarity = p.rarity if p and p.rarity else 9
        return (below, -prio, cur, -rarity)

    def _upgrade_worth(self, card: Card, state: RunState, vals: dict) -> bool:
        """필수 찾기 리롤을 멈추고 집을 카드: 점수가 붙은 필수/다다익선 (목표 넘은 다다익선(3) 레벨업은 빼고).
        새 카드도 점수가 붙었으면 (다다익선(3) 새 Lv3는 남는 기회가 있을 때만 점수가 붙는다) 집는다."""
        if (vals[card.slot][0] or 0) <= 0:
            return False
        if card.is_new and card.level_to < 3:
            return False  # 새 Lv2 때문에 찾기를 멈추진 않는다 (찾던 필수의 Lv2는 hit이 따로 잡는다)
        goal = self.goals.get(card.potential.id) if card.potential else None
        return bool(goal and goal.mark in ("필수", "다다익선") and not self._weak(card, state))

    def _mains_high(self, state: RunState) -> bool:
        """필수가 다 잡혀 5레벨 이상이고 다다익선(6)도 다 4레벨 이상인지 (사용자 2026-10-02 23시):
        이쯤이면 다다익선(3)을 목표 넘게 올리는 것도 괜찮다 — 약한 카드 취급(리롤 한 번)을 그만둔다."""
        for g in self.goals.values():
            if g.kind == "core":
                continue
            lv = state.owned.get(g.id, 0)
            if g.mark == "필수" and lv < 5:
                return False
            if g.mark == "다다익선" and g.target_level >= self.w.plenty_high_min_target and lv < 4:
                return False
        return True

    def _weak(self, card: Card, state: RunState) -> bool:
        """목표를 넘은 다다익선(목표 3) 레벨업: 집긴 하지만 리롤을 한 번 해 볼 만한 카드."""
        goal = self.goals.get(card.potential.id) if card.potential else None
        return bool(goal and goal.mark == "다다익선" and goal.target_level < self.w.plenty_high_min_target
                    and (card.level_from or 0) >= goal.target_level)

    # -- 결정 -------------------------------------------------------------
    def missing_essentials(self, state: RunState) -> list[PotentialGoal]:
        """아직 0레벨인 필수 (이게 하나라도 있으면 기록은 버릴 조건)."""
        return [g for g in self.goals.values() if g.kind != "core" and g.mark == "필수" and state.owned.get(g.id, 0) == 0]

    def choose(self, cards: list[Card], state: RunState, *, can_reroll: bool = True) -> Decision:
        vals = {c.slot: self.value(c, state) for c in cards}
        ok = [c for c in cards if vals[c.slot][0] is not None and vals[c.slot][0] > 0]
        values = {s: v for s, (v, _) in vals.items()}
        last = state.floor >= LAST_FLOOR
        fishing = state.floor >= FISH_FLOOR and (state.keep_after_gamble or last)
        # 새 잠재가 막힌 캐릭터의 필수는 리롤로 못 찾는다 (그 캐릭터 잠재를 6레벨로 올리는 카드에 점수를 더 준다)
        missing = ([g for g in self.missing_essentials(state) if not self.char_locked(self.char_of.get(g.id, -1), state)]
                   if fishing else [])
        if missing and can_reroll:
            ids = {g.id for g in missing}
            hit = any(c.potential and c.potential.id in ids and c.level_to >= 2 for c in cards)
            # 프리셋 코어(1000점)가 떠 있으면 그냥 집는다 (13층에서 집중 속사를 리롤로 넘겼다)
            core = any((v or 0) >= SAFEGUARD_BONUS for v in values.values())
            # 필수/다다익선의 레벨업이 떠 있으면 리롤하지 말고 집는다 (사용자 2026-10-02 23시:
            # "리롤해도 되는 건 다다익선(3)이 이미 목표(3렙) 이상인 경우뿐". 22:53 15층: 결전의 순간 5→6을 넘겼다)
            upgrade = any(self._upgrade_worth(c, state, vals) for c in cards)
            # 20층은 마지막이라 한 화면 횟수 제한 없이, 남길 돈 없이 (사용자: "먹을 때까지 리롤")
            more = last or state.fish_this_pick < FISH_PER_PICK
            money = state.gold >= REROLL_COST + (0 if last else FISH_RESERVE)
            # 그 캐릭터 잠재를 6레벨 쪽으로 올리는 카드가 떠 있으면 리롤하지 않고 그걸 집는다 (사용자 설명)
            helps = any(vals[c.slot][0] and self.help_bonus(c, state) > 0 for c in cards)
            if not hit and not core and not helps and not upgrade and more and money:
                state.fish_this_pick += 1
                state.fish_total += 1
                return Decision("reroll", None, f"필수 찾기 리롤 (0레벨: {', '.join(g.name for g in missing)})", values)
        if ok:
            best = max(ok, key=lambda c: (vals[c.slot][0], self._tiebreak(c)))
            if (can_reroll and all(self._weak(c, state) for c in ok) and state.rerolls_this_pick == 0
                    and state.gold >= REROLL_COST + WEAK_REROLL_RESERVE and not self._mains_high(state)):
                return Decision("reroll", None, "목표 넘은 다다익선(3)만 떠서 한 번 리롤", values)
            return Decision("pick", best, f"{best.potential.name}: {vals[best.slot][1]} +{vals[best.slot][0]:.0f}", values)
        if state.keep_after_gamble or last:
            if state.rerolls_this_pick >= MAX_REROLLS_KEEP:
                can_reroll = False  # 아래에서 덜 나쁜 것을 고른다
        elif state.rerolls_this_pick >= MAX_REROLLS_PER_PICK:
            return Decision("restart", None, f"한 선택에서 리롤 {state.rerolls_this_pick}번", values)
        elif state.floor <= EARLY_FLOOR and state.rerolls_early >= MAX_REROLLS_EARLY:
            return Decision("restart", None, f"{EARLY_FLOOR}층까지 리롤 {state.rerolls_early}번", values)
        if can_reroll and state.gold >= REROLL_COST:
            return Decision("reroll", None, "고를 카드가 없음", values)
        # 리롤도 못 하면: 프리셋에 있는 카드 중 점수 높은 것, 그것도 없으면 첫 카드.
        # 단 필수/다다익선의 새 Lv1은 절대 안 집는다 (사용자 규칙) — 차라리 프리셋에 없는 카드
        def main_lv1(c: Card) -> bool:
            g = self.goals.get(c.potential.id) if c.potential else None
            return bool(c.is_new and c.level_to <= 1 and g and g.mark in ("필수", "다다익선"))
        allowed = [c for c in cards if not main_lv1(c)] or cards
        fallback = [c for c in allowed if c.potential and c.potential.id in self.goals]
        pick = max(fallback, key=lambda c: self.value(c, state, ignore_floor=True)[0] or 0) if fallback else allowed[0]
        return Decision("pick", pick, "리롤 못 함, 덜 나쁜 것", values)

    def choose_enhance(self, cards: list[Card], state: RunState) -> Decision:
        """강화머신: 오르는 점수가 가장 큰 카드. 같으면 5레벨을 만들지 않는 쪽 → 캐릭터 순서.

        +2 강화(판에 5번)는 6레벨에서 잘린다 (사용자 2026-10-03): +1 강화로 필수/다다익선(6)을 5레벨에
        세워 두면 나중에 +2가 떴을 때 1레벨을 버린다. 그래서 +2가 남아 있는 동안은 동점이면
        4레벨에 멈추는 쪽(3→4)을 5레벨을 만드는 쪽(4→5)보다 먼저 고른다. 5→6(마무리)은 괜찮다."""
        scored = [(self.value(c, state, ignore_floor=True)[0] or 0.0, c) for c in cards]
        values = {c.slot: v for v, c in scored}

        plus2_screen = any(c.level_from is not None and c.gain >= 2 for c in cards)

        def strands(c: Card) -> bool:
            goal = self.goals.get(c.potential.id) if c.potential else None
            main_high = goal and (goal.mark == "필수" or (goal.mark == "다다익선"
                                                          and goal.target_level >= self.w.plenty_high_min_target))
            return bool(main_high and c.level_to == 5 and state.plus2_taken < PLUS2_PER_RUN)

        def wasteful(c: Card) -> bool:
            # +2가 명함만/후순위/프리셋 밖 카드에 꽂히는 건 낭비, 다다익선(목표 3)은 13층 전까지 낭비
            # (사용자 2026-10-03). 낭비끼리만 남으면 그나마 점수 높은 것
            if not plus2_screen:
                return False
            goal = self.goals.get(c.potential.id) if c.potential else None
            if goal is None or goal.mark in (None, "후순위", "명함만"):
                return True
            return bool(goal.mark == "다다익선" and goal.target_level < self.w.plenty_high_min_target
                        and state.floor < FISH_FLOOR)

        best = max(scored, key=lambda t: (not wasteful(t[1]), t[0], not strands(t[1]), self._tiebreak(t[1])))[1]
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
        state.fish_this_pick = 0

    def record_reroll(self, state: RunState) -> None:
        state.rerolls_this_pick += 1
        state.gold -= REROLL_COST
        if state.floor <= EARLY_FLOOR:
            state.rerolls_early += 1
