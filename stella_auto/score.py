"""완성된 도자기 점수 계산 (docs/tower-rules.md "도자기 점수").

배율은 사용자와 같이 조절해 나갈 값이라 ScoreWeights로 따로 뺐다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .preset import Preset


@dataclass
class ScoreWeights:
    essential: float = 100  # 필수, 레벨당 (보라/금 구분 없음)
    plenty_high: float = 70  # 다다익선 (목표 6렙), 레벨당
    plenty_low: float = 30  # 다다익선 (목표 3렙), 레벨당
    low: float = 5  # 후순위, 레벨당
    plenty_high_min_target: int = 6  # 다다익선 목표 레벨이 이 이상이면 plenty_high
    max_scored_level: int = 6  # 7–9레벨은 탑 밖 아이템으로 올리는 거라 도자기 가치에서 뺀다
    ensemble_level: float = 250  # 협주스킬 활성화 레벨 총합당
    note: float = 1  # 소리 1개당
    record_level_threshold: int = 31  # 도자기 평점 레벨이 이 이상이면 공격력 버프
    record_level_bonus: float = 200
    record_level_penalty: float = -200


@dataclass
class RecordResult:
    """기록 화면에서 읽은 완성 도자기."""

    potential_levels: dict[int, int]  # 잠재력 ID → 레벨 (없으면 0)
    ensemble_levels: list[int]  # 협주스킬 6개의 활성화 레벨
    notes: dict[str, int]  # 소리 종류 → 개수
    record_level: int  # 기록 화면 왼쪽 위 숫자 (도자기 평점 레벨)


@dataclass
class ScoreResult:
    total: float
    discard_reasons: list[str] = field(default_factory=list)
    breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def discard(self) -> bool:
        return bool(self.discard_reasons)


def potential_weight(mark: str | None, target_level: int, w: ScoreWeights) -> float:
    if mark == "필수":
        return w.essential
    if mark == "다다익선":
        return w.plenty_high if target_level >= w.plenty_high_min_target else w.plenty_low
    if mark == "후순위":
        return w.low
    return 0.0  # 명함만은 있는지만 본다. 표시 없는 잠재도 점수 없음


def score_record(preset: Preset, record: RecordResult, w: ScoreWeights | None = None) -> ScoreResult:
    w = w or ScoreWeights()
    reasons: list[str] = []
    breakdown = {"필수": 0.0, "다다익선": 0.0, "후순위": 0.0}

    for ch in preset.characters:
        for p in ch.potentials:
            if p.kind == "core":
                continue  # 코어는 탑에서 무조건 고를 수 있어서 점수에 안 넣는다
            level = record.potential_levels.get(p.id, 0)
            if p.mark in ("필수", "명함만") and level == 0:
                reasons.append(f"{p.mark} '{p.name}'({ch.name})이 없음")
            if p.mark in breakdown:
                scored = min(level, w.max_scored_level)  # 목표 레벨은 넘어도 6까지는 센다
                breakdown[p.mark] += scored * potential_weight(p.mark, p.target_level, w)

    breakdown["협주스킬"] = sum(record.ensemble_levels) * w.ensemble_level
    breakdown["소리"] = sum(record.notes.values()) * w.note
    breakdown["평점"] = (
        w.record_level_bonus if record.record_level >= w.record_level_threshold else w.record_level_penalty
    )
    return ScoreResult(sum(breakdown.values()), reasons, breakdown)
