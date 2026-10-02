"""sstoy 빌드 → 자동화가 쓸 목표 도자기(프리셋)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..gamedata import GameData, default_gamedata
from .sstoy_codec import MARKS, RawBuild, decode_share_code, extract_code

SLOT_LABELS = {"master": "메인", "assist1": "지원1", "assist2": "지원2"}
MAX_CORE_PER_CHAR = 2  # sstoy togglePotential: 코어(Stype 42)는 캐릭터당 2개까지


@dataclass
class PotentialGoal:
    id: int
    name: str
    kind: str  # core / normal / common
    card_color: str  # pink / purple / gold
    order: int  # sstoy 목록에서의 순서 (0부터). 우선순위인지는 아직 미정
    target_level: int
    max_level: int
    mark: str | None  # 필수 / 후순위 / 다다익선 / 명함만 / None(표시 없음)


@dataclass
class CharacterGoal:
    slot: str  # master / assist1 / assist2
    char_id: int
    name: str
    potentials: list[PotentialGoal]


@dataclass
class Preset:
    title: str
    characters: list[CharacterGoal]
    share_code: str
    warnings: list[str] = field(default_factory=list)
    # 같은 점수일 때 강화할 캐릭터 순서 (char_id, 앞일수록 먼저). sstoy 링크에는 없어서 사용자가 적는다
    priority: list[int] = field(default_factory=list)
    # 강화 우선 캐릭터 (char_id). 보통은 메인 캐릭터 잠재가 제일 잘 나와서 필요 없지만, 지원 캐릭터
    # 잠재를 메인보다 우선해야 하는 프리셋이 있다 (사용자 2026-10-02 22시, 바람 프리셋의 엘레노어).
    # 켜져 있으면 거래의 방에서: 이 캐릭터의 강화할 잠재가 없으면 상점 음료부터 사서 얻고 나서 강화머신에 간다
    enhance_first: int | None = None

    def set_enhance_first(self, name: str | None) -> None:
        if name is None:
            self.enhance_first = None
            return
        by_key = {c.name: c.char_id for c in self.characters} | {str(c.char_id): c.char_id for c in self.characters}
        if name not in by_key:
            raise ValueError(f"팀에 없는 캐릭터: {name} (팀: {', '.join(c.name for c in self.characters)})")
        self.enhance_first = by_key[name]

    def set_priority(self, names: list[str]) -> None:
        """'엘레노어,안즈,레이스'처럼 이름(또는 ID)으로 순서를 정한다."""
        by_key = {c.name: c.char_id for c in self.characters} | {str(c.char_id): c.char_id for c in self.characters}
        unknown = [n for n in names if n not in by_key]
        if unknown:
            team = ", ".join(c.name for c in self.characters)
            raise ValueError(f"팀에 없는 캐릭터: {', '.join(unknown)} (팀: {team})")
        order = [by_key[n] for n in names]
        if len(set(order)) != len(order):
            raise ValueError("같은 캐릭터가 두 번 들어감")
        self.priority = order

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def save(self, path: Path) -> None:
        path.write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Preset":
        raw = json.loads(path.read_text(encoding="utf-8"))
        chars = [
            CharacterGoal(**{**c, "potentials": [PotentialGoal(**p) for p in c["potentials"]]})
            for c in raw["characters"]
        ]
        return cls(raw["title"], chars, raw["share_code"], raw.get("warnings", []), raw.get("priority", []),
                   raw.get("enhance_first"))


def build_preset(raw: RawBuild, share_code: str, gd: GameData) -> Preset:
    warnings: list[str] = []
    chars: list[CharacterGoal] = []

    for slot, rc in raw.characters.items():
        label = SLOT_LABELS[slot]
        role = "master" if slot == "master" else "assist"
        info = gd.characters.get(rc.char_id)
        if info is None:
            warnings.append(f"{label}: 모르는 캐릭터 ID {rc.char_id} (tools/update_gamedata.py로 데이터 갱신 필요?)")
        char_name = info.name if info else str(rc.char_id)
        pool = set(info.pool(role)) if info else set()

        goals: list[PotentialGoal] = []
        # 선택 해제한 잠재력의 레벨/표시도 sstoy가 남겨두므로 levels/marks는
        # 선택된 목록(rc.potentials)에 있는 것만 본다.
        for order, pid in enumerate(rc.potentials):
            pi = gd.potentials.get(pid)
            if pi is None:
                warnings.append(f"{label} {char_name}: 모르는 잠재력 ID {pid}")
                continue
            if info and pid not in pool:
                warnings.append(f"{label} {char_name}: '{pi.name}'({pid})는 {label} 역할로는 안 나오는 잠재력")
            level = rc.levels.get(pid, 1)
            if level > pi.max_level:
                warnings.append(f"{label} {char_name}: '{pi.name}' 목표 레벨 {level} > 최대 {pi.max_level}")
            goals.append(
                PotentialGoal(
                    id=pid,
                    name=pi.name,
                    kind=pi.kind,
                    card_color=pi.card_color,
                    order=order,
                    target_level=level,
                    max_level=pi.max_level,
                    mark=MARKS.get(rc.marks.get(pid, 0)),
                )
            )

        core_count = sum(g.kind == "core" for g in goals)
        if core_count > MAX_CORE_PER_CHAR:
            warnings.append(f"{label} {char_name}: 코어 잠재력이 {core_count}개 (최대 {MAX_CORE_PER_CHAR})")
        chars.append(CharacterGoal(slot, rc.char_id, char_name, goals))

    return Preset(raw.name, chars, share_code, warnings)


def load_preset(link: str, gd: GameData | None = None) -> Preset:
    """sstoy 공유 링크(또는 v3d-... 코드)로 프리셋을 만든다."""
    return build_preset(decode_share_code(link), extract_code(link), gd or default_gamedata())
