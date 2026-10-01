"""data/gamedata.json 로더. 파일은 tools/update_gamedata.py로 만든다."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "gamedata.json"

# 영상(2026-10-01 녹화)으로 확인한 카드 테두리 색
CARD_COLORS = {"core": "pink", 1: "purple", 2: "gold"}


@dataclass(frozen=True)
class PotentialInfo:
    id: int
    char_id: int
    name: str
    kind: str  # core / normal / common
    role: str  # master / assist / both
    max_level: int
    rarity: int | None
    build: int | None  # Potential.json Build 값. 유파와 관련 있어 보이지만 확인 안 됨
    brief: str

    @property
    def card_color(self) -> str:
        if self.kind == "core":
            return CARD_COLORS["core"]
        return CARD_COLORS.get(self.rarity, "unknown")


@dataclass(frozen=True)
class CharacterInfo:
    id: int
    name: str
    potentials: dict[str, tuple[int, ...]]  # master_core, master_normal, assist_core, assist_normal, common

    def pool(self, role: str) -> tuple[int, ...]:
        """해당 역할(master/assist)로 나왔을 때 뜰 수 있는 잠재력 전체."""
        p = self.potentials
        return p[f"{role}_core"] + p[f"{role}_normal"] + p["common"]


class GameData:
    def __init__(self, raw: dict):
        self.source = raw.get("source", {})
        self.characters = {
            int(k): CharacterInfo(int(k), v["name"], {lk: tuple(ids) for lk, ids in v["potentials"].items()})
            for k, v in raw["characters"].items()
        }
        self.potentials = {
            int(k): PotentialInfo(
                id=int(k),
                char_id=v["char_id"],
                name=v["name"],
                kind=v["kind"],
                role=v["role"],
                max_level=v["max_level"],
                rarity=v.get("rarity"),
                build=v.get("build"),
                brief=v.get("brief", ""),
            )
            for k, v in raw["potentials"].items()
        }

    @classmethod
    def load(cls, path: Path = DATA_PATH) -> "GameData":
        return cls(json.loads(path.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def default_gamedata() -> GameData:
    return GameData.load()
