"""방 지도 기억: 층마다 나오는 지도는 2~3개로 정해져 있다 (사용자 설명).

방에 들어온 첫 화면(입구 화면)으로 어떤 지도인지 알아보고, 전에 찾은 출구 위치와 막힌 칸을
꺼내 쓴다. 출구로 나가면 그 자리를 출구로 적어 둔다.

data/rooms/index.json 에 방 목록, data/rooms/<id>.png 에 입구 화면(1/4 크기 흑백)을 둔다.
월드 좌표는 그 방 입구 화면 기준 (pathing.Navigator._identify가 맞춰 옮긴다).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .navmap import NavMap
from .odometry import shift_between, small_gray

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "data" / "rooms"
MATCH_RESPONSE = 0.2  # 입구 화면이 이만큼 맞아야 같은 지도
MATCH_NCC = 0.7  # 같은 지도 입구는 0.9~0.99. 0.57로 엉뚱한 지도를 알아본 일이 있었다


@dataclass
class Room:
    id: str
    room: str  # 방 종류 (전투/선택/강적/거래/리더)
    floors: list[int] = field(default_factory=list)
    visits: int = 0
    exit: list[float] | None = None  # 출구 월드 좌표 (나간 자리들의 평균)
    exit_count: int = 0
    exit_conflicts: int = 0  # 기억한 출구와 크게 다른 자리로 나간 횟수 (연속)
    npcs: list[list] = field(default_factory=list)  # [x, y, 종류] 말을 건 자리 (종류: npc, shop, enhance)
    nav: dict = field(default_factory=dict)  # NavMap.to_json()
    updated: str = ""


class RoomMemory:
    def __init__(self, directory: Path = DEFAULT_DIR):
        self.dir = Path(directory)
        self.rooms: dict[str, Room] = {}
        self._refs: dict[str, np.ndarray] = {}
        self.last_candidates: list = []
        index = self.dir / "index.json"
        if index.exists():
            for d in json.loads(index.read_text(encoding="utf-8")):
                self.rooms[d["id"]] = Room(**d)

    def ref(self, rid: str) -> np.ndarray | None:
        if rid not in self._refs:
            path = self.dir / f"{rid}.png"
            if not path.exists():
                return None
            self._refs[rid] = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        return self._refs[rid]

    def identify(self, room: str, img: np.ndarray) -> tuple[Room, tuple[float, float], float] | None:
        """입구 화면으로 아는 지도인지 찾는다. (방, 밀린 양, 점수) 또는 None."""
        cur = small_gray(img)
        best = None
        self.last_candidates: list = []  # 기준엔 못 미쳤지만 비슷했던 지도 (기록용)
        for r in self.rooms.values():
            if room and r.room and r.room != room:
                continue
            ref = self.ref(r.id)
            if ref is None:
                continue
            (dx, dy), resp = shift_between(ref, cur)
            if resp < MATCH_RESPONSE or math.hypot(dx, dy) > 900:
                continue
            ncc = _aligned_ncc(ref, cur, dx, dy)
            if ncc >= MATCH_NCC and (best is None or ncc > best[2]):
                best = (r, (dx, dy), ncc)
            if ncc >= 0.4:
                self.last_candidates.append((r.id, round(ncc, 2), round(resp, 2)))
        return best

    def add(self, room: str, floor: int, img: np.ndarray) -> Room:
        rid = f"r{len(self.rooms) + 1:04d}"
        while rid in self.rooms:
            rid = f"r{int(rid[1:]) + 1:04d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        ref = small_gray(img)
        cv2.imencode(".png", np.clip(ref, 0, 255).astype(np.uint8))[1].tofile(str(self.dir / f"{rid}.png"))
        self._refs[rid] = ref
        r = Room(rid, room, [floor])
        self.rooms[rid] = r
        self.save()
        return r

    def record(self, r: Room, floor: int, nav: NavMap, exit_pos: tuple[float, float] | None) -> None:
        """방을 나갈 때: 출구 자리, 막힌 칸을 적어 둔다."""
        r.visits += 1
        if floor not in r.floors:
            r.floors.append(floor)
        if exit_pos is not None:
            far = r.exit is not None and math.hypot(exit_pos[0] - r.exit[0], exit_pos[1] - r.exit[1]) > 300
            if far and r.exit_count >= 2 and r.exit_conflicts < 1:
                # 여러 번 확인한 출구와 많이 다르다: 이번 판 위치 재기가 틀어졌을 수 있어서 한 번은 무시한다
                # (5층에서 오래 헤맨 뒤 출구 자리를 500px 틀리게 덮어썼다)
                r.exit_conflicts += 1
            elif r.exit is None or far:
                r.exit, r.exit_count, r.exit_conflicts = [round(exit_pos[0]), round(exit_pos[1])], 1, 0
            else:
                n = r.exit_count
                r.exit = [round((r.exit[0] * n + exit_pos[0]) / (n + 1)), round((r.exit[1] * n + exit_pos[1]) / (n + 1))]
                r.exit_count = n + 1
                r.exit_conflicts = 0
        r.nav = nav.to_json()  # 이번 지도는 기억에서 꺼낸 지도에 이어 그린 것이라 그대로 저장
        r.updated = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def add_spot(self, r: Room, pos: tuple[float, float], kind: str) -> bool:
        """NPC/상점/강화머신 앞에 섰던 자리. 같은 지도에서 NPC는 바뀌어도 서 있는 자리는 같다
        (16층 베아트리스, 2층 포셔가 같은 지도 같은 자리). 가까운 자리가 이미 있으면 안 넣는다."""
        for x, y, k in r.npcs:
            if k == kind and math.hypot(pos[0] - x, pos[1] - y) < 200:
                return False
        r.npcs.append([round(pos[0]), round(pos[1]), kind])
        self.save()
        return True

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        # 방 하나를 한 줄로 (칸 목록이 길어서 들여쓰기하면 파일이 너무 길어진다)
        rows = [json.dumps(r.__dict__, ensure_ascii=False) for r in sorted(self.rooms.values(), key=lambda r: r.id)]
        tmp = self.dir / "index.json.tmp"
        tmp.write_text("[\n" + ",\n".join(rows) + "\n]\n", encoding="utf-8")
        tmp.replace(self.dir / "index.json")


def _aligned_ncc(ref: np.ndarray, cur: np.ndarray, dx: float, dy: float) -> float:
    """cur를 ref에 맞춰 겹친 부분의 정규화 상관 (같은 지도인지 한 번 더 확인)."""
    sx, sy = int(round(dx / 4)), int(round(dy / 4))
    h, w = ref.shape
    a = ref[max(0, -sy):h - max(0, sy), max(0, -sx):w - max(0, sx)]
    b = cur[max(0, sy):h - max(0, -sy), max(0, sx):w - max(0, -sx)]
    if a.size < 0.3 * ref.size or a.shape != b.shape:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum())) or 1.0
    return float((a * b).sum() / den)
