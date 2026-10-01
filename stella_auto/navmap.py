"""방 지도: 걸어 본 칸, 막힌 칸, 피할 곳을 칸(cell) 단위로 적어 두고 A*로 길을 찾는다.

좌표는 odometry의 월드 좌표(방에 들어왔을 때 화면 픽셀). 칸 하나는 CELL px.
- 걸어 본 칸(free): 캐릭터가 서 있었던 칸. 길 찾기에서 조금 싸다.
- 막힌 칸(blocked): 그쪽으로 키를 눌렀는데 안 움직인 칸. 지나갈 수 없다.
- 전에 막혔던 칸(soft): 지난번에 이 지도에서 막혔던 칸. 위치 재기가 조금씩 틀어질 수 있어서
  막지는 않고 비싸게만 둔다.
- 피할 곳(avoid): 아직 나가면 안 될 때 출구 문 주변 같은 곳. 지나갈 수 없다.
- 모르는 칸: 지나갈 수 있다고 보되 걸어 본 칸보다 조금 비싸다.
- 바닥 같아 보이는 정도(looks, 0~1): 지금 화면에서 그 칸이 걸어 본 바닥과 색이 얼마나 비슷한지.
  안 걸어 본 칸 중 바닥 같지 않은 칸(벽, 가구)은 비싸게 보고, 둘러볼 때도 덜 쳐준다.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

CELL = 48
MAX_NODES = 40000
SEARCH_MARGIN = 12  # 시작/목표/아는 칸 둘레로 이만큼 칸을 더 본다

Cell = tuple[int, int]


def to_cell(p: tuple[float, float]) -> Cell:
    return int(math.floor(p[0] / CELL)), int(math.floor(p[1] / CELL))


def cell_center(c: Cell) -> tuple[float, float]:
    return (c[0] + 0.5) * CELL, (c[1] + 0.5) * CELL


@dataclass
class NavMap:
    free: set[Cell] = field(default_factory=set)
    blocked: dict[Cell, int] = field(default_factory=dict)
    soft: dict[Cell, int] = field(default_factory=dict)
    avoid: list[tuple[tuple[float, float], float]] = field(default_factory=list)  # (가운데, 반지름)
    visits: dict[Cell, int] = field(default_factory=dict)
    looks: dict[Cell, float] = field(default_factory=dict)

    def visit(self, p: tuple[float, float]) -> None:
        c = to_cell(p)
        self.free.add(c)
        self.visits[c] = self.visits.get(c, 0) + 1
        # 서 있는 칸은 막힌 칸이 아니다 (막힘 판단이 틀렸던 것)
        self.blocked.pop(c, None)
        self.soft.pop(c, None)

    def block(self, p: tuple[float, float], force: bool = False) -> Cell | None:
        c = to_cell(p)
        if not force and c in self.free and self.visits.get(c, 0) >= 2:
            return None  # 여러 번 서 본 칸은 막혔다고 하지 않는다 (같은 자리에서 또 막히면 force로 막는다)
        self.blocked[c] = self.blocked.get(c, 0) + 1
        self.free.discard(c)
        return c

    def forget_blocks(self) -> int:
        """이번에 막혔다고 적은 칸을 전부 '전에 막혔던 칸'(비싸지만 지나갈 수 있음)으로 돌린다.
        잘못 적은 막힌 칸 때문에 길이 없어져 엉뚱한 구석으로 갈 때 처음부터 다시 찾게 한다."""
        n = len(self.blocked)
        for c, k in self.blocked.items():
            self.soft[c] = max(self.soft.get(c, 0), k)
        self.blocked.clear()
        return n

    def is_avoided(self, c: Cell) -> bool:
        x, y = cell_center(c)
        return any(math.hypot(x - ax, y - ay) < r for (ax, ay), r in self.avoid)

    def passable(self, c: Cell) -> bool:
        return c not in self.blocked and not self.is_avoided(c)

    def step_cost(self, c: Cell) -> float:
        cost = 1.0 if c in self.free else 1.3
        if c in self.soft:
            cost += 4.0
        if c not in self.free:
            cost += 6.0 * (1.0 - self.looks.get(c, 0.6)) ** 2  # 바닥으로 안 보이면 크게 돌아가는 편이 낫다
        x, y = c
        near = sum((x + dx, y + dy) in self.blocked for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        return cost + 0.8 * near  # 벽에 바짝 붙어 가지 않게

    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]] | None:
        """start에서 goal까지 월드 좌표 경로 (칸 가운데 점들, 마지막은 goal). 못 가면 None."""
        s, g = to_cell(start), to_cell(goal)
        if s == g:
            return [goal]
        saved_avoid = self.avoid
        # 이미 피할 곳 안에 서 있으면 그 피할 곳은 빼고 찾는다 (안 그러면 한 발짝도 못 움직인다)
        self.avoid = [(c, r) for c, r in self.avoid if math.hypot(start[0] - c[0], start[1] - c[1]) >= r]
        try:
            return self._plan(s, g, goal)
        finally:
            self.avoid = saved_avoid

    def _plan(self, s: Cell, g: Cell, goal: tuple[float, float]) -> list[tuple[float, float]] | None:
        xs = [s[0], g[0]] + [c[0] for c in self.blocked] + [c[0] for c in self.free]
        ys = [s[1], g[1]] + [c[1] for c in self.blocked] + [c[1] for c in self.free]
        x0, x1 = min(xs) - SEARCH_MARGIN, max(xs) + SEARCH_MARGIN
        y0, y1 = min(ys) - SEARCH_MARGIN, max(ys) + SEARCH_MARGIN
        goal_ok = self.passable(g)

        def h(c: Cell) -> float:
            return math.hypot(c[0] - g[0], c[1] - g[1])

        best_g = {s: 0.0}
        came: dict[Cell, Cell] = {}
        heap = [(h(s), 0.0, s)]
        closest = (h(s), s)
        n = 0
        while heap and n < MAX_NODES:
            _, cost, c = heapq.heappop(heap)
            if cost > best_g.get(c, math.inf):
                continue
            n += 1
            if c == g:
                break
            if h(c) < closest[0]:
                closest = (h(c), c)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (c[0] + dx, c[1] + dy)
                    if not (x0 <= nb[0] <= x1 and y0 <= nb[1] <= y1):
                        continue
                    if not self.passable(nb) and not (nb == g and not goal_ok):
                        continue
                    if dx and dy and not (self.passable((c[0] + dx, c[1])) or self.passable((c[0], c[1] + dy))):
                        continue  # 대각선으로 벽 모서리를 뚫고 가지 않게
                    nc = cost + self.step_cost(nb) * (1.4142 if dx and dy else 1.0)
                    if nc < best_g.get(nb, math.inf):
                        best_g[nb] = nc
                        came[nb] = c
                        heapq.heappush(heap, (nc + h(nb), nc, nb))
        # 목표까지 못 가면 목표에 가장 가까이 갈 수 있는 칸까지 (막힌 칸이 더 밝혀지면 다시 찾는다)
        end = g if g in best_g else closest[1]
        if end == s:
            return None
        cells = [end]
        while cells[-1] != s:
            cells.append(came[cells[-1]])
        cells.reverse()
        return [cell_center(c) for c in cells[1:-1]] + [goal if end == g else cell_center(end)]

    def frontier_score(self, start: tuple[float, float], angle: float, dist: float = 700) -> float:
        """start에서 angle(화면 각도) 쪽으로 가면 모르는 칸을 얼마나 볼 수 있는지. 막히면 거기까지만."""
        rad = math.radians(angle)
        ux, uy = math.cos(rad), -math.sin(rad)
        score = 0.0
        saved_avoid = self.avoid
        # 피할 곳 안에 서 있으면 그 피할 곳은 빼고 본다 (20층: 바닥 무늬를 출구로 알고 그 옆에서 "더 가 볼 곳이 없음"만 150초)
        self.avoid = [(c, r) for c, r in self.avoid if math.hypot(start[0] - c[0], start[1] - c[1]) >= r]
        try:
            for k in range(1, int(dist / (CELL / 2)) + 1):
                d = k * CELL / 2
                c = to_cell((start[0] + ux * d, start[1] + uy * d))
                if not self.passable(c):
                    break
                if c in self.free:
                    score += 0.2
                else:
                    look = self.looks.get(c, 0.6)
                    if look < 0.05:
                        break  # 바닥으로 안 보이는 곳 너머는 셈하지 않는다
                    score += look
        finally:
            self.avoid = saved_avoid
        return score

    # -- 저장 -------------------------------------------------------------------
    def to_json(self) -> dict:
        """저장용. 이번에 막힌 칸과 전에 막혔던 칸을 합쳐 "blocked"로."""
        allb = dict(self.soft)
        for c, n in self.blocked.items():
            allb[c] = max(allb.get(c, 0), n)
        return {"free": [list(c) for c in sorted(self.free)],
                "blocked": [list(c) + [n] for c, n in sorted(allb.items()) if c not in self.free]}

    @classmethod
    def from_json(cls, d: dict) -> "NavMap":
        """기억에서 꺼낸 지도: 막혔던 칸은 soft로 (이번에 다시 막히면 그때 진짜로 막는다)."""
        m = cls()
        m.free = {(c[0], c[1]) for c in d.get("free", [])}
        m.soft = {(c[0], c[1]): c[2] for c in d.get("blocked", [])}
        return m
