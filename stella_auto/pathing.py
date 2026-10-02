"""방 안에서 생각하며 걷기.

예전 방식(목표 쪽으로 무작정 걷다가 막히면 옆으로 비키기, 안 보이면 사방 둘러보기)은 같은 벽에
계속 박고, 책장 그림의 동그라미를 출구로 착각해 45걸음씩 헛걸었다. 이제는:

1. odometry로 캐릭터가 방 안 어디 있는지 잰다 (방에 들어온 자리 기준).
2. 걸어 본 칸과 막힌 칸을 지도(NavMap)에 적고, A*로 막힌 칸을 피해 길을 찾는다.
   키를 누르고 있는데 그 방향으로 안 움직이면 앞 칸을 막힌 칸으로 적는다.
3. 목표는 화면에서 본 것을 믿는다:
   - 화면 가장자리에 출구 표시(문 아이콘)가 있으면 출구는 화면 밖이다. 표시 방향으로 간다
     (이때 화면 안의 문양 비슷한 것은 무시한다).
   - 표시가 없고 출구 문양이 보이면(색이 다음 방 종류와 맞아야) 그 자리로 간다.
   - 이번에 본 출구 자리나 전에 이 지도에서 나갔던 자리를 기억해 둔다.
4. 방을 나가면 그 지도(입구 화면), 출구 자리, 막힌 칸을 저장해 다음에 쓴다 (roommemory).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from . import navigate as nv
from .navmap import CELL, NavMap, to_cell
from .odometry import Odometry
from .roommemory import Room, RoomMemory, exit_tolerance

SCREEN_CENTER = (960.0, 540.0)
CHAR_FALLBACK = (960.0, 555.0)  # 체력바를 못 찾으면 캐릭터는 대개 화면 가운데 (카메라가 따라감)
TICK = 0.03
BLOCK_WINDOW = 1.0  # 이 시간 동안
BLOCK_MIN_MOVE = 40  # 누른 방향으로 이만큼도 못 가면 막힌 것 (걸으면 1초에 약 370px)
KEEP_KEYS_DEG = 28  # 원하는 방향이 지금 누르는 방향과 이만큼 안쪽이면 키를 안 바꾼다 (지그재그 방지)
NO_PROGRESS_SEC = 9.0
EXIT_AVOID_R = 280  # 아직 나가면 안 될 때 출구 둘레 이만큼은 안 간다
MARKER_GOAL_DIST = 1400  # 출구 표시 방향으로 이만큼 앞을 목표로 둔다
LOOK_EVERY = 0.5  # 초. 화면에서 바닥 같아 보이는 칸 매기기
LOOK_SCALE = 8
LOOK_SIGMA = 14.0  # Lab 색 거리. 걸어 본 바닥과 이만큼 다르면 바닥 같음이 e^-1
MARKER_KEEP_SEC = 5.0  # 출구 표시가 사라져도 (문이 화면에 들어오는 중) 이만큼은 그 목표로 계속 간다


@dataclass
class Goal:
    pos: tuple[float, float]  # 월드 좌표
    kind: str  # "door", "marker", "seen", "memory", "label", "explore"
    arrive: float = 0.0  # 이만큼 가까워지면 도착 (출구는 0: 문에 들어갈 때까지)


@dataclass
class NavResult:
    reason: str  # arrived, stopped, lost, no_progress, timeout, no_goal, no_path
    secs: float


GoalFn = Callable[[np.ndarray, tuple[float, float], tuple[float, float]], "Goal | None"]


def _hud(x: float, y: float) -> bool:
    return (x < 420 and y < 200) or (x > 1420 and y < 85) or (x > 1780 and y < 230) or y > 930         or (x > 1450 and y > 760)


class Navigator:
    def __init__(self, gi, grab: Callable[[], np.ndarray], log: Callable[[str, str], None],
                 memory: RoomMemory | None = None):
        self.gi = gi
        self.grab = grab
        self.log = log
        self.memory = memory if memory is not None else RoomMemory()
        self.odo = Odometry()
        self.map = NavMap()
        self.room: Room | None = None
        self.floor = 0
        self.kind = ""
        self.next_kind = ""
        self.pending: tuple[int, str, str, bool] | None = None  # 층/방 종류를 알았는데 아직 지도를 안 찾아봄
        self.await_entry = False  # 다음 필드 화면이 새 방의 입구 화면
        self.entry_img: np.ndarray | None = None
        self.door_seen: tuple[float, float] | None = None  # 이번에 본 출구 문양 자리
        self.door_candidate: tuple[float, float] | None = None  # 한 번 본 문양 자리 (한 번 더 보면 믿는다)
        self.false_doors: list[tuple[float, float]] = []  # 가 봤는데 문이 아니었던 자리 (이 방에서만)
        self.ignored_doors = 0  # 믿는 출구 자리와 멀어서 무시한 문양 수 (이 방에서만)
        self.memory_exit: tuple[float, float] | None = None
        self.marker_goal: tuple[float, float] | None = None
        self.marker_goal_at = 0.0
        self.last_pos: tuple[float, float] | None = None
        self.explore_last: float | None = None
        self.last_stop: tuple[float, tuple[float, float]] | None = None  # 걷다가 화면이 바뀌어 멈춘 때와 자리
        self.finished = True
        self.looked_at = 0.0
        self.floor_colors: deque = deque(maxlen=300)  # 이 방에서 걸어 본 바닥 칸들의 색

    # -- 방 드나들기 ------------------------------------------------------------
    # 로딩 화면을 보면(loading_seen) 그다음 첫 필드 화면이 새 방의 입구 화면이다. 위치는 그 화면을
    # (0, 0) 기준으로 잰다. 방 제목을 읽어 층/방 종류를 알게 되면(new_room) 그 입구 화면으로 어떤
    # 지도인지 알아본다. 제목은 입구 화면보다 늦게 읽힐 때가 많다.
    def loading_seen(self) -> None:
        self.left_by_exit()
        self.await_entry = True

    def new_room(self, floor: int, kind: str, next_kind: str, at_entrance: bool = True) -> None:
        """층이 바뀜 (방 제목을 읽었거나 로딩 뒤 시간이 지남).
        at_entrance=False: 봇을 방 중간에서 켰다. 그 화면은 입구가 아니라서 새 지도로 기억하지 않는다."""
        self.pending = (floor, kind, next_kind, at_entrance)
        if self.entry_img is None:  # 로딩 뒤 입구 화면을 아직 못 봤다 (처음 출발, 중간에 켬)
            self.await_entry = True

    def _begin(self, img: np.ndarray) -> None:
        if not self.finished:
            self.finish(exit_reached=False)
        self.await_entry = False
        self.entry_img = img
        self.odo.reset()
        self.odo.update(img)
        self.map = NavMap()
        self.room = None
        self.door_seen = None
        self.door_candidate = None
        self.false_doors = []
        self.ignored_doors = 0
        self.memory_exit = None
        self.marker_goal = None
        self.explore_last = None
        self.floor_colors.clear()
        self.finished = False

    def _identify(self) -> None:
        floor, kind, next_kind, at_entrance = self.pending
        self.pending = None
        entry, self.entry_img = self.entry_img, None
        self.floor, self.kind, self.next_kind = floor, kind, next_kind
        found = self.memory.identify(kind, entry)
        if found:
            room, (dx, dy), score = found
            self.room = room
            # 입구 화면은 기억해 둔 입구 화면에서 (dx, dy)만큼 밀려 있다: 좌표를 기억 쪽 기준으로 옮긴다
            self.odo.shift_origin(-dx, -dy)
            self.map = NavMap.from_json(room.nav) if room.nav else NavMap()
            scattered = self.memory.exit_scattered(room)
            if room.exit and not scattered:
                self.memory_exit = (float(room.exit[0]), float(room.exit[1]))
            self.log("지도", f"아는 지도 {room.id} (일치 {score:.2f}, {room.visits}번 와 봄, "
                           f"출구 {'(%d, %d)' % tuple(room.exit) if room.exit else '모름'}"
                           f"{' - 나간 자리가 들쭉날쭉해서 안 씀' if room.exit and scattered else ''}, "
                           f"전에 막힌 칸 {len(self.map.soft)}개)")
        elif at_entrance:
            self.room = self.memory.add(kind, floor, entry)
            near = f" (비슷했던 지도 {self.memory.last_candidates})" if self.memory.last_candidates else ""
            self.log("지도", f"처음 보는 지도 -> {self.room.id}로 기억{near}")
        else:
            self.room = None
            self.log("지도", "방 중간이라 어떤 지도인지 모름 (이번 방은 기억하지 않음)")

    def left_by_exit(self) -> None:
        """로딩 화면을 봄: 방금 걷다가 화면이 바뀌어 멈췄다면 그 자리가 출구 (문에 들어간 것)."""
        if self.last_stop and time.monotonic() - self.last_stop[0] < 6 and not self.finished:
            self.finish(exit_reached=True, pos=self.last_stop[1])

    def finish(self, exit_reached: bool, pos: tuple[float, float] | None = None) -> None:
        """방을 떠남. 출구로 나갔으면 그 자리를 출구로 적는다."""
        if self.finished or self.room is None:
            self.finished = True
            return
        self.finished = True
        exit_pos = (pos or self.last_pos) if exit_reached else None
        self.memory.record(self.room, self.floor, self.map, exit_pos)
        if exit_pos:
            self.log("지도", f"{self.room.id} 출구 자리 기억 ({exit_pos[0]:.0f}, {exit_pos[1]:.0f})")

    # -- 보기 -------------------------------------------------------------------
    def observe(self, img: np.ndarray, char: tuple[float, float] | None = None) -> tuple[float, float]:
        """화면 하나 볼 때마다 부른다: 위치 갱신, 걸어 본 칸 표시. 캐릭터 월드 좌표를 돌려준다."""
        if self.await_entry:
            self._begin(img)
        else:
            self.odo.update(img)
        if self.pending and self.entry_img is not None:
            self._identify()
        if char is None:
            char = nv.find_character(img) or CHAR_FALLBACK
        pos = self.odo.to_world(char)
        self.map.visit(pos)
        self.last_pos = pos
        now = time.monotonic()
        if now - self.looked_at > LOOK_EVERY:
            self.looked_at = now
            self.look(img, char)
        return pos

    def look(self, img: np.ndarray, char: tuple[float, float]) -> None:
        """지금 화면에 보이는 칸마다 바닥 같아 보이는지 매긴다 (map.looks).

        걸어 본 칸이 화면에 보이면 그 칸의 색(Lab 평균)을 바닥 표본으로 모으고, 안 걸어 본 칸은
        표본과 색이 얼마나 가까운지로 0~1을 준다. 방마다 바닥이 달라서 방에 들어올 때마다 새로 모은다.
        """
        small = cv2.cvtColor(cv2.resize(img, (1920 // LOOK_SCALE, 1080 // LOOK_SCALE), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2LAB).astype(np.float32)
        k = CELL // LOOK_SCALE
        cx0, cy0 = self.odo.cam
        cells, colors = [], []
        for gy in range(int(cy0 // CELL), int((cy0 + 1080) // CELL) + 1):
            for gx in range(int(cx0 // CELL), int((cx0 + 1920) // CELL) + 1):
                sx, sy = gx * CELL - cx0, gy * CELL - cy0  # 칸 왼쪽 위의 화면 좌표
                if sx < 0 or sy < 0 or sx + CELL > 1920 or sy + CELL > 1080:
                    continue
                mx, my = sx + CELL / 2, sy + CELL / 2
                if _hud(mx, my) or math.hypot(mx - char[0], my - (char[1] - 60)) < 170:
                    continue  # HUD, 캐릭터 몸
                px, py = int(sx // LOOK_SCALE), int(sy // LOOK_SCALE)
                cells.append((gx, gy))
                colors.append(small[py:py + k, px:px + k].reshape(-1, 3).mean(axis=0))
        if not cells:
            return
        colors = np.array(colors)
        for c, col in zip(cells, colors):
            if c in self.map.free:
                self.floor_colors.append(col)
        if len(self.floor_colors) < 5:
            return
        ref = np.array(self.floor_colors)
        d = np.sqrt(((colors[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        for c, dist in zip(cells, d):
            if c not in self.map.free:
                self.map.looks[c] = float(math.exp(-(dist / LOOK_SIGMA) ** 2))

    def exit_markers(self, img: np.ndarray, char: tuple[float, float]) -> list[nv.Marker]:
        return [m for m in nv.find_markers(img, char) if m.kind == "exit"]

    def exit_goal(self, img: np.ndarray, char: tuple[float, float], pos: tuple[float, float]) -> Goal | None:
        """출구로 가려면 어디로? 화면에서 본 것 > 이번에 본 자리 > 전에 기억한 자리."""
        if self.door_seen and math.hypot(self.door_seen[0] - pos[0], self.door_seen[1] - pos[1]) < 450:
            # 문 바로 앞: 가장자리 표시는 문 근처에서 흔들려서 안 본다. 대신 화면의 문양으로 자리를 다시 잡는다
            # (위치 재기가 틀어져서 3층에서 문 바로 옆에 서서 엉뚱한 자리로 가려고 했다)
            hue = nv.DOOR_HUE.get(self.next_kind, (None, 0))[0]
            door = nv.find_exit_door(img, char, hue)
            if door and nv.door_matches(door, self.next_kind):
                w = self.odo.to_world(door.center)
                if math.hypot(w[0] - self.door_seen[0], w[1] - self.door_seen[1]) < 700 and not self._is_false_door(w):
                    self.door_seen = w
                    return Goal(w, "door")
            return Goal(self.door_seen, "seen")
        markers = self.exit_markers(img, char)
        if markers:
            m = markers[0]
            heading = nv.angle_of(m.icon[0] - SCREEN_CENTER[0], m.icon[1] - SCREEN_CENTER[1])
            if self.door_seen and nv.angle_diff(
                    heading, nv.angle_of(self.door_seen[0] - pos[0], self.door_seen[1] - pos[1])) > 70:
                # 출구는 화면 밖 표시 쪽에 있다. 크게 다른 쪽에 기억한 문양은 무늬였다
                # (14층: 오른쪽 가짜 문과 왼쪽 위 표시 사이를 4분 동안 왔다 갔다 했다)
                self.mark_false_door(self.door_seen, "seen")
            known = self.door_seen or self.memory_exit
            if known and nv.angle_diff(heading, nv.angle_of(known[0] - pos[0], known[1] - pos[1])) < 50:
                return Goal(known, "seen" if known == self.door_seen else "memory")
            rad = math.radians(heading)
            cam_center = self.odo.to_world(SCREEN_CENTER)
            self.marker_goal = (cam_center[0] + math.cos(rad) * MARKER_GOAL_DIST,
                                cam_center[1] - math.sin(rad) * MARKER_GOAL_DIST)
            self.marker_goal_at = time.monotonic()
            return Goal(self.marker_goal, "marker")
        hue = nv.DOOR_HUE.get(self.next_kind, (None, 0))[0]
        door = nv.find_exit_door(img, char, hue)
        if door and nv.door_matches(door, self.next_kind) and not self._is_false_door(self.odo.to_world(door.center)) \
                and self._far_from_trusted_exit(self.odo.to_world(door.center)):
            self.ignored_doors += 1  # 믿는 출구와 먼 문양을 본 화면 수 (출구로 못 가는 게 겹치면 기억 대신 이 문을 믿는다)
        if door and self.door_seen and nv.door_matches(door, self.next_kind):
            w = self.odo.to_world(door.center)
            if math.hypot(w[0] - self.door_seen[0], w[1] - self.door_seen[1]) > 300:
                # 이미 고른 문이 있으면 다른 문양으로 바꾸지 않는다 (14층: 진짜 문과 나무문 장식 사이를 1분 오갔다).
                # 고른 문이 가짜면 문 앞에서 확인해 지우고, 못 가면 runner가 지운다
                return Goal(self.door_seen, "seen")
        if door and nv.door_matches(door, self.next_kind) and not self._is_false_door(self.odo.to_world(door.center)) \
                and not self._far_from_trusted_exit(self.odo.to_world(door.center)):
            w = self.odo.to_world(door.center)
            # 같은 자리(월드 좌표)에서 두 번 보여야 문으로 믿는다 (한 번 우연히 잡힌 무늬에 끌려가지 않게)
            c = self.door_candidate
            if c is None:  # 처음 봤으면 화면을 바로 한 번 더 보고 확인한다 (그사이 둘러보러 가 버리지 않게)
                img2 = self.grab()
                char2 = nv.find_character(img2) or CHAR_FALLBACK
                self.odo.update(img2)
                d2 = nv.find_exit_door(img2, char2, hue)
                c = self.odo.to_world(d2.center) if d2 and nv.door_matches(d2, self.next_kind) else None
            if c and math.hypot(w[0] - c[0], w[1] - c[1]) < 200:
                self.door_seen = w
                return Goal(w, "door")
            self.door_candidate = w
        if self.door_seen:
            return Goal(self.door_seen, "seen")
        if self.memory_exit:
            return Goal(self.memory_exit, "memory")
        if self.marker_goal and time.monotonic() - self.marker_goal_at < MARKER_KEEP_SEC:
            return Goal(self.marker_goal, "marker")
        return None

    # -- 기억해 둔 NPC 자리 -------------------------------------------------------
    def remember_spot(self, kind: str) -> None:
        """지금 선 자리를 이 지도의 NPC/상점/강화머신 자리로 기억 (말을 걸었을 때)."""
        if self.room is not None and self.last_pos is not None:
            if self.memory.add_spot(self.room, self.last_pos, kind):
                self.log("지도", f"{self.room.id}에 {kind} 자리 기억 ({self.last_pos[0]:.0f}, {self.last_pos[1]:.0f})")

    def spots(self, kind: str) -> list[tuple[float, float]]:
        if self.room is None:
            return []
        return [(float(x), float(y)) for x, y, k in self.room.npcs if k == kind]

    def _far_from_trusted_exit(self, w: tuple[float, float]) -> bool:
        """이 지도에서 두 번 넘게 나가 본 출구가 있는데, 새로 본 문양이 거기서 멀면 무늬일 가능성이 크다."""
        if self.memory_exit is None or self.room is None or self.room.exit_count < 2:
            return False
        if math.hypot(w[0] - self.memory_exit[0], w[1] - self.memory_exit[1]) <= exit_tolerance(self.memory_exit):
            return False
        # 두 번 넘게 나가 본 다른 출구 근처면 믿는다 (출구가 둘인 방)
        return not any(n >= 2 and math.hypot(w[0] - c[0], w[1] - c[1]) <= exit_tolerance(c)
                       for c, n in self.memory.exit_clusters(self.room))

    def _is_false_door(self, w: tuple[float, float]) -> bool:
        return any(math.hypot(w[0] - f[0], w[1] - f[1]) < 200 for f in self.false_doors)

    def mark_false_door(self, w: tuple[float, float], kind: str) -> None:
        """그 자리에 가 봤는데 로딩이 안 됐다: 문이 아니다 (11층에서 문 아닌 무늬 앞에서 계속 서성였다)."""
        self.false_doors.append(w)
        if kind == "memory":
            self.memory_exit = None  # 이번엔 기억한 출구가 틀렸다 (지도를 잘못 알아봤을 수 있다)
        self.door_seen = None
        self.door_candidate = None
        self.log("이동", f"({w[0]:.0f}, {w[1]:.0f})에 가 봤는데 출구가 아님, 앞으로 무시")

    def known_exit(self) -> tuple[float, float] | None:
        return self.door_seen or self.memory_exit

    # -- 걷기 -------------------------------------------------------------------
    def walk(self, goal_fn: GoalFn, stop: Callable[[np.ndarray], bool], *, max_sec: float = 30.0,
             avoid_exit: bool = False, why: str = "") -> NavResult:
        """goal_fn이 주는 목표로 막힌 칸을 피해 걷는다. 화면을 볼 때마다 목표와 길을 다시 정한다."""
        t0 = time.monotonic()
        self.last_stop = None  # 걷는 중에 본 로딩은 걷기가 끝난 뒤에 출구로 적는다 (예전 멈춘 자리를 쓰지 않게)
        self.map.avoid = []
        if avoid_exit and self.known_exit():
            self.map.avoid = [(self.known_exit(), EXIT_AVOID_R)]
        best_d, best_t = math.inf, t0
        hist: deque = deque()  # (시각, 월드 좌표)
        held: tuple[str, ...] = ()
        held_since = t0
        lost = 0
        blocks = 0
        stuck_at: deque = deque(maxlen=6)  # 막힌 자리들 (같은 자리에서 또 막히면 억지로 적고, 세 번이면 빠져나온다)
        last_goal_kind = ""
        last_goal_log = 0.0
        at_door_since: float | None = None
        try:
            while True:
                now = time.monotonic()
                img = self.grab()
                if stop(img):
                    if self.last_pos:
                        self.last_stop = (now, self.last_pos)
                    return NavResult("stopped", now - t0)
                char = nv.find_character(img)
                if char is None:
                    lost += 1
                    # 카메라가 캐릭터를 따라가니 못 찾아도 화면 가운데로 보고 걷는다 (17층: 체력바를 못 읽어
                    # 6번 만에 "lost"로 그만두기를 1분 넘게 되풀이했다). 아주 오래 못 찾을 때만 그만둔다
                    if lost >= 150:
                        return NavResult("lost", now - t0)
                else:
                    lost = 0
                pos = self.observe(img, char)
                goal = goal_fn(img, char or CHAR_FALLBACK, pos)
                if avoid_exit and self._exit_too_close(img, char or CHAR_FALLBACK, pos, goal.pos if goal else None):
                    return NavResult("near_exit", now - t0)
                if goal is None:
                    return NavResult("no_goal", now - t0)
                if goal.kind != last_goal_kind and now - last_goal_log > 1.5:
                    self.log("이동", f"{why} 목표 {goal.kind} ({goal.pos[0]:.0f}, {goal.pos[1]:.0f}), 지금 ({pos[0]:.0f}, {pos[1]:.0f})")
                    last_goal_kind, last_goal_log = goal.kind, now
                d = math.hypot(goal.pos[0] - pos[0], goal.pos[1] - pos[1])
                if goal.arrive and d < goal.arrive:
                    return NavResult("arrived", now - t0)
                if goal.kind in ("door", "seen", "memory"):
                    if d < 160:
                        if at_door_since is None:
                            at_door_since = now
                        elif now - at_door_since > (3.0 if d < 90 else 5.0):
                            # 문 자리 바로 앞에서 몇 초 동안 로딩이 안 된다 (16층: 무늬 앞 80~130px에서 90초 막혔다)
                            self.mark_false_door(goal.pos, goal.kind)
                            return NavResult("false_door", now - t0)
                    else:
                        at_door_since = None
                if d < best_d - 40:
                    best_d, best_t = d, now
                elif now - best_t > NO_PROGRESS_SEC:
                    return NavResult("no_progress", now - t0)
                if now - t0 > max_sec:
                    return NavResult("timeout", now - t0)

                # 막힘: 같은 쪽으로 BLOCK_WINDOW 넘게 눌렀는데 그쪽으로 거의 안 갔다
                hist.append((now, pos))
                while len(hist) > 2 and now - hist[1][0] >= BLOCK_WINDOW:
                    hist.popleft()
                if held and now - held_since > BLOCK_WINDOW + 0.3 and now - hist[0][0] >= BLOCK_WINDOW * 0.8:
                    ang = nv.angle_of_keys(held)
                    ux, uy = math.cos(math.radians(ang)), -math.sin(math.radians(ang))
                    moved = (pos[0] - hist[0][1][0]) * ux + (pos[1] - hist[0][1][1]) * uy
                    if moved < BLOCK_MIN_MOVE:
                        # 앞 칸과 그 양옆 칸 (벽은 대개 옆으로 이어진다).
                        # 대각선으로 48px 앞은 지금 칸과 같을 수 있어서 지금 칸을 벗어날 때까지 앞으로 본다
                        # (8층 술통 벽에서 앞 칸이 늘 지금 칸이라 못 적고 4분 동안 같은 쪽으로 밀었다)
                        here = to_cell(pos)
                        d = CELL
                        while to_cell((pos[0] + ux * d, pos[1] + uy * d)) == here and d < CELL * 2:
                            d += CELL / 4
                        fx, fy = pos[0] + ux * d, pos[1] + uy * d
                        again = any(math.hypot(pos[0] - p[0], pos[1] - p[1]) < 60 for p in stuck_at)
                        c = self.map.block((fx, fy), force=again)
                        if c is not None:
                            # 또 막혔으면 옆 칸도 억지로: 위치 재기가 조금 틀어져 벽 칸을 '서 본 칸'으로 적었을 수 있다
                            # (가짜 게임에서 대각선 길이 그 칸을 지나가서 키는 d인데 계속 벽으로 밀었다)
                            for side in (-1, 1):
                                self.map.block((fx - uy * CELL * side, fy + ux * CELL * side), force=again)
                        stuck_at.append(pos)
                        blocks += 1
                        self.log("이동", f"{'+'.join(held)} 쪽이 막힘 ({moved:.0f}px), 칸 {c} 막힌 칸으로 적음"
                                       + (" (같은 자리에서 또 막혀서 억지로)" if again and c is not None else ""))
                        held = ()
                        self.gi.set_held(())
                        held_since = now
                        hist.clear()
                        continue

                path = self.map.plan(pos, goal.pos)
                if not path:
                    return NavResult("no_path", now - t0)
                wp = self._lookahead(pos, path)
                want = nv.angle_of(wp[0] - pos[0], wp[1] - pos[1])
                cur = nv.angle_of_keys(held) if held else None
                if cur is None or nv.angle_diff(want, cur) > KEEP_KEYS_DEG:
                    keys = nv.keys_for_angle(want)
                    if keys != held:
                        if cur is None or nv.angle_diff(nv.angle_of_keys(keys), cur) > 60:
                            held_since = now  # 크게 꺾으면 제자리에서 도는 시간이 있다
                            hist.clear()
                        held = keys
                        self.gi.set_held(held)
                time.sleep(TICK)
        finally:
            self.gi.set_held(())

    def _exit_too_close(self, img: np.ndarray, char: tuple[float, float], pos: tuple[float, float],
                        toward: tuple[float, float] | None = None) -> bool:
        """아직 나가면 안 될 때: 화면에 출구 문양이 보이면 그 둘레를 피할 곳으로 정하고, 이미 가까우면 True.
        (5층 거래의 방에서 상점을 찾으며 둘러보다 출구에 들어가 버렸다)
        toward(가려는 곳)가 문양과 다른 쪽이면 가까워도 멈추지 않는다 (20층: 시작 자리 옆 바닥 무늬를 출구로 알고
        기억한 강화머신 자리로 한 발짝도 못 가 150초를 날렸다)."""
        hue = nv.DOOR_HUE.get(self.next_kind, (None, 0))[0]
        door = nv.find_exit_door(img, char, hue)
        if door and nv.door_matches(door, self.next_kind):
            w = self.odo.to_world(door.center)
            if self._is_false_door(w) or self._far_from_trusted_exit(w):
                return False  # 문이 아니었던 자리, 또는 여러 번 나가 본 출구와 먼 무늬
            if self.door_seen is None:
                self.door_seen = w
            if not any(math.hypot(c[0] - w[0], c[1] - w[1]) < 100 for c, _ in self.map.avoid):
                self.map.avoid.append((w, EXIT_AVOID_R))
            if math.hypot(w[0] - pos[0], w[1] - pos[1]) < EXIT_AVOID_R + 60:
                if toward is not None and nv.angle_diff(nv.angle_of(w[0] - pos[0], w[1] - pos[1]),
                                                        nv.angle_of(toward[0] - pos[0], toward[1] - pos[1])) > 75:
                    return False  # 문양에서 멀어지는 쪽으로 간다
                self.log("이동", f"출구 문양이 가까워서 멈춤 ({w[0]:.0f}, {w[1]:.0f})")
                return True
        return False

    def _lookahead(self, pos: tuple[float, float], path: list[tuple[float, float]]) -> tuple[float, float]:
        """경로에서 막힌 칸 없이 곧장 갈 수 있는 가장 먼 점 (최대 400px 앞)."""
        best = path[0]
        for p in path:
            if math.hypot(p[0] - pos[0], p[1] - pos[1]) > 400:
                break
            if self._clear_line(pos, p):
                best = p
            else:
                break
        return best

    def _clear_line(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (CELL / 3)))
        for k in range(1, n + 1):
            c = to_cell((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n))
            if not self.map.passable(c):
                return False
        return True

    # -- 둘러보기 ---------------------------------------------------------------
    def explore(self, stop: Callable[[np.ndarray], bool], *, max_sec: float = 3.0, avoid_exit: bool = False,
                why: str = "둘러보기") -> NavResult:
        """아직 안 가 본 쪽 중 막히지 않은 방향으로 조금 걷는다. 출구를 피해야 하면 출구 쪽은 안 간다."""
        img = self.grab()
        char = nv.find_character(img)
        pos = self.observe(img, char)
        avoid_angles = []
        if avoid_exit:
            for m in self.exit_markers(img, char or CHAR_FALLBACK):
                avoid_angles.append(nv.angle_of(m.icon[0] - SCREEN_CENTER[0], m.icon[1] - SCREEN_CENTER[1]))
            if self.known_exit():
                ex = self.known_exit()
                avoid_angles.append(nv.angle_of(ex[0] - pos[0], ex[1] - pos[1]))
        best = None
        for ang, _ in nv.DIRECTIONS:
            if any(nv.angle_diff(ang, a) < 60 for a in avoid_angles):
                continue
            score = self.map.frontier_score(pos, ang)
            if self.explore_last is not None:
                diff = nv.angle_diff(ang, self.explore_last)
                score += 3 if diff < 1 else (-4 if diff > 150 else 0)  # 가던 쪽 조금 선호, 되돌아가기는 덜
            if best is None or score > best[0]:
                best = (score, ang)
        if best is None or best[0] <= 0:
            self.log("이동", f"{why}: 더 가 볼 곳이 없음")
            self.explore_last = None
            return NavResult("no_goal", 0.0)
        ang = best[1]
        self.explore_last = ang
        rad = math.radians(ang)
        target = (pos[0] + math.cos(rad) * 700, pos[1] - math.sin(rad) * 700)
        self.log("이동", f"{why}: {ang:+.0f}도 쪽 (모르는 칸 점수 {best[0]:.0f})")
        return self.walk(lambda im, ch, p: Goal(target, "explore", arrive=120), stop, max_sec=max_sec,
                         avoid_exit=avoid_exit, why=why)
