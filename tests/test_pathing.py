"""위치 재기(odometry), 방 지도(navmap), 지도 기억(roommemory), 생각하며 걷기(pathing)."""

import math
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from stella_auto import navigate as nv
from stella_auto.navmap import CELL, NavMap, to_cell
from stella_auto.odometry import Odometry
from stella_auto.pathing import Goal, Navigator
from stella_auto.roommemory import RoomMemory

SCREENS = Path(__file__).parent / "screens"


def load(name: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(SCREENS / name), np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(img, (1920, 1080)) if img.shape[1] != 1920 else img


# -- odometry ------------------------------------------------------------------
def test_odometry_real_walk():
    # 13층에서 W+D로 걷다가(5->8) S+A로 돌아옴(24->27). 실제 기록: (+104, -81), (-87, +65) 정도
    odo = Odometry()
    odo.update(load("odo/walk13_005.jpg"))
    assert odo.update(load("odo/walk13_008.jpg"))
    dx, dy = odo.cam
    assert 70 < dx < 140 and -110 < dy < -50
    odo.reset()
    odo.update(load("odo/walk13_024.jpg"))
    odo.update(load("odo/walk13_027.jpg"))
    dx, dy = odo.cam
    assert -120 < dx < -60 and 40 < dy < 95


def test_odometry_still_and_hud_ignored():
    img = load("odo/walk13_036.jpg")
    odo = Odometry()
    odo.update(img)
    odo.update(img.copy())
    assert np.hypot(*odo.cam) < 2


# -- navmap ----------------------------------------------------------------------
def test_plan_goes_around_wall():
    m = NavMap()
    for y in range(-5, 6):  # x=5 칸에 세로 벽 (y -5~5)
        m.block(((5 + 0.5) * CELL, (y + 0.5) * CELL))
    path = m.plan((0.5 * CELL, 0.5 * CELL), (10.5 * CELL, 0.5 * CELL))
    assert path is not None
    assert all(to_cell(p) not in m.blocked for p in path)
    assert any(abs(to_cell(p)[1]) >= 6 for p in path)  # 벽 끝을 돌아간다


def test_plan_avoid_zone_and_soft_cells():
    m = NavMap()
    m.avoid = [((5 * CELL, 0), 2 * CELL)]
    path = m.plan((0, 0), (10 * CELL, 0))
    assert all(math.hypot(p[0] - 5 * CELL, p[1]) >= 2 * CELL - CELL for p in path)
    soft = NavMap.from_json({"free": [], "blocked": [[3, 0, 2], [3, -1, 1]]})
    assert soft.passable((3, 0)) and (3, 0) in soft.soft


def test_visit_clears_wrong_block_and_json_roundtrip():
    m = NavMap()
    m.block((100, 100))
    m.visit((100, 100))
    assert to_cell((100, 100)) not in m.blocked
    m.block((500, 100))
    d = m.to_json()
    m2 = NavMap.from_json(d)
    assert to_cell((500, 100)) in m2.soft and to_cell((100, 100)) in m2.free


def test_frontier_prefers_unknown():
    m = NavMap()
    for x in range(0, 10):
        m.visit((x * CELL, 0))
    assert m.frontier_score((0, 0), 90) > m.frontier_score((0, 0), 0)


# -- roommemory ------------------------------------------------------------------
def test_room_memory_identify(tmp_path):
    mem = RoomMemory(tmp_path)
    a = load("odo/walk13_036.jpg")
    r = mem.add("리더", 13, a)
    # 입구에서 조금 움직인 화면도 같은 지도로 알아보고, 밀린 양을 준다
    moved = np.roll(a, (-24, 40), axis=(0, 1))
    found = mem.identify("리더", moved)
    assert found is not None and found[0].id == r.id
    assert abs(found[1][0] - 40) < 6 and abs(found[1][1] + 24) < 6
    # 다른 방 종류로는 안 찾는다, 다른 지도는 아니라고 한다
    assert mem.identify("전투", a) is None
    assert mem.identify("리더", load("field__live_marker.jpg")) is None
    # 저장했다가 다시 열기
    mem.record(r, 13, NavMap(), (1500.0, -200.0))
    mem2 = RoomMemory(tmp_path)
    assert mem2.rooms[r.id].exit == [1500, -200]


# -- 가짜 게임에서 걷기 ----------------------------------------------------------
class FakeWorld:
    """큰 무늬 바닥 위를 걷는 가짜 게임. 카메라는 캐릭터를 따라가고, 벽(사각형)은 못 지나간다."""

    SPEED = 370.0  # px/초 (실제 게임과 비슷)

    def __init__(self, walls, start, paint=False):
        rng = np.random.default_rng(1)
        noise = rng.integers(60, 140, (900, 1200, 3), dtype=np.uint8)  # 회색 무늬 바닥
        self.world = cv2.resize(cv2.GaussianBlur(noise, (0, 0), 1.2), (4800, 3600), interpolation=cv2.INTER_LINEAR)
        if paint:  # 벽을 바닥과 다른 색으로 칠한다 (진짜 게임처럼 보이는 벽)
            for x0, y0, x1, y1 in walls:
                self.world[y0:y1, x0:x1] = (40, 40, 200)
        self.walls = walls
        self.pos = np.array(start, float)
        self.held: tuple = ()
        self.t = time.monotonic()
        self.trace = []

    # GameInput 흉내
    def set_held(self, keys):
        self._advance()
        self.held = tuple(keys)

    def _blocked(self, p) -> bool:
        return any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for x0, y0, x1, y1 in self.walls)

    def _advance(self):
        now = time.monotonic()
        dt, self.t = now - self.t, now
        ang = nv.angle_of_keys(self.held) if self.held else None
        if ang is None:
            return
        step = np.array([math.cos(math.radians(ang)), -math.sin(math.radians(ang))]) * self.SPEED * dt
        for s in (step, np.array([step[0], 0]), np.array([0, step[1]])):  # 벽에 비스듬히 닿으면 미끄러진다
            if not self._blocked(self.pos + s):
                self.pos = self.pos + s
                break
        self.trace.append(self.pos.copy())

    def grab(self):
        self._advance()
        cx, cy = int(self.pos[0] - 960), int(self.pos[1] - 555)
        img = self.world[cy:cy + 1080, cx:cx + 1920].copy()
        for i in range(5):  # 체력바 (초록 칸 5개)
            cv2.rectangle(img, (912 + i * 20, 610), (912 + i * 20 + 17, 620), (60, 210, 60), -1)
        return img


def make_nav(world, tmp_path):
    nav = Navigator(world, world.grab, lambda k, m: None, memory=RoomMemory(tmp_path))
    nav.new_room(1, "전투", "선택")
    nav.observe(world.grab())
    return nav


def test_walk_straight_to_goal(tmp_path):
    world = FakeWorld([], (1500, 1800))
    nav = make_nav(world, tmp_path)
    start = nav.last_pos
    goal = (start[0] + 600, start[1])
    res = nav.walk(lambda im, ch, p: Goal(goal, "test", arrive=60), lambda im: False, max_sec=8)
    assert res.reason == "arrived"
    assert world.pos[0] - 1500 > 500


def test_walk_marks_wall_and_goes_around(tmp_path):
    # 오른쪽 목표 사이에 세로 벽. 벽에 막히면 막힌 칸을 적고 돌아가야 한다
    world = FakeWorld([(1900, 1300, 2000, 2300)], (1500, 1800))
    nav = make_nav(world, tmp_path)
    start = nav.last_pos
    goal = (start[0] + 900, start[1])
    res = None
    for _ in range(4):  # 진척이 없으면 runner가 다시 부르는 것처럼
        res = nav.walk(lambda im, ch, p: Goal(goal, "test", arrive=80), lambda im: False, max_sec=12)
        if res.reason == "arrived":
            break
    assert res.reason == "arrived", res
    assert nav.map.blocked  # 벽을 막힌 칸으로 적었다
    assert world.pos[0] > 2000


def test_room_entry_is_first_frame_after_loading(tmp_path):
    # 로딩 뒤 첫 필드 화면이 입구. 방 제목은 조금 늦게 읽혀도 좌표는 입구 기준으로 이어진다
    world = FakeWorld([], (1500, 1800))
    mem = RoomMemory(tmp_path)
    nav = Navigator(world, world.grab, lambda k, m: None, memory=mem)
    nav.loading_seen()
    nav.observe(world.grab())
    world.pos += (60, 0)
    nav.observe(world.grab())
    nav.new_room(3, "전투", "강적")
    p = nav.observe(world.grab())
    assert nav.room is not None and len(mem.rooms) == 1
    assert abs(p[0] - (960 + 60)) < 15 and abs(p[1] - 555) < 15
    nav.finish(exit_reached=True, pos=(1800.0, 400.0))
    # 다음 판: 같은 지도에 조금 다른 자리로 들어와도 알아보고, 기억 기준 좌표로 맞춘다
    world.pos[:] = (1540, 1780)
    nav.loading_seen()
    nav.observe(world.grab())
    nav.new_room(3, "전투", "강적")
    p2 = nav.observe(world.grab())
    assert len(mem.rooms) == 1 and nav.memory_exit == (1800.0, 400.0)
    assert abs(p2[0] - (960 + 40)) < 15 and abs(p2[1] - (555 - 20)) < 15


def test_visible_wall_is_avoided_before_bumping(tmp_path):
    # 벽이 바닥과 다른 색으로 보이면, 부딪혀 보기 전에 바닥 같은 쪽으로 돌아간다
    walls = [(1900, 1400, 2250, 2200)]  # 큰 가구처럼 두꺼운 벽
    bumps = {}
    for paint in (False, True):
        world = FakeWorld(walls, (1500, 1800), paint=paint)
        nav = make_nav(world, tmp_path / str(paint))
        start = nav.last_pos
        # 바닥 색을 배우게 잠깐 왼쪽으로 걸었다 돌아온다
        nav.walk(lambda im, ch, p: Goal((start[0] - 250, start[1]), "test", arrive=60), lambda im: False, max_sec=4)
        goal = (start[0] + 1100, start[1])
        for _ in range(4):
            res = nav.walk(lambda im, ch, p: Goal(goal, "test", arrive=80), lambda im: False, max_sec=12)
            if res.reason == "arrived":
                break
        assert res.reason == "arrived"
        bumps[paint] = len(nav.map.blocked)
    assert bumps[True] < bumps[False]
