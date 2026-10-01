"""게임에 마우스 클릭과 키 입력 보내기 (Windows SendInput).

안전장치: 입력을 보내기 직전에 매번 맨 앞 창이 게임인지 확인한다. 게임을 앞으로
가져오지 못하면 아무것도 보내지 않고 NotFocusedError를 낸다. 다른 창에
WASD나 클릭이 들어가는 사고를 막기 위해서다.

좌표는 게임 화면(클라이언트 영역) 1920x1080 기준.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

from .capture import GameWindow, describe, find_game_window, restore, user32

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
VK_MENU = 0x12

# Unity 게임은 가상 키코드보다 스캔코드를 더 잘 받는다
SCANCODES = {
    "esc": 0x01, "1": 0x02, "2": 0x03, "3": 0x04, "q": 0x10, "w": 0x11, "e": 0x12, "r": 0x13,
    "a": 0x1E, "s": 0x1F, "d": 0x20, "f": 0x21, "z": 0x2C, "x": 0x2D, "b": 0x30,
    "space": 0x39, "enter": 0x1C, "shift": 0x2A, "tab": 0x0F,
}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_void_p)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_void_p)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT
user32.GetForegroundWindow.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]


class NotFocusedError(RuntimeError):
    pass


def _send(*inputs: INPUT) -> None:
    arr = (INPUT * len(inputs))(*inputs)
    if user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT)) != len(inputs):
        raise OSError(f"SendInput 실패 (error {ctypes.get_last_error()})")


def _key_input(scan: int, up: bool) -> INPUT:
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    return INPUT(INPUT_KEYBOARD, _INPUTUNION(ki=KEYBDINPUT(0, scan, flags, 0, None)))


def _mouse_input(flags: int) -> INPUT:
    return INPUT(INPUT_MOUSE, _INPUTUNION(mi=MOUSEINPUT(0, 0, 0, flags, 0, None)))


def release_all_keys() -> None:
    """봇이 누를 수 있는 키를 전부 뗀다 (비상 정지, 강제 종료 뒤 정리용)."""
    _send(*(_key_input(s, up=True) for s in SCANCODES.values()))
    _send(_mouse_input(MOUSEEVENTF_LEFTUP))


class GameInput:
    def __init__(self, focus_timeout: float = 2.0):
        self.focus_timeout = focus_timeout
        self._held: set[str] = set()

    def focus(self) -> GameWindow:
        """게임 창을 맨 앞으로. 실패하면 NotFocusedError."""
        win = find_game_window()
        if win is None:
            raise NotFocusedError("게임 창을 못 찾음")
        if win.minimized:
            win = restore(win)
        if user32.GetForegroundWindow() == win.hwnd:
            return win
        # 다른 프로그램이 앞에 있으면 SetForegroundWindow가 막힌다. Alt를 한 번 눌러 풀어준다
        _send(INPUT(INPUT_KEYBOARD, _INPUTUNION(ki=KEYBDINPUT(VK_MENU, 0, 0, 0, None))),
              INPUT(INPUT_KEYBOARD, _INPUTUNION(ki=KEYBDINPUT(VK_MENU, 0, KEYEVENTF_KEYUP, 0, None))))
        user32.SetForegroundWindow(win.hwnd)
        deadline = time.monotonic() + self.focus_timeout
        while time.monotonic() < deadline:
            if user32.GetForegroundWindow() == win.hwnd:
                time.sleep(0.1)
                return describe(win.hwnd)
            time.sleep(0.05)
        raise NotFocusedError("게임 창을 맨 앞으로 가져오지 못함")

    def _check_focus(self) -> GameWindow:
        win = self.focus()
        if user32.GetForegroundWindow() != win.hwnd:
            self.release_all()
            raise NotFocusedError("맨 앞 창이 게임이 아님")
        return win

    def click(self, x: int, y: int, *, pause: float = 0.05) -> None:
        """게임 화면 좌표 (x, y)를 왼쪽 클릭."""
        if not (0 <= x < 1920 and 0 <= y < 1080):
            raise ValueError(f"게임 화면 밖 좌표: ({x}, {y})")
        win = self._check_focus()
        sx, sy = win.to_screen(x, y)
        user32.SetCursorPos(sx, sy)
        time.sleep(pause)
        self._check_focus()
        _send(_mouse_input(MOUSEEVENTF_LEFTDOWN))
        time.sleep(0.05)
        _send(_mouse_input(MOUSEEVENTF_LEFTUP))

    def key(self, name: str, hold: float = 0.06) -> None:
        """키 하나를 눌렀다 뗀다."""
        self.hold([name], hold)

    def hold(self, names: list[str], seconds: float) -> None:
        """여러 키를 같이 누르고 seconds 동안 있다가 뗀다 (예: ['w', 'd'] 대각선 이동)."""
        scans = [SCANCODES[n] for n in names]
        self._check_focus()
        try:
            _send(*(_key_input(s, up=False) for s in scans))
            self._held.update(names)
            end = time.monotonic() + seconds
            while True:
                left = end - time.monotonic()
                if left <= 0:
                    break
                time.sleep(min(0.1, left))
                if user32.GetForegroundWindow() != find_game_window().hwnd:
                    raise NotFocusedError("키를 누르는 중에 다른 창이 앞으로 옴")
        finally:
            _send(*(_key_input(s, up=True) for s in scans))
            self._held.difference_update(names)

    def set_held(self, names: tuple[str, ...]) -> None:
        """지금 누르고 있는 이동 키를 names로 바꾼다 (같은 키는 떼지 않고 계속 누른다).
        걸으면서 화면을 보고 방향을 고칠 때 쓴다. 다 떼려면 빈 튜플."""
        if names:
            self._check_focus()
        up =[n for n in self._held if n not in names]
        down = [n for n in names if n not in self._held]
        if up:
            _send(*(_key_input(SCANCODES[n], up=True) for n in up))
            self._held.difference_update(up)
        if down:
            _send(*(_key_input(SCANCODES[n], up=False) for n in down))
            self._held.update(down)

    def release_all(self) -> None:
        if self._held:
            _send(*(_key_input(SCANCODES[n], up=True) for n in self._held))
            self._held.clear()
