"""게임 창 찾기와 화면 캡처 (Windows 전용, ctypes만 사용).

    python -m stella_auto.capture                 # 창 정보 출력 + 한 장 저장
    python -m stella_auto.capture --every 2 -n 30 # 2초마다 30장
    python -m stella_auto.capture --method screen # 화면에서 그대로 복사 (가려지면 같이 찍힘)

캡처 방식
  printwindow  PrintWindow(PW_RENDERFULLCONTENT). 다른 창에 가려져도 게임 화면만 찍힌다.
  screen       데스크톱에서 클라이언트 영역을 복사. 위에 뭐가 있으면 그것도 찍힌다.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROCESS_NAME = "StellaSora.exe"

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

PW_RENDERFULLCONTENT = 0x2
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# 64비트에서 핸들이 int로 잘리지 않게 타입을 지정한다
_H = wintypes.HANDLE
for _fn, _res, _args in [
    (user32.GetDC, _H, [wintypes.HWND]),
    (user32.GetWindowDC, _H, [wintypes.HWND]),
    (user32.ReleaseDC, ctypes.c_int, [wintypes.HWND, _H]),
    (user32.PrintWindow, wintypes.BOOL, [wintypes.HWND, _H, wintypes.UINT]),
    (gdi32.CreateCompatibleDC, _H, [_H]),
    (gdi32.CreateCompatibleBitmap, _H, [_H, ctypes.c_int, ctypes.c_int]),
    (gdi32.SelectObject, _H, [_H, _H]),
    (gdi32.DeleteObject, wintypes.BOOL, [_H]),
    (gdi32.DeleteDC, wintypes.BOOL, [_H]),
    (gdi32.BitBlt, wintypes.BOOL,
     [_H, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, _H, ctypes.c_int, ctypes.c_int, wintypes.DWORD]),
    (gdi32.GetDIBits, ctypes.c_int,
     [_H, _H, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]),
    (kernel32.OpenProcess, _H, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]),
    (kernel32.CloseHandle, wintypes.BOOL, [_H]),
]:
    _fn.restype, _fn.argtypes = _res, _args


def _set_dpi_aware() -> None:
    # 좌표를 실제 픽셀로 받기 위해. 배율이 100%가 아니어도 창 크기가 맞게 나온다.
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2
    except (AttributeError, OSError):
        ctypes.windll.shcore.SetProcessDpiAwareness(2)


_set_dpi_aware()


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


@dataclass(frozen=True)
class GameWindow:
    hwnd: int
    title: str
    left: int  # 클라이언트 영역의 화면 좌표
    top: int
    width: int
    height: int
    dpi: int
    minimized: bool

    def to_screen(self, x: int, y: int) -> tuple[int, int]:
        """클라이언트 좌표 → 화면 좌표 (클릭할 때 쓴다)."""
        return self.left + x, self.top + y


def _process_name(hwnd: int) -> str:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buf))
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return Path(buf.value).name
        return ""
    finally:
        kernel32.CloseHandle(h)


def find_game_window(process_name: str = PROCESS_NAME) -> GameWindow | None:
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd) and _process_name(hwnd).lower() == process_name.lower():
            rect = wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            if rect.right > 0 and rect.bottom > 0:
                found.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    if not found:
        return None
    # 게임은 보통 창이 하나지만, 혹시 여러 개면 클라이언트 영역이 제일 큰 걸 쓴다
    return max((describe(h) for h in found), key=lambda w: w.width * w.height)


def describe(hwnd: int) -> GameWindow:
    title = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, title, 256)
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    return GameWindow(
        hwnd=hwnd,
        title=title.value,
        left=origin.x,
        top=origin.y,
        width=rect.right,
        height=rect.bottom,
        dpi=user32.GetDpiForWindow(hwnd),
        minimized=bool(user32.IsIconic(hwnd)),
    )


def _grab(hwnd: int, width: int, height: int, method: str, src: tuple[int, int] = (0, 0)) -> np.ndarray:
    window_dc = user32.GetWindowDC(hwnd) if method == "printwindow" else user32.GetDC(0)
    mem_dc = gdi32.CreateCompatibleDC(window_dc)
    bmp = gdi32.CreateCompatibleBitmap(window_dc, width, height)
    old = gdi32.SelectObject(mem_dc, bmp)
    try:
        if method == "printwindow":
            ok = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)
        else:
            ok = gdi32.BitBlt(mem_dc, 0, 0, width, height, window_dc, src[0], src[1], SRCCOPY)
        if not ok:
            raise OSError(f"{method} 실패 (error {ctypes.get_last_error()})")

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = width
        bmi.biHeight = -height  # 위에서 아래로
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        buf = np.empty((height, width, 4), dtype=np.uint8)
        gdi32.GetDIBits(mem_dc, bmp, 0, height, buf.ctypes.data, ctypes.byref(bmi), DIB_RGB_COLORS)
        return buf[:, :, :3].copy()
    finally:
        gdi32.SelectObject(mem_dc, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd if method == "printwindow" else 0, window_dc)


def capture(win: GameWindow, method: str = "printwindow") -> np.ndarray:
    """게임 클라이언트 영역을 BGR 이미지로 돌려준다."""
    if win.minimized:
        raise RuntimeError("게임 창이 최소화돼 있음")
    if method == "screen":
        return _grab(win.hwnd, win.width, win.height, "screen", (win.left, win.top))

    # PrintWindow는 테두리와 제목표시줄까지 포함한 창 전체를 그린다
    wrect = wintypes.RECT()
    user32.GetWindowRect(win.hwnd, ctypes.byref(wrect))
    full = _grab(win.hwnd, wrect.right - wrect.left, wrect.bottom - wrect.top, "printwindow")
    ox, oy = win.left - wrect.left, win.top - wrect.top
    return full[oy : oy + win.height, ox : ox + win.width].copy()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stella_auto.capture", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", type=Path, default=Path("captures"), help="저장 폴더 (기본: captures)")
    ap.add_argument("--method", choices=["printwindow", "screen"], default="printwindow")
    ap.add_argument("--every", type=float, default=0, help="N초마다 반복 캡처")
    ap.add_argument("-n", "--count", type=int, default=1, help="캡처할 장수")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    win = find_game_window()
    if win is None:
        print(f"{PROCESS_NAME} 창을 못 찾음. 게임이 켜져 있어?", file=sys.stderr)
        return 1
    print(f"창: '{win.title}' 클라이언트 {win.width}x{win.height} @ ({win.left},{win.top}) "
          f"DPI {win.dpi} ({win.dpi * 100 // 96}%)" + (" [최소화됨]" if win.minimized else ""))
    if (win.width, win.height) != (1920, 1080):
        print("주의: 게임 화면이 1920x1080이 아님. 화면 인식 좌표가 어긋날 수 있음")

    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        if i:
            time.sleep(args.every)
        win = find_game_window() or win  # 창을 옮겼을 수도 있어서 매번 다시 잰다
        img = capture(win, args.method)
        path = args.out / f"{dt.datetime.now():%Y%m%d_%H%M%S_%f}"[:-3]
        path = path.with_suffix(".png")
        cv2.imencode(".png", img)[1].tofile(str(path))  # 한글 경로에서도 저장되게
        print(f"저장: {path} (평균 밝기 {img.mean():.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
