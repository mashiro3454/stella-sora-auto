"""봇 켜고 끄는 전역 단축키 (사용자 요청 2026-10-05). 봇단축키.bat로 띄운다.

- F9: 봇 토글. 꺼져 있으면 650모드(--mode650 --fresh)로 켜고, 켜져 있으면 끈다.
- F10: "다음" (650 성공으로 멈춰 있을 때 logs/step.go를 만들어 진행).
- F12: 봇 자체의 비상 정지 (이 프로그램과 무관하게 원래 동작. 그 뒤 F9로 다시 켤 수 있다).

삑 소리: 높은 삑 2번 = 켜짐, 낮은 삑 = 꺼짐, 짧은 삑 = 다음, 아주 낮고 긴 삑 = 오류.
창 없이 돌아간다. 끝내려면 작업 관리자에서 pythonw.exe를 끝내면 된다.
"""

import ctypes
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path

import winsound

ROOT = Path(__file__).resolve().parent.parent
_VENV = Path(r"C:\Users\masir\.venvs\stella-sora-auto\Scripts\python.exe")
PYTHON = _VENV if _VENV.exists() else Path(sys.executable).with_name("python.exe")
PRESET = ROOT / "presets" / "물.json"  # 다른 팀을 쓰려면 이 줄만 바꾼다
ARGS = ["--mode650", "--fresh"]
PID_FILE = ROOT / "logs" / "bot.pid"
LOG = ROOT / "logs" / "night1.log"
CREATE_NO_WINDOW = 0x08000000

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.CreateMutexW.restype = ctypes.c_void_p
kernel32.OpenProcess.restype = ctypes.c_void_p


def bot_pid() -> int | None:
    """bot.pid가 가리키는 파이썬 프로세스가 살아 있으면 그 pid."""
    try:
        pid = int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None
    h = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        buf = ctypes.create_unicode_buffer(512)
        size = wintypes.DWORD(len(buf))
        if not kernel32.QueryFullProcessImageNameW(ctypes.c_void_p(h), 0, buf, ctypes.byref(size)):
            return None
        # pid가 재사용돼서 엉뚱한 프로그램을 죽이지 않게 이름까지 확인
        return pid if "python" in Path(buf.value).name.lower() else None
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(h))


def release_keys() -> None:
    subprocess.run([str(PYTHON), "-c", "from stella_auto.input import release_all_keys; release_all_keys()"],
                   cwd=str(ROOT), creationflags=CREATE_NO_WINDOW, timeout=30)


def toggle() -> None:
    pid = bot_pid()
    if pid:
        winsound.Beep(330, 350)  # 낮은 삑 = 끄는 중
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       creationflags=CREATE_NO_WINDOW, timeout=30)
        PID_FILE.unlink(missing_ok=True)
        release_keys()  # 강제 종료 순간 누르고 있던 키를 뗀다
        return
    winsound.Beep(880, 120)
    winsound.Beep(1100, 120)  # 높은 삑 2번 = 켜는 중
    (ROOT / "logs" / "step.go").unlink(missing_ok=True)  # run_state는 --fresh가 지운다
    with open(LOG, "a", encoding="utf-8") as f:
        subprocess.Popen([str(PYTHON), "-u", "-m", "stella_auto.runner", str(PRESET)] + ARGS,
                         cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                         creationflags=CREATE_NO_WINDOW)


def next_floor() -> None:
    (ROOT / "logs" / "step.go").touch()
    winsound.Beep(660, 90)  # 짧은 삑 = 다음


def main() -> int:
    # 이 프로그램도 하나만 뜨게 (가비지 컬렉션으로 핸들이 닫히지 않게 들고 있는다)
    main.mutex = kernel32.CreateMutexW(None, True, r"Local\stella_hotkeys")
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        winsound.Beep(660, 90)  # 이미 떠 있다는 뜻
        return 0

    MOD_NOREPEAT = 0x4000
    VK_F9, VK_F10 = 0x78, 0x79
    if not user32.RegisterHotKey(None, 1, MOD_NOREPEAT, VK_F9):
        winsound.Beep(200, 600)  # 다른 프로그램이 F9를 쓰고 있음
        return 1
    user32.RegisterHotKey(None, 2, MOD_NOREPEAT, VK_F10)
    winsound.Beep(880, 80)  # 준비됨

    WM_HOTKEY = 0x0312
    msg = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        if msg.message == WM_HOTKEY:
            try:
                if msg.wParam == 1:
                    toggle()
                elif msg.wParam == 2:
                    next_floor()
            except Exception:
                winsound.Beep(200, 600)
    return 0


if __name__ == "__main__":
    sys.exit(main())
