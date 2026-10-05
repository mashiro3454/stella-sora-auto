"""봇 켜고 끄는 버튼 창 (사용자 요청 2026-10-05). 봇스위치.bat로 띄운다.

- [켜기]: 650모드(--mode650 --fresh)로 봇을 켠다. 켜져 있으면 [끄기]로 바뀐다.
- [끄기]: 봇을 끄고 누르고 있던 키를 뗀다.
- [다음]: 650 성공으로 멈춰 있을 때 logs/step.go를 만들어 진행.
- F12: 봇 자체의 비상 정지 (이 창과 무관하게 원래 동작. 그 뒤 [켜기]로 다시).

창을 닫아도 봇은 계속 돈다 (이 창은 리모컨일 뿐). 전역 단축키는 안 쓴다.
"""

import ctypes
import subprocess
import sys
import threading
import tkinter as tk
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


def start_bot() -> None:
    (ROOT / "logs" / "step.go").unlink(missing_ok=True)  # run_state는 --fresh가 지운다
    with open(LOG, "a", encoding="utf-8") as f:
        subprocess.Popen([str(PYTHON), "-u", "-m", "stella_auto.runner", str(PRESET)] + ARGS,
                         cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                         creationflags=CREATE_NO_WINDOW)


def stop_bot(pid: int) -> None:
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                   creationflags=CREATE_NO_WINDOW, timeout=30)
    PID_FILE.unlink(missing_ok=True)
    release_keys()  # 강제 종료 순간 누르고 있던 키를 뗀다


def last_log_line() -> str:
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 2000))
            lines = f.read().decode("utf-8", "ignore").strip().splitlines()
            return lines[-1].strip() if lines else ""
    except OSError:
        return ""


class Panel:
    def __init__(self) -> None:
        self.busy = False
        self.root = tk.Tk()
        self.root.title("텔라리 봇")
        self.root.geometry("340x190")
        self.root.resizable(False, False)

        self.status = tk.Label(self.root, text="…", font=("맑은 고딕", 14, "bold"))
        self.status.pack(pady=(12, 4))

        self.toggle_btn = tk.Button(self.root, text="켜기 (650모드)", font=("맑은 고딕", 12),
                                    width=20, height=1, command=self.on_toggle)
        self.toggle_btn.pack(pady=2)

        self.next_btn = tk.Button(self.root, text="다음 (650 멈춤 풀기)", font=("맑은 고딕", 10),
                                  width=24, command=self.on_next)
        self.next_btn.pack(pady=2)

        self.log = tk.Label(self.root, text="", font=("맑은 고딕", 8), fg="#666666",
                            wraplength=320, justify="left")
        self.log.pack(pady=(6, 0), padx=8)

        self.topmost = tk.BooleanVar(value=False)
        tk.Checkbutton(self.root, text="항상 위", variable=self.topmost,
                       command=lambda: self.root.attributes("-topmost", self.topmost.get())
                       ).place(x=8, y=4)

        self.refresh()

    def on_toggle(self) -> None:
        if self.busy:
            return
        self.busy = True
        pid = bot_pid()
        if pid:
            self.status.config(text="끄는 중…", fg="#996600")
            self.toggle_btn.config(state="disabled")
            winsound.Beep(330, 200)
            threading.Thread(target=self._stop_work, args=(pid,), daemon=True).start()
        else:
            winsound.Beep(880, 100)
            try:
                start_bot()
            except Exception as e:
                self.status.config(text=f"켜기 실패: {e}", fg="#cc0000")
            self.busy = False
            self.refresh_once()

    def _stop_work(self, pid: int) -> None:
        try:
            stop_bot(pid)
        finally:
            self.root.after(0, self._stop_done)

    def _stop_done(self) -> None:
        self.busy = False
        self.toggle_btn.config(state="normal")
        self.refresh_once()

    def on_next(self) -> None:
        (ROOT / "logs" / "step.go").touch()
        winsound.Beep(660, 80)

    def refresh_once(self) -> None:
        if self.busy:
            return
        if bot_pid():
            self.status.config(text="봇: 켜짐", fg="#007700")
            self.toggle_btn.config(text="끄기")
        else:
            self.status.config(text="봇: 꺼짐", fg="#cc0000")
            self.toggle_btn.config(text="켜기 (650모드)")
        line = last_log_line()
        self.log.config(text=line[-90:] if line else "")

    def refresh(self) -> None:
        self.refresh_once()
        self.root.after(1000, self.refresh)


def main() -> int:
    # 이 창도 하나만 뜨게 (가비지 컬렉션으로 핸들이 닫히지 않게 들고 있는다)
    main.mutex = kernel32.CreateMutexW(None, True, r"Local\stella_botpanel")
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        hwnd = user32.FindWindowW(None, "텔라리 봇")
        if hwnd:
            user32.SetForegroundWindow(hwnd)  # 이미 떠 있으면 그 창을 앞으로
        return 0
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # 고해상도에서 창이 흐릿하지 않게
    except OSError:
        pass
    Panel().root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
