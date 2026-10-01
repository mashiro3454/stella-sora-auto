"""F12 비상 정지.

봇 본체와 따로 도는 감시 스레드가 F12를 0.03초마다 확인하고, 눌리는 즉시 눌려 있는 키를 전부 떼고
프로세스를 바로 끝낸다. 봇이 걷기, 재시작, 글자 인식처럼 오래 걸리는 동작 중이어도 기다리지 않는다.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from typing import Callable

VK_F12 = 0x7B
POLL_SEC = 0.03

_user32 = ctypes.WinDLL("user32")


def _pressed(vk: int) -> bool:
    # 0x8000: 지금 눌려 있음, 0x0001: 지난번 확인 뒤로 눌린 적 있음 (톡 누르고 뗀 것도 잡는다)
    return bool(_user32.GetAsyncKeyState(vk) & 0x8001)


def start(on_stop: Callable[[], None], vk: int = VK_F12) -> threading.Thread:
    """vk 키가 눌리면 on_stop()을 부르고 프로세스를 끝낸다. on_stop에서 오류가 나도 끝낸다."""
    _pressed(vk)  # 시작 전에 눌렸던 기록은 지운다

    def watch() -> None:
        while True:
            if _pressed(vk):
                try:
                    on_stop()
                finally:
                    os._exit(0)
            time.sleep(POLL_SEC)

    t = threading.Thread(target=watch, name="killswitch", daemon=True)
    t.start()
    return t
