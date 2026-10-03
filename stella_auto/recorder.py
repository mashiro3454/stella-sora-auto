"""650원 이긴 판 녹화 (사용자 요청 2026-10-03).

전용 스레드가 게임 창을 일정 간격(10fps)으로 직접 찍어서 mp4로 쓴다. 봇이 보는 프레임만 쓰면
판단 사이가 길 때 영상이 뚝뚝 끊겨서 보기 힘들다 (사용자 지적). 판이 시작되면 임시 폴더
(OneDrive 밖)에 쓰다가, 650원을 이긴 판만 logs/videos/로 옮기고 나머지는 지운다.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from pathlib import Path

import cv2

FPS = 10
SIZE = (960, 540)  # mp4v라 비트레이트 조절이 없어서 크기/fps로 용량을 잡는다
KEEP_DAYS = 2
KEEP_COUNT = 8


class Recorder:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.writer: cv2.VideoWriter | None = None
        self.tmp_path: Path | None = None
        self.stamp = ""
        self._thread: threading.Thread | None = None
        self._run = False
        self._lock = threading.Lock()
        self.prune()

    @property
    def recording(self) -> bool:
        return self._run

    def start(self) -> None:
        if self._run:
            self.stop(keep=False)
        self.stamp = time.strftime("%m%d_%H%M%S")
        self.tmp_path = Path(tempfile.gettempdir()) / f"stella_rec_{self.stamp}.mp4"
        self.writer = cv2.VideoWriter(str(self.tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
        self._run = True
        self._thread = threading.Thread(target=self._loop, name="recorder", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        from .capture import capture, find_game_window

        next_t = time.monotonic()
        frame = None
        while self._run:
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.05, next_t - now))
                continue
            next_t += 1.0 / FPS
            if next_t < now - 1.0:  # 한참 밀렸으면 (창을 못 찍는 동안) 따라잡으려 몰아 쓰지 않는다
                next_t = now
            try:
                win = find_game_window()
                if win is not None and not win.minimized:
                    frame = cv2.resize(capture(win), SIZE, interpolation=cv2.INTER_AREA)
            except Exception:
                pass  # 창이 잠깐 없어도 (재시작 로딩 등) 녹화는 계속
            with self._lock:
                if self._run and self.writer is not None and frame is not None:
                    self.writer.write(frame)

    def push(self, img) -> None:
        """예전 호환용 (이제 전용 스레드가 찍는다)."""

    def stop(self, keep: bool) -> Path | None:
        """녹화 끝. keep이면 logs/videos/로 옮기고 경로를 돌려준다, 아니면 지운다."""
        if not self._run and self.writer is None:
            return None
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            if self.writer is not None:
                self.writer.release()
                self.writer = None
        path, self.tmp_path = self.tmp_path, None
        if path is None or not path.exists():
            return None
        if not keep:
            path.unlink(missing_ok=True)
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        dest = self.out_dir / f"{self.stamp}.mp4"
        shutil.move(str(path), str(dest))
        self.prune()
        return dest

    def prune(self, keep_days: int = KEEP_DAYS, keep_count: int = KEEP_COUNT) -> None:
        """이틀 지난 것과 최신 8개를 넘는 것은 지운다 (판당 수백 MB가 OneDrive로 올라가서 빡빡하게)."""
        if not self.out_dir.exists():
            return
        cutoff = time.time() - keep_days * 86400
        files = sorted(self.out_dir.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
        for i, f in enumerate(files):
            if i >= keep_count or f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
