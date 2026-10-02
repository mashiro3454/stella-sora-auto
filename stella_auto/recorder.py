"""650원 이긴 판 녹화 (사용자 요청 2026-10-03).

봇이 어차피 매 판단마다 화면을 찍으니, 그 프레임을 고정 10fps 타임라인에 이어 붙여 mp4로 만든다
(프레임 사이가 길면 마지막 프레임을 반복해서 실제 시간과 비슷하게). 판이 시작되면 임시 폴더
(OneDrive 밖)에 쓰다가, 650원을 이긴 판만 logs/videos/로 옮기고 나머지는 지운다.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

FPS = 10
SIZE = (1280, 720)
MAX_BURST = 30  # 프레임 사이가 아주 길어도 (멈춤 등) 이만큼만 반복해 쓴다 (3초 분량)
KEEP_DAYS = 2


class Recorder:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.writer: cv2.VideoWriter | None = None
        self.tmp_path: Path | None = None
        self.last_t = 0.0
        self.stamp = ""
        self.prune()

    @property
    def recording(self) -> bool:
        return self.writer is not None

    def start(self) -> None:
        if self.writer is not None:
            self.stop(keep=False)
        self.stamp = time.strftime("%m%d_%H%M%S")
        self.tmp_path = Path(tempfile.gettempdir()) / f"stella_rec_{self.stamp}.mp4"
        self.writer = cv2.VideoWriter(str(self.tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
        self.last_t = time.monotonic()

    def push(self, img: np.ndarray) -> None:
        """봇이 화면을 볼 때마다 부른다. 지난 시간만큼 (10fps 기준) 반복해서 쓴다."""
        if self.writer is None:
            return
        now = time.monotonic()
        n = min(MAX_BURST, max(1, round((now - self.last_t) * FPS)))
        self.last_t = now
        frame = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)
        for _ in range(n):
            self.writer.write(frame)

    def stop(self, keep: bool) -> Path | None:
        """녹화 끝. keep이면 logs/videos/로 옮기고 경로를 돌려준다, 아니면 지운다."""
        if self.writer is None:
            return None
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
        return dest

    def prune(self, keep_days: int = KEEP_DAYS, keep_count: int = 15) -> None:
        """이틀 지난 것과 최신 15개를 넘는 것은 지운다 (저장 공간/OneDrive 보호)."""
        if not self.out_dir.exists():
            return
        cutoff = time.time() - keep_days * 86400
        files = sorted(self.out_dir.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
        for i, f in enumerate(files):
            if i >= keep_count or f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
