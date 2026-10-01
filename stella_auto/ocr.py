"""Windows 기본 OCR(한국어)로 화면 글자 읽기.

Windows에 한국어 OCR이 깔려 있어야 한다 (한국어 Windows는 기본으로 있음).
작은 글자는 잘 못 읽어서 기본으로 2배 키워서 읽는다.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import cv2
import numpy as np
from winrt.windows.globalization import Language
from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.storage.streams import DataWriter


@dataclass
class OcrWord:
    text: str
    box: tuple[int, int, int, int]  # (x, y, w, h) 원래 화면 좌표


@dataclass
class OcrLine:
    text: str
    words: list[OcrWord]

    @property
    def box(self) -> tuple[int, int, int, int]:
        x0 = min(w.box[0] for w in self.words)
        y0 = min(w.box[1] for w in self.words)
        x1 = max(w.box[0] + w.box[2] for w in self.words)
        y1 = max(w.box[1] + w.box[3] for w in self.words)
        return x0, y0, x1 - x0, y1 - y0


class KoreanOcr:
    def __init__(self, lang: str = "ko"):
        self.engine = OcrEngine.try_create_from_language(Language(lang))
        if self.engine is None:
            raise RuntimeError(f"Windows OCR에 '{lang}' 언어가 없음 (설정 > 언어에서 한국어 추가)")
        self._loop = asyncio.new_event_loop()

    def _bitmap(self, img: np.ndarray) -> SoftwareBitmap:
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA if img.ndim == 3 else cv2.COLOR_GRAY2BGRA)
        writer = DataWriter()
        writer.write_bytes(np.ascontiguousarray(bgra).tobytes())
        return SoftwareBitmap.create_copy_from_buffer(
            writer.detach_buffer(), BitmapPixelFormat.BGRA8, bgra.shape[1], bgra.shape[0]
        )

    def read(self, img: np.ndarray, box: tuple[int, int, int, int] | None = None, scale: float = 2.0) -> list[OcrLine]:
        """img의 box(x0, y0, x1, y1) 부분을 읽는다. 좌표는 원래 img 기준으로 돌려준다."""
        ox, oy = 0, 0
        if box:
            x0, y0, x1, y1 = box
            img, ox, oy = img[y0:y1, x0:x1], x0, y0
        if scale != 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        result = self._loop.run_until_complete(self.engine.recognize_async(self._bitmap(img)))
        lines = []
        for line in result.lines:
            words = []
            for w in line.words:
                r = w.bounding_rect
                words.append(OcrWord(w.text, (int(ox + r.x / scale), int(oy + r.y / scale),
                                              int(r.width / scale), int(r.height / scale))))
            lines.append(OcrLine(line.text, words))
        return lines

    def text(self, img: np.ndarray, box: tuple[int, int, int, int] | None = None, scale: float = 2.0) -> str:
        return "\n".join(l.text for l in self.read(img, box, scale))
