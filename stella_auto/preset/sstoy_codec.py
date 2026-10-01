"""sstoy 공유 링크(#build=...) 해독기.

원본 구현: JforPlay/sstoy src/modules/app-saveload.ts
  - unpackSharePayload / decompressSharePayload / base91Decode / readVarint

지원 형식
  v3d-  deflate(raw) + base91, ID를 그대로 저장 (현재 sstoy 기본값)
  v2d-  deflate(raw) + base91, ID를 인덱스로 저장 (구버전, 고정 ID 표로 복원)

지원 안 함
  v3r-/v2r-  sstoy의 base32768 표가 511글자뿐이라 인코딩 단계에서 이미
             'undefined' 문자열로 깨진다. sstoy 자신도 못 읽는다.
  N4Ig...    더 옛날 LZ-String JSON 형식.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from urllib.parse import quote, unquote

BASE91_CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "0123456789!#$%&()*+,./:;<=>?@[]^_`{|}~\""
)
_BASE91_INDEX = {c: i for i, c in enumerate(BASE91_CHARS)}

SHARE_POSITIONS = ("master", "assist1", "assist2")
DISC_SLOTS = ("main1", "main2", "main3", "sub1", "sub2", "sub3")

# app-saveload.ts MARK_CODES (한국어 표기 기준)
MARKS = {1: "필수", 2: "후순위", 3: "다다익선", 4: "명함만"}

# v2 링크용 고정 ID 표 (app-saveload.ts LEGACY_V2_CHAR_IDS / LEGACY_V2_POT_IDS)
_LEGACY_V2_CHAR_IDS = [
    103, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    123, 125, 126, 127, 129, 130, 132, 133, 134, 135, 136, 141, 142, 143, 144, 145,
    147, 149, 150, 155, 156, 158, 159,
]
# 캐릭터마다 01~13, 21~33, 41~43 패턴이고 아래 캐릭터 목록 순서와 같다.
_LEGACY_V2_POT_CHARS = [
    103, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 123, 125,
    126, 127, 130, 132, 133, 134, 135, 136, 141, 142, 143, 144, 145, 147, 149, 150,
    155, 156, 158, 159,
]
_POT_SUFFIXES = [*range(1, 14), *range(21, 34), *range(41, 44)]


def _legacy_v2_pot_ids() -> list[int]:
    return sorted(500000 + c * 100 + s for c in _LEGACY_V2_POT_CHARS for s in _POT_SUFFIXES)


class ShareCodeError(ValueError):
    pass


@dataclass
class RawCharacter:
    """sstoy가 저장한 캐릭터 한 명분 데이터 (가공 전)."""

    char_id: int
    potentials: list[int]  # sstoy에 선택된 순서 그대로
    levels: dict[int, int] = field(default_factory=dict)  # 목표 레벨(1이 아닌 것만)
    marks: dict[int, int] = field(default_factory=dict)  # 1 필수, 2 후순위, 3 다다익선, 4 명함만


@dataclass
class RawDisc:
    disc_id: int
    limit_break: int
    sub_level: int


@dataclass
class RawBuild:
    version: int
    name: str
    characters: dict[str, RawCharacter]  # key: master / assist1 / assist2
    discs: dict[str, RawDisc] | None


# ---------------------------------------------------------------------------
# 저수준 디코딩
# ---------------------------------------------------------------------------


def base91_decode(text: str) -> bytes:
    b = n = 0
    v = -1
    out = bytearray()
    for ch in text:
        p = _BASE91_INDEX.get(ch)
        if p is None:
            continue
        if v < 0:
            v = p
            continue
        v += p * 91
        b |= v << n
        n += 13 if (v & 8191) > 88 else 14
        while n > 7:
            out.append(b & 255)
            b >>= 8
            n -= 8
        v = -1
    if v >= 0:
        out.append((b | (v << n)) & 255)
    return bytes(out)


def base91_encode(data: bytes) -> str:
    b = n = 0
    out = []
    for byte in data:
        b |= byte << n
        n += 8
        if n > 13:
            v = b & 8191
            if v > 88:
                b >>= 13
                n -= 13
            else:
                v = b & 16383
                b >>= 14
                n -= 14
            out.append(BASE91_CHARS[v % 91] + BASE91_CHARS[v // 91])
    if n:
        out.append(BASE91_CHARS[b % 91])
        if n > 7 or b > 90:
            out.append(BASE91_CHARS[b // 91])
    return "".join(out)


def _write_varint(out: bytearray, value: int) -> None:
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def byte(self) -> int:
        if self.pos >= len(self.data):
            raise ShareCodeError("데이터가 중간에 끊김")
        b = self.data[self.pos]
        self.pos += 1
        return b

    def varint(self) -> int:
        result = shift = 0
        while True:
            b = self.byte()
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                return result
            shift += 7
            if shift > 35:
                raise ShareCodeError("varint가 너무 김")

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ShareCodeError("데이터가 중간에 끊김")
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        return chunk

    @property
    def remaining(self) -> int:
        return len(self.data) - self.pos


def unpack_payload(data: bytes) -> RawBuild:
    r = _Reader(data)
    version = r.varint()
    if version not in (2, 3):
        raise ShareCodeError(f"모르는 버전: {version}")

    if version == 2:
        char_map = {i + 1: cid for i, cid in enumerate(sorted(set(_LEGACY_V2_CHAR_IDS)))}
        pot_map = {i + 1: pid for i, pid in enumerate(_legacy_v2_pot_ids())}
    else:
        char_map = pot_map = {}

    name = r.take(r.varint()).decode("utf-8")
    slot_mask = r.byte()

    characters: dict[str, RawCharacter] = {}
    for idx, pos in enumerate(SHARE_POSITIONS):
        if not slot_mask & (1 << idx):
            continue
        raw_id = r.varint()
        char_id = char_map.get(raw_id, raw_id)

        pots = [r.varint() for _ in range(r.varint())]
        pots = [pot_map.get(p, p) for p in pots]

        levels: dict[int, int] = {}
        key = 0
        for _ in range(r.varint()):
            key += r.varint()
            levels[pot_map.get(key, key)] = r.varint() + 1

        marks: dict[int, int] = {}
        key = 0
        for _ in range(r.varint()):
            key += r.varint()
            code = r.byte()
            if code in MARKS:
                marks[pot_map.get(key, key)] = code

        characters[pos] = RawCharacter(char_id, pots, levels, marks)

    discs: dict[str, RawDisc] | None = None
    if r.remaining and r.byte() == 1 and r.remaining:
        disc_mask = r.byte() & 0x3F
        discs = {}
        for idx, slot in enumerate(DISC_SLOTS):
            if not disc_mask & (1 << idx):
                continue
            disc_id = r.varint()
            packed = r.byte()
            discs[slot] = RawDisc(disc_id, ((packed >> 3) & 0x07) + 1, packed & 0x07)

    if r.remaining:
        raise ShareCodeError(f"해석 후 {r.remaining}바이트가 남음")
    return RawBuild(version, name, characters, discs)


def pack_payload(build: RawBuild) -> bytes:
    """packSharePayload의 v3 포팅. 항상 version 3으로 쓴다."""
    out = bytearray()
    _write_varint(out, 3)
    name = build.name.encode("utf-8")
    _write_varint(out, len(name))
    out += name

    out.append(sum(1 << i for i, pos in enumerate(SHARE_POSITIONS) if pos in build.characters))
    for pos in SHARE_POSITIONS:
        ch = build.characters.get(pos)
        if ch is None:
            continue
        _write_varint(out, ch.char_id)
        _write_varint(out, len(ch.potentials))
        for pid in ch.potentials:
            _write_varint(out, pid)

        levels = sorted((k, v) for k, v in ch.levels.items() if v != 1)
        _write_varint(out, len(levels))
        prev = 0
        for pid, level in levels:
            _write_varint(out, pid - prev)
            _write_varint(out, max(0, level - 1))
            prev = pid

        marks = sorted((k, v) for k, v in ch.marks.items() if v in MARKS)
        _write_varint(out, len(marks))
        prev = 0
        for pid, code in marks:
            _write_varint(out, pid - prev)
            out.append(code)
            prev = pid

    if build.discs is None:
        out.append(0)
    else:
        out.append(1)
        out.append(sum(1 << i for i, slot in enumerate(DISC_SLOTS) if slot in build.discs))
        for slot in DISC_SLOTS:
            d = build.discs.get(slot)
            if d is None:
                continue
            _write_varint(out, d.disc_id)
            out.append(((max(0, d.limit_break - 1) & 0x07) << 3) | (d.sub_level & 0x07))
    return bytes(out)


# ---------------------------------------------------------------------------
# 링크 → RawBuild
# ---------------------------------------------------------------------------

# base91 글자에 '#', '&', '='도 있어서 URL을 그 글자들로 자르지 않고 접두사부터 찾는다.
_CODE_RE = re.compile(r"v[23][dr]-\S+")
_NEXT_PARAM_RE = re.compile(r"&\w+=")


def _inflate(data: bytes) -> bytes:
    d = zlib.decompressobj(-15)
    out = d.decompress(data)
    if not d.eof or d.unused_data:
        raise ShareCodeError("압축 스트림이 깨졌거나 뒤에 다른 데이터가 붙어 있음")
    return out


# encodeURIComponent 결과에 나올 수 있는 글자. 이 밖의 글자가 있으면 인코딩 안 된 원문이다.
_URI_ENCODED_RE = re.compile(r"[A-Za-z0-9\-_.!~*'()%]*")


def _unescape_candidates(body: str) -> tuple[str, ...]:
    if _URI_ENCODED_RE.fullmatch(body):
        return unquote(body), body, unquote(unquote(body))
    # 원문에 우연히 '%2B' 같은 조각이 있으면 풀어도 그럴듯하게 읽힐 수 있어서 원문을 먼저 본다.
    return body, unquote(body)


def extract_code(text: str) -> str:
    """전체 URL, '#build=...', 'v3d-...' 무엇을 넣어도 코드 부분만 꺼낸다."""
    text = text.strip()
    m = _CODE_RE.search(text)
    if m:
        return m.group(0)
    if text.startswith("N4Ig") or "build=N4Ig" in text:
        raise ShareCodeError("LZ-String 형식의 아주 옛날 링크는 지원 안 함. sstoy에서 열어서 다시 공유해줘")
    raise ShareCodeError("sstoy 공유 코드(v3d-...)를 못 찾음")


def decode_share_code(text: str) -> RawBuild:
    code = extract_code(text)
    prefix, body = code[:4], code[4:]
    if prefix in ("v3r-", "v2r-"):
        raise ShareCodeError(
            "무압축(v3r) 링크는 sstoy 버그로 데이터가 깨져 있어서 복원 불가. "
            "잠재력을 조금 더 넣으면 v3d 링크가 나옴"
        )

    # 보통은 퍼센트 인코딩을 한 번 풀면 된다. 하지만 주소창에서 복사하면서 이미
    # 풀린 문자열이 오거나, 뒤에 다른 해시 파라미터(&key=)가 붙었을 수도 있다.
    # 그래서 후보를 여러 개 만들고 압축 스트림과 페이로드가 끝까지 딱 맞게
    # 읽히는 첫 번째 후보를 쓴다.
    bodies = [body] + [body[: m.start()] for m in _NEXT_PARAM_RE.finditer(body)]
    candidates = [c for b in bodies for c in _unescape_candidates(b)]
    errors = []
    for cand in dict.fromkeys(candidates):
        try:
            return unpack_payload(_inflate(base91_decode(cand)))
        except (zlib.error, ShareCodeError, UnicodeDecodeError) as e:
            errors.append(f"{type(e).__name__}: {e}")
    raise ShareCodeError("해독 실패 (" + " / ".join(errors) + ")")


def encode_share_code(build: RawBuild) -> str:
    """RawBuild → 'v3d-...' (퍼센트 인코딩 전). sstoy 링크는 app.html#build=<quote(코드)>."""
    comp = zlib.compressobj(9, zlib.DEFLATED, -15)
    return "v3d-" + base91_encode(comp.compress(pack_payload(build)) + comp.flush())


def share_url(build: RawBuild) -> str:
    return "https://jforplay.github.io/sstoy/app.html#build=" + quote(encode_share_code(build), safe="")
