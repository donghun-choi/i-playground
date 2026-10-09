"""양자화 가중치(정수) 엔트로피 부호: 출력 채널마다 이산 라플라스(기하) 분포 하나, 파라미터 1바이트.

P(q) ∝ f(|q|),  f(0) = 2^24,  f(k+1) = max(1, f(k)·r >> 8),  r = 1..255,  q ∈ [-qmax, qmax]
빈도를 정수 연산만으로 만들고 정확한 나눗셈으로 확률을 내므로 (segcodec 과 같은 방식) 기계와 무관하게 같은 부호가 된다.

pack_state 형식 (model.py) 의 바이트를 그대로 받아 같은 바이트로 되돌린다:
    ec_pack(buf, template) → 'EC' 형식,  ec_unpack(buf, template) → pack_state 형식
EC 형식: "<HB" (n, 0xEC) + u32 len + xz(스케일·1차원 fp16·채널별 r·텐서별 qmax) + range coder 단어들
"""

from __future__ import annotations

import lzma
import struct

import numpy as np

MARK = 0xEC
_F0 = 1 << 24
_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
_TABLES: dict = {}


def _probs(r: int, qmax: int) -> np.ndarray:
    key = (r, qmax)
    if key not in _TABLES:
        f = [_F0]
        for _ in range(qmax):
            f.append(max(1, (f[-1] * r) >> 8))
        a = np.array([f[abs(q)] for q in range(-qmax, qmax + 1)], np.float64)
        _TABLES[key] = np.ascontiguousarray(a / a.sum())
    return _TABLES[key]


def _best_r(row: np.ndarray, qmax: int) -> int:
    """교차 엔트로피가 가장 작은 r (|q| 히스토그램으로 계산)."""
    hist = np.bincount(np.abs(row), minlength=qmax + 1)
    best, best_bits = 1, None
    for r in range(1, 256):
        p = _probs(r, qmax)
        bits = -(hist[0] * np.log2(p[qmax]) + sum(hist[k] * np.log2(2 * p[qmax + k]) for k in range(1, qmax + 1)))
        if best_bits is None or bits < best_bits:
            best, best_bits = r, bits
    return best


def _enc_rows(enc, fam, q: np.ndarray):
    """(c, m) 정수 → 행마다 r 을 골라 부호화. → (qmax, r 바이트들)"""
    qmax = int(np.abs(q).max())
    rs = [_best_r(row, qmax) if qmax else 1 for row in q]
    if qmax:
        for row, r in zip(q, rs):
            sym = np.ascontiguousarray((row + qmax).astype(np.int32))
            enc.encode(sym, fam, np.ascontiguousarray(np.broadcast_to(_probs(r, qmax), (len(sym), 2 * qmax + 1))))
    return qmax, bytes(rs)


def _dec_rows(dec, fam, qmax: int, rs: bytes, m: int) -> np.ndarray:
    q = np.zeros((len(rs), m), np.int64)
    if qmax:
        for i, r in enumerate(rs):
            q[i] = dec.decode(fam, np.ascontiguousarray(np.broadcast_to(_probs(r, qmax), (m, 2 * qmax + 1)))) - qmax
    return q


def _parse(buf: bytes, template: dict):
    """pack_state 바이트 → (n, bits, [(kind, scales_bytes | None, q 또는 fp16 바이트)])"""
    n, bits = struct.unpack_from("<HB", buf, 0)
    off, out = 3, []
    for v in template.values():
        (kind,) = struct.unpack_from("<B", buf, off)
        off += 1
        if kind == 1:
            c = v.shape[0]
            sc = buf[off : off + 2 * c]
            off += 2 * c
            q = np.frombuffer(buf, np.int8, v.numel(), off).reshape(c, -1).astype(np.int64)
            off += v.numel()
            out.append((1, sc, q))
        else:
            out.append((0, None, buf[off : off + 2 * v.numel()]))
            off += 2 * v.numel()
    assert off == len(buf)
    return n, bits, out


def ec_pack(buf: bytes, template: dict) -> bytes:
    import constriction

    n, bits, items = _parse(buf, template)
    raw, enc = [struct.pack("<B", bits)], constriction.stream.queue.RangeEncoder()
    fam = constriction.stream.model.Categorical(perfect=False)
    for kind, sc, q in items:
        raw.append(struct.pack("<B", kind))
        if kind == 0:
            raw.append(q)
            continue
        qmax, rs = _enc_rows(enc, fam, q)
        raw += [struct.pack("<B", qmax), sc, rs]
    z = lzma.compress(b"".join(raw), format=lzma.FORMAT_RAW, filters=_FILTERS)
    return struct.pack("<HBI", n, MARK, len(z)) + z + enc.get_compressed().tobytes()


def is_ec(buf: bytes) -> bool:
    return len(buf) >= 3 and buf[2] == MARK


def ec_unpack(buf: bytes, template: dict) -> bytes:
    import constriction

    n, mark, lz = struct.unpack_from("<HBI", buf, 0)
    assert mark == MARK
    hl = struct.calcsize("<HBI")
    raw = lzma.decompress(buf[hl : hl + lz], format=lzma.FORMAT_RAW, filters=_FILTERS)
    dec = constriction.stream.queue.RangeDecoder(np.frombuffer(buf[hl + lz :], np.uint32))
    fam = constriction.stream.model.Categorical(perfect=False)
    out, off = [struct.pack("<HB", n, raw[0])], 1
    for v in template.values():
        kind = raw[off]
        off += 1
        out.append(bytes([kind]))
        if kind == 0:
            out.append(raw[off : off + 2 * v.numel()])
            off += 2 * v.numel()
            continue
        c, m = v.shape[0], v.numel() // v.shape[0]
        qmax = raw[off]
        sc = raw[off + 1 : off + 1 + 2 * c]
        rs = raw[off + 1 + 2 * c : off + 1 + 3 * c]
        off += 1 + 3 * c
        out += [sc, _dec_rows(dec, fam, qmax, rs, m).astype(np.int8).tobytes()]
    assert off == len(raw)
    return b"".join(out)


# ---------------------------------------------------------------- 문맥 모델 (segcodec.QNet.to_bytes 형식, 모양이 바이트 안에 있다)
def ec_pack_qnet(buf: bytes) -> bytes:
    """QNet 바이트 → 정수 가중치는 range coder, 헤더·B·M (int32) 는 xz."""
    import constriction

    enc, fam = constriction.stream.queue.RangeEncoder(), constriction.stream.model.Categorical(perfect=False)
    (n,) = struct.unpack_from("<B", buf, 0)
    off, raw = 1, [buf[:1]]
    for _ in range(n):
        o, i, k, _d, _r = struct.unpack_from("<HHBBB", buf, off)
        raw.append(buf[off : off + 7])
        off += 7
        W = np.frombuffer(buf, np.int8, o * i * k * k, off).reshape(o, -1).astype(np.int64)
        off += o * i * k * k
        qmax, rs = _enc_rows(enc, fam, W)
        raw += [struct.pack("<B", qmax), rs, buf[off : off + 8 * o]]
        off += 8 * o
    assert off == len(buf)
    z = lzma.compress(b"".join(raw), format=lzma.FORMAT_RAW, filters=_FILTERS)
    return struct.pack("<I", len(z)) + z + enc.get_compressed().tobytes()


def ec_unpack_qnet(buf: bytes) -> bytes:
    import constriction

    (lz,) = struct.unpack_from("<I", buf, 0)
    raw = lzma.decompress(buf[4 : 4 + lz], format=lzma.FORMAT_RAW, filters=_FILTERS)
    dec, fam = constriction.stream.queue.RangeDecoder(np.frombuffer(buf[4 + lz :], np.uint32)), constriction.stream.model.Categorical(perfect=False)
    n = raw[0]
    off, out = 1, [raw[:1]]
    for _ in range(n):
        o, i, k, _d, _r = struct.unpack_from("<HHBBB", raw, off)
        out.append(raw[off : off + 7])
        off += 7
        qmax, rs = raw[off], raw[off + 1 : off + 1 + o]
        off += 1 + o
        out += [_dec_rows(dec, fam, qmax, rs, i * k * k).astype(np.int8).tobytes(), raw[off : off + 8 * o]]
        off += 8 * o
    assert off == len(raw)
    return b"".join(out)


# ---------------------------------------------------------------- 정수 행렬 하나 (pose 기저 등)
def ec_pack_rows(q: np.ndarray) -> bytes:
    """(r, m) 정수 → [u8 qmax][r 바이트들][u32 단어 수][단어들]."""
    import constriction

    enc, fam = constriction.stream.queue.RangeEncoder(), constriction.stream.model.Categorical(perfect=False)
    qmax, rs = _enc_rows(enc, fam, q.astype(np.int64))
    words = enc.get_compressed()
    return struct.pack("<B", qmax) + rs + struct.pack("<I", len(words)) + words.tobytes()


def ec_unpack_rows(buf: bytes, r: int, m: int, off: int = 0):
    """→ ((r, m) int64, 다음 오프셋)"""
    import constriction

    qmax = buf[off]
    rs = buf[off + 1 : off + 1 + r]
    (nw,) = struct.unpack_from("<I", buf, off + 1 + r)
    w0 = off + 5 + r
    dec = constriction.stream.queue.RangeDecoder(np.frombuffer(buf, np.uint32, nw, w0))
    fam = constriction.stream.model.Categorical(perfect=False)
    return _dec_rows(dec, fam, qmax, rs, m), w0 + 4 * nw
