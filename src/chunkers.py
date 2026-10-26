# -*- coding: utf-8 -*-
"""阶段 2 · 两种切块策略，供 A/B 对比用。

- hard_split：阶段 1 的现状算法（固定 400 字窗口、尽量在句末收口、带重叠）。
- structure_split：内容感知。标题行、表格、``` 代码块都是原子块，段落按空行切；
  这些块顺序打包到 500 字上限，标题不会被留在上一块末尾当孤儿，超长段落再硬切，
  块之间带 80 字重叠。

两个函数都只做纯字符串处理，不依赖模型，方便单测和对比。
"""
from __future__ import annotations

import re

from config import CHUNK_OVERLAP, CHUNK_SIZE, SENTENCE_ENDS

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


# --------------------------------------------------------------------------
# A：现状算法
# --------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """A 策略切块前会做的规范化：去掉空行、行首尾空白。"""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def hard_windows(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, int]]:
    """hard_split 的切块区间，给需要核对边界的地方用（文本应已 normalize_text）。"""
    windows: list[tuple[int, int]] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window_start = start + int(size * 0.6)
            cuts = [text.rfind(ch, window_start, end) for ch in SENTENCE_ENDS]
            cut = max(cuts)
            if cut != -1:
                end = cut + 1
        if text[start:end].strip():
            windows.append((start, end))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return windows


def hard_split(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """按字符滑窗切块，尽量让切口落在句号/问号/换行上（阶段 1 的原样逻辑）。"""
    text = normalize_text(text)
    if not text:
        return []
    return [
        text[a:b].strip() for a, b in hard_windows(text, size, overlap) if text[a:b].strip()
    ]


# --------------------------------------------------------------------------
# B：内容感知
# --------------------------------------------------------------------------
def _tail_overlap(text: str, overlap: int) -> str:
    """从 text 末尾取一段作重叠内容，尽量从一个句子开头开始。"""
    if overlap <= 0 or not text:
        return ""
    cut = -1
    for ch in SENTENCE_ENDS:
        idx = text.rfind(ch, max(0, len(text) - overlap - 1))
        if idx != -1:
            cut = max(cut, idx + 1)
    if cut != -1 and len(text) - cut <= overlap + 40:
        return text[cut:].strip()
    return text[-overlap:].strip()


def _blocks_with_headings(text: str) -> list[str]:
    """把全文切成原子块：标题行、代码块、表格、段落，顺序不变。"""
    lines = text.splitlines()
    blocks: list[str] = []
    buf: list[str] = []
    i, n = 0, len(lines)
    in_fence = False

    def flush() -> None:
        nonlocal buf
        piece = "\n".join(buf).strip()
        if piece:
            blocks.append(piece)
        buf = []

    while i < n:
        line = lines[i]
        if HEADING_RE.match(line):
            flush()
            blocks.append(line.strip())
            i += 1
            continue
        if FENCE_RE.match(line):
            if not in_fence:
                flush()
                in_fence = True
            buf.append(line)
            in_fence = not in_fence
            if not in_fence:
                flush()
            i += 1
            continue
        if in_fence:
            buf.append(line)
            i += 1
            continue
        if not line.strip():
            flush()
            i += 1
            continue
        if line.lstrip().startswith("|"):
            flush()
            table = []
            while i < n and lines[i].lstrip().startswith("|"):
                table.append(lines[i])
                i += 1
            blocks.append("\n".join(table).strip())
            continue
        buf.append(line)
        i += 1
    flush()
    return blocks


def structure_split(
    text: str,
    max_size: int = 500,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """标题/段落感知切块：原子块按顺序打包到 max_size，块间带 overlap。"""
    if not text.strip():
        return []

    chunks: list[str] = []

    def push(piece: str) -> None:
        piece = piece.strip()
        if piece:
            chunks.append(piece)

    cur = ""
    for block in _blocks_with_headings(text):
        if len(block) > max_size:  # 单块就超限：先冲刷，再硬切这一段
            push(cur)
            cur = ""
            for piece in hard_split(block, max_size, overlap):
                push(piece)
            continue

        is_heading = bool(HEADING_RE.match(block))
        # 标题不留在上一块末尾：当前块过半就先收口，让标题开新块
        if is_heading and cur and len(cur) >= max_size * 0.5:
            push(cur)
            cur = block
            continue

        if not cur:
            cur = block
            continue

        candidate = cur + "\n\n" + block
        if len(candidate) <= max_size:
            cur = candidate
            continue

        push(cur)
        seed = _tail_overlap(cur, overlap)
        if seed and len(seed) + 2 + len(block) <= max_size and len(seed) < len(cur) * 0.5:
            cur = seed + "\n\n" + block
        else:
            cur = block
    push(cur)
    return chunks


STRATEGIES = {
    "A": ("hard_split", hard_split),
    "B": ("structure_split", structure_split),
}


def get_strategy(name: str):
    key = name.strip().upper()
    if key not in STRATEGIES:
        raise SystemExit(f"未知策略：{name}（可选 A / B）")
    return STRATEGIES[key]
