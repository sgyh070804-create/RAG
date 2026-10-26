# -*- coding: utf-8 -*-
"""阶段 1 配置：所有路径和参数集中在这里，改一处即可生效。"""
import os
from pathlib import Path

# 项目根目录 = src/ 的上一级
ROOT = Path(__file__).resolve().parent.parent

# 语料目录：放 .md / .txt 文件
DOC_DIR = ROOT / "data" / "docs"

# 向量库持久化目录（重启后索引仍然存在）
CHROMA_DIR = ROOT / "data" / "index" / "chroma"
COLLECTION_NAME = "yunfan_docs"

# embedding 模型：优先用本地目录，没有就回退到 HF 仓库名（会联网下载）
LOCAL_MODEL_DIR = ROOT / ".models" / "bge-small-zh-v1.5"
MODEL_NAME = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else "BAAI/bge-small-zh-v1.5"

# 切块参数（按字符数）
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# 检索参数
TOP_K = 3

# 生成端：本地 Ollama 的 OpenAI 兼容接口
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")

# 中文标点，用于让切块尽量落在句子边界
SENTENCE_ENDS = "。！？；\n"
