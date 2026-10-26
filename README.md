# 第一版 RAG：本地文档问答

第一版 RAG 的本地问答界面、检索/生成代码和最小评测素材，包含 10 篇短文档及 3 篇长文档。简历、面试/答辩讲稿、演示包、视频和评测汇报未纳入。

## 启动

1. 安装 Python 3.12，运行 `pip install -r requirements.txt`。
2. 准备运行目录，包含本地依赖（可选 `.pylibs`）和 `.models/bge-small-zh-v1.5`、`.models/qwen2.5-3b-instruct`。本仓库已包含 `tools/llm_server.py`。
3. 复制 `runtime.local.example.json` 为 `runtime.local.json`，将 `runtime_root` 改成该运行目录的路径。此机本地配置会被 Git 忽略。
4. 双击 `启动演示.cmd`。界面默认打开 `http://127.0.0.1:8024/`。

如果需要重建索引，运行 `python src/ingest.py`。首次构建需要可用的 BGE 模型。问答服务只绑定本机回环地址。

## 目录

- `app.py`、`start.py`、`index.html`：本地问答界面和启动入口。
- `src/`：切块、向量检索、回答生成和阶段评测代码。
- `data/docs/`、`data/docs-long/`：示例知识资料。
- `teach/`：检索评测题库，不含答辩讲稿。
- `data/index/`：运行时生成的索引及结果，不提交。

第一版使用向量距离阈值做资料外拒答。语义相似不等于事实正确，回答应结合召回原文核对。
