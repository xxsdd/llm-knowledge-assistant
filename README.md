# 个人知识库助手

基于 Datawhale [llm-universe](https://github.com/datawhalechina/llm-universe) C3/C4 的个人学习演示。

- GLM-5.3-Flash 生成回答；密钥通过环境变量或 Streamlit Secrets 读取。
- BAAI/bge-small-zh-v1.5 在 CPU 上执行中文 Embedding，不调用付费 Embedding API。
- 1,004 个教程文本片段及预先计算的 512 维向量，加载到 Chroma 后按余弦相似度检索。
- LangChain / LCEL 连接检索、提示词和流式回答，并显示参考资料。

## 本机运行

使用 Python 3.11，安装 requirements.txt，把 .env.example 复制成 .env 并填写密钥，然后运行：

```bash
streamlit run streamlit_app.py
```

## Streamlit Community Cloud

选择本仓库、main 分支和 streamlit_app.py，Python 版本选择 3.11。在 Advanced settings 的 Secrets 中填写：

```toml
ZHIPUAI_API_KEY = "填入自己的密钥"
ZHIPUAI_MODEL = "glm-5.3-flash"
```

真实 .env 和 Secrets 文件不进入 Git。首次启动需要下载公开的中文向量模型，随后使用缓存。

## 资料与限制

知识库来自 Datawhale llm-universe 自带的 Prompt Engineering 教程与南瓜书，版权与原项目的授权声明归各自作者；本仓库不额外授予第三方资料的权利。具体来源见 ATTRIBUTION.md。本项目用于个人学习演示。

此演示只使用有限资料，回答可能不准确。每次检索 4 个片段，问题最多 2,000 字，最多保留最近 12 条对话，单次生成上限 2,048 tokens。问题会发送给智谱 API，请勿提交敏感信息。清空聊天会清除当前会话记录。
