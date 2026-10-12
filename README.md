# Personal Knowledge Assistant

[Open the live app](https://han-knowledge-assistant.streamlit.app/)

A personal learning demo based on C3/C4 of Datawhale's [llm-universe](https://github.com/datawhalechina/llm-universe). Ask questions about RAG, prompting, and machine learning through an English interface.

- Answers are generated with GLM-5.3-Flash, in English by default. Users can explicitly request another language.
- English questions are translated into Chinese search queries to match the source materials. Follow-up questions are rewritten using recent chat history.
- BAAI/bge-small-zh-v1.5 runs Chinese embeddings on the CPU without a paid embedding API.
- The knowledge base contains 1,004 tutorial excerpts and precomputed 512-dimensional vectors, loaded into Chroma for cosine-similarity retrieval.
- LangChain / LCEL connects retrieval, prompts, and streaming answers. Source excerpts remain in their original language for verification.

## Run locally

Use Python 3.11, install `requirements.txt`, copy `.env.example` to `.env`, and enter your API key. Then run:

```bash
streamlit run streamlit_app.py
```

## Streamlit Community Cloud

Select this repository, the `main` branch, and `streamlit_app.py`. Choose Python 3.11. Configure the following under Advanced settings > Secrets:

```toml
ZHIPUAI_API_KEY = "your_key_here"
ZHIPUAI_MODEL = "glm-5.3-flash"
```

Real `.env` and Secrets files are excluded from Git. The public embedding model is downloaded on first startup and cached afterwards.

## Sources and limitations

The knowledge base uses the Prompt Engineering materials and Pumpkin Book bundled with Datawhale llm-universe. Copyright and licensing remain with their respective authors. This repository does not grant additional rights to third-party materials; see `ATTRIBUTION.md`. This project is a personal learning demo.

The app uses a limited set of sources and may produce inaccurate answers. It retrieves four excerpts per question, accepts up to 2,000 characters, includes the latest 12 chat messages as history, and limits each model response to 2,048 tokens. English queries and follow-up questions require an additional model call before retrieval, which also uses API quota.

Questions and relevant chat history are sent to the Zhipu API. Do not submit sensitive information. Clear chat removes the current session's conversation history. The app does not perform live web searches.
