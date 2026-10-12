import streamlit as st
st.set_page_config(page_title="Personal Knowledge Assistant", page_icon="📚")
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=True)
if not os.getenv("ZHIPUAI_API_KEY"):
    try:
        for name in ("ZHIPUAI_API_KEY", "ZHIPUAI_MODEL"):
            if name in st.secrets:
                os.environ[name] = str(st.secrets[name])
    except FileNotFoundError:
        pass
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zhipuai_llm import ZhipuaiLLM
from local_retriever import get_local_retriever

def get_retriever():
    # 本地中文 Embedding 不消耗智谱 API 额度；C3 原有 Chroma 库保留。
    return get_local_retriever()

def combine_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('source', '')}\n{doc.page_content}" for doc in docs["context"])

def get_qa_history_chain():
    retriever = get_retriever()
    llm = ZhipuaiLLM(
        model_name=os.getenv("ZHIPUAI_MODEL", "glm-5.3-flash"),
        temperature=0,
        max_tokens=2048,
        timeout=90,
    )
    condense_question_system_template = (
        "Rewrite the user's latest question as a standalone search query. "
        "Use the chat history only to resolve references and preserve the user's meaning. "
        "The knowledge base contains Chinese learning materials, so write the search query "
        "in Simplified Chinese. Preserve technical names, acronyms, numbers, and conditions. "
        "Return only the search query, without answering it or adding explanations."
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    retrieve_docs = RunnableBranch(
        (
            lambda x: not x.get("chat_history")
            and re.search(r"[\u3400-\u9fff]", x["input"]) is not None,
            (lambda x: x["input"]) | retriever,
        ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "You are a knowledge-base question-answering assistant. "
        "Use the retrieved source excerpts below to answer the user's question. "
        "The excerpts may be in Chinese; explain their relevant content in English. "
        "Respond in English by default, unless the user explicitly requests another language. "
        "If you do not know the answer, say so rather than inventing information. "
        "Keep answers clear and concise, using paragraphs or bullets when helpful."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs, 
        ).assign(answer=qa_chain)
    return qa_history_chain

def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "context" in res:
            st.session_state.last_sources = res["context"]
        if "answer" in res.keys():
            yield res["answer"]

# Streamlit 应用程序界面
def main():
    st.title('📚 Personal Knowledge Assistant')
    st.caption('Explore RAG, prompting, and machine learning with answers based on Datawhale learning resources.')
    if not os.getenv("ZHIPUAI_API_KEY"):
        st.error("Set ZHIPUAI_API_KEY in your local .env file or Streamlit Secrets before using the assistant.")
        st.stop()
    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()
    st.sidebar.markdown("**Example questions**\n\n- What is RAG?\n- How can I write clear prompts?\n- How does the Pumpkin Book explain overfitting?")
    st.sidebar.caption("Answers may contain errors. Check the source excerpts and do not submit sensitive information.")

    # 用于跟踪对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state or st.session_state.get("qa_chain_language") != "en":
        st.session_state.qa_history_chain = get_qa_history_chain()
        st.session_state.qa_chain_language = "en"
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages:
            with messages.chat_message(message[0]):
                st.write(message[1])
    if prompt := st.chat_input("Ask a question about the knowledge base…", max_chars=2000):
        previous_history = st.session_state.messages[-12:]
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        with messages.chat_message("human"):
            st.write(prompt)

        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=previous_history
        )
        with messages.chat_message("ai"):
            try:
                with st.spinner("Searching the knowledge base and preparing your answer…"):
                    output = st.write_stream(answer)
                if not output:
                    st.warning("The model returned no answer. Try a shorter question.")
                else:
                    st.session_state.messages.append(("ai", output))
            except Exception as exc:
                # 调试时仅在服务端记录经过密钥脱敏的错误，网页不展示内部信息。
                safe_error = str(exc).replace(os.getenv("ZHIPUAI_API_KEY", ""), "[redacted]")
                print(f"Knowledge assistant error: {type(exc).__name__}: {safe_error}", flush=True)
                st.error("Unable to complete your request. Check the model quota, API key configuration, and network connection, then try again.")
                st.session_state.messages.pop()
    if st.session_state.get("last_sources"):
        with st.expander("View source excerpts"):
            st.caption("Excerpts are shown in their original language, which may be Chinese.")
            for doc in st.session_state.last_sources:
                page = doc.metadata.get("page")
                title = doc.metadata.get("source", "Source") + (f" · Page {int(page)+1}" if page is not None else "")
                st.markdown(f"**{title}**")
                st.write(doc.page_content)


if __name__ == "__main__":
    main()
