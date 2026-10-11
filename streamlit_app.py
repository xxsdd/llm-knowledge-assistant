import streamlit as st
st.set_page_config(page_title="个人知识库助手", page_icon="📚")
import os
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
    return "\n\n".join(f"来源：{doc.metadata.get('source', '')}\n{doc.page_content}" for doc in docs["context"])

def get_qa_history_chain():
    retriever = get_retriever()
    llm = ZhipuaiLLM(
        model_name=os.getenv("ZHIPUAI_MODEL", "glm-5.3-flash"),
        temperature=0,
        max_tokens=2048,
        timeout=90,
    )
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
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
    st.title('📚 个人知识库助手')
    st.caption('基于 Datawhale 教程资料检索 · GLM-5.3-Flash 回答 · 本地免费中文 Embedding')
    if not os.getenv("ZHIPUAI_API_KEY"):
        st.error("请先在本机 .env 或云端 Secrets 中设置 ZHIPUAI_API_KEY。")
        st.stop()
    if st.sidebar.button("清空本次聊天"):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.rerun()
    st.sidebar.markdown("**示例问题**\n\n- RAG 是什么？\n- 如何写清晰的提示词？\n- 南瓜书中的过拟合是什么意思？")
    st.sidebar.caption("资料来自公开学习教程。回答可能有误，请核对原文。")

    # 用于跟踪对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages:
            with messages.chat_message(message[0]):
                st.write(message[1])
    if prompt := st.chat_input("输入你想问的知识库问题…", max_chars=2000):
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
                with st.spinner("正在检索资料并生成回答…"):
                    output = st.write_stream(answer)
                if not output:
                    st.warning("模型没有返回正文，请缩短问题后重试。")
                else:
                    st.session_state.messages.append(("ai", output))
            except Exception:
                st.error("暂时无法完成回答。请检查智谱模型额度、云端密钥及网络连接后重试。")
                st.session_state.messages.pop()
    if st.session_state.get("last_sources"):
        with st.expander("查看本次检索到的参考资料"):
            for doc in st.session_state.last_sources:
                page = doc.metadata.get("page")
                title = doc.metadata.get("source", "资料") + (f" · 第 {int(page)+1} 页" if page is not None else "")
                st.markdown(f"**{title}**")
                st.write(doc.page_content)


if __name__ == "__main__":
    main()
