# _*_ coding: utf-8 _*_
# @Time    : 2026/3/2 14:28
# @Author  : ClarkeChan
# @File    : app
# @Project : industrial-ai-assistant

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import config
from knowledge_graph import build_kg, trace_defect

# ========== 初始化（缓存，只跑一次）==========
@st.cache_resource
def init_system():
    llm = ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url
    )

    # RAG
    loader = DirectoryLoader("data/quality_docs/", glob="*.txt",
                             loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # KG
    kg = build_kg()

    return llm, retriever, kg


llm, retriever, KG = init_system()

# ========== Tools ==========
@tool
def search_quality_standard(query: str) -> str:
    """从质量标准文档库中检索相关信息。当需要查询检测标准、判定依据、处理流程时调用。"""
    docs = retriever.invoke(query)
    if not docs:
        return "未找到相关标准文档"
    return "\n\n".join([d.page_content for d in docs])

@tool
def trace_defect_source(defect_type: str) -> str:
    """根据缺陷类型追溯到工站、供应商和变更记录。当需要分析缺陷原因、追溯根因时调用。参数为缺陷类型名称，如：螺纹孔异物、表面划伤、尺寸偏差"""
    return trace_defect(KG, defect_type)

@tool
def generate_report(analysis_data: str) -> str:
    """将分析数据整理成结构化质量分析报告。当已收集足够信息需要生成最终报告时调用。"""
    response = llm.invoke(
        f"请将以下数据整理成结构化质量报告（含缺陷概况、根因分析、改善建议）：\n{analysis_data}"
    )
    return response.content

tools = [search_quality_standard, trace_defect_source, generate_report]
tool_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

# ========== Agent ==========
def run_agent(user_input, max_rounds=5):
    messages = [
        SystemMessage(content="""你是工业质量分析专家。
收到缺陷分析请求时：
1. 用trace_defect_source追溯根因
2. 用search_quality_standard查询标准
3. 用generate_report生成报告
所有结论必须基于工具返回的数据。"""),
        HumanMessage(content=user_input)
    ]

    steps = []

    for i in range(max_rounds):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content, steps

        for tc in response.tool_calls:
            name = tc["name"]
            args = tc["args"]
            arg_value = list(args.values())[0]
            steps.append({"tool": name, "input": arg_value})
            result = tool_map[name].invoke(arg_value)
            steps[-1]["output"] = str(result)[:500]
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return "分析未完成（达到最大轮次）", steps

# ========== UI ==========
st.set_page_config(page_title="工业AI质量助手", page_icon="🏭", layout="wide")

st.title("🏭 Industrial AI Quality Assistant")
st.caption("Powered by Agentic AI + RAG + Knowledge Graph")

# 侧边栏
with st.sidebar:
    st.header("系统信息")
    st.success(f"✅ KG: {KG.number_of_nodes()} 节点, {KG.number_of_edges()} 边")
    st.success(f"✅ RAG: 文档库已加载")
    st.success(f"✅ LLM: {config.model}")

    st.divider()
    st.header("示例问题")
    examples = [
        "CNC-03工站最近螺纹孔异物缺陷增多，分析原因",
        "螺纹孔异物怎么判定？",
        "AI检测系统故障了怎么处理？",
        "表面划伤的检测标准是什么？",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["input"] = ex

# 输入框
user_input = st.text_input(
    "请输入您的质量问题：",
    value=st.session_state.get("input", ""),
    placeholder="例如：CNC-03工站螺纹孔异物增多，分析原因"
)

if st.button("开始分析", type="primary", use_container_width=True) and user_input:
    with st.spinner("Agent 分析中..."):
        answer, steps = run_agent(user_input)

    # 显示Agent推理过程
    if steps:
        st.subheader("Agent 推理过程")
        for i, step in enumerate(steps):
            with st.expander(f"Step {i+1}: {step['tool']}({step['input']})", expanded=False):
                st.code(step.get("output", ""), language="text")

    # 显示最终结果
    st.subheader("分析结果")
    st.markdown(answer)