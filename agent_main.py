# _*_ coding: utf-8 _*_
# @Time    : 2026/3/2 11:04
# @Author  : ClarkeChan
# @File    : agent_main
# @Project : industrial-ai-assistant

# agent_main.py
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import config
from knowledge_graph import build_kg, trace_defect

# ========== 初始化 ==========
print("初始化LLM...")
llm = ChatOpenAI(
    model=config.model,
    api_key=config.api_key,
    base_url=config.base_url
)

# 初始化RAG
print("初始化RAG...")
loader = DirectoryLoader("data/quality_docs/", glob="*.txt",
                         loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
print(f"RAG就绪: {len(docs)}个文档, {len(chunks)}个文本块")

# 初始化KG
print("初始化KG...")
KG = build_kg()
print(f"KG就绪: {KG.number_of_nodes()}个节点, {KG.number_of_edges()}条边")

# ========== 3个Tool ==========
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
    report_prompt = f"""请将以下分析数据整理成结构化的质量分析报告：

{analysis_data}

报告格式要求：
1. 缺陷概况
2. 根因分析
3. 影响范围
4. 改善建议（具体可执行的措施）
5. 后续跟踪项"""
    response = llm.invoke(report_prompt)
    return response.content

# ========== Agent ==========
tools = [search_quality_standard, trace_defect_source, generate_report]
tool_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

def run_agent(user_input, max_rounds=5):
    messages = [
        SystemMessage(content="""你是博世BCI工业质量分析专家。

【工作原则】
1. 收到缺陷分析请求时，必须同时使用trace_defect_source追溯根因和search_quality_standard查询标准
2. 收集完信息后，使用generate_report生成结构化报告
3. 所有结论必须基于工具返回的数据，不要猜测

【工作流程】
第一步：用trace_defect_source追溯缺陷的工站、供应商、变更记录
第二步：用search_quality_standard查询相关的检测标准和处理流程
第三步：用generate_report综合以上信息生成分析报告"""),
        HumanMessage(content=user_input)
    ]

    print(f"\n{'='*60}")
    print(f"[用户] {user_input}")
    print(f"{'='*60}")

    for i in range(max_rounds):
        print(f"\n--- 第{i+1}轮 ---")
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"\n[最终输出]\n{response.content}")
            return response.content

        for tc in response.tool_calls:
            name = tc["name"]
            args = tc["args"]
            arg_value = list(args.values())[0]
            print(f"[调用工具] {name}({arg_value})")
            result = tool_map[name].invoke(arg_value)
            print(f"[返回] {result[:200]}...")  # 只打印前200字
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    print("[达到最大轮次]")
    return "分析未完成"


# ========== 测试 ==========
print("\n" + "="*60)
print("系统就绪，开始测试")
print("="*60)

run_agent("CNC-03工站最近螺纹孔异物缺陷增多，请帮我分析原因并给出处理建议")