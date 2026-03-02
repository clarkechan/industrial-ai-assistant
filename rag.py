# _*_ coding: utf-8 _*_
# @Time    : 2026/3/1 15:09
# @Author  : ClarkeChan
# @File    : rag
# @Project : industrial-ai-assistant

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import config

llm = ChatOpenAI(
    model=config.model,
    api_key=config.api_key,
    base_url=config.base_url
)

# 2. 加载文档
print("加载文档...")
loader = DirectoryLoader(
    "data/quality_docs/",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()
print(f"加载了 {len(docs)} 个文档")

# 3. 文本分割
print("分割文档...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,     # 每段300字
    chunk_overlap=50    # 重叠50字，防止上下文断裂
)
chunks = splitter.split_documents(docs)
print(f"分割成 {len(chunks)} 个文本块")

# 4. 向量化 + 存入ChromaDB
print("向量化存储中...（首次需要下载模型，稍等）")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 每次检索3段
print("向量数据库就绪")

# 5. RAG Chain
prompt = ChatPromptTemplate.from_template("""你是工业质量专家。根据以下参考文档回答问题。
如果文档中没有相关信息，请说明。

参考文档：
{context}

用户问题：{question}

请给出专业、准确的回答：""")


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 6. 测试
print("\n" + "=" * 50)
questions = [
    "螺纹孔异物怎么判定？",
    "SN-2产品检测要注意什么？",
    "AI检测系统故障了怎么办？",
]

for q in questions:
    print(f"\n[问题] {q}")
    answer = rag_chain.invoke(q)
    print(f"[回答] {answer.content}")
    print("-" * 50)