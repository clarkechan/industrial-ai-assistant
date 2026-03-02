# 🏭 Industrial AI Quality Assistant

> An intelligent quality analysis system for manufacturing, powered by **Agentic AI + RAG + Knowledge Graph**.

[中文版本](#中文说明)

## Architecture

```
User Input (defect description / batch number / question)
    │
    ▼
┌─────────────────────────────────────────────┐
│              AI Agent (ReAct)               │
│         LLM-driven Planning & Routing       │
├─────────────┬──────────────┬────────────────┤
│   Tool 1    │    Tool 2    │    Tool 3      │
│  RAG Search │  KG Tracing  │ Report Gen     │
│             │              │                │
│ Quality     │ Defect →     │ Structured     │
│ Standard    │ Station →    │ Analysis       │
│ Documents   │ Supplier     │ Report         │
└─────────────┴──────────────┴────────────────┘
    │
    ▼
Structured Output (root cause + standard reference + recommendations)
```

## Features

- **Agentic AI**: LLM autonomously decides which tools to use based on the query — no hardcoded if-else logic
- **RAG (Retrieval-Augmented Generation)**: Searches quality standard documents and returns accurate, evidence-based answers
- **Knowledge Graph**: Traces defect root causes through entity relationships (Defect → Station → Supplier → Batch)
- **Hybrid AI**: Combines KG reasoning ("why did this happen?") with RAG retrieval ("what does the standard say?")

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | GLM-4 (ZhipuAI) via OpenAI-compatible API |
| Agent Framework | LangChain + Custom ReAct Agent |
| Vector Database | FAISS |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Knowledge Graph | NetworkX (Python) |
| Frontend | Streamlit |
| Language | Python 3.10+ |

## Project Structure

```
industrial-ai-assistant/
├── README.md
├── config.example.py        # API key configuration template
├── requirements.txt
├── app.py                   # Streamlit entry point
├── rag.py                   # RAG module: document loading → chunking → vectorization → retrieval
├── agent.py                 # Agent module: LLM + tools + planning loop
├── knowledge_graph.py       # KG module: entity-relationship graph + tracing queries
├── data/
│   └── quality_docs/        # Quality standard documents (simulated)
│       ├── 螺纹孔检测标准.txt
│       ├── AOI缺陷分类指南.txt
│       └── 产线异常处理流程.txt
└── screenshots/
    └── demo.png
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/industrial-ai-assistant.git
cd industrial-ai-assistant
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp config.example.py config.py
# Edit config.py and add your API key
```

### 3. Run

```bash
python rag.py          # Test RAG module standalone
python agent.py        # Test Agent module standalone
streamlit run app.py   # Launch full application
```

## How It Works

### Example Query

**Input**: "CNC-03工站SN-2最近螺纹孔异物缺陷增多，帮我分析原因"

**Agent Reasoning Process**:

```
[Agent Planning]  User mentions CNC-03, SN-2, and defect increase.
                  I need to: 1) check defect standard  2) trace station info  3) check product color

[Tool Call]       check_defect_standard("螺纹孔异物")
[Result]          A级: >0.5mm² → NG, B级: 0.1-0.5mm² → 人工复判, 置信度阈值0.85

[Tool Call]       get_station_info("CNC-03")
[Result]          供应商A, 最近变更: 2月换了新批次刀具

[Tool Call]       get_colorinfo("SN-2")
[Result]          本色, 容易有异物

[Final Analysis]  Root cause: Supplier A changed to new batch tooling in Feb.
                  SN-2 products are more susceptible to foreign matter.
                  Recommendation: Isolate batch, notify supplier, submit 8D report.
```

### RAG Pipeline

```
Quality Documents → Text Splitting (500 chars, 100 overlap)
                  → Embedding (all-MiniLM-L6-v2)
                  → FAISS Vector Store
                  → Semantic Retrieval (top-5)
                  → LLM generates answer based on retrieved context
```

### Knowledge Graph

```
[Defect: 螺纹孔异物] --detected_at--> [Station: CNC-03]
[Station: CNC-03]     --parts_from-->  [Supplier: 供应商A]
[Supplier: 供应商A]   --recent_change--> [2月换了新批次刀具]
[Defect: 螺纹孔异物] --method-->       [Algorithm: AI深度学习]
```

## Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Controllable** | Agent actions bounded by defined tool set; max iteration limit prevents infinite loops |
| **Observable** | Every reasoning step and tool call is logged with verbose output |
| **Explainable** | Agent shows its reasoning chain: why it chose each tool and how it reached conclusions |

These three principles — controllable, observable, explainable — are essential for deploying AI agents in industrial environments where reliability and auditability are non-negotiable.

## Background

This project demonstrates how **Agentic AI** can be applied to manufacturing quality management. It was built based on real-world experience developing AOI (Automated Optical Inspection) systems for Apple supply chain projects, where I:

- Designed dual-robot concurrent inspection systems with state machine scheduling
- Integrated AI inference services (Python) into C#/.NET industrial software via REST APIs
- Built traceability platforms connecting defect data across multiple production sites

The transition from rule-based automation to LLM-driven agents represents the next evolution in industrial AI — moving from "hardcoded workflows" to "intelligent, adaptive systems."

## Future Improvements

- [ ] Neo4j for production-grade Knowledge Graph storage
- [ ] Multi-agent coordination (e.g., separate agents for quality, logistics, maintenance)
- [ ] Real-time data integration with MES/WMS systems
- [ ] Model evaluation and drift detection for production deployment
- [ ] Multi-site deployment with centralized agent management

---

<a name="中文说明"></a>
## 中文说明

### 项目简介

这是一个基于 **Agentic AI + RAG + Knowledge Graph** 的工业质量智能分析助手。

系统通过AI Agent自主决策调用工具，结合RAG从质量标准文档中检索信息，利用Knowledge Graph进行缺陷根因追溯，最终生成专业的质量分析报告。

### 核心亮点

- **不是写死的if-else**：Agent根据用户输入动态规划调用哪些工具、什么顺序
- **有据可查**：RAG确保回答基于实际质量标准文档，而非LLM凭空生成
- **链式追溯**：KG支持从缺陷→工站→供应商→批次的完整根因追溯
- **可控可观测可解释**：每一步推理过程都可追踪和审计

### 适用场景

- 产线质量异常分析
- 缺陷根因追溯
- 质量标准查询
- 自动生成质量分析报告

---

## License

MIT

## Author

**Clarke Chan (陈星宇)**
- Blog: [mrcxy.com](https://mrcxy.com)
- Background: 2.5 years of industrial software development for Apple supply chain