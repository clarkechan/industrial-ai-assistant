# _*_ coding: utf-8 _*_
# @Time    : 2026/3/2 10:59
# @Author  : ClarkeChan
# @File    : knowledge_graph
# @Project : industrial-ai-assistant

import networkx as nx

def build_kg():
    G = nx.DiGraph()

    # 缺陷节点
    G.add_node("螺纹孔异物", type="defect", level="A级")
    G.add_node("表面划伤", type="defect", level="B级")
    G.add_node("尺寸偏差", type="defect", level="A级")

    # 工站节点
    G.add_node("CNC-03", type="station", line="Line-2")
    G.add_node("CNC-07", type="station", line="Line-3")
    G.add_node("AOI-01", type="station", line="Line-2")

    # 供应商节点
    G.add_node("供应商A", type="supplier", contact="张经理")
    G.add_node("供应商B", type="supplier", contact="李经理")

    # 人员节点
    G.add_node("张三", type="operator", shift="白班")
    G.add_node("李四", type="operator", shift="夜班")

    # 变更记录节点
    G.add_node("2026-02刀具批次变更", type="change", date="2026-02-15")
    G.add_node("2026-01设备维护", type="change", date="2026-01-20")

    # 关系：缺陷 → 工站
    G.add_edge("螺纹孔异物", "CNC-03", relation="detected_at")
    G.add_edge("螺纹孔异物", "CNC-07", relation="detected_at")
    G.add_edge("表面划伤", "AOI-01", relation="detected_at")
    G.add_edge("尺寸偏差", "CNC-03", relation="detected_at")

    # 关系：工站 → 供应商
    G.add_edge("CNC-03", "供应商A", relation="parts_from")
    G.add_edge("CNC-07", "供应商B", relation="parts_from")
    G.add_edge("AOI-01", "供应商A", relation="parts_from")

    # 关系：工站 → 操作员
    G.add_edge("CNC-03", "张三", relation="operated_by")
    G.add_edge("CNC-03", "李四", relation="operated_by")
    G.add_edge("CNC-07", "张三", relation="operated_by")

    # 关系：供应商 → 变更记录
    G.add_edge("供应商A", "2026-02刀具批次变更", relation="recent_change")
    G.add_edge("CNC-03", "2026-01设备维护", relation="recent_change")

    return G


def trace_defect(G, defect_type):
    """从缺陷出发，沿着关系链追溯根因"""
    if defect_type not in G:
        return f"未找到缺陷类型: {defect_type}"

    results = []
    results.append(f"=== {defect_type} 追溯报告 ===")
    results.append(f"缺陷等级: {G.nodes[defect_type].get('level', '未知')}")

    # 第1层：缺陷 → 工站
    for station in G.successors(defect_type):
        edge_data = G.edges[defect_type, station]
        if edge_data["relation"] == "detected_at":
            results.append(f"\n发生工站: {station} (产线: {G.nodes[station].get('line', '未知')})")

            # 第2层：工站 → 供应商/操作员/变更
            for target in G.successors(station):
                rel = G.edges[station, target]["relation"]
                node_type = G.nodes[target].get("type", "")

                if rel == "parts_from":
                    results.append(f"  供应商: {target} (联系人: {G.nodes[target].get('contact', '未知')})")
                    # 第3层：供应商 → 变更记录
                    for change in G.successors(target):
                        if G.edges[target, change]["relation"] == "recent_change":
                            results.append(f"    ⚠️ 最近变更: {change} (日期: {G.nodes[change].get('date', '未知')})")

                elif rel == "operated_by":
                    results.append(f"  操作员: {target} (班次: {G.nodes[target].get('shift', '未知')})")

                elif rel == "recent_change":
                    results.append(f"  ⚠️ 工站变更: {target} (日期: {G.nodes[change].get('date', '未知')})")

    return "\n".join(results)


# 测试
if __name__ == "__main__":
    G = build_kg()
    print(trace_defect(G, "螺纹孔异物"))