"""
Agent Trace 可视化组件 - DAG 流程图 + 数据表格
使用 plotly 绘制 Agent 执行的有向图，展示管道流、条件分支
"""

import streamlit as st
from typing import Optional


# Agent 配色方案
AGENT_COLORS = {
    "retrieve":  "#4ECDC4",  # 青色
    "metadata":  "#45B7D1",  # 蓝
    "deepread":  "#96CEB4",  # 绿
    "critique":  "#FFEAA7",  # 黄
    "synthesis": "#DDA0DD",  # 紫
    "code":      "#F8B500",  # 金
    "rag_qa":    "#7BC8A4",  # 浅绿
}

AGENT_LABELS = {
    "retrieve":  "检索",
    "metadata":  "元数据提取",
    "deepread":  "精读分析",
    "critique":  "批判审查",
    "synthesis": "综述生成",
    "code":      "代码复现",
    "rag_qa":    "RAG 问答",
}


def _get_flow_structure(traces: list[dict]) -> list[dict]:
    """
    分析 trace 列表，构建带分支标记的节点序列。
    检测 revision loop（critique→deepread 回边）和 code branch。
    """
    nodes = []
    for i, t in enumerate(traces):
        name = t.get("agent_name", "unknown")
        nodes.append({
            "agent": name,
            "label": AGENT_LABELS.get(name, name),
            "color": AGENT_COLORS.get(name, "#B0B0B0"),
            "duration_ms": t.get("duration_ms", 0),
            "status": t.get("status", "running"),
            "tool_calls": len(t.get("tool_calls", [])),
            "input_summary": (t.get("input_summary", "") or "")[:100],
            "output_summary": (t.get("output_summary", "") or "")[:100],
            "index": i,
        })
    return nodes


def _build_edges(nodes: list[dict]) -> list[dict]:
    """
    根据节点序列构建边，识别回边(revision loop)和分支。
    回边：deepread → critique → deepread (critique 后面又出现 deepread)
    分支：synthesis → code 是条件分支
    """
    edges = []
    n = len(nodes)
    for i in range(n - 1):
        src = nodes[i]
        dst = nodes[i + 1]
        # 检测回边: critique 后回到 deepread
        is_back = (src["agent"] == "critique" and dst["agent"] == "deepread")
        # 检测条件边: synthesis → code (special branch)
        is_conditional = (src["agent"] == "synthesis" and dst["agent"] == "code")
        edges.append({
            "from": i,
            "to": i + 1,
            "is_back": is_back,
            "is_conditional": is_conditional,
        })
    return edges


def render_trace_dag(
    traces: list[dict],
    title: str = "Agent 执行流程图",
    key: str = "trace_dag",
):
    """
    渲染 Agent 执行的 DAG 流程图。

    特点:
    - 从上到下的管线布局
    - 不同 Agent 类型用不同颜色
    - 节点大小反映执行耗时
    - 回边(revision loop)用弯曲虚线表示
    - 条件分支(code)用特殊样式标记
    - 悬停显示详细信息
    """
    if not traces:
        st.info("暂无追踪数据")
        return

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("需要 plotly 来渲染 Trace 图")
        render_trace_table(traces, key=key + "_table")
        return

    nodes = _get_flow_structure(traces)
    edges = _build_edges(nodes)

    if not nodes:
        return

    # 布局: 从上到下，x 根据 agent type 微调，y 等距排列
    spacing_y = 1.0
    total_height = (len(nodes) - 1) * spacing_y

    # 为回边留出水平偏移
    for i, node in enumerate(nodes):
        node["x"] = 0
        node["y"] = total_height - i * spacing_y

    fig = go.Figure()

    # ---- 先画边 ----
    for e in edges:
        src = nodes[e["from"]]
        dst = nodes[e["to"]]
        x0, y0 = src["x"], src["y"]
        x1, y1 = dst["x"], dst["y"]

        if e["is_back"]:
            # 回边: 红色虚线，从右边绕行
            mid_y = (y0 + y1) / 2
            offset_x = 0.35
            fig.add_trace(go.Scatter(
                x=[x0, x0 + offset_x, x0 + offset_x, x1],
                y=[y0, y0, y1, y1],
                mode="lines",
                line=dict(color="#E74C3C", width=2, dash="dash"),
                hoverinfo="text",
                hovertext="修订循环 (Revision Loop)",
                showlegend=False,
            ))
            # 箭头标记
            fig.add_annotation(
                x=x1, y=y1,
                ax=x0 + offset_x, ay=y1,
                xref="x", yref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowcolor="#E74C3C",
                text="",
            )
        elif e["is_conditional"]:
            # 条件分支: 蓝色虚线
            mid_y = (y0 + y1) / 2
            offset_x = 0.25
            fig.add_trace(go.Scatter(
                x=[x0, x0 + offset_x, x0 + offset_x, x1],
                y=[y0, y0, y1, y1],
                mode="lines",
                line=dict(color="#3498DB", width=2, dash="dot"),
                hoverinfo="text",
                hovertext="条件分支 (含算法触发)",
                showlegend=False,
            ))
            fig.add_annotation(
                x=x1, y=y1,
                ax=x0 + offset_x, ay=y1,
                xref="x", yref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowcolor="#3498DB",
                text="",
            )
        else:
            # 正常边: 灰色实线
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="#888888", width=2),
                hoverinfo="none",
                showlegend=False,
            ))
            # 箭头
            fig.add_annotation(
                x=x1, y=y1,
                ax=x0, ay=y0,
                xref="x", yref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.5,
                arrowcolor="#888888",
                text="",
            )

    # ---- 再画节点 ----
    node_x = [n["x"] for n in nodes]
    node_y = [n["y"] for n in nodes]
    node_colors = [n["color"] for n in nodes]
    node_sizes = [max(25, min(60, (n["duration_ms"] or 100) / 200)) for n in nodes]

    # 节点内的状态标记
    node_symbols = []
    for n in nodes:
        if n["status"] == "completed":
            node_symbols.append("circle")
        elif n["status"] == "error":
            node_symbols.append("x")
        else:
            node_symbols.append("circle-open")

    # 主节点散点
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[n["label"] for n in nodes],
        textposition="middle left",
        textfont=dict(size=13, family="Arial"),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            symbol=node_symbols,
            line=dict(width=3, color="white"),
        ),
        hovertext=[
            f"<b>{n['label']}</b><br>"
            f"状态: {'✅' if n['status'] == 'completed' else '❌'}<br>"
            f"耗时: {n['duration_ms']}ms<br>"
            f"工具调用: {n['tool_calls']}次<br>"
            f"输入: {n['input_summary']}<br>"
            f"输出: {n['output_summary']}"
            for n in nodes
        ],
        hoverinfo="text",
        showlegend=False,
    ))

    # 耗时标签(在节点右侧)
    for n in nodes:
        dur_text = f"{n['duration_ms']}ms" if n['duration_ms'] else "?"
        fig.add_annotation(
            x=n["x"] + 0.15,
            y=n["y"],
            text=dur_text,
            showarrow=False,
            font=dict(size=9, color="#666"),
            xanchor="left",
        )

    # ---- 布局 ----
    y_padding = 0.6
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=80, r=120, t=50),
        xaxis=dict(
            range=[-0.1, 1.0],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
        ),
        yaxis=dict(
            range=[-y_padding, total_height + y_padding],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
        ),
        height=max(300, len(nodes) * 90),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # ---- 图例 ----
    # 手动添加图例项
    for agent_name, color in AGENT_COLORS.items():
        if agent_name in [n["agent"] for n in nodes]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=color),
                name=AGENT_LABELS.get(agent_name, agent_name),
                showlegend=True,
            ))

    # 特殊边的图例
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color="#E74C3C", width=2, dash="dash"),
        name="修订循环",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color="#3498DB", width=2, dash="dot"),
        name="条件分支",
        showlegend=True,
    ))

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.15,
        xanchor="center",
        x=0.5,
        font=dict(size=10),
    ))

    st.plotly_chart(fig, use_container_width=True, key=key)


def render_trace_table(traces: list[dict], key: str = "trace_table"):
    """以表格形式渲染 Agent Trace"""
    if not traces:
        st.info("暂无追踪数据")
        return

    import pandas as pd

    rows = []
    for t in traces:
        agent_name = t.get("agent_name", "")
        rows.append({
            "Agent": AGENT_LABELS.get(agent_name, agent_name),
            "耗时(ms)": t.get("duration_ms", 0),
            "状态": "✅" if t.get("status") == "completed" else "❌",
            "工具调用": len(t.get("tool_calls", [])),
            "输入摘要": (t.get("input_summary", "") or "")[:60],
            "输出摘要": (t.get("output_summary", "") or "")[:60],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, key=key, hide_index=True)
