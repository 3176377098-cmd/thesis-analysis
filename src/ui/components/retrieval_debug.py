"""
检索调试面板 - 展示每次检索返回的 chunk 及其分数和来源
"""

import streamlit as st


def render_retrieval_debug(
    retrieved_docs: list[dict],
    key: str = "retrieval_debug",
):
    """
    渲染检索结果调试面板。

    Args:
        retrieved_docs: 检索结果列表 [{chunk_id, text, score, retrieval_method, metadata}]
        key: Streamlit 组件 key
    """
    if not retrieved_docs:
        st.info("暂无检索结果")
        return

    st.markdown(f"**检索结果** ({len(retrieved_docs)} 个文档)")

    for i, doc in enumerate(retrieved_docs):
        score = doc.get("score", 0)
        method = doc.get("retrieval_method", "unknown")
        section = doc.get("metadata", {}).get("section_title", "N/A")

        # 分数颜色
        if score > 0.8:
            score_color = "green"
        elif score > 0.5:
            score_color = "orange"
        else:
            score_color = "red"

        with st.expander(
            f"[{i+1}] :{score_color}[{score:.3f}] {method} | Section: {section}",
            expanded=(i < 3),  # 前 3 个默认展开
        ):
            st.caption(f"Chunk ID: {doc.get('chunk_id', 'N/A')}")
            st.text(doc.get("text", "")[:500])
            if len(doc.get("text", "")) > 500:
                st.caption(f"... ({len(doc['text'])} 字符)")

            # 元数据
            meta = doc.get("metadata", {})
            if meta:
                cols = st.columns(3)
                cols[0].metric("页码", meta.get("page_start", "N/A"))
                cols[1].metric("Token数", meta.get("token_count", "N/A"))
                cols[2].metric("含表格", "是" if meta.get("contains_table") else "否")
