"""
Paper Analysis System - Streamlit UI
All heavy imports are lazy (inside button handlers) to keep page load fast
"""
import sys
import os
from pathlib import Path

# Ensure project root is on Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import tempfile

st.set_page_config(page_title="Paper Analyzer", layout="wide")

# 检测是否在云端运行（无 Ollama）
IS_CLOUD = os.environ.get("STREAMLIT_CLOUD", "") == "1" or os.environ.get("STREAMLIT_RUNTIME", {}).get("isCloud", False) if isinstance(os.environ.get("STREAMLIT_RUNTIME", {}), dict) else False
# 简易检测：非 Windows 且无 ollama 命令 = 大概率云端
if not IS_CLOUD:
    import shutil
    IS_CLOUD = sys.platform != "win32" and shutil.which("ollama") is None

# ====== SIDEBAR (lightweight, no heavy imports) ======
st.sidebar.title("Paper Analyzer")

# -- Model selector --
st.sidebar.markdown("### Model")
if "provider" not in st.session_state:
    st.session_state.provider = "deepseek" if IS_CLOUD else "ollama"
if "api_key_deepseek" not in st.session_state:
    st.session_state.api_key_deepseek = ""
if "api_key_openai" not in st.session_state:
    st.session_state.api_key_openai = ""

provider_options = ["deepseek", "openai"] if IS_CLOUD else ["ollama", "deepseek", "openai"]
provider_labels = {
    "ollama": "[Local] Ollama (Free)",
    "deepseek": "[Cloud] DeepSeek",
    "openai": "[Cloud] OpenAI",
}
provider = st.sidebar.selectbox(
    "Provider",
    options=provider_options,
    format_func=lambda x: provider_labels[x],
    index=provider_options.index(st.session_state.provider),
)
st.session_state.provider = provider

# -- API Key (cloud mode) --
if provider == "ollama":
    st.sidebar.success("Local mode - no key needed")
    st.sidebar.caption("Requires: ollama pull qwen3:14b")
else:
    label = {"deepseek": "DeepSeek", "openai": "OpenAI"}[provider]
    skey = f"api_key_{provider}"

    key_val = st.sidebar.text_input(
        f"{label} API Key",
        value=st.session_state[skey],
        type="password",
        placeholder="sk-... paste here",
        key=f"key_input_{provider}",
    )
    st.session_state[skey] = key_val

    if key_val:
        masked = key_val[:6] + "..." + key_val[-4:] if len(key_val) > 10 else key_val
        st.sidebar.success(f"Key: {masked}")
    else:
        st.sidebar.warning(f"Enter {label} API Key")
        if provider == "deepseek":
            st.sidebar.caption("Get: platform.deepseek.com")

st.sidebar.markdown("---")

# -- Analysis options --
st.sidebar.markdown("### Analysis")
depth = st.sidebar.selectbox("Depth", ["basic", "advanced", "full"],
    format_func=lambda x: {"basic": "Basic RAG", "advanced": "Multi-Agent", "full": "Full + Code"}[x], index=2)
strategy = st.sidebar.selectbox("Retrieval", ["hybrid", "dense", "hyde", "multi_query"],
    format_func=lambda x: {"hybrid": "Hybrid", "dense": "Vector", "hyde": "HyDE", "multi_query": "Multi-Query"}[x])

st.sidebar.markdown("---")

# -- Status --
if "paper_id" in st.session_state:
    st.sidebar.success(f"Paper: {st.session_state.paper_id[:50]}...")
else:
    st.sidebar.warning("No paper loaded")

# ====== HELPER: get or create pipeline (lazy, inside button handlers) ======
def _get_pipeline():
    """Create or return cached pipeline, rebuild only if provider changed."""
    prov = st.session_state.provider
    skey = f"api_key_{prov}"
    api_key = st.session_state.get(skey, "")

    # Rebuild pipeline if provider changed or not yet created
    rebuild = (
        "pipeline" not in st.session_state
        or st.session_state.get("_pipeline_provider") != prov
    )

    if rebuild:
        from config.settings import get_settings
        s = get_settings()
        s.LLM_PROVIDER = prov

        from config.llm_config import set_ui_credentials
        if prov != "ollama" and api_key:
            set_ui_credentials(prov, api_key)

        from src.pipeline import PaperAnalysisPipeline
        st.session_state.pipeline = PaperAnalysisPipeline()
        st.session_state._pipeline_provider = prov

    return st.session_state.pipeline


def _ingest_file(filepath):
    """Shared ingestion logic - used by all 3 sources"""
    prov = st.session_state.provider
    if prov != "ollama":
        key_val = st.session_state.get(f"api_key_{prov}", "")
        if not key_val:
            st.error(f"Enter {prov} API Key in the sidebar first.")
            return

    with st.spinner("Ingesting... (parsing + formulas + images + embedding)"):
        try:
            p = _get_pipeline()
            pid = p.ingest(str(filepath))
            st.session_state.paper_id = pid
            st.success(f"Ingestion complete! Paper ID: {pid}")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
            import traceback
            st.code(traceback.format_exc())


# ====== REPORT RENDERER ======
import re

_FIELD_LABELS = {
    "executive_summary":     "执行摘要",
    "integrated_analysis":   "整合分析",
    "key_findings":          "核心发现",
    "contradictions_resolved": "矛盾辨析",
    "practical_implications":"实践意义",
    "limitations_acknowledged":"研究局限",
    "future_work_suggested": "未来方向",
    "future_work":           "未来方向",
    "citation_map":          "引用映射",
    "content":               None,  # not a heading
}

def _clean_field_labels(text: str) -> str:
    """Replace bare snake_case field-name lines with Chinese markdown headings."""
    if not text:
        return text
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    for key, title in _FIELD_LABELS.items():
        if title is None:
            continue
        # Match all 4 variants the LLM produces on its own line
        for variant in [
            f'### **{key}**\n',     # H3 + bold (most common)
            f'### **{key}**\n\n',   # H3 + bold + blank line
            f'### {key}\n',         # H3 only
            f'**{key}**\n',         # bold only
            f'{key}\n',             # bare label
        ]:
            if variant in text:
                text = text.replace(variant, f'### {title}\n')
    return text


def _render_report(rep: dict):
    """Render synthesis result as a structured academic report."""
    from loguru import logger
    logger.warning(f"RENDERER: rep keys={list(rep.keys())}, has_content={'content' in rep}")
    if "content" in rep:
        text = str(rep.get("content", ""))
        logger.warning(f"RENDERER: content[:150]={repr(text[:150])}")
    if not rep:
        st.info("(empty response)")
        return

    if "content" in rep:
        text = str(rep.get("content", ""))
        text = _clean_field_labels(text)
        logger.warning(f"RENDERER: after clean[:150]={repr(text[:150])}")
        st.markdown(text)
        return

    rendered = set()
    found_any = False

    has_structured = any(k in rep for k in _FIELD_LABELS if k != "content")
    if not has_structured:
        st.markdown(str(rep))
        return

    for key, title in _FIELD_LABELS.items():
        if title is None or key in rendered:
            continue
        val = rep.get(key)
        if not val:
            continue
        rendered.add(key)
        found_any = True
        st.markdown(f"### {title}")
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    st.json(item)
                else:
                    st.markdown(f"- {item}")
        elif isinstance(val, str):
            st.markdown(_clean_field_labels(val.strip()))
        else:
            st.markdown(str(val))

    for k, v in rep.items():
        if k in rendered or k == "content":
            continue
        found_any = True
        st.markdown(f"### {k}")
        if isinstance(v, list):
            for item in v:
                st.markdown(f"- {item}")
        else:
            st.markdown(str(v))

    if not found_any:
        st.markdown(str(rep))

# ====== MAIN CONTENT ======
st.title("Academic Paper Analysis")

tab1, tab2, tab3 = st.tabs(["1. Upload", "2. Analysis", "3. Evaluation"])

# ---- TAB 1: Upload ----
with tab1:
    st.subheader("Get a Paper")

    # Source selector
    source = st.radio(
        "Source",
        options=["upload", "arxiv", "semantic_scholar"],
        format_func=lambda x: {
            "upload": "Local Upload (PDF from your computer)",
            "arxiv": "arXiv Search (free, 2M+ CS papers)",
            "semantic_scholar": "Semantic Scholar Search (free, 200M+ papers)",
        }[x],
        horizontal=True,
    )

    # ===== LOCAL UPLOAD =====
    if source == "upload":
        f = st.file_uploader("Choose a PDF file", type=["pdf"])
        if f:
            tmp = Path(tempfile.gettempdir()) / "paper_analyzer" / f.name
            tmp.parent.mkdir(exist_ok=True)
            tmp.write_bytes(f.read())
            st.info(f"Ready: {f.name} ({f.size/1024:.0f} KB)")

            if st.button("Start Ingestion", type="primary"):
                _ingest_file(tmp)

    # ===== ARXIV SEARCH =====
    elif source == "arxiv":
        col1, col2 = st.columns([3, 1])
        with col1:
            arxiv_query = st.text_input("Search arXiv", placeholder="e.g. transformer attention mechanism, bearing fault diagnosis...")
        with col2:
            arxiv_limit = st.number_input("Max results", 1, 50, 10)
            st.write("")

        if arxiv_query and st.button("Search arXiv", type="primary"):
            with st.spinner(f"Searching arXiv: {arxiv_query}..."):
                try:
                    from src.tools.search_tools import ArxivSearchTool
                    arxiv = ArxivSearchTool()
                    papers = arxiv.search(arxiv_query, max_results=arxiv_limit)
                    st.session_state.arxiv_results = papers
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.caption("arXiv API may be temporarily unavailable. Try again in a few seconds.")

        if "arxiv_results" in st.session_state and st.session_state.arxiv_results:
            papers = st.session_state.arxiv_results
            st.success(f"Found {len(papers)} papers")

            for i, paper in enumerate(papers):
                with st.container():
                    cols = st.columns([6, 1])
                    cols[0].markdown(f"**{paper.title}**")
                    cols[0].caption(f"{', '.join(paper.authors[:3])} | {paper.year} | arXiv:{paper.arxiv_id}")
                    if paper.abstract:
                        cols[0].caption(paper.abstract[:300] + ("..." if len(paper.abstract) > 300 else ""))
                    if cols[1].button("Import", key=f"arxiv_{i}"):
                        with st.spinner(f"Downloading + ingesting {paper.arxiv_id}..."):
                            try:
                                from src.sources.arxiv_connector import ArxivConnector
                                conn = ArxivConnector()
                                tmp_dir = Path(tempfile.gettempdir()) / "paper_analyzer"
                                tmp_dir.mkdir(exist_ok=True)
                                pdf_path = conn.download(paper.arxiv_id, tmp_dir)
                                _ingest_file(pdf_path)
                            except Exception as e:
                                st.error(f"Download failed: {e}")
                    st.divider()

    # ===== SEMANTIC SCHOLAR SEARCH =====
    elif source == "semantic_scholar":
        col1, col2 = st.columns([3, 1])
        with col1:
            s2_query = st.text_input("Search Semantic Scholar", placeholder="e.g. deep learning fault diagnosis, NLP transformer...")
        with col2:
            s2_limit = st.number_input("Max results", 1, 50, 10)
            st.write("")

        if s2_query and st.button("Search Semantic Scholar", type="primary"):
            with st.spinner(f"Searching: {s2_query}..."):
                try:
                    from src.tools.search_tools import SemanticScholarSearchTool
                    s2 = SemanticScholarSearchTool()
                    papers = s2.search(s2_query, limit=s2_limit)
                    st.session_state.s2_results = papers
                except Exception as e:
                    st.error(f"Search failed: {e}")

        if "s2_results" in st.session_state and st.session_state.s2_results:
            papers = st.session_state.s2_results
            st.success(f"Found {len(papers)} papers")

            for i, paper in enumerate(papers):
                with st.container():
                    cols = st.columns([6, 1])
                    cols[0].markdown(f"**{paper.title}**")
                    cols[0].caption(f"{', '.join(paper.authors[:3])} | {paper.year} | {paper.venue}")
                    if paper.abstract:
                        cols[0].caption(paper.abstract[:300] + ("..." if len(paper.abstract) > 300 else ""))
                    source_id = paper.arxiv_id or paper.doi
                    if source_id and cols[1].button("Import", key=f"s2_{i}"):
                        with st.spinner(f"Downloading {source_id}..."):
                            try:
                                if paper.arxiv_id:
                                    from src.sources.arxiv_connector import ArxivConnector
                                    conn = ArxivConnector()
                                else:
                                    from src.sources.semantic_scholar import SemanticScholarConnector
                                    conn = SemanticScholarConnector()
                                tmp_dir = Path(tempfile.gettempdir()) / "paper_analyzer"
                                tmp_dir.mkdir(exist_ok=True)
                                pdf_path = conn.download(source_id, tmp_dir)
                                _ingest_file(pdf_path)
                            except Exception as e:
                                st.error(f"Download failed: {e}")
                    st.divider()

# ---- TAB 2: Analysis ----
with tab2:
    if "paper_id" not in st.session_state:
        st.info("Upload and ingest a paper in Tab 1 first.")
    else:
        st.subheader(f"Ask about: {st.session_state.paper_id[:60]}")
        q = st.chat_input("Your question about the paper...")
        if q:
            with st.spinner(f"Analyzing ({depth} mode)..."):
                try:
                    p = _get_pipeline()
                    r = p.analyze(st.session_state.paper_id, q, depth, strategy)
                    rep = r.get("final_report", {})
                    if isinstance(rep, dict):
                        _render_report(rep)
                    else:
                        st.markdown(str(rep))

                    # ---- Code Reproduction ----
                    code_rep = r.get("code_reproduction", {})
                    if code_rep and code_rep.get("algorithm_identified"):
                        st.markdown("---")
                        st.subheader("💻 代码复现")
                        algo_name = code_rep.get("algorithm_name", "Algorithm")
                        st.markdown(f"**算法**: {algo_name}")

                        python_code = code_rep.get("python_code", "")
                        if python_code:
                            st.code(python_code, language="python")

                        deps = code_rep.get("dependencies", [])
                        if deps:
                            st.caption(f"依赖: {', '.join(deps)}")

                        # Execution result
                        exec_result = code_rep.get("execution_result", {})
                        if exec_result:
                            st.markdown("#### 执行结果")
                            col1, col2, col3 = st.columns(3)
                            status_text = "✅ 成功" if exec_result.get("success") else "❌ 失败"
                            col1.metric("状态", status_text)
                            col2.metric("耗时", f"{exec_result.get('elapsed_ms', 0)}ms")
                            col3.metric("Exit Code", exec_result.get("exit_code", -1))

                            stdout = exec_result.get("stdout", "")
                            stderr = exec_result.get("stderr", "")
                            error_msg = exec_result.get("error_message", "")

                            if stdout:
                                st.text_area("stdout", stdout, height=150, key="code_stdout")
                            if stderr:
                                with st.expander("stderr"):
                                    st.code(stderr)
                            if error_msg:
                                st.error(error_msg)

                        # Original pseudocode
                        pseudocode = code_rep.get("original_pseudocode", "")
                        if pseudocode:
                            with st.expander("原始伪代码/算法描述"):
                                st.text(pseudocode)

                        notes = code_rep.get("validation_notes", "")
                        if notes:
                            st.caption(f"验证说明: {notes}")

                    # ---- Agent Trace ----
                    traces = r.get("agent_traces", [])
                    if traces:
                        st.markdown("---")
                        st.subheader("Agent Trace")
                        from src.ui.components.trace_viewer import render_trace_dag, render_trace_table

                        # DAG 流程图
                        render_trace_dag(traces, key="trace_dag")

                        # 详细表格
                        with st.expander("详细信息 (表格 + 单步展开)"):
                            render_trace_table(traces, key="trace_table")

                            for t in traces:
                                name = t.get("agent_name", "?")
                                dur = t.get("duration_ms", 0)
                                status_icon = "✅" if t.get("status") == "completed" else "❌"
                                with st.expander(f"{status_icon} {name} ({dur}ms)"):
                                    st.caption(f"Input: {t.get('input_summary', 'N/A')[:200]}")
                                    st.caption(f"Output: {t.get('output_summary', 'N/A')[:300]}")
                                    tc = t.get("tool_calls", [])
                                    if tc:
                                        st.caption(f"Tool calls: {len(tc)}")
                    else:
                        st.caption("No agent trace recorded (basic mode)")

                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# ---- TAB 3: Evaluation ----
with tab3:
    st.subheader("Retrieval Quality Evaluation")

    if "paper_id" not in st.session_state:
        st.info("Upload and ingest a paper first (Tab 1), then run evaluation here.")
    else:
        st.markdown(f"Paper: `{st.session_state.paper_id[:60]}`")

        col1, col2 = st.columns([1, 3])
        with col1:
            num_queries = st.number_input("Test queries", min_value=5, max_value=50, value=10)
        with col2:
            if st.button("Run Benchmark", type="primary"):
                with st.spinner("Generating test queries + evaluating retrieval..."):
                    try:
                        p = _get_pipeline()

                        # Generate test queries from paper chunks
                        from src.evaluation.test_queries import QueryGenerator
                        from src.processing.chunker import Chunk, ChunkMetadata

                        # Get chunks from chroma
                        chunk_ids = p.chroma_store.get_paper_chunk_ids(st.session_state.paper_id)
                        if not chunk_ids:
                            st.error("No chunks found. Re-ingest the paper first.")
                        else:
                            # Build simple chunk objects for query generation
                            # Query from the BM25 index
                            bm25_idx = p.bm25_index._indices.get(st.session_state.paper_id)
                            if bm25_idx:
                                chunks_for_gen = bm25_idx.chunks
                            else:
                                st.error("Paper index not found. Re-ingest the paper.")
                                st.stop()

                            gen = QueryGenerator(seed=42)
                            test_queries = gen.generate(chunks_for_gen, num_queries=num_queries)
                            st.success(f"Generated {len(test_queries)} test queries")

                            # Evaluate each retrieval strategy
                            strategies = ["hybrid", "dense", "sparse"]
                            from src.evaluation.metrics import RetrievalMetrics
                            metrics = RetrievalMetrics()

                            results_by_strategy = {}
                            for strat in strategies:
                                all_retrieved = []
                                for tq in test_queries:
                                    docs = p.retrieval.retrieve(
                                        query=tq["query"],
                                        paper_id=st.session_state.paper_id,
                                        strategy=strat,
                                        k=20,
                                    )
                                    all_retrieved.append([doc.chunk_id for doc in docs])

                                relevant = [tq["relevant_chunk_ids"] for tq in test_queries]
                                result = metrics.evaluate(
                                    [tq["query"] for tq in test_queries],
                                    relevant,
                                    all_retrieved,
                                )
                                results_by_strategy[strat] = result

                            # Display results
                            st.markdown("---")
                            st.subheader("Results by Strategy")

                            # Metrics cards
                            for strat in strategies:
                                r = results_by_strategy[strat]
                                st.markdown(f"#### {strat.upper()}")
                                c1, c2, c3, c4, c5 = st.columns(5)
                                c1.metric("Hit Rate@1", f"{r.hit_rate_k.get(1, 0):.2f}")
                                c2.metric("Hit Rate@5", f"{r.hit_rate_k.get(5, 0):.2f}")
                                c3.metric("Hit Rate@10", f"{r.hit_rate_k.get(10, 0):.2f}")
                                c4.metric("MRR", f"{r.mrr:.3f}")
                                c5.metric("NDCG@10", f"{r.ndcg_k.get(10, 0):.3f}")

                                # Bar chart comparison
                                import pandas as pd
                                chart_data = pd.DataFrame({
                                    "Metric": ["HR@1", "HR@5", "HR@10", "MRR", "NDCG@10"],
                                    "Score": [
                                        r.hit_rate_k.get(1, 0),
                                        r.hit_rate_k.get(5, 0),
                                        r.hit_rate_k.get(10, 0),
                                        r.mrr,
                                        r.ndcg_k.get(10, 0),
                                    ],
                                })
                                st.bar_chart(chart_data, x="Metric", y="Score", height=200)

                            # Strategy comparison table
                            st.markdown("---")
                            st.subheader("Strategy Comparison")
                            comp_data = {
                                "Metric": ["Hit Rate@5", "HR@10", "MRR", "NDCG@10"],
                            }
                            for strat in strategies:
                                r = results_by_strategy[strat]
                                comp_data[strat] = [
                                    f"{r.hit_rate_k.get(5, 0):.3f}",
                                    f"{r.hit_rate_k.get(10, 0):.3f}",
                                    f"{r.mrr:.3f}",
                                    f"{r.ndcg_k.get(10, 0):.3f}",
                                ]
                            st.dataframe(pd.DataFrame(comp_data), hide_index=True)

                            # Test queries preview
                            with st.expander("Test Queries (generated)"):
                                for i, tq in enumerate(test_queries[:5]):
                                    st.caption(f"Q{i+1}: {tq['query']}")
                                    st.caption(f"  Section: {tq.get('section', 'N/A')} | "
                                              f"Relevant chunks: {len(tq['relevant_chunk_ids'])}")

                            # Store results for display
                            st.session_state.eval_results = results_by_strategy
                            st.session_state.eval_queries = test_queries

                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        # Show previous results if available
        if "eval_results" in st.session_state:
            st.markdown("---")
            st.caption("Last evaluation results are shown above. Run again to refresh.")
