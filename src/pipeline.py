"""
核心分析管线 - 串联所有模块的端到端流程
PDF -> 解析(文本+公式+图片) -> 分块 -> 嵌入 -> 检索 -> Multi-Agent -> 报告
"""

import uuid
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import get_settings
from config.llm_config import create_chat_model, create_fast_model, create_deep_model

from src.parsing import ParserFactory, TableExtractor
from src.parsing.base import ParsedDocument
from src.processing import (
    TextPreprocessor,
    MetadataExtractor,
    AcademicChunker,
    FormulaDetector,
    ImageExtractor,
    VisionAnalyzer,
)
from src.indexing.embedder import Embedder
from src.indexing.chroma_store import ChromaStore
from src.indexing.bm25_index import BM25Index
from src.indexing.retrieval import RetrievalOrchestrator

from src.agents.state import PaperAnalysisState
from src.agents.nodes import (
    MetadataAgent,
    DeepReadAgent,
    CritiqueAgent,
    SynthesisAgent,
    CodeReproductionAgent,
)
from src.graph import build_analysis_graph
from src.graph.tools import create_retrieval_tool, create_python_repl_tool
from src.evaluation.tracer import TraceCollector


class PaperAnalysisPipeline:
    """
    学术论文分析全流程管线。

    用法:
        pipeline = PaperAnalysisPipeline()
        paper_id = pipeline.ingest("paper.pdf")
        report = pipeline.analyze(paper_id, "What is the main contribution?")
    """

    def __init__(self):
        settings = get_settings()

        # ---- LLM (lazy init - only create when needed) ----
        self._llm = None
        self._llm_fast = None
        self._llm_deep = None
        self._vision_llm = None

        # ---- PDF 解析 ----
        self.parser = ParserFactory.create(preference=settings.PARSER_PREFERENCE)
        self.table_extractor = TableExtractor()
        self.preprocessor = TextPreprocessor()
        self.formula_detector = FormulaDetector()
        self.image_extractor = ImageExtractor()
        self.vision_analyzer = VisionAnalyzer(llm=self.vision_llm)

        # ---- 分块 & 嵌入 ----
        self.metadata_extractor = MetadataExtractor(self.llm_fast)
        self.chunker = AcademicChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self.embedder = Embedder(
            model_name=settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
        )

        # ---- 向量存储 ----
        self.chroma_store = ChromaStore(persist_dir=settings.CHROMA_DIR)
        self.bm25_index = BM25Index()

        # ---- 检索 ----
        self.retrieval = RetrievalOrchestrator(
            chroma_store=self.chroma_store,
            bm25_index=self.bm25_index,
            embedder=self.embedder,
            llm=self.llm,
        )

        # ---- Agent ----
        self.metadata_agent = MetadataAgent(self.llm_fast)
        retrieval_tool = None  # Will be bound per-paper
        self.deepread_agent = DeepReadAgent(self.llm, tools=None)
        self.critique_agent = CritiqueAgent(self.llm_deep)
        self.synthesis_agent = SynthesisAgent(self.llm)
        self.code_agent = CodeReproductionAgent(
            self.llm,
            tools=[create_python_repl_tool()],
        )

        # ---- 追踪 ----
        self.trace_collector = TraceCollector(settings.TRACES_DIR)

    # =================================================================
    #  Lazy LLM properties (avoid blocking init)
    # =================================================================

    @property
    def llm(self):
        if self._llm is None:
            self._llm = create_chat_model()
        return self._llm

    @property
    def llm_fast(self):
        if self._llm_fast is None:
            self._llm_fast = create_fast_model()
        return self._llm_fast

    @property
    def llm_deep(self):
        if self._llm_deep is None:
            self._llm_deep = create_deep_model()
        return self._llm_deep

    @property
    def vision_llm(self):
        if self._vision_llm is None:
            self._vision_llm = create_chat_model()
        return self._vision_llm

    # =================================================================
    #  Phase 1: Ingestion (PDF -> Vectors)
    # =================================================================

    def ingest(self, pdf_path: str | Path) -> str:
        """
        完整导入流程: PDF -> 文本+公式+图片 -> 分块 -> 嵌入 -> 存储。

        Returns:
            paper_id
        """
        pdf_path = Path(pdf_path)
        paper_id = pdf_path.stem
        t0 = time.time()

        logger.info(f"{'='*60}")
        logger.info(f"Ingesting: {paper_id}")
        logger.info(f"{'='*60}")

        # --- Step 1: Parse PDF ---
        logger.info("[1/7] Parsing PDF...")
        parsed = self.parser.parse(pdf_path)
        logger.info(f"  -> {parsed.page_count} pages, {len(parsed.text)} chars")

        # --- Step 2: Extract tables ---
        logger.info("[2/7] Extracting tables...")
        tables = self.table_extractor.extract(pdf_path)
        parsed.tables = tables
        logger.info(f"  -> {len(tables)} tables found")

        # --- Step 3: Extract & analyze images ---
        logger.info("[3/7] Extracting and analyzing images/figures...")
        image_descriptions = []
        try:
            images = self.image_extractor.extract(pdf_path)
            for img in images:
                try:
                    desc = self.vision_analyzer.analyze(img)
                    image_descriptions.append({
                        "page": img.page_number,
                        "caption": img.caption,
                        "description": desc,
                        "width": img.width,
                        "height": img.height,
                    })
                except Exception as e:
                    logger.warning(f"Vision analysis failed for image on page {img.page_number}: {e}")
                    image_descriptions.append({
                        "page": img.page_number,
                        "caption": img.caption,
                        "description": f"[Figure on page {img.page_number}, {img.width}x{img.height}px. Caption: {img.caption}]",
                    })
            logger.info(f"  -> {len(images)} images analyzed ({len(image_descriptions)} with descriptions)")
        except Exception as e:
            logger.warning(f"Image extraction failed (non-critical): {e}")

        # --- Step 4: Preprocess text ---
        logger.info("[4/7] Preprocessing text...")
        parsed = self.preprocessor.preprocess(parsed)

        # --- Step 5: Detect formulas ---
        logger.info("[5/7] Detecting formulas...")
        formulas = self.formula_detector.detect(parsed.text)
        protected_text, formula_map = self.formula_detector.protect_formulas(parsed.text)
        parsed.text = protected_text  # Use protected text for chunking
        logger.info(f"  -> {len(formulas)} formulas detected")

        # --- Step 6: Extract metadata ---
        logger.info("[6/7] Extracting metadata...")
        metadata = self.metadata_extractor.extract(parsed.text)
        logger.info(f"  -> Title: {metadata.get('title', 'N/A')[:80]}")

        # --- Step 7: Chunk + Embed + Store ---
        logger.info("[7/7] Chunking, embedding, and storing...")
        chunks = self.chunker.chunk(parsed, paper_id=paper_id)

        # Enrich chunks: add image descriptions to relevant chunks
        chunks = self._enrich_chunks_with_images(chunks, image_descriptions)

        # Embed and store
        texts = [c.text for c in chunks]
        self.embedder.fit(texts)  # TF-IDF needs fit first
        embeddings = self.embedder.embed(texts)
        self.chroma_store.add_chunks(paper_id, chunks, embeddings)
        self.bm25_index.build(paper_id, chunks)

        elapsed = time.time() - t0
        logger.info(f"Ingestion complete: {paper_id} ({len(chunks)} chunks, {elapsed:.1f}s)")
        return paper_id

    def _enrich_chunks_with_images(self, chunks, image_descriptions):
        """将图片分析结果注入到对应的 chunk 中"""
        if not image_descriptions:
            return chunks

        for img_desc in image_descriptions:
            page = img_desc["page"]
            text = (
                f"\n\n[FIGURE - Page {page}]\n"
                f"Caption: {img_desc['caption']}\n"
                f"Analysis: {img_desc['description']}\n"
            )
            # Find chunks from the same page and append
            for chunk in chunks:
                if chunk.metadata.page_range[0] == page:
                    chunk.text += text
                    break
            else:
                # No matching chunk, add to last chunk
                if chunks:
                    chunks[-1].text += text

        return chunks

    # =================================================================
    #  Phase 2: Analysis (Query -> Multi-Agent -> Report)
    # =================================================================

    def analyze(
        self,
        paper_id: str,
        query: str,
        depth: str = "full",
        strategy: str = "hybrid",
    ) -> dict:
        """
        对已导入的论文执行 Multi-Agent 分析。

        Args:
            paper_id: 论文标识
            query: 分析查询
            depth: "basic" | "advanced" | "full"
            strategy: 检索策略

        Returns:
            包含 final_report, agent_traces, retrieved_docs 的字典
        """
        run_id = str(uuid.uuid4())[:8]
        logger.info(f"[Analysis] run={run_id}, paper={paper_id}, query='{query[:60]}...'")

        # --- Step 0: Ensure embedder is fit (recover from ChromaDB) ---
        if not self.embedder.is_fitted():
            try:
                collection = self.chroma_store.client.get_collection(
                    self.chroma_store._collection_name(paper_id)
                )
                stored = collection.get(include=["documents"])
                if stored and stored.get("documents"):
                    docs = stored["documents"]
                    logger.info(f"Recovering TF-IDF from {len(docs)} ChromaDB documents")
                    self.embedder.fit(docs)
                else:
                    logger.warning("No ChromaDB documents to fit TF-IDF")
            except Exception as e:
                logger.warning(f"TF-IDF recovery failed: {e}")

        # --- Step 1: Retrieve ---
        retrieved = self.retrieval.retrieve(
            query=query, paper_id=paper_id, strategy=strategy, k=20
        )

        # --- Step 2: Build initial state ---
        state: PaperAnalysisState = {
            "paper_id": paper_id,
            "query": query,
            "analysis_depth": depth,
            "retrieval_strategy": strategy,
            "retrieved_docs": [
                {
                    "chunk_id": doc.chunk_id,
                    "text": doc.text,
                    "score": doc.score,
                    "retrieval_method": doc.retrieval_method,
                    "metadata": doc.metadata,
                }
                for doc in retrieved
            ],
            "query_history": [],
            "metadata_analysis": {},
            "deep_read_analysis": {},
            "critique_result": {},
            "synthesis_result": {},
            "code_reproduction": {},
            "agent_traces": [],
            "execution_path": [],
            "revision_count": 0,
            "current_step": "init",
            "status": "initialized",
            "errors": [],
        }

        # --- Step 3: Run the analysis ---
        if depth == "basic":
            report = self._run_basic_analysis(state)
        else:
            report = self._run_multi_agent_analysis(state)

        # --- Step 4: Save traces ---
        traces = state.get("agent_traces", [])
        logger.info(f"[Analysis] Traces collected: {len(traces)}, "
                    f"agents: {[t.get('agent_name','?') for t in traces]}")
        if traces:
            self.trace_collector.save_run(run_id, traces, {
                "paper_id": paper_id,
                "query": query,
                "depth": depth,
            })

        return {
            "run_id": run_id,
            "final_report": report,
            "agent_traces": traces,
            "retrieved_docs": state["retrieved_docs"],
            "code_reproduction": state.get("code_reproduction", {}),
        }

    def _run_basic_analysis(self, state: PaperAnalysisState) -> dict:
        """Basic mode: retrieve + single LLM answer"""
        docs = state["retrieved_docs"]
        context = "\n\n---\n\n".join(
            f"[{i+1}] {doc['text'][:500]}" for i, doc in enumerate(docs[:5])
        )

        prompt = f"""Based on the following paper excerpts, answer the question.
If the answer cannot be found, say so clearly.

Paper content:
{context}

Question: {state['query']}

Answer:"""

        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)

        return {
            "analysis_depth": "basic",
            "executive_summary": answer[:300],
            "integrated_analysis": answer,
            "key_findings": [],
            "retrieved_docs_count": len(docs),
        }

    def _run_multi_agent_analysis(self, state: PaperAnalysisState) -> dict:
        """Advanced/Full mode: Multi-Agent pipeline (runs agents sequentially)"""
        import asyncio

        async def run_agents():
            # Helper: merge agent result into state, accumulating traces
            def merge(state, r):
                trace_count = len(r.get("agent_traces", []))
                logger.info(f"  [merge] agent_traces: +{trace_count}, "
                           f"keys: {[k for k in r.keys() if k != 'agent_traces']}")
                for k, v in r.items():
                    if k in ("agent_traces", "execution_path"):
                        if k in state:
                            state[k].extend(v)
                        else:
                            state[k] = list(v)
                    else:
                        state[k] = v
                logger.info(f"  [merge] total agent_traces now: {len(state.get('agent_traces', []))}")

            # 1. Metadata
            logger.info("  [metadata] Running...")
            r = await self.metadata_agent.run(state)
            merge(state, r)

            # 2. Deep Read
            logger.info("  [deepread] Running...")
            r = await self.deepread_agent.run(state)
            merge(state, r)

            # 3. Critique
            logger.info("  [critique] Running...")
            r = await self.critique_agent.run(state)
            merge(state, r)

            # Revision loop (max 2)
            revision = 0
            while state.get("critique_result", {}).get("needs_revision") and revision < 2:
                revision += 1
                logger.info(f"  [deepread] Revision {revision}...")
                state["revision_count"] = revision
                r = await self.deepread_agent.run(state)
                merge(state, r)
                r = await self.critique_agent.run(state)
                merge(state, r)

            # 4. Synthesis
            logger.info("  [synthesis] Running...")
            r = await self.synthesis_agent.run(state)
            merge(state, r)

            # 5. Code (full mode only, if algorithm detected)
            if state.get("analysis_depth") == "full":
                dr = state.get("deep_read_analysis", {})
                if dr.get("contains_algorithm"):
                    logger.info("  [code] Running...")
                    r = await self.code_agent.run(state)
                    merge(state, r)

                    # Auto-execute generated code in sandbox
                    code_output = state.get("code_reproduction", {})
                    python_code = code_output.get("python_code", "")
                    if python_code and len(python_code) > 20:
                        logger.info("  [code] Executing in sandbox...")
                        try:
                            from src.tools.code_executor import execute_code
                            exec_result = execute_code(python_code, timeout=30)
                            code_output["execution_result"] = {
                                "success": exec_result.success,
                                "stdout": exec_result.stdout,
                                "stderr": exec_result.stderr,
                                "exit_code": exec_result.exit_code,
                                "elapsed_ms": exec_result.elapsed_ms,
                                "error_message": exec_result.error_message,
                            }
                            state["code_reproduction"] = code_output
                            logger.info(f"  [code] Sandbox done: success={exec_result.success}")
                        except Exception as e:
                            logger.warning(f"  [code] Sandbox error: {e}")

        asyncio.run(run_agents())
        return state.get("synthesis_result", {})

    # =================================================================
    #  Convenience
    # =================================================================

    def ingest_and_analyze(
        self,
        pdf_path: str | Path,
        query: str,
        depth: str = "full",
    ) -> dict:
        """一键导入+分析"""
        paper_id = self.ingest(pdf_path)
        return self.analyze(paper_id, query, depth=depth)

    def evaluate_retrieval(
        self,
        paper_id: str,
        test_queries: list[dict],
    ) -> dict:
        """
        评测检索质量。

        Args:
            paper_id: 论文标识
            test_queries: [{"query": str, "relevant_chunk_ids": [str]}]

        Returns:
            评测结果字典
        """
        from src.evaluation.metrics import RetrievalMetrics

        metrics = RetrievalMetrics()

        queries = [q["query"] for q in test_queries]
        relevant = [q["relevant_chunk_ids"] for q in test_queries]
        retrieved = []

        for query in queries:
            results = self.retrieval.retrieve(query, paper_id, k=20)
            retrieved.append([doc.chunk_id for doc in results])

        result = metrics.evaluate(queries, relevant, retrieved)

        return {
            "hit_rate_k": result.hit_rate_k,
            "mrr": result.mrr,
            "ndcg_k": result.ndcg_k,
            "recall_k": result.recall_k,
            "num_queries": result.num_queries,
        }
