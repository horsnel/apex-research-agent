"""
Seed script — populates the database with initial documents.
Run: python -m db.seed

Seeds 50-100 core documents from:
- arXiv (via API)
- Sample PDF corpus
- Pre-configured seed URLs
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from datetime import date

import asyncpg
from dotenv import load_dotenv

# Load env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://apex:apex_secret@localhost:5432/apex_db")


# ── Sample seed data for initial population ──
# These are placeholder documents for development/testing.
# In production, the ingest pipeline fills these from real sources.

SEED_DOCUMENTS = [
    {
        "source_url": "https://arxiv.org/abs/2312.10997",
        "source_tier": "P1",
        "domain": "arxiv.org",
        "doc_type": "paper",
        "published_date": date(2023, 12, 18),
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Lewis, P.", "Perez, E.", "Piktus, A."],
        "raw_text": (
            "Retrieval-Augmented Generation (RAG) combines pre-trained parametric memory with "
            "non-parametric memory for language generation. The model retrieves documents from "
            "a knowledge source (e.g., Wikipedia) and conditions on them alongside the input "
            "to generate outputs. RAG models achieve strong performance on knowledge-intensive "
            "benchmarks including Natural Questions, TriviaQA, and MS MARCO. The approach "
            "addresses the problem of hallucination by grounding generation in retrieved evidence."
        ),
        "metadata": {"arxiv_id": "2312.10997", "categories": ["cs.CL", "cs.AI"]},
    },
    {
        "source_url": "https://arxiv.org/abs/2310.06825",
        "source_tier": "P1",
        "domain": "arxiv.org",
        "doc_type": "paper",
        "published_date": date(2023, 10, 10),
        "title": "Vector Search at Scale: Optimizing IVFFlat and HNSW Indexes",
        "authors": ["Johnson, J.", "Douze, M.", "Jegou, H."],
        "raw_text": (
            "Approximate nearest neighbor search is a critical component of modern retrieval systems. "
            "This paper benchmarks IVFFlat and HNSW indexing strategies for vector similarity search "
            "at billion-scale. We find that HNSW provides superior recall at sub-millisecond latency "
            "for datasets up to 100M vectors, while IVFFlat with optimized lists counts offers better "
            "memory efficiency for larger collections. Hybrid approaches combining both methods "
            "achieve the best tradeoff between accuracy and throughput."
        ),
        "metadata": {"arxiv_id": "2310.06825", "categories": ["cs.IR"]},
    },
    {
        "source_url": "https://arxiv.org/abs/2305.14314",
        "source_tier": "P1",
        "domain": "arxiv.org",
        "doc_type": "paper",
        "published_date": date(2023, 5, 23),
        "title": "LLM Agents: Tool Use and Autonomous Reasoning",
        "authors": ["Yao, S.", "Zhao, J.", "Yu, D."],
        "raw_text": (
            "Large language models can be augmented with tool-use capabilities, enabling them to "
            "call external APIs, search the web, execute code, and interact with databases during "
            "inference. This paper surveys the landscape of LLM agent architectures, including "
            "ReAct, Toolformer, and HuggingGPT. We propose a unified framework for evaluating "
            "agent capabilities across planning, tool selection, and error recovery dimensions. "
            "Results show that structured prompting with explicit tool schemas outperforms "
            "free-form tool descriptions by 23% on multi-step reasoning tasks."
        ),
        "metadata": {"arxiv_id": "2305.14314", "categories": ["cs.AI", "cs.CL"]},
    },
    {
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/38084231/",
        "source_tier": "P1",
        "domain": "pubmed.ncbi.nlm.nih.gov",
        "doc_type": "paper",
        "published_date": date(2023, 12, 15),
        "title": "Clinical Applications of Large Language Models in Healthcare",
        "authors": ["Singhal, K.", "Azizi, S.", "Tu, T."],
        "raw_text": (
            "Large language models show significant promise for clinical applications including "
            "diagnostic reasoning, medical record summarization, and patient communication. "
            "We evaluate GPT-4 and Med-PaLM 2 on USMLE-style questions, finding that these models "
            "achieve expert-level performance on differential diagnosis tasks. However, safety "
            "concerns remain around hallucination rates in medication dosing and contraindication "
            "identification. We recommend a human-in-the-loop deployment strategy for clinical settings."
        ),
        "metadata": {"pmid": "38084231"},
    },
    {
        "source_url": "https://www.nature.com/articles/s41586-023-06600-9",
        "source_tier": "P1",
        "domain": "nature.com",
        "doc_type": "paper",
        "published_date": date(2023, 10, 4),
        "title": "Scaling Laws for Neural Language Models: Implications for Efficiency",
        "authors": ["Kaplan, J.", "McCandlish, S."],
        "raw_text": (
            "Scaling laws for neural language models describe how performance improves with increases "
            "in model size, dataset size, and compute budget. We extend previous findings by showing "
            "that the optimal allocation of compute between model size and training duration follows "
            "predictable power-law relationships. For a fixed compute budget, training a larger model "
            "for fewer steps consistently outperforms training a smaller model to convergence. These "
            "results have significant implications for the efficient design of frontier AI systems."
        ),
        "metadata": {"doi": "10.1038/s41586-023-06600-9"},
    },
    {
        "source_url": "https://www.science.org/doi/10.1126/science.adf2900",
        "source_tier": "P1",
        "domain": "science.org",
        "doc_type": "paper",
        "published_date": date(2023, 8, 18),
        "title": "Protein Structure Prediction with AI: From AlphaFold to Drug Discovery",
        "authors": ["Jumper, J.", "Hassabis, D."],
        "raw_text": (
            "AlphaFold and its successors have transformed structural biology by predicting protein "
            "structures with atomic accuracy. This review covers advances from AlphaFold2 to "
            "AlphaFold3, including predictions of protein-ligand, protein-DNA, and protein-RNA "
            "complexes. We discuss how these predictions are being used in drug discovery pipelines, "
            "enabling virtual screening against previously intractable targets. Limitations include "
            "dynamic conformational predictions and membrane protein accuracy."
        ),
        "metadata": {"doi": "10.1126/science.adf2900"},
    },
    {
        "source_url": "https://www.nih.gov/research-training/medical-research-initiatives",
        "source_tier": "P2",
        "domain": "nih.gov",
        "doc_type": "report",
        "published_date": date(2023, 6, 1),
        "title": "NIH Research Initiatives in Artificial Intelligence for Biomedical Research",
        "authors": ["NIH Office of the Director"],
        "raw_text": (
            "The National Institutes of Health has identified artificial intelligence as a strategic "
            "priority for accelerating biomedical research. Key initiatives include the Bridge2AI "
            "program, which aims to generate flagship datasets for AI modeling, and the All of Us "
            "research program's precision medicine data platform. NIH is investing $300M over five "
            "years in AI research infrastructure, with emphasis on data sharing standards, ethical "
            "AI frameworks, and workforce development."
        ),
        "metadata": {"program": "Bridge2AI"},
    },
    {
        "source_url": "https://nasa.gov/mission_pages/ai-earth-science",
        "source_tier": "P2",
        "domain": "nasa.gov",
        "doc_type": "report",
        "published_date": date(2023, 9, 15),
        "title": "NASA AI Applications in Earth Science and Climate Modeling",
        "authors": ["NASA Earth Science Division"],
        "raw_text": (
            "NASA is deploying foundation models for Earth observation data analysis, including "
            "Prithvi, a geospatial foundation model trained on Harmonized Landsat Sentinel-2 data. "
            "Applications include flood mapping, wildfire prediction, and agricultural monitoring. "
            "The model achieves state-of-the-art performance on multi-temporal classification tasks "
            "with 40% less labeled data than traditional approaches. NASA's Open Science policy "
            "ensures all model weights and datasets are publicly available."
        ),
        "metadata": {"program": "NASA Open Science"},
    },
    {
        "source_url": "https://cdc.gov/ai-strategy/2023-framework",
        "source_tier": "P2",
        "domain": "cdc.gov",
        "doc_type": "report",
        "published_date": date(2023, 3, 20),
        "title": "CDC AI Strategy Framework for Public Health Surveillance",
        "authors": ["CDC Center for Surveillance"],
        "raw_text": (
            "The CDC has released an AI strategy framework for integrating machine learning into "
            "public health surveillance systems. Priority use cases include syndromic surveillance "
            "for early outbreak detection, vaccine adverse event monitoring using NLP on VAERS data, "
            "and predictive modeling for respiratory disease seasonality. The framework establishes "
            "governance requirements including algorithmic bias audits, data privacy protections, "
            "and model validation before deployment in public health decision-making."
        ),
        "metadata": {"framework_version": "2023.1"},
    },
    {
        "source_url": "https://medium.com/@research-team/rag-optimization-techniques",
        "source_tier": "P3",
        "domain": "medium.com",
        "doc_type": "article",
        "published_date": date(2024, 1, 5),
        "title": "10 RAG Optimization Techniques That Actually Work",
        "authors": ["Research Team Blog"],
        "raw_text": (
            "This article summarizes practical techniques for optimizing RAG systems. Key strategies "
            "include: (1) query rewriting with HyDE, (2) hybrid search combining dense and sparse "
            "retrieval, (3) cross-encoder reranking, (4) metadata filtering before vector search, "
            "(5) chunk overlap tuning, (6) context window budget management, (7) citation grounding, "
            "(8) failure fallback to web search, (9) cache warming for common queries, and "
            "(10) evaluation with RAGAS metrics. Each technique is benchmarked on a standardized "
            "QA dataset showing 5-15% improvements in answer accuracy."
        ),
        "metadata": {"platform": "medium"},
    },
    {
        "source_url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "source_tier": "P3",
        "domain": "wikipedia.org",
        "doc_type": "article",
        "published_date": date(2024, 2, 1),
        "title": "Retrieval-Augmented Generation — Wikipedia",
        "authors": ["Wikipedia Contributors"],
        "raw_text": (
            "Retrieval-augmented generation (RAG) is a technique that enhances large language model "
            "responses by incorporating information retrieved from an external knowledge base. The "
            "approach was introduced by Lewis et al. in 2020. RAG addresses two key limitations of "
            "LLMs: knowledge cutoff and hallucination. By grounding responses in retrieved documents, "
            "RAG systems can provide up-to-date and verifiable information. Modern RAG implementations "
            "use dense vector retrieval with models like FAISS, Qdrant, or Pinecone for efficient "
            "similarity search over large document collections."
        ),
        "metadata": {"platform": "wikipedia"},
    },
    {
        "source_url": "https://www.anthropic.com/research/constitutional-ai",
        "source_tier": "P2",
        "domain": "anthropic.com",
        "doc_type": "article",
        "published_date": date(2023, 12, 12),
        "title": "Constitutional AI: Harmlessness from AI Feedback",
        "authors": ["Bai, Y.", "Kadavath, S.", "Kundu, S."],
        "raw_text": (
            "Constitutional AI (CAI) is an approach to training AI systems to be helpful and harmless "
            "using a set of principles (a constitution) rather than human-labeled data. The method "
            "works in two stages: first, the AI critiques its own responses using constitutional "
            "principles; second, the AI trains on its revised responses using RL from AI Feedback "
            "(RLAIF). Experiments show that CAI reduces harmful outputs by 50% compared to standard "
            "RLHF while maintaining helpfulness. The approach is scalable and reduces dependence on "
            "human annotators for safety training."
        ),
        "metadata": {"company": "Anthropic"},
    },
    {
        "source_url": "https://openreview.net/forum?id=abc123",
        "source_tier": "P1",
        "domain": "openreview.net",
        "doc_type": "paper",
        "published_date": date(2024, 1, 20),
        "title": "Efficient RAG: Chunking Strategies and Their Impact on Retrieval Quality",
        "authors": ["Chen, L.", "Wang, R.", "Patel, A."],
        "raw_text": (
            "We present a systematic study of chunking strategies for RAG systems, evaluating "
            "fixed-size, sentence-level, semantic, and recursive chunking across five QA benchmarks. "
            "Key findings: (1) Semantic chunking with 512-token target size and 20% overlap achieves "
            "the best F1 score across all benchmarks. (2) Fixed-size chunking at 256 tokens degrades "
            "performance by 12% due to context fragmentation. (3) Recursive chunking with markdown "
            "header awareness preserves document structure best for technical documents. We release "
            "our evaluation framework as an open-source toolkit."
        ),
        "metadata": {"venue": "ICLR 2024"},
    },
    {
        "source_url": "https://biorxiv.org/content/10.1101/2024.01.15.575498",
        "source_tier": "P1",
        "domain": "biorxiv.org",
        "doc_type": "paper",
        "published_date": date(2024, 1, 15),
        "title": "Foundation Models for Genomics: From Sequence to Function Prediction",
        "authors": ["Nguyen, T.", "Patel, R."],
        "raw_text": (
            "Foundation models trained on genomic sequences are enabling breakthroughs in functional "
            "genomics. We present GenomicBERT, a model pre-trained on 500B nucleotides from diverse "
            "organisms. The model achieves state-of-the-art on promoter prediction (AUROC 0.94), "
            "variant effect prediction (Spearman r=0.78), and chromatin accessibility prediction "
            "(AUROC 0.91). Transfer learning experiments show that fine-tuning on as few as 1,000 "
            "labeled examples suffices for specialist genomics tasks, making the approach viable "
            "for rare disease variant interpretation."
        ),
        "metadata": {"preprint_server": "bioRxiv"},
    },
    {
        "source_url": "https://dl.acm.org/doi/10.1145/3589334",
        "source_tier": "P1",
        "domain": "dl.acm.org",
        "doc_type": "paper",
        "published_date": date(2023, 7, 10),
        "title": "Query Classification for Mixed-Initiative Search Systems",
        "authors": ["Zhang, W.", "Liu, H."],
        "raw_text": (
            "We propose a hybrid query classification approach combining rule-based and LLM-based "
            "routing for mixed-initiative search. Rules capture temporal signals (latest, today, "
            "breaking) with 98% precision. A lightweight BERT classifier handles ambiguous queries "
            "with 91% accuracy. The combined system routes 73% of queries to cached/stored results "
            "and 27% to live retrieval, reducing average response latency by 2.3x and API costs by "
            "65%. We also introduce a confidence calibration mechanism that triggers live retrieval "
            "when cached result similarity falls below a learned threshold."
        ),
        "metadata": {"venue": "SIGIR 2023"},
    },
    {
        "source_url": "https://iee.org/publications/tech-predictions-2024",
        "source_tier": "P2",
        "domain": "ieee.org",
        "doc_type": "report",
        "published_date": date(2024, 1, 8),
        "title": "IEEE Technology Predictions 2024: AI Systems and Infrastructure",
        "authors": ["IEEE Spectrum Editorial Board"],
        "raw_text": (
            "IEEE's 2024 technology predictions highlight advances in AI systems and infrastructure. "
            "Key trends include: the shift from monolithic LLMs to modular agent architectures, "
            "the emergence of small language models (SLMs) optimized for edge deployment, advances "
            "in RAG systems for enterprise knowledge management, and the growing importance of AI "
            "safety evaluation frameworks. The report also notes the convergence of retrieval systems "
            "and generative models, with RAG becoming the default architecture for knowledge-intensive "
            "applications."
        ),
        "metadata": {"publication": "IEEE Spectrum"},
    },
    {
        "source_url": "https://www.who.int/publications/ai-health-guidelines",
        "source_tier": "P2",
        "domain": "who.int",
        "doc_type": "report",
        "published_date": date(2023, 10, 30),
        "title": "WHO Guidelines on Ethics and Governance of AI for Health",
        "authors": ["WHO Digital Health Department"],
        "raw_text": (
            "The World Health Organization has published comprehensive guidelines on the ethical "
            "deployment of AI in healthcare. Six core principles are outlined: (1) Protect autonomy, "
            "(2) Promote human well-being, (3) Ensure transparency, (4) Foster responsibility, "
            "(5) Ensure inclusiveness, (6) Promote responsive AI. The guidelines recommend mandatory "
            "clinical validation before deployment, ongoing post-market surveillance, and the "
            "establishment of national AI-in-health regulatory frameworks. Special attention is given "
            "to data privacy in cross-border health data sharing scenarios."
        ),
        "metadata": {"guideline_type": "ethics"},
    },
    {
        "source_url": "https://springer.com/article/10.1007/s10579-023-09658-x",
        "source_tier": "P1",
        "domain": "springer.com",
        "doc_type": "paper",
        "published_date": date(2023, 11, 5),
        "title": "Cross-Encoder Reranking for Multi-Stage Retrieval Pipelines",
        "authors": ["Müller, B.", "Rossi, F."],
        "raw_text": (
            "Cross-encoder reranking significantly improves retrieval quality in multi-stage pipelines. "
            "We evaluate six cross-encoder architectures on MS MARCO and TREC DL benchmarks. The "
            "distilroberta-base cross-encoder achieves the best efficiency-accuracy tradeoff, improving "
            "MRR@10 by 18% over bi-encoder only retrieval while adding only 12ms latency per query. "
            "For budget-constrained deployments, we show that reranking top-20 bi-encoder results "
            "captures 95% of the quality gain of reranking top-100 at 5x lower cost."
        ),
        "metadata": {"venue": "Language Resources and Evaluation"},
    },
    {
        "source_url": "https://arxiv.org/abs/2401.00123",
        "source_tier": "P1",
        "domain": "arxiv.org",
        "doc_type": "paper",
        "published_date": date(2024, 1, 2),
        "title": "Token-Efficient LLM Output: Compression and Citation Strategies",
        "authors": ["Kumar, A.", "Santos, M."],
        "raw_text": (
            "We study methods for reducing LLM output token count while preserving information density. "
            "Key approaches include: (1) Table-over-prose formatting reduces tokens by 40% for "
            "comparative data, (2) Inline citations replace verbose attribution, saving 25% tokens, "
            "(3) Pass-through mode for direct source quoting avoids redundant synthesis. We introduce "
            "a token budget controller that dynamically adjusts output length based on query complexity. "
            "On a 500-query evaluation set, the system achieves 150-token average output with 94% "
            "information preservation compared to unconstrained generation."
        ),
        "metadata": {"arxiv_id": "2401.00123"},
    },
    {
        "source_url": "https://wiley.com/doi/10.1002/ai.2023.456",
        "source_tier": "P1",
        "domain": "wiley.com",
        "doc_type": "paper",
        "published_date": date(2023, 8, 22),
        "title": "Semantic Chunking for Document Retrieval: Beyond Fixed-Size Windows",
        "authors": ["Park, S.", "Kim, J."],
        "raw_text": (
            "Fixed-size chunking fails to respect document structure, leading to fragmented context "
            "in retrieval. We propose Semantic Boundary Detection (SBD), which uses embedding "
            "similarity between adjacent sentences to identify natural topic boundaries. SBD achieves "
            "12% higher recall than fixed-size chunking at equivalent token budgets on the "
            "SciFact benchmark. When combined with header-aware splitting for structured documents, "
            "recall improves further to 18% over baseline. The method adds negligible latency "
            "(2ms per document) and requires no additional training data."
        ),
        "metadata": {"venue": "AI Journal"},
    },
]


async def seed_database():
    """Insert seed documents into the database."""
    logger.info(f"Connecting to database at {DATABASE_URL.split('@')[-1]}")
    try:
        conn = await asyncpg.connect(DATABASE_URL)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        logger.info("Make sure PostgreSQL is running and the database exists.")
        logger.info("Run: docker-compose up -d db")
        sys.exit(1)

    # Check schema exists
    schema_exists = await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'documents')"
    )
    if not schema_exists:
        logger.error("Schema not found. Run db/schema.sql first.")
        sys.exit(1)

    inserted = 0
    skipped = 0

    for doc in SEED_DOCUMENTS:
        # Check if already exists
        exists = await conn.fetchval(
            "SELECT 1 FROM documents WHERE source_url = $1 AND chunk_index = 0",
            doc["source_url"],
        )
        if exists:
            skipped += 1
            continue

        try:
            await conn.execute(
                """
                INSERT INTO documents (
                    source_url, source_tier, domain, doc_type,
                    published_date, title, authors, raw_text,
                    content_vector, metadata, chunk_index, total_chunks
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, $12)
                ON CONFLICT (source_url, chunk_index) DO NOTHING
                """,
                doc["source_url"],
                doc["source_tier"],
                doc["domain"],
                doc["doc_type"],
                doc["published_date"],
                doc["title"],
                doc["authors"],
                doc["raw_text"],
                None,  # content_vector will be populated by embedder
                __import__("json").dumps(doc.get("metadata", {})),
                0,
                1,
            )
            inserted += 1
            logger.info(f"  Inserted: {doc['title'][:60]}...")
        except Exception as e:
            logger.warning(f"  Failed to insert {doc['source_url']}: {e}")

    total = await conn.fetchval("SELECT COUNT(*) FROM documents")
    logger.info(f"Seed complete: {inserted} inserted, {skipped} skipped. Total docs in DB: {total}")

    await conn.close()
    logger.info("Database connection closed.")


if __name__ == "__main__":
    asyncio.run(seed_database())
