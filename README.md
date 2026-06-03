# APEX Research Agent

Token-efficient hybrid RAG + Live Scraper research AI with 9-model Cloudflare Workers AI fallback chain.

## Architecture

```
User Query → Query Classifier → Vector DB (RAG) → [Fallback: Live Scraper] → 9-Model Synthesizer → Answer
```

**Core principle**: RAG is default. Live scrape is exception. Output is minimal. LLM calls are tiered by confidence.

## 9-Model Fallback Chain

All LLM calls route through Cloudflare Workers AI with automatic fallback:

| # | Model | Provider | Params | Price (per M tok) | Use Case |
|---|-------|----------|--------|-------------------|----------|
| 1 | Pass-through | — | — | Free | sim > 0.85, direct quote |
| 2 | Granite-4.0-Micro | Cloudflare | ~3B | $0.017 / $0.112 | Classification, simple Q&A |
| 3 | Llama-3.2-1B | Cloudflare | ~1B | $0.008 / $0.032 | Ultra-cheap fallback |
| 4 | GLM-4.7-Flash | Cloudflare | ~4.7B | $0.060 / $0.400 | Multilingual synthesis |
| 5 | Llama-3.2-3B | Cloudflare | ~3B | $0.022 / $0.089 | Better instruction following |
| 6 | Qwen3-30B-MoE | Cloudflare | 30B MoE | $0.051 / $0.335 | Table queries |
| 7 | Llama-3.1-8B | Cloudflare | ~8B | $0.075 / $0.300 | Complex synthesis |
| 8 | Mistral-Small-3.1-24B | Cloudflare | ~24B | $0.351 / $0.555 | Multi-source synthesis |
| 9 | Llama-3.3-70B | Cloudflare | ~70B | $0.650 / $1.300 | Hardest queries |

**Estimated cost: ~$4/month for 30K queries** (60% pass-through, 25% cheap, 10% mid, 5% capable)

### Tier Selection Logic

| Similarity | Models Used | Example Cost/Query |
|-----------|-------------|-------------------|
| > 0.85 | Pass-through only | $0.000 |
| 0.72 - 0.85 | Granite → Llama-1B → GLM → Llama-3B → Qwen3 | ~$0.0001 |
| < 0.72 | Full 9-model chain | ~$0.001 |
| Classification | Cheapest 3 models | ~$0.00005 |
| Table queries | Qwen3 → Llama-8B → Mistral-24B → Llama-70B | ~$0.0005 |

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env — set CLOUDFLARE_API_TOKEN (required) and OPENAI_API_KEY (for embeddings)

# 2. Start services
docker-compose up -d

# 3. Seed the database
docker-compose exec api python -m db.seed

# 4. Ingest additional content
curl -X POST http://localhost:8000/ingest/arxiv \
  -H "Content-Type: application/json" \
  -d '{"category": "cs.AI", "max_results": 25}'

# 5. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?"}'

# 6. Test all 9 LLM models
curl -X POST http://localhost:8000/router/test
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Main research query |
| POST | `/classify` | Classify query (RAG vs live) |
| POST | `/search` | Direct corpus search |
| POST | `/scrape` | Direct live scrape |
| POST | `/ingest/url` | Ingest a web URL |
| POST | `/ingest/arxiv` | Ingest arXiv papers |
| POST | `/ingest/pubmed` | Ingest PubMed papers |
| POST | `/ingest/pdf` | Ingest a PDF |
| POST | `/ingest/embed-pending` | Embed unprocessed chunks |
| GET | `/health` | Health check |
| GET | `/router/status` | LLM router status & credentials |
| GET | `/router/chain` | Full 9-model fallback chain |
| POST | `/router/test` | Test connectivity for all 9 models |
| POST | `/router/reload` | Reload model config from YAML |
| GET | `/mcp/tools` | List MCP tools |
| POST | `/mcp/call` | Execute MCP tool |

## Token Rules

| Step | Max Tokens | Action |
|------|-----------|--------|
| RAG retrieval | 2,000 | Context injected into prompt |
| Live scrape | 3,000 | Scraped markdown injected |
| Synthesis output | 300 | Hard cap. 150 default. |
| Citation | Inline | `[Author Year, P1]` or `[Source URL, P2]` |

## Source Hierarchy

| Tier | Description | Boost |
|------|-------------|-------|
| P1 | Peer-reviewed, institutional | 1.5x |
| P2 | Government, university | 1.2x |
| P3 | Community, blogs | 1.0x |
| UNV | Unverified | 0.8x |

## Configuration

### Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `CLOUDFLARE_API_TOKEN` | Yes | CF Workers AI token (auto-detects account ID) |
| `CLOUDFLARE_ACCOUNT_ID` | No | Auto-detected from token. Set to skip detection. |
| `OPENAI_API_KEY` | Yes* | For embeddings (*required for RAG) |
| `GITHUB_TOKEN` | No | GitHub Models fallback (150 req/day free) |
| `DATABASE_URL` | Yes | PostgreSQL + pgvector connection |
| `COHERE_API_KEY` | No | For search result reranking |
| `FIRECRAWL_API_KEY` | No | For live web scraping |

### YAML Model Configuration

Edit `config/llm_models.yaml` to customize the 9-model chain:
- Add/remove models
- Change fallback order
- Adjust tier selection thresholds
- Enable/disable individual models
- Add GitHub Models as bonus fallbacks (uncomment in YAML)

Changes take effect immediately via `POST /router/reload`.

## MCP Integration

The MCP server exposes two tools:

1. `search_corpus(query, domain, top_k)` — Vector + keyword search
2. `scrape_live(url, query)` — Live web scraping

Connect from Claude Desktop, Cursor, or any MCP client:

```json
{
  "mcpServers": {
    "apex-research": {
      "url": "http://localhost:8081/mcp/rpc"
    }
  }
}
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run dev server
uvicorn api.main:app --reload --port 8000

# Run MCP server
python -m tools.mcp_server
```

## Deployment

```bash
# Build and deploy
docker-compose up -d

# Weekly reingest via GitHub Actions
# (configured in .github/workflows/weekly_ingest.yml)
```

## Cloudflare Workers AI Setup

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/profile/api-tokens)
2. Create an API token with **Workers AI** permission
3. Set `CLOUDFLARE_API_TOKEN` in your `.env`
4. The account ID is auto-detected on first API call
5. Run `POST /router/test` to verify all models are reachable
