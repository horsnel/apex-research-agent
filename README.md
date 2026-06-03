# APEX Research Agent

Token-efficient hybrid RAG + Live Scraper research AI.

## Architecture

```
User Query → Query Classifier → Vector DB (RAG) → [Fallback: Live Scraper] → Synthesizer → Answer
```

**Core principle**: RAG is default. Live scrape is exception. Output is minimal.

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your API keys

# 2. Start services
docker-compose up -d

# 3. Seed the database (50-100 core docs)
docker-compose exec api python -m db.seed

# 4. Ingest additional content
curl -X POST http://localhost:8000/ingest/arxiv \
  -H "Content-Type: application/json" \
  -d '{"category": "cs.AI", "max_results": 25}'

# 5. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?"}'
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

## Configuration

All config via `.env` file (see `.env.example`):

- `DATABASE_URL` — PostgreSQL connection string
- `OPENAI_API_KEY` — For embeddings + LLM
- `ANTHROPIC_API_KEY` — For Claude synthesis
- `FIRECRAWL_API_KEY` — For live scraping
- `COHERE_API_KEY` — For reranking

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
