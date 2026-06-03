"""
MCP Server — Model Context Protocol server exposing APEX tools.

Exposes two tools to any MCP client (Claude, Cursor, etc.):
1. search_corpus(query, domain, top_k) — Returns chunks from vector DB
2. scrape_live(url, query) — Returns live scraped markdown

Protocol: Anthropic MCP standard (JSON-RPC over stdio/SSE)
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.retriever import retrieve, RetrievedChunk
from agent.query_classifier import classify_query
from tools.live_scraper import live_scrape, ScrapeResult
from tools.citation_validator import format_source_citation

logger = logging.getLogger(__name__)

# ── MCP Server App ──
mcp_app = FastAPI(title="APEX MCP Server", version="1.0.0")

# ── Tool Definitions (MCP schema) ──
MCP_TOOLS = [
    {
        "name": "search_corpus",
        "description": (
            "Search the APEX research corpus using hybrid vector + keyword search. "
            "Returns the most relevant document chunks with source metadata and tier ratings. "
            "Use this as the primary search tool for research queries."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — natural language question or keywords.",
                },
                "domain": {
                    "type": "string",
                    "description": "Optional domain filter (e.g., 'arxiv.org', 'nature.com').",
                    "default": None,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "scrape_live",
        "description": (
            "Scrape live web content when the corpus doesn't have the answer. "
            "Use for current events, latest news, or topics not covered in the database. "
            "Returns clean markdown with source URL and tier."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Specific URL to scrape (optional).",
                    "default": None,
                },
                "query": {
                    "type": "string",
                    "description": "Search query for finding relevant pages (used if no URL provided).",
                    "default": None,
                },
            },
            "required": [],
        },
    },
]


def _chunks_to_text(chunks: List[RetrievedChunk]) -> str:
    """Convert retrieved chunks to a readable text format."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        citation = format_source_citation(
            chunk.source_url,
            chunk.source_tier,
            chunk.title,
            chunk.authors,
        )
        parts.append(
            f"--- Result {i} {citation} ---\n"
            f"Title: {chunk.title}\n"
            f"Source: {chunk.source_url}\n"
            f"Tier: {chunk.source_tier} | Similarity: {chunk.similarity_score:.3f}\n\n"
            f"{chunk.raw_text}\n"
        )
    return "\n".join(parts)


def _scrape_results_to_text(results: List[ScrapeResult]) -> str:
    """Convert scrape results to readable text format."""
    parts = []
    for i, result in enumerate(results, 1):
        status = "SUCCESS" if result.success else "FAILED"
        parts.append(
            f"--- Live Source {i} [{status}] ---\n"
            f"URL: {result.url}\n"
            f"Title: {result.title}\n\n"
            f"{result.markdown}\n"
        )
    return "\n".join(parts)


async def handle_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool call and return the result.

    Args:
        tool_name: Name of the tool to call
        arguments: Tool arguments

    Returns:
        Dict with 'content' key containing the result
    """
    if tool_name == "search_corpus":
        query = arguments.get("query", "")
        domain = arguments.get("domain")
        top_k = min(arguments.get("top_k", 5), 10)

        chunks, avg_similarity = await retrieve(
            query=query,
            top_k=top_k,
            final_k=top_k,
            domain_filter=domain,
        )

        result_text = _chunks_to_text(chunks)
        if not chunks:
            result_text = "No relevant results found in corpus. Consider using scrape_live."

        return {
            "content": [
                {
                    "type": "text",
                    "text": result_text,
                }
            ],
            "metadata": {
                "avg_similarity": avg_similarity,
                "result_count": len(chunks),
            },
        }

    elif tool_name == "scrape_live":
        url = arguments.get("url")
        query = arguments.get("query")

        if not url and not query:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: Provide either a 'url' or 'query' parameter.",
                    }
                ],
                "isError": True,
            }

        urls = [url] if url else None
        results = await live_scrape(query=query or "", urls=urls)

        result_text = _scrape_results_to_text(results)

        return {
            "content": [
                {
                    "type": "text",
                    "text": result_text,
                }
            ],
            "metadata": {
                "scraped_count": len(results),
                "success_count": sum(1 for r in results if r.success),
            },
        }

    else:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Unknown tool: {tool_name}",
                }
            ],
            "isError": True,
        }


# ── MCP Protocol Endpoints ──

@mcp_app.get("/mcp/tools")
async def list_tools():
    """List available MCP tools."""
    return {"tools": MCP_TOOLS}


@mcp_app.post("/mcp/call")
async def call_tool(request: Request):
    """Execute an MCP tool call."""
    body = await request.json()
    tool_name = body.get("name")
    arguments = body.get("arguments", {})

    result = await handle_tool_call(tool_name, arguments)
    return JSONResponse(content=result)


# ── JSON-RPC Endpoint (stdio-compatible) ──

@mcp_app.post("/mcp/rpc")
async def json_rpc(request: Request):
    """Handle JSON-RPC requests for MCP protocol compliance."""
    body = await request.json()
    method = body.get("method")
    params = body.get("params", {})
    req_id = body.get("id")

    if method == "tools/list":
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": MCP_TOOLS},
        })

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        result = await handle_tool_call(tool_name, arguments)
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        })

    elif method == "initialize":
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "apex-research-mcp",
                    "version": "1.0.0",
                },
            },
        })

    else:
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        })


def run_mcp_server():
    """Run the MCP server."""
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8081"))

    logger.info(f"Starting APEX MCP Server on {host}:{port}")
    uvicorn.run(mcp_app, host=host, port=port)


if __name__ == "__main__":
    run_mcp_server()
