#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# APEX Research Agent — Quick Deploy Script
#
# Options:
#   1. Railway (recommended) — Docker + PostgreSQL + pgvector
#   2. Fly.io — Docker + PostgreSQL
#   3. Local + Cloudflare Tunnel — No cloud account needed
#
# After deployment, your API URL will be something like:
#   Railway: https://apex-research.up.railway.app
#   Fly.io:  https://apex-research.fly.dev
#   Tunnel:  https://apex.your-tunnel-name.cfargotunnel.com
#
# Then set APEX_API_URL in your Cloudflare Pages dashboard
# so kovira.pages.dev/research proxies to it.
# ──────────────────────────────────────────────────────────────

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     APEX Research Agent — Deploy to Cloud       ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Check prerequisites ──
check_cli() {
    if command -v "$1" &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} $1 found"
        return 0
    else
        echo -e "  ${RED}✗${NC} $1 not found"
        return 1
    fi
}

echo -e "${YELLOW}Checking prerequisites...${NC}"
check_cli docker
check_cli git

echo ""
echo "Choose deployment target:"
echo "  1) Railway (recommended — free $5 credit, PostgreSQL + pgvector)"
echo "  2) Fly.io (free tier, Docker + PostgreSQL)"
echo "  3) Local + Cloudflare Tunnel (no cloud account needed)"
echo "  4) Just show me the API endpoint map (skip deploy)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo -e "${CYAN}═══ Railway Deployment ═══${NC}"
        echo ""

        if ! command -v railway &> /dev/null; then
            echo "Installing Railway CLI..."
            npm install -g @railway/cli
        fi

        echo "Step 1: Login to Railway..."
        railway login

        echo "Step 2: Initialize project..."
        railway init

        echo "Step 3: Add PostgreSQL database..."
        railway add --database postgres

        echo "Step 4: Set environment variables..."
        # Load from .env file
        if [ -f .env ]; then
            echo "Loading API keys from .env..."
            while IFS='=' read -r key value; do
                if [[ ! -z "$key" && ! "$key" =~ ^# ]]; then
                    railway variables set "${key}=${value}" 2>/dev/null || true
                fi
            done < .env
        fi

        echo "Step 5: Deploy..."
        railway up

        echo ""
        echo -e "${GREEN}✓ Deployed!${NC}"
        echo ""
        echo "Your API URL:"
        railway domain
        echo ""
        echo "Set this as APEX_API_URL in your Cloudflare Pages dashboard"
        echo "(kovira.pages.dev → Settings → Environment variables)"
        ;;

    2)
        echo ""
        echo -e "${CYAN}═══ Fly.io Deployment ═══${NC}"
        echo ""

        if ! command -v fly &> /dev/null; then
            echo "Installing Fly CLI..."
            curl -L https://fly.io/install.sh | sh
        fi

        echo "Step 1: Login to Fly.io..."
        fly auth login

        echo "Step 2: Launch app..."
        fly launch --dockerfile Dockerfile --name apex-research

        echo "Step 3: Add PostgreSQL..."
        fly postgres create
        fly postgres attach apex-research-db

        echo "Step 4: Set secrets (environment variables)..."
        if [ -f .env ]; then
            echo "Loading API keys from .env..."
            while IFS='=' read -r key value; do
                if [[ ! -z "$key" && ! "$key" =~ ^# ]]; then
                    fly secrets set "${key}=${value}" 2>/dev/null || true
                fi
            done < .env
        fi

        echo "Step 5: Deploy..."
        fly deploy

        echo ""
        echo -e "${GREEN}✓ Deployed!${NC}"
        echo ""
        echo "Your API URL: https://apex-research.fly.dev"
        echo ""
        echo "Set this as APEX_API_URL in your Cloudflare Pages dashboard"
        ;;

    3)
        echo ""
        echo -e "${CYAN}═══ Cloudflare Tunnel Deployment ═══${NC}"
        echo ""

        if ! command -v cloudflared &> /dev/null; then
            echo "Installing cloudflared..."
            curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
            chmod +x /usr/local/bin/cloudflared
        fi

        echo "Step 1: Start local API (Docker Compose)..."
        echo "  docker-compose up -d"
        echo ""

        echo "Step 2: Create tunnel..."
        cloudflared tunnel create apex-research

        echo "Step 3: Configure tunnel to point localhost:8000..."
        cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: apex-research
credentials-file: /root/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: apex.your-domain.com
    service: http://localhost:8000
  - service: http_status:404
EOF

        echo ""
        echo "Step 4: Route DNS..."
        echo "  cloudflared tunnel route dns apex-research apex.your-domain.com"
        echo ""
        echo "Step 5: Run tunnel..."
        echo "  cloudflared tunnel run apex-research"
        ;;

    4)
        echo ""
        echo -e "${CYAN}═══ APEX API Endpoint Map ═══${NC}"
        echo ""
        echo "Base URL depends on your deployment:"
        echo "  Railway:  https://apex-research.up.railway.app"
        echo "  Fly.io:   https://apex-research.fly.dev"
        echo "  Tunnel:   https://apex.your-domain.com"
        echo ""
        echo "─────────────────────────────────────────────"
        echo "ENDPOINTS FOR kovira.pages.dev/research:"
        echo "─────────────────────────────────────────────"
        echo ""
        echo "  POST /query              Main research query"
        echo "    Body: { query, force_live, tier_filter, depth }"
        echo ""
        echo "  POST /research           Deep research with report"
        echo "    Body: { query, classification, depth, verify }"
        echo ""
        echo "  POST /verify             Claim verification"
        echo "    Body: { query }"
        echo ""
        echo "  POST /classify           Query classification"
        echo "    Body: { query }"
        echo ""
        echo "  POST /search             Direct corpus search"
        echo "    Body: { query, domain, top_k }"
        echo ""
        echo "  POST /scrape             Live web scrape"
        echo "    Body: { query }"
        echo ""
        echo "  GET  /health             Health check"
        echo "  GET  /research/status    Engine status + upgrade info"
        echo ""
        echo "─────────────────────────────────────────────"
        echo "CLOUDFLARE PAGES PROXY SETUP:"
        echo "─────────────────────────────────────────────"
        echo ""
        echo "1. Copy cloudflare-pages/ to your kovira project root"
        echo "2. Set env var in CF Pages dashboard:"
        echo "   APEX_API_URL = https://your-deployed-api-url"
        echo "3. Deploy — kovira.pages.dev/research/* → APEX API"
        echo ""
        echo "Example fetch from kovira.pages.dev:"
        echo ""
        echo '  const res = await fetch("/research/query", {'
        echo '    method: "POST",'
        echo '    headers: { "Content-Type": "application/json" },'
        echo '    body: JSON.stringify({ query: "What is RAG?" })'
        echo '  });'
        echo '  const data = await res.json();'
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
