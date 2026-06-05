/**
 * Cloudflare Pages Function — Production Proxy for APEX Research Agent API
 *
 * Routes: /research/* → APEX API (strips /research prefix)
 * Example: /research/health → https://apex-api/health
 *          /research/api/v1/query?q=x → https://apex-api/api/v1/query?q=x
 *
 * Features:
 *   - Forwards ALL HTTP methods (GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD)
 *   - Strips /research prefix when forwarding
 *   - Passes query parameters through
 *   - Forwards request body for POST/PUT/PATCH
 *   - Full CORS support (preflight + response headers)
 *   - SSE streaming response pass-through (text/event-stream)
 *   - Configurable timeouts (30s default, 120s for /research & /query)
 *   - Structured JSON error responses (502 upstream, 504 timeout)
 *   - APEX_API_URL from env var with hardcoded fallback
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const FALLBACK_API_URL = 'https://apex-research-agent-production.up.railway.app';

/** Endpoints that may run long LLM pipelines — extended timeout */
const LONG_RUNNING_PATTERNS = ['/research', '/query'];

const DEFAULT_TIMEOUT_MS = 30_000;   // 30 seconds
const LONG_TIMEOUT_MS    = 120_000;  // 120 seconds

const ALLOWED_ORIGINS = [
  'https://kovira.pages.dev',
  'http://localhost:3000',
  'http://localhost:8000',
];

const CORS_METHODS  = 'GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD';
const CORS_HEADERS  = 'Content-Type, Authorization, X-Requested-With, Accept, X-Request-ID';
const MAX_AGE       = 86_400; // 24 h

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Resolve the effective origin from the request's Origin header.
 * Returns the origin only if it is in the allow-list, otherwise falls back
 * to the first allowed origin so the header is always set.
 */
function resolveOrigin(request) {
  const origin = request.headers.get('Origin') || '';
  if (ALLOWED_ORIGINS.includes(origin)) return origin;
  return ALLOWED_ORIGINS[0];
}

/**
 * Attach standard CORS response headers to a Response object.
 */
function withCORS(response, request) {
  const origin = resolveOrigin(request);
  response.headers.set('Access-Control-Allow-Origin', origin);
  response.headers.set('Access-Control-Allow-Methods', CORS_METHODS);
  response.headers.set('Access-Control-Allow-Headers', CORS_HEADERS);
  response.headers.set('Access-Control-Max-Age', String(MAX_AGE));
  response.headers.set('Access-Control-Allow-Credentials', 'true');
  response.headers.set('Vary', 'Origin');
  return response;
}

/**
 * Build a JSON error response.
 */
function errorResponse(status, code, message, target, request) {
  const body = JSON.stringify({ error: code, message, target });
  const res = new Response(body, {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
  return withCORS(res, request);
}

/**
 * Decide the timeout based on the target path.
 */
function getTimeout(path) {
  const normalized = path.replace(/\/$/, ''); // strip trailing slash
  for (const pattern of LONG_RUNNING_PATTERNS) {
    if (normalized === pattern || normalized.startsWith(pattern + '/')) {
      return LONG_TIMEOUT_MS;
    }
  }
  return DEFAULT_TIMEOUT_MS;
}

/**
 * AbortSignal that fires after `ms` milliseconds.
 */
function timeoutSignal(ms) {
  return AbortSignal.timeout(ms);
}

// ---------------------------------------------------------------------------
// Main request handler
// ---------------------------------------------------------------------------

export async function onRequest(context) {
  const { request, env } = context;

  // --- Resolve API URL & key ---
  const apiUrl = env?.APEX_API_URL || FALLBACK_API_URL;
  const apiKey = env?.APEX_API_KEY || '';

  // --- Build target URL (strip /research prefix) ---
  const url    = new URL(request.url);
  const targetPath = url.pathname.replace(/^\/research/, '') || '/';
  const targetUrl  = `${apiUrl}${targetPath}${url.search}`;

  // --- Forwarding headers ---
  const headers = new Headers(request.headers);
  headers.set('Origin', apiUrl);
  headers.delete('host');
  if (apiKey) {
    headers.set('Authorization', `Bearer ${apiKey}`);
  }

  // --- Determine timeout ---
  const timeout = getTimeout(targetPath);

  // --- Body handling ---
  const hasBody = !['GET', 'HEAD'].includes(request.method);

  try {
    const proxyResponse = await fetch(targetUrl, {
      method:  request.method,
      headers,
      body:    hasBody ? request.body : undefined,
      signal:  timeoutSignal(timeout),
      redirect: 'follow',
    });

    // --- SSE streaming detection ---
    const contentType    = proxyResponse.headers.get('Content-Type') || '';
    const isSSE          = contentType.includes('text/event-stream');

    // For SSE we must pass the body through as a stream without buffering
    // so the client receives events in real time.
    const responseBody = proxyResponse.body;

    // --- Build response ---
    const response = new Response(responseBody, {
      status:     proxyResponse.status,
      statusText: proxyResponse.statusText,
      headers:    proxyResponse.headers,
    });

    // Tag the response
    response.headers.set('X-Proxied-By', 'apex-cf-pages-proxy');

    // Ensure streaming-related headers are correct for SSE
    if (isSSE) {
      response.headers.set('Cache-Control', 'no-cache');
      response.headers.set('Connection', 'keep-alive');
      response.headers.delete('Content-Encoding'); // don't re-compress SSE
    }

    return withCORS(response, request);

  } catch (err) {
    // Distinguish timeout (504) from upstream failures (502)
    if (err.name === 'TimeoutError' || err.name === 'AbortError') {
      return errorResponse(
        504,
        'gateway_timeout',
        `APEX API did not respond within ${timeout / 1000}s (path: ${targetPath})`,
        targetUrl,
        request,
      );
    }

    return errorResponse(
      502,
      'bad_gateway',
      `APEX API unreachable: ${err.message}`,
      targetUrl,
      request,
    );
  }
}

// ---------------------------------------------------------------------------
// CORS preflight handler
// ---------------------------------------------------------------------------

export async function onRequestOptions(context) {
  const { request } = context;
  const origin = resolveOrigin(request);

  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin':      origin,
      'Access-Control-Allow-Methods':      CORS_METHODS,
      'Access-Control-Allow-Headers':      CORS_HEADERS,
      'Access-Control-Max-Age':            String(MAX_AGE),
      'Access-Control-Allow-Credentials':  'true',
      'Vary': 'Origin',
    },
  });
}
