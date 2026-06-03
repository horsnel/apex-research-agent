/**
 * Cloudflare Pages Function — Proxy /research/* to APEX API
 *
 * This runs at the edge on kovira.pages.dev
 * All requests to kovira.pages.dev/research/* are proxied
 * to the APEX Research Agent API.
 *
 * Setup:
 * 1. Deploy this to your Cloudflare Pages project (kovira.pages.dev)
 * 2. Set APEX_API_URL environment variable in Cloudflare Pages settings
 *    e.g. https://apex-research.up.railway.app
 * 3. Optionally set APEX_API_KEY for auth
 */

const APEX_API_URL = (typeof APEX_API_URL_ENV !== 'undefined')
  ? APEX_API_URL_ENV
  : 'https://apex-research.up.railway.app';

const APEX_API_KEY = (typeof APEX_API_KEY_ENV !== 'undefined')
  ? APEX_API_KEY_ENV
  : '';

export async function onRequest(context) {
  const { request, env } = context;

  // Get the APEX API URL from env vars (set in CF Pages dashboard)
  const apiUrl = env?.APEX_API_URL || APEX_API_URL;
  const apiKey = env?.APEX_API_KEY || APEX_API_KEY;

  // Parse the incoming request
  const url = new URL(request.url);
  const targetPath = url.pathname.replace(/^\/research/, '') || '/';
  const targetUrl = `${apiUrl}${targetPath}${url.search}`;

  // Build proxied request headers
  const headers = new Headers(request.headers);
  headers.set('Origin', apiUrl);
  headers.delete('host');
  if (apiKey) {
    headers.set('Authorization', `Bearer ${apiKey}`);
  }

  try {
    const proxyResponse = await fetch(targetUrl, {
      method: request.method,
      headers,
      body: ['GET', 'HEAD'].includes(request.method) ? undefined : request.body,
    });

    // Build response with CORS headers
    const response = new Response(proxyResponse.body, {
      status: proxyResponse.status,
      statusText: proxyResponse.statusText,
      headers: proxyResponse.headers,
    });

    response.headers.set('Access-Control-Allow-Origin', 'https://kovira.pages.dev');
    response.headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    response.headers.set('X-Proxied-By', 'kovira-pages-proxy');

    return response;
  } catch (err) {
    return new Response(JSON.stringify({
      error: 'APEX API unavailable',
      message: err.message,
      target: targetUrl,
    }), {
      status: 502,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': 'https://kovira.pages.dev',
      },
    });
  }
}

// Handle CORS preflight
export async function onRequestOptions(context) {
  return new Response(null, {
    headers: {
      'Access-Control-Allow-Origin': 'https://kovira.pages.dev',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400',
    },
  });
}
