/**
 * APEX Research Agent — LLM Router
 * 9-model fallback chain using Workers AI binding
 *
 * Fallback order (cheapest → most capable):
 *   1. Pass-through (no LLM, similarity > 0.85)
 *   2. Granite-4.0-Micro   @cf/ibm-granite/granite-4.0-h-micro
 *   3. Llama-3.2-1B        @cf/meta/llama-3.2-1b-instruct
 *   4. GLM-4.7-Flash       @cf/zai-org/glm-4.7-flash
 *   5. Llama-3.2-3B        @cf/meta/llama-3.2-3b-instruct
 *   6. Qwen3-30B-MoE       @cf/qwen/qwen3-30b-a3b-fp8
 *   7. Llama-3.1-8B        @cf/meta/llama-3.1-8b-instruct-fp8
 *   8. Mistral-Small-24B   @cf/mistralai/mistral-small-3.1-24b-instruct
 *   9. Llama-3.3-70B       @cf/meta/llama-3.3-70b-instruct-fp8-fast
 */

import { Env, ModelConfig, LLMCallResult, RouterResult, RouterAttempt, ModelTestResult, Provider } from './types';

// ── Fallback Chain ──

export const FALLBACK_CHAIN: ModelConfig[] = [
  {
    name: 'pass-through', provider: 'passthrough', modelId: 'none',
    contextWindow: 0, maxOutputTokens: 150, priceInputPerM: 0, priceOutputPerM: 0,
    supportsTools: false, tier: 'free', enabled: true,
    description: 'Direct source quote when similarity > 0.85. Zero LLM cost.',
  },
  {
    name: 'Granite-4.0-Micro', provider: 'cloudflare',
    modelId: '@cf/ibm-granite/granite-4.0-h-micro',
    contextWindow: 131000, maxOutputTokens: 150, priceInputPerM: 0.017, priceOutputPerM: 0.112,
    supportsTools: true, tier: 'cheap', enabled: true,
    description: 'Cheapest LLM tier. Good for classification and simple Q&A.',
  },
  {
    name: 'Llama-3.2-1B', provider: 'cloudflare',
    modelId: '@cf/meta/llama-3.2-1b-instruct',
    contextWindow: 131000, maxOutputTokens: 150, priceInputPerM: 0.008, priceOutputPerM: 0.032,
    supportsTools: false, tier: 'cheap', enabled: true,
    description: 'Ultra-cheap fallback. 1B params, fastest inference on CF.',
  },
  {
    name: 'GLM-4.7-Flash', provider: 'cloudflare',
    modelId: '@cf/zai-org/glm-4.7-flash',
    contextWindow: 131072, maxOutputTokens: 150, priceInputPerM: 0.060, priceOutputPerM: 0.400,
    supportsTools: true, tier: 'cheap', enabled: true,
    description: 'Strong multilingual model. Excellent for synthesis.',
  },
  {
    name: 'Llama-3.2-3B', provider: 'cloudflare',
    modelId: '@cf/meta/llama-3.2-3b-instruct',
    contextWindow: 131000, maxOutputTokens: 200, priceInputPerM: 0.022, priceOutputPerM: 0.089,
    supportsTools: false, tier: 'cheap', enabled: true,
    description: 'Step up from 1B. Better instruction following.',
  },
  {
    name: 'Qwen3-30B-MoE', provider: 'cloudflare',
    modelId: '@cf/qwen/qwen3-30b-a3b-fp8',
    contextWindow: 32768, maxOutputTokens: 4096, priceInputPerM: 0.051, priceOutputPerM: 0.335,
    supportsTools: true, tier: 'mid', enabled: true,
    description: '30B MoE with 3B active. Great quality/cost for tables.',
  },
  {
    name: 'Llama-3.1-8B', provider: 'cloudflare',
    modelId: '@cf/meta/llama-3.1-8b-instruct-fp8',
    contextWindow: 131000, maxOutputTokens: 4096, priceInputPerM: 0.075, priceOutputPerM: 0.300,
    supportsTools: true, tier: 'mid', enabled: true,
    description: '8B dense model. Reliable for complex synthesis.',
  },
  {
    name: 'Mistral-Small-3.1-24B', provider: 'cloudflare',
    modelId: '@cf/mistralai/mistral-small-3.1-24b-instruct',
    contextWindow: 128000, maxOutputTokens: 4096, priceInputPerM: 0.351, priceOutputPerM: 0.555,
    supportsTools: true, tier: 'mid', enabled: true,
    description: '24B model. Best for complex multi-source synthesis.',
  },
  {
    name: 'Llama-3.3-70B', provider: 'cloudflare',
    modelId: '@cf/meta/llama-3.3-70b-instruct-fp8-fast',
    contextWindow: 131000, maxOutputTokens: 8192, priceInputPerM: 0.650, priceOutputPerM: 1.300,
    supportsTools: true, tier: 'capable', enabled: true,
    description: '70B flagship. Final fallback. Highest quality on CF.',
  },
];

// ── Tier Selection ──

export function selectTier(
  similarity?: number,
  tableNeeded?: boolean,
  isClassification?: boolean,
  forceModel?: string,
): ModelConfig[] {
  const available = FALLBACK_CHAIN.filter(m => m.enabled);

  if (forceModel) {
    const found = available.filter(m => m.name === forceModel);
    return found.length > 0 ? found : FALLBACK_CHAIN.slice(0, 1);
  }

  if (isClassification) {
    const classModels = available.filter(m => m.tier === 'cheap' || m.tier === 'free');
    return classModels.filter(m => m.provider !== 'passthrough').slice(0, 3);
  }

  if (similarity !== undefined && similarity > 0.85) {
    return [FALLBACK_CHAIN[0]];
  }

  if (tableNeeded) {
    const tableModels = available.filter(m => m.tier === 'mid' || m.tier === 'capable');
    return tableModels.length > 0 ? tableModels : available.slice(5);
  }

  if (similarity !== undefined && similarity < 0.72) {
    return available;
  }

  // Default: cheap + mid models
  const defaultModels = available.filter(m => m.tier === 'cheap' || m.tier === 'mid' || m.tier === 'free');
  return defaultModels.length > 0 ? defaultModels : available.slice(1, 6);
}

// ── Workers AI Call ──

async function callWorkersAI(
  env: Env,
  modelId: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number,
  temperature = 0.0,
): Promise<LLMCallResult> {
  const start = Date.now();

  try {
    const response = await env.AI.run(modelId, {
      messages,
      max_tokens: maxTokens,
      temperature,
    }) as { response?: string; choices?: Array<{ message?: { content?: string } }> };

    const latencyMs = Date.now() - start;

    // Workers AI returns different formats depending on model
    let content = '';
    if (response) {
      // Some models return { response: "..." }
      content = (response as unknown as string || '').toString().trim();
    } else if ((response as any).choices?.[0]?.message?.content) {
      content = ((response as any).choices[0].message.content || '').trim();
    }

    // Handle None/empty content (e.g. Qwen3 MoE reasoning mode)
    if (!content) {
      return {
        success: false, content: '', modelName: modelId, modelId,
        provider: 'cloudflare', latencyMs, tokensUsed: 0,
        error: 'Empty response (model returned None/empty content)',
      };
    }

    return {
      success: true, content, modelName: modelId, modelId,
      provider: 'cloudflare', latencyMs, tokensUsed: estimateTokens(content),
    };
  } catch (err: any) {
    const latencyMs = Date.now() - start;
    return {
      success: false, content: '', modelName: modelId, modelId,
      provider: 'cloudflare', latencyMs, tokensUsed: 0,
      error: err.message || String(err),
    };
  }
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

// ── Main Router ──

export async function routeLLMCall(
  env: Env,
  messages: Array<{ role: string; content: string }>,
  maxTokens = 150,
  temperature = 0.0,
  similarity?: number,
  tableNeeded?: boolean,
  isClassification?: boolean,
  forceModel?: string,
): Promise<RouterResult> {
  const startTime = Date.now();
  const modelsToTry = selectTier(similarity, tableNeeded, isClassification, forceModel);
  const attempts: RouterAttempt[] = [];
  let fallbackCount = 0;

  for (const model of modelsToTry) {
    // Skip pass-through (handled upstream)
    if (model.provider === 'passthrough') {
      attempts.push({
        model: model.name, provider: 'passthrough',
        status: 'skipped', reason: 'Pass-through handled upstream',
      });
      fallbackCount++;
      continue;
    }

    // Only Cloudflare models in this Worker
    const modelMax = Math.min(maxTokens, model.maxOutputTokens);

    const result = await callWorkersAI(env, model.modelId, messages, modelMax, temperature);

    attempts.push({
      model: model.name,
      modelId: model.modelId,
      provider: model.provider,
      status: result.success ? 'success' : 'failed',
      latencyMs: result.latencyMs,
      tokensUsed: result.tokensUsed,
      error: result.success ? null : result.error,
    });

    if (result.success && result.content) {
      const totalLatencyMs = Date.now() - startTime;
      return {
        content: result.content,
        modelName: model.name,
        modelId: model.modelId,
        provider: model.provider,
        fallbackCount,
        totalLatencyMs,
        attempts,
      };
    }

    fallbackCount++;
  }

  // All models failed
  const totalLatencyMs = Date.now() - startTime;
  return {
    content: '[ALL_LLM_FAILED] No model could generate a response.',
    modelName: 'none', modelId: 'none', provider: 'none',
    fallbackCount, totalLatencyMs, attempts,
  };
}

// ── Convenience Functions ──

export async function classifyWithRouter(
  env: Env, query: string,
): Promise<{ route: string; reason: string; domain_hint: string; model_used?: string }> {
  const systemPrompt = `You are a query router for a research AI. Classify the query as needing:
- "rag": if it can likely be answered from a pre-loaded academic/research corpus
- "live": if it needs current/real-time data from the web

Respond ONLY with valid JSON: {"route": "rag"|"live", "reason": "...", "domain_hint": "..."}`;

  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: query },
  ];

  const result = await routeLLMCall(env, messages, 50, 0.0, undefined, false, true);

  if (result.content.startsWith('[ALL_LLM_FAILED]')) {
    return { route: 'rag', reason: 'LLM fallback failed, defaulting to RAG', domain_hint: '' };
  }

  try {
    let content = result.content.trim();
    if (content.startsWith('```')) {
      content = content.replace(/^```(?:json)?\s*/, '').replace(/\s*```$/, '');
    }
    const parsed = JSON.parse(content);
    return {
      route: parsed.route || 'rag',
      reason: parsed.reason || 'LLM classified',
      domain_hint: parsed.domain_hint || '',
      model_used: result.modelName,
    };
  } catch {
    return { route: 'rag', reason: `LLM output not JSON: ${result.content.slice(0, 50)}`, domain_hint: '' };
  }
}

export async function synthesizeWithRouter(
  env: Env,
  query: string,
  context: string,
  maxTokens = 150,
  similarity?: number,
  tableNeeded?: boolean,
  systemPrompt = '',
): Promise<RouterResult> {
  if (!systemPrompt) {
    systemPrompt = APEX_SYSTEM_PROMPT;
  }

  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: `Query: ${query}\n\nContext:\n${context}` },
  ];

  return routeLLMCall(env, messages, maxTokens, 0.0, similarity, tableNeeded);
}

// ── APEX System Prompt ──

export const APEX_SYSTEM_PROMPT = `You are APEX, a token-efficient research synthesis AI. Your job is to compress multiple source texts into a single, dense answer.

Rules:
1. MAXIMUM INFORMATION DENSITY — every token must carry information
2. NEVER repeat the question or use filler phrases ("Based on the sources...", "According to...")
3. Use inline citations: [Source N, Tier] where N is the source number and Tier is P1/P2/P3
4. Preserve key numbers, dates, and proper nouns exactly
5. If sources conflict, note it: "Source A claims X [S1, P1], while Source B claims Y [S2, P2]"
6. Maximum 150 tokens unless asked for more
7. Use semicolons to join related facts instead of separate sentences
8. Mark speculative claims: [SPECULATIVE]
9. Mark contested claims: [CONTESTED]
10. For P3/UNV sources, add a warning: "[UNVERIFIED SOURCE]"`;

// ── Router Status ──

export function getRouterStatus(): Record<string, any> {
  return {
    total_models: FALLBACK_CHAIN.length,
    configured_models: FALLBACK_CHAIN.length, // All CF models are always configured in Worker
    models: FALLBACK_CHAIN.map(m => ({
      name: m.name,
      provider: m.provider,
      model_id: m.modelId,
      tier: m.tier,
      configured: true,
      supports_tools: m.supportsTools,
      price_input_per_m: m.priceInputPerM,
      price_output_per_m: m.priceOutputPerM,
      context_window: m.contextWindow,
      description: m.description,
    })),
    provider: 'cloudflare_workers_ai',
    note: 'All models use native Workers AI binding — no HTTP calls needed',
  };
}

// ── Test All Models ──

export async function testAllModels(env: Env): Promise<ModelTestResult[]> {
  const results: ModelTestResult[] = [];

  for (const model of FALLBACK_CHAIN) {
    if (model.provider === 'passthrough') {
      results.push({
        modelName: model.name, modelId: model.modelId, provider: 'passthrough',
        configured: true, reachable: true, latencyMs: 0,
        sampleOutput: '[Pass-through: no LLM call needed]', error: '',
      });
      continue;
    }

    const start = Date.now();
    try {
      const callResult = await callWorkersAI(
        env, model.modelId,
        [{ role: 'user', content: 'Say hello in exactly 5 words.' }],
        20,
      );
      const latencyMs = Date.now() - start;
      results.push({
        modelName: model.name, modelId: model.modelId, provider: model.provider,
        configured: true, reachable: callResult.success, latencyMs,
        error: callResult.success ? '' : callResult.error,
        sampleOutput: callResult.success ? callResult.content.slice(0, 100) : '',
      });
    } catch (err: any) {
      results.push({
        modelName: model.name, modelId: model.modelId, provider: model.provider,
        configured: true, reachable: false, latencyMs: Date.now() - start,
        error: err.message || String(err), sampleOutput: '',
      });
    }
  }

  return results;
}
