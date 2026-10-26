import type { Analysis, Improvement } from '../types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init.headers || {}) },
    });
  } catch {
    throw new Error(`Could not reach the analyzer API at ${API_BASE}. Check that the backend is running and VITE_API_BASE_URL is correct.`);
  }
  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try {
      const body = await response.json() as { detail?: unknown };
      if (typeof body.detail === 'string') {
        detail = body.detail;
      } else if (Array.isArray(body.detail)) {
        const messages = body.detail.map((item) => {
          if (item && typeof item === 'object' && 'msg' in item && typeof item.msg === 'string') return item.msg;
          return '';
        }).filter(Boolean);
        if (messages.length) detail = messages.join('; ');
      }
    } catch {
      // Keep a safe generic message if the server didn't return JSON.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function analyzePrompt(prompt: string): Promise<Analysis> {
  return request<Analysis>('/analyze', { method: 'POST', body: JSON.stringify({ prompt }) });
}

export function improvePrompt(prompt: string, useProvider: boolean): Promise<Improvement> {
  return request<Improvement>('/improve', {
    method: 'POST',
    body: JSON.stringify({ prompt, use_provider: useProvider }),
  });
}
