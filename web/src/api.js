// Thin client for the seg_server API. Every call resolves to the response's `data` plus the
// server's `meta`, and the browser-measured round trip; API errors throw ApiError.

export class ApiError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function call(method, path, { body, json, signal } = {}) {
  const started = performance.now();
  const response = await fetch(path, {
    method,
    signal,
    body: json !== undefined ? JSON.stringify(json) : body,
    headers: json !== undefined ? { "Content-Type": "application/json" } : undefined,
  });
  const payload = await response.json().catch(() => null);
  const roundTripMs = performance.now() - started;
  if (!response.ok) {
    const error = payload?.error;
    throw new ApiError(response.status, error?.code ?? "http_error", error?.message ?? response.statusText);
  }
  return { data: payload.data, meta: payload.meta ?? {}, roundTripMs, status: response.status };
}

export const api = {
  upload(file, signal) {
    const form = new FormData();
    form.append("image", file);
    return call("POST", "/api/runs", { body: form, signal });
  },
  segment(runId, params, signal) {
    return call("POST", `/api/runs/${runId}/results`, { json: params, signal });
  },
  save(resultId) {
    return call("PUT", `/api/experiments/${resultId}`);
  },
  experiments(signal) {
    return call("GET", "/api/experiments?limit=100", { signal });
  },
  clearHistory() {
    return call("DELETE", "/api/experiments");
  },
};

export const imageUrl = (resultId, name) => `/api/results/${resultId}/images/${name}`;
