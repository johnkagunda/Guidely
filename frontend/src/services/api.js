// Thin fetch wrapper around the Guidely backend API.
// In dev, Vite proxies /api -> http://localhost:8000 (see vite.config.js).

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

async function handleResponse(res) {
  let body = null;
  try {
    body = await res.json();
  } catch {
    // no JSON body (e.g. network error page)
  }
  if (!res.ok) {
    const message = body?.detail || body?.error || `Request failed (HTTP ${res.status})`;
    const error = new Error(message);
    error.status = res.status;
    throw error;
  }
  return body;
}

export async function search(query, topK) {
  const res = await fetch(`${BASE_URL}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK ?? null }),
  });
  return handleResponse(res);
}

export async function listDocuments() {
  const res = await fetch(`${BASE_URL}/documents`);
  return handleResponse(res);
}

export async function uploadDocument(file, onProgress) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BASE_URL}/documents/upload`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(res);
}

export async function deleteDocument(id) {
  const res = await fetch(`${BASE_URL}/documents/${id}`, { method: "DELETE" });
  return handleResponse(res);
}

export async function reindexDocument(id) {
  const res = await fetch(`${BASE_URL}/documents/${id}/reindex`, { method: "POST" });
  return handleResponse(res);
}

export async function getMetrics() {
  const res = await fetch(`${BASE_URL}/metrics`);
  return handleResponse(res);
}

export async function getHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  return handleResponse(res);
}
