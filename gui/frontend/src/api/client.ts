import type { SubjectSummary, SubjectDetail, SliceParams, PerSubjectMetrics } from '../types'

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}: ${body}`)
  }
  return res.json() as Promise<T>
}

export function fetchSubjects(): Promise<SubjectSummary[]> {
  return apiFetch('/api/subjects')
}

export function fetchSubject(id: string): Promise<SubjectDetail> {
  return apiFetch(`/api/subjects/${encodeURIComponent(id)}`)
}

export function fetchMetrics(id: string): Promise<PerSubjectMetrics> {
  return apiFetch(`/api/subjects/${encodeURIComponent(id)}/metrics`)
}

export function buildSliceUrl(p: SliceParams): string {
  const params = new URLSearchParams({
    volume: p.volume,
    axis: String(p.axis),
    index: String(p.index),
    overlay: p.overlay,
    threshold: String(p.threshold),
  })
  return `/api/subjects/${encodeURIComponent(p.subjectId)}/slice?${params}`
}

export async function postSetPseudoGt(subjectId: string, sampleIndex: number): Promise<{ written_path: string; subject_id: string }> {
  const res = await fetch(`/api/subjects/${encodeURIComponent(subjectId)}/set-pseudo-gt`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sample_index: sampleIndex }),
  })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}: ${body}`)
  }
  return res.json()
}
