import { useQuery } from '@tanstack/react-query'
import { buildSliceUrl } from '../api/client'
import type { SliceParams } from '../types'

async function fetchSliceBlob(params: SliceParams): Promise<string> {
  const url = buildSliceUrl(params)
  const res = await fetch(url)
  if (res.status === 404) throw Object.assign(new Error('not_found'), { status: 404 })
  if (!res.ok) throw new Error(`Slice fetch failed: ${res.status}`)
  const blob = await res.blob()
  return URL.createObjectURL(blob)
}

export function useSlice(params: SliceParams | null) {
  return useQuery({
    queryKey: ['slice', params],
    queryFn: () => fetchSliceBlob(params!),
    enabled: params !== null,
    staleTime: 1000 * 60 * 10,
  })
}
