import { useQuery } from '@tanstack/react-query'
import { fetchMetrics } from '../api/client'

export function useMetrics(id: string | null) {
  return useQuery({
    queryKey: ['metrics', id],
    queryFn: () => fetchMetrics(id!),
    enabled: id !== null,
  })
}
