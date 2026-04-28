import { useQuery } from '@tanstack/react-query'
import { fetchSubject } from '../api/client'

export function useSubject(id: string | null) {
  return useQuery({
    queryKey: ['subject', id],
    queryFn: () => fetchSubject(id!),
    enabled: id !== null,
  })
}
