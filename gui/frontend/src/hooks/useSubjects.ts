import { useQuery } from '@tanstack/react-query'
import { fetchSubjects } from '../api/client'

export function useSubjects() {
  return useQuery({
    queryKey: ['subjects'],
    queryFn: fetchSubjects,
  })
}
