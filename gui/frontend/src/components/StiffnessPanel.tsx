import { SliceViewer } from './SliceViewer'
import type { SliceParams } from '../types'

interface Props {
  subjectId: string
  available: boolean
  axis: number
  index: number
}

export function StiffnessPanel({ subjectId, available, axis, index }: Props) {
  if (!available) {
    return (
      <div className="flex flex-col items-center gap-1">
        <div className="text-xs font-bold tracking-wide uppercase text-gray-600 px-2 py-0.5 rounded">
          Stiffness (NLI)
        </div>
        <div className="w-44 h-44 bg-gray-900 border border-gray-800 rounded-lg flex items-center justify-center">
          <span className="text-gray-600 text-xs text-center px-3">
            Not yet available
          </span>
        </div>
      </div>
    )
  }

  const params: SliceParams = {
    subjectId,
    volume: 'stiffness',
    axis,
    index,
    overlay: 'none',
    threshold: 0.5,
  }

  return (
    <SliceViewer
      label="Stiffness (NLI)"
      params={params}
    />
  )
}
