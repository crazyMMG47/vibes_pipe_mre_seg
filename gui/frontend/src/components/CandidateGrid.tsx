import { SliceViewer } from './SliceViewer'
import type { SliceParams } from '../types'

interface Props {
  subjectId: string
  nSamples: number
  axis: number
  index: number
  overlay: 'gt' | 'none'
  selectedSample: number | null
  onSelectSample: (k: number) => void
}

export function CandidateGrid({ subjectId, nSamples, axis, index, overlay, selectedSample, onSelectSample }: Props) {
  const base: Omit<SliceParams, 'volume'> = {
    subjectId,
    axis,
    index,
    overlay,
    threshold: 0.5,
  }

  const columns: { label: string; volume: string; selectable?: boolean; sampleIdx?: number }[] = [
    { label: 'Input', volume: 'raw' },
    { label: 'Ground Truth', volume: 'gt' },
    ...Array.from({ length: nSamples }, (_, k) => ({
      label: `Sample ${k + 1}`,
      volume: `sample_${k}`,
      selectable: true,
      sampleIdx: k,
    })),
  ]

  return (
    <div className="overflow-x-auto">
      <div className="flex gap-3 pb-2" style={{ minWidth: 'max-content' }}>
        {columns.map(col => (
          <SliceViewer
            key={col.volume}
            label={col.label}
            params={{ ...base, volume: col.volume }}
            selected={col.selectable && col.sampleIdx === selectedSample}
            onSelect={col.selectable ? () => onSelectSample(col.sampleIdx!) : undefined}
          />
        ))}
      </div>
    </div>
  )
}
