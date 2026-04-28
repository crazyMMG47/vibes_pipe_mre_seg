interface Props {
  axis: number
  onAxisChange: (a: number) => void
  index: number
  maxIndex: number
  onIndexChange: (i: number) => void
  overlay: 'gt' | 'none'
  onOverlayChange: (o: 'gt' | 'none') => void
}

const AXES = [
  { label: 'Axial', value: 2 },
  { label: 'Sagittal', value: 0 },
  { label: 'Coronal', value: 1 },
]

export function SliceScrubber({ axis, onAxisChange, index, maxIndex, onIndexChange, overlay, onOverlayChange }: Props) {
  return (
    <div className="flex items-center gap-4 bg-gray-900 border border-gray-800 rounded-xl px-4 py-2">
      {/* Axis toggle */}
      <div className="flex gap-1">
        {AXES.map(a => (
          <button
            key={a.value}
            onClick={() => { onAxisChange(a.value); onIndexChange(-1) }}
            className={`text-xs px-2 py-1 rounded transition-colors
              ${axis === a.value
                ? 'bg-indigo-600 text-white font-bold'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'}`}
          >
            {a.label}
          </button>
        ))}
      </div>

      <div className="h-4 border-l border-gray-700" />

      {/* Slice slider */}
      <div className="flex items-center gap-2 flex-1 min-w-40">
        <span className="text-xs text-gray-500 whitespace-nowrap">Slice</span>
        <input
          type="range"
          min={0}
          max={Math.max(maxIndex, 0)}
          value={index < 0 ? Math.floor(maxIndex / 2) : index}
          onChange={e => onIndexChange(Number(e.target.value))}
          className="flex-1 accent-indigo-500"
        />
        <span className="text-xs font-mono text-gray-300 w-8 text-right">
          {index < 0 ? Math.floor(maxIndex / 2) : index}
        </span>
      </div>

      <div className="h-4 border-l border-gray-700" />

      {/* GT overlay toggle */}
      <button
        onClick={() => onOverlayChange(overlay === 'gt' ? 'none' : 'gt')}
        className={`text-xs px-2 py-1 rounded transition-colors
          ${overlay === 'gt'
            ? 'bg-violet-700 text-white font-bold'
            : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'}`}
      >
        GT Contour {overlay === 'gt' ? 'ON' : 'OFF'}
      </button>
    </div>
  )
}
