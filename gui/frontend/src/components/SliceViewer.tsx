import { useSlice } from '../hooks/useSlice'
import type { SliceParams } from '../types'

interface Props {
  label: string
  params: SliceParams | null
  selected?: boolean
  onSelect?: () => void
}

export function SliceViewer({ label, params, selected = false, onSelect }: Props) {
  const { data: url, isLoading, isError, error } = useSlice(params)

  const is404 = isError && (error as { status?: number })?.status === 404

  return (
    <div
      onClick={onSelect}
      className={`flex flex-col items-center gap-1 ${onSelect ? 'cursor-pointer' : ''}
        ${selected ? 'ring-2 ring-amber-400 rounded-lg' : ''}`}
    >
      <div className={`text-xs font-bold tracking-wide uppercase px-2 py-0.5 rounded
        ${selected ? 'text-amber-300 bg-amber-950' : 'text-gray-400'}`}>
        {label}
        {selected && <span className="ml-1 text-amber-400">✓</span>}
      </div>
      <div className="w-44 h-44 bg-gray-900 border border-gray-700 rounded-lg overflow-hidden flex items-center justify-center">
        {isLoading && (
          <div className="w-full h-full bg-gray-800 animate-pulse" />
        )}
        {is404 && (
          <span className="text-gray-600 text-xs text-center px-2">Not available</span>
        )}
        {isError && !is404 && (
          <span className="text-red-500 text-xs text-center px-2">Load error</span>
        )}
        {url && !isLoading && (
          <img
            src={url}
            alt={label}
            className="w-full h-full object-contain"
            draggable={false}
          />
        )}
      </div>
    </div>
  )
}
