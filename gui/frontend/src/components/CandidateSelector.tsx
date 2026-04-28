import { useState } from 'react'
import { postSetPseudoGt } from '../api/client'

interface Props {
  subjectId: string
  selectedSample: number | null
  onExportSuccess: (path: string) => void
  onExportError: (msg: string) => void
}

export function CandidateSelector({ subjectId, selectedSample, onExportSuccess, onExportError }: Props) {
  const [loading, setLoading] = useState(false)

  async function handleExport() {
    if (selectedSample === null) return
    setLoading(true)
    try {
      const result = await postSetPseudoGt(subjectId, selectedSample)
      onExportSuccess(result.written_path)
    } catch (e) {
      onExportError(e instanceof Error ? e.message : 'Export failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex items-center gap-3">
      <span className="text-sm text-gray-400">
        {selectedSample !== null
          ? <><span className="text-amber-300 font-bold">Sample {selectedSample}</span> selected</>
          : 'Click a sample column or bar to select'}
      </span>
      <button
        onClick={handleExport}
        disabled={selectedSample === null || loading}
        className={`px-4 py-1.5 rounded-lg text-sm font-bold transition-all
          ${selectedSample !== null && !loading
            ? 'bg-emerald-600 hover:bg-emerald-500 text-white cursor-pointer'
            : 'bg-gray-800 text-gray-600 cursor-not-allowed'}`}
      >
        {loading ? (
          <span className="flex items-center gap-2">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z" />
            </svg>
            Exporting…
          </span>
        ) : 'Export as Pseudo-GT'}
      </button>
    </div>
  )
}
